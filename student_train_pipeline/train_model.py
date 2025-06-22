import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import DroneCommandRegressor
from drone_dataset import DroneCommandDataset
from tqdm import tqdm
import os
from pathlib import Path
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.min_validation_loss = float('inf')

    def __call__(self, validation_loss):
        if self.best_loss is None:
            self.best_loss = validation_loss
        elif validation_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = validation_loss
            self.counter = 0
        return self.early_stop

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    checkpoint_dir: Path,
    early_stopping: EarlyStopping,
    writer: SummaryWriter,
) -> tuple[list[float], list[float]]:
    """
    Train the model with validation and checkpointing.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        num_epochs: Number of epochs to train
        checkpoint_dir: Directory to save checkpoints
        early_stopping: Early stopping handler
        writer: TensorBoard writer
    
    Returns:
        Lists of training and validation losses
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Log model graph
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    writer.add_graph(model, dummy_input)
    
    global_step = 0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        train_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for images, labels in train_pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            # Log training metrics
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            
            # Log model parameters histograms (every 100 steps)
            if global_step % 100 == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram(f'Parameters/{name}', param.data, global_step)
                    if param.grad is not None:
                        writer.add_histogram(f'Gradients/{name}', param.grad, global_step)
            
            total_train_loss += loss.item()
            train_batches += 1
            train_pbar.set_postfix({'loss': loss.item()})
            global_step += 1
        
        avg_train_loss = total_train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
            for images, labels in val_pbar:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_val_loss += loss.item()
                val_batches += 1
                val_pbar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = total_val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # Log epoch metrics
        writer.add_scalars('Loss/epoch', {
            'train': avg_train_loss,
            'validation': avg_val_loss
        }, epoch)
        
        # Log learning rate
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = checkpoint_dir / f'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f'Saved best model checkpoint to {checkpoint_path}')
        
        # Regular checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
        
        # Early stopping check
        if early_stopping(avg_val_loss):
            print(f'Early stopping triggered after epoch {epoch + 1}')
            break
    
    writer.close()
    return train_losses, val_losses

def setup_logging(log_dir: Path):
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )

if __name__ == "__main__":
    # Setup directories
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    normalized_factors = [-24.0, 9.0, 41.0]
    run_dir = Path(f"runs/run_{timestamp}_norm_factor_{normalized_factors}")
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(run_dir)
    logging.info(f"Starting training run {timestamp}")
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(tensorboard_dir)
    print(f"TensorBoard logs will be saved to {tensorboard_dir}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets and dataloaders
    dataset_train = DroneCommandDataset(data_root="sim_ds/train")
    dataset_val = DroneCommandDataset(data_root="sim_ds/validation")
    
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=128,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    # Initialize model, criterion, optimizer
    model = DroneCommandRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    early_stopping = EarlyStopping(patience=3, min_delta=1e-4)
    
    # Log hyperparameters
    writer.add_hparams(
        {
            'learning_rate': 0.001,
            'batch_size': 16,
            'optimizer': 'Adam',
            'model_type': 'DroneCommandRegressor',
            'patience': 10,
            'min_delta': 1e-4
        },
        {}  # Metrics dict will be populated during training
    )
    
    # Log model summary
    model.display_summary()
    
    # Train the model
    train_losses, val_losses = train_model(
        model=model,
        train_loader=dataloader_train,
        val_loader=dataloader_val,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=10,  # Maximum epochs
        checkpoint_dir=checkpoint_dir,
        early_stopping=early_stopping,
        writer=writer
    )
    
    logging.info("Training completed!")