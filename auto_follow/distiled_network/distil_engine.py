import torch
import numpy as np
from torchvision import transforms
from auto_follow.distiled_network.drone_command_regressor import DroneCommandRegressor


class StudentEngine:
    def __init__(self, model_path: str, device: str | None = None, img_size: tuple[int, int] = (224, 224)):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.img_size = img_size
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.denormalize = torch.tensor([9.0, 5.0, 24.0], device=self.device)
        
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        image = torch.from_numpy(image).to(self.device).permute(2, 0, 1).float() / 255.0
        image = self.transform(image)
        return image.unsqueeze(0)
    
    
    def _load_model(self, model_path: str) -> DroneCommandRegressor:
        model = DroneCommandRegressor()
        model.load_state_dict(torch.load(model_path)["model_state_dict"])
        model.eval()
        model.to(self.device)
        return model
    
    def predict(self, image: np.ndarray) -> torch.Tensor:
        image = self.preprocess(image)
        
        with torch.inference_mode():
            output = self.model(image)
            output *= self.denormalize
            output = output.squeeze(0).cpu().numpy()
            
        return output