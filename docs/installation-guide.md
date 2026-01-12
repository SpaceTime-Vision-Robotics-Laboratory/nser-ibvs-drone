# Installation Guide
This guide covers the complete setup process for the Auto-Follow visual servoing framework.

## System Requirements
- **OS:** Ubuntu 22.04 or 24.04 or Debian 11 (required for Parrot Sphinx and Parrot Olympe)
- **Python:** Version 3.10+
- **GPU:** NVIDIA GPU with CUDA support (recommended for real-time inference, however can run on CPU but might require modifications in `processors` package)
- **RAM:** 8GB minimum, 32GB recommended
- **Storage:** ~30GB total (project dependencies ~9GB, bunker custom built environment ~3.5GB, rest for Parrot Sphinx)

> Note: For real-world experiments hardware requirements are far less constraining 

## Step 1: Clone the Repository
```bash
git clone --recursive https://github.com/SpaceTime-Vision-Robotics-Laboratory/auto-follow.git
cd auto-follow
```

If you already cloned without `--recursive`, initialize submodules:
```bash
git submodule update --init --recursive
```

## Step 2: Python Environment
Create and activate a virtual environment:
```bash
python3 -m venv ./venv
source venv/bin/activate
```

Install the main package and dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Step 3: Install Parrot Sphinx Simulator
Parrot Sphinx is the drone simulation platform used for experiments. 
Follow their [documentation](https://developer.parrot.com/docs/sphinx/installation.html) for instructions.

Install Sphinx empty environment to be able to run our code:
```
sudo apt install parrot-ue4-empty
```

### Configure Environment File for Firmwared Service
The firmwared service manages drone firmware. Create a `.env` file in the project root:
```bash
echo "AUTH=your-sudo-password" > .env
```

This is required by `drone-sim-runner` submodule to restart the service when needed:
```bash
sudo systemctl restart firmwared.service
```

## Step 4: Custom Bunker Environment
For experiments using the detailed bunker environment, you need the custom UE4 build.

#### Option A: Use Pre-built Environment
Contact the authors for access to the pre-built [bunker environment](https://drive.google.com/file/d/1kHqJtTq7CGoazUUn8tPFnijV3lYY4toO/view?usp=drive_link).

#### Option B: Use Empty Environment with Injection
The `parrot-ue4-empty` environment can be populated dynamically:
```bash
./scripts/start_empty_bunker_env.sh
```
However, this method won't provide the best results (or paper results).


## Verification
Run tests which verify imports and functionality:
```
python -m unittest discover ./tests
```

Check Sphinx (requires display or -RenderOffScreen):
```bash
parrot-ue4-empty -level=main -quality=low &
sphinx "/opt/parrot-sphinx/usr/share/sphinx/drones/anafi.drone"::firmware="https://firmware.parrot.com/Versions/anafi/pc/%23latest/images/anafi-pc.ext2.zip"
```

Check custom-built UE4 bunker application:
```bash
# Provide path as argument
./start_bunker_env.sh /path/to/downloaded/bunker/UnrealApp.sh

# Or set environment variable
export BUNKER_APP_PATH=/path/to/downloaded/bunker/UnrealApp.sh
./start_bunker_env.sh
```

## Troubleshooting
### "parrot-ue4-empty is required but not installed"
Ensure Parrot packages are installed and the repository was added correctly:
```bash
sudo apt update
sudo apt install parrot-ue4-empty
```

### Firmwared Service Issues
If the simulator crashes or behaves unexpectedly:
```bash
sudo systemctl restart firmwared.service
```

### Import Errors for External Packages
Ensure all submodules are installed in development mode:
```bash
pip install -e external/drone_base
pip install -e external/drone_sim_runner  
pip install -e external/mask_splitter
```