# Bunker UE4 Environment

Important levels for this repository:
- `main` -> The default option when the -level= parameter is not given. It contains the bunker, carpet and, lights. The default spawn point is at (0, 0, 30)
- `main_details` -> Enhanced version of the default option. It also contains bunker details (e.g., doors, handles, sockets, exit signs etc.). The default spawn point is (-153.75, 0, 30)
- `main_center_car` → Similar to `main_details` but with an already spawned car in the center (0, 0, 12). 

To open it, use:
```bash
<PATH_TO_PROJECT_DIR>/installed/LinuxNoEditor/UnrealApp.sh -level=main_details
```

### Run environment script
You can load the car and the bunker with a simple carpet but without lights by running the 
[run_environment_car.sh](run_environment_car.sh) with default parameters. Those are the simple `parrot-ue4-empty` on 
`main` level using [config/bunker_with_car.yaml](config/bunker_with_car.yaml).

The best approach is to run with the custom build bunker environment, and load the car dynamically from 
[config/add_mc_laren.yaml](config/add_mc_laren.yaml).
```bash
./run_environment_car.sh -command=<PATH_TO_BUILD_ENV> -level=<LEVEL_NAME> -config=<YAML_FILE_NAME_WITHOUT_EXTENSION> -quality=<LOW_OR_HIGH>
```

Example of a full command, which will run the built UE4 environment on `main_details` level with a car added in the middle (0, 0, 12) with low quality setting:
```bash
./run_environment_car.sh -command=/home/brittle/Games/MyGames/DroneSimulation/bunker/installed/LinuxNoEditor/UnrealApp.sh -level=main_details -config=add_mc_laren -quality=low
```