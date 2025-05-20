# How to Load Custom Environment

Start the environment with the following command:
```
./installed/LinuxNoEditor/UnrealApp.sh -level=simple_car -ams-path="BP_SplineDef,Pickup:*" -quality=low
```


Start the drone in the simulator with the following command:
```
sphinx "/opt/parrot-sphinx/usr/share/sphinx/drones/anafi.drone"::firmware="https://firmware.parrot.com/Versions/anafi/pc/%23latest/images/anafi-pc.ext2.zip"::pose="Pickup"
```