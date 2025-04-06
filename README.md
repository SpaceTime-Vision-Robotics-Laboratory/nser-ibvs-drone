# Auto Follow

## How to clone

```bash
git clone --recursive https://github.com/SpaceTime-Vision-Robotics-Laboratory/auto-follow.git
```

## Setup for scene

Use the `parrot-ue4-carla` scene run the following command:

```
parrot-ue4-carla -level=town_10 -ams-path="DefaultPath,Pickup:*" -quality=low
```

Command to spawn the Anafi in the car:

```
sphinx "/opt/parrot-sphinx/usr/share/sphinx/drones/anafi.drone"::firmware="https://firmware.parrot.com/Versions/anafi/pc/%23latest/images/anafi-pc.ext2.zip"::pose="Pickup"
```

Start the car: 

```
sphinx-cli param -m world actors pause false
```