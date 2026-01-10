# Scene Load

## Setup the scene

Download the `parrot-ue4-carla` scene by running the following commands:

```bash
sudo apt install parrot-ue4-carla
```

More general ```sudo apt install parrot-ue4-<MAP_NAME>```

## Options to run the scene

To specify the map follow this example:   

```bash
parrot-ue4-carla -level=town_10
```

Setup the quality of the scene, follow this example:

```bash
parrot-ue4-carla -quality=low
```

Headless mode:

```bash
parrot-ue4-carla -RenderOffScreen
```

Full command to setup the scene:

```bash
parrot-ue4-carla -level=town_10 -ams-path="DefaultPath,Pickup:*"
```

Command to only spawn the Anafi AI:

```bash
sphinx "/opt/parrot-sphinx/usr/share/sphinx/drones/anafi.drone"::firmware="https://firmware.parrot.com/Versions/anafi/pc/%23latest/images/anafi-pc.ext2.zip"
```

Command to spawn the Anafi AI in the car:

```bash
sphinx "/opt/parrot-sphinx/usr/share/sphinx/drones/anafi_ai.drone"::firmware="https://firmware.parrot.com/Versions/anafi2/pc/%23latest/images/anafi2-pc.ext2.zip"::pose="Pickup"
```

Start the car: 

```
sphinx-cli param -m world actors pause false
```