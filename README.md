# Auto Follow
[![Ubuntu](https://github.com/SpaceTime-Vision-Robotics-Laboratory/auto-follow/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/SpaceTime-Vision-Robotics-Laboratory/auto-follow/actions/workflows/ubuntu.yml)
[![Ruff Linter](https://github.com/SpaceTime-Vision-Robotics-Laboratory/auto-follow/actions/workflows/ruff_linter.yml/badge.svg)](https://github.com/SpaceTime-Vision-Robotics-Laboratory/auto-follow/actions/workflows/ruff_linter.yml)

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: AFL-3.0](https://img.shields.io/badge/License-AFL3.0-yellow.svg)](https://opensource.org/license/afl-3-0-php)

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

Command to spawn the Anafi Parrot drone:
```
sphinx "/opt/parrot-sphinx/usr/share/sphinx/drones/anafi.drone"::firmware="https://firmware.parrot.com/Versions/anafi/pc/%23latest/images/anafi-pc.ext2.zip"
```

Start the car: 

```
sphinx-cli param -m world actors pause false
```