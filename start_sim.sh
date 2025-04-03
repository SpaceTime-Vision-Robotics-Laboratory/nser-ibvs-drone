#!/bin/bash

# First terminal - parrot-ue4-carla
gnome-terminal -- bash -c 'parrot-ue4-carla -level=town_10 -ams-path="DefaultPath,Pickup:*" -quality=low; exec bash'

# Second terminal - sphinx
gnome-terminal -- bash -c 'sphinx "/opt/parrot-sphinx/usr/share/sphinx/drones/anafi.drone"::firmware="https://firmware.parrot.com/Versions/anafi/pc/%23latest/images/anafi-pc.ext2.zip"::pose="Pickup"; exec bash'

sleep 10


# Third terminal - sphinx-cli
gnome-terminal -- bash -c 'sphinx-cli param -m world actors pause false; exec bash'

echo "Simulation started in separate terminals"
