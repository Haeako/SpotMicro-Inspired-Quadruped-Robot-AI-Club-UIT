#!/bin/bash
DEVICE_MAC="46:DE:22:04:67:DB"

echo "Turning on Bluetooth and connecting to $DEVICE_MAC..."

bluetoothctl << EOF
connect $DEVICE_MAC
EOF
echo "Bluetooth connection attempt finished."


source $HOME/catkin_ws/devel/setup.bash

roslaunch spot_micro_joy everything.launch
