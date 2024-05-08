#!/bin/bash

cd /sys/class/gpio
echo 399 > export
cd /sys/class/gpio/PI.00
echo "out" > direction
