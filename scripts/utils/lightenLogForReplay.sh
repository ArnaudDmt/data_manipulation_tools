#!/bin/zsh

logPath="$1"

mc_bin_utils extract $logPath light_log --keys "t" "qIn" "JointSensor*" "ground_Default*" "qOut*" "FloatingBase_*" "Accelerometer_*" "tauIn*" "RightFootForceSensor*" "LeftFootForceSensor*" "LeftHandForceSensor*" "RightHandForceSensor*" "alphaIn*" "ff*"