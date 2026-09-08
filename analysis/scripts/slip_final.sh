#!/bin/bash
set -e
source /opt/ros/*/setup.bash 2>/dev/null || true
cd /home/arnaud/devel/src/catkin_ws
colcon build --merge-install --packages-select test_state_obs_ros2 --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
echo "ROS BUILD DONE"
cd /home/arnaud/devel/src/data_manipulation_tools/scripts
P=KO_TRO_2024_RHPS1_SLIPPAGE_1,KO_TRO_2024_RHPS1_SLIPPAGE_2,KO_TRO_2024_RHPS1_SLIPPAGE_3,KO_TRO2024_RHPS1_1,KO_TRO2024_RHPS1_3
../.venv/bin/python kinetics_eval.py --projects "$P" prepare
echo "PREPARE DONE"
../.venv/bin/python kinetics_eval.py --projects "$P" run --label slipvel --no-plots --no-open --no-latest
echo "SLIPVEL DONE"
