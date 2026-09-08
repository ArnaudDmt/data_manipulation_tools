#!/bin/bash
set -e
source /opt/ros/*/setup.bash 2>/dev/null || true
SP="$(dirname "$0")"
cd /home/arnaud/devel/src/catkin_ws
colcon build --merge-install --packages-select kinetics_observer_ros2 test_state_obs_ros2 \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
echo "=== ROS BUILD DONE ==="
cd /home/arnaud/devel/src/data_manipulation_tools/scripts
P=HRP5_MultiContact_1,HRP5_MultiContact_2,HRP5_MultiContact_3,HRP5_MultiContact_4
../.venv/bin/python kinetics_eval.py --projects "$P" prepare
echo "=== PREPARE DONE ==="
../.venv/bin/python kinetics_eval.py --projects "$P" run --label nfbase --no-plots --no-open --no-latest
echo "=== CONTROL DONE ==="
../.venv/bin/python kinetics_eval.py --projects "$P" --covariance-overlay "$SP/sensor_noise.yaml" \
  run --label nfix --no-plots --no-open --no-latest
echo "=== NOISEFIX DONE ==="
