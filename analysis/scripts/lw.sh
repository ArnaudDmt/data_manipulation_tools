#!/bin/bash
set -e
source /opt/ros/*/setup.bash 2>/dev/null || true
SP="$(dirname "$0")"
cd /home/arnaud/devel/src/catkin_ws
colcon build --merge-install --packages-select kinetics_observer_ros2 test_state_obs_ros2 \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
echo "=== ROS BUILD DONE ==="
cd /home/arnaud/devel/src/data_manipulation_tools/scripts
P=HRP5_MultiContact_1,HRP5_MultiContact_2,HRP5_MultiContact_3,HRP5_MultiContact_4,KO_TRO_2024_RHPS1_SLIPPAGE_1,KO_TRO_2024_RHPS1_SLIPPAGE_2,KO_TRO_2024_RHPS1_SLIPPAGE_3,KO_TRO2024_RHPS1_1
../.venv/bin/python kinetics_eval.py --projects "$P" prepare --force
echo "=== PREPARE DONE ==="
../.venv/bin/python kinetics_eval.py --projects "$P" --covariance-overlay "$SP/lw0.yaml" \
  run --label lw0 --no-plots --no-open --no-latest
echo "=== ALPHA 0 DONE ==="
../.venv/bin/python kinetics_eval.py --projects "$P" --covariance-overlay "$SP/lw1.yaml" \
  run --label lw1 --no-plots --no-open --no-latest
echo "=== ALPHA 1 DONE ==="
