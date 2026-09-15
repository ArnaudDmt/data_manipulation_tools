#!/bin/zsh
# Re-run from the spatial alignment onwards. Used when only the mocap tilt
# offset changed: resampling and temporal alignment are unaffected.
set -e
cwd=$(cd "$(dirname "$0")/../.." && pwd)
cd $cwd
source env/bin/activate
scriptsPath="$cwd/scripts"
plotResults=false
for projectName in "$@"; do
    echo "================ $projectName ================"
    projectPath="$cwd/Projects/$projectName"
    outputDataPath="$projectPath/output_data"
    projectConfig="$projectPath/projectConfig.yaml"
    cd $scriptsPath
    python matchInitPose.py 0 false "y" "$projectPath"
    cd $cwd
    source $scriptsPath/routine_scripts/computeMetrics.sh
done
deactivate
