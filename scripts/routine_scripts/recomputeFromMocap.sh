#!/bin/zsh
# Re-run the pipeline from the mocap resampling onwards, non-interactively.
# Used when the ground truth itself changed (a fix in
# resampleMocapAndExtractPose.py): the replay of the estimators is unaffected
# and is deliberately NOT redone, but everything downstream of the mocap pose
# -- temporal alignment, spatial alignment, formatting and metrics -- is.
#
# Usage: scripts/routine_scripts/recomputeFromMocap.sh <project> [<project> ...]

set -e

cwd=$(cd "$(dirname "$0")/../.." && pwd)
cd $cwd
source env/bin/activate

scriptsPath="$cwd/scripts"
plotResults=false
displayLogs=false

for projectName in "$@"; do
    echo "================ $projectName ================"
    projectPath="$cwd/Projects/$projectName"
    outputDataPath="$projectPath/output_data"
    projectConfig="$projectPath/projectConfig.yaml"

    cd $scriptsPath
    echo "--- resampling the mocap signal"
    python resampleMocapAndExtractPose.py "$displayLogs" "y" "$projectPath"
    echo "--- temporal alignment"
    python crossCorrelation.py "$displayLogs" "y" "$projectPath"
    echo "--- spatial alignment"
    python matchInitPose.py 0 "$displayLogs" "y" "$projectPath"

    cd $cwd
    source $scriptsPath/routine_scripts/computeMetrics.sh
done

deactivate
