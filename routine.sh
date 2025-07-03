#!/bin/zsh


# current working directory
cwd=$(pwd)


set -e


if ! [ -d "env" ]; then
    sudo apt install python3.8-venv
    python3.8 -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    deactivate
fi

source env/bin/activate

# Check if yq is installed
if ! command -v yq &> /dev/null
then
    echo "yq could not be found. Please install yq to use this script."
    sudo wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq &&\
    sudo chmod +x /usr/bin/yq
    exit
fi


if [[ "$1" == "-h" ]]; then
  echo "If the results of a script seem incorrect, please run this script with the argument "--debug""
  exit 0
fi


############################ Absolute paths initialization ############################

replay_yaml="$HOME/.config/mc_rtc/controllers/Passthrough.yaml"
mc_rtc_yaml="$HOME/.config/mc_rtc/mc_rtc.yaml"
mocapPlugin_yaml="$HOME/.config/mc_rtc/plugins/MocapAligner.yaml"
mocapMarkers_yaml="markersPlacements.yaml"



############################ Variables initialization ############################

# indicates if the scripts must be run
runScript=false
# indicates if we must run in debug mode
debug=false
# indicates if we want to compute only metrics
compute_metrics_only=false
# indicates if we only want to plot results
plot_results_only=false
plotResults=false
# indicates if the script must ask for inputs once the script started. If no, the default values will be used.
runFromZero=false


############################ Reading of parameters ############################

for arg in "$@"; do
    if [[ "$arg" == "--debug" ]]; then
        debug=true
        break
    elif [[ "$arg" == "--compute-metrics" ]]; then
        compute_metrics_only=true
    elif [[ "$arg" == "--plot-results" ]]; then
        plot_results_only=true
    fi
done
displayLogs=$debug


############################ Computing only the metrics if required ############################



# Function to select folders
select_folders() {
        onPress=true

    # Change to your project directory
    cd $cwd/Projects

    if $compute_metrics_only; then
        # Get folders containing the specific file, remove './' prefix, and sort them alphabetically
        folders=($(find . -type d -exec test -e '{}/output_data/finalData.csv' \; -print | sed 's|^\./||' | grep -v '^$' | sort))
    elif $plot_results_only; then
        folders=($(find . -type d -exec test -e '{}/output_data/evals/mocap_loc_vel.pickle' \; -print | sed 's|^\./||' | grep -v '^$' | sort))
    else
        folders=(*/)
        # Remove trailing slashes
        folders=("${folders[@]%/}")
    fi
    # Check if no folders found
    if [[ ${#folders[@]} -eq 0 ]]; then
        exit 1
    fi

    selected_folders=()
    local folder_count=${#folders[@]}

    while true; do
        clear
        echo "Select folders (Type the numbers to select/unselect, 'y' to confirm. If you want to select folders above 9, please press 't' to type the number and press Enter):"
        # Display the folders with selection indication
        for ((i = 1; i <= folder_count; i++)); do
        folder="${folders[$((i))]}"
        if [[ " ${selected_folders[@]} " == *" $folder "* ]]; then
            printf "\033[31m* %d: %s\033[0m\n" "$i" "$folder"  # Selected folders in red
        else
            printf "  %d: %s\n" "$i" "$folder"  # Unselected folders
        fi
        done

        # Read user input for folder selection
        if $onPress; then
            read -rs -k1 input  # Read a single character without echoing it
        else
            read input
        fi
        # If input is 'y', confirm selection
        if [[ $input == "y" ]]; then
            break
        fi
        # If input is 'y', confirm selection
        if [[ $input == "t" ]]; then
            onPress=false
            continue
        fi
        if [[ -n $input && $input =~ ^[0-9]+$ ]]; then
            index=$input
        else
            continue
        fi

        # Toggle selection if the input corresponds to a valid index
        if (( 1 <= index && index <= folder_count )); then
        folder="${folders[$((index))]}"
        if [[ " ${selected_folders[@]} " == *" $folder "* ]]; then
            # Unselect
            selected_folders=(${(@)selected_folders:#$folder})
        else
            # Select
            selected_folders+=("$folder")
        fi
        fi
    done

    # Print final selected folders
    echo "Final selected folders:"
    echo $selected_folders
    }






createNewProject=false

# Create an array with the names of the directories
cd Projects
if [ -n "$(find . -maxdepth 1 -type d -not -path .)" ]; then
    echo "Do you want to create a new project or run an existing one?"
    select createNew in "Run existing project" "Create new project" "Run several existing projects"; do
        case $createNew in
            "Run existing project" ) 
                projectNames=(*/)
                # Remove trailing slashes
                projectNames=("${projectNames[@]%/}")

                echo "Please select a project:"
                select projectName in "${projectNames[@]}"; do
                    if [[ -n $projectName ]]; then
                        break
                    else
                        echo "Invalid selection. Please try again."
                    fi
                done
                selected_folders=$projectName
                break;;
            "Create new project" ) 
                createNewProject=true
                break;;
            "Run several existing projects" ) 
                select_folders
                runFromZero=true
                break;;
        esac
    done
else
    createNewProject=true
fi

cd $cwd

if $createNewProject; then
    echo "Please enter the name of the name of the new project: "; 
    read projectName;

    projectPath="$cwd/Projects/$projectName"
    mkdir $projectPath
    mkdir "$projectPath/raw_data"
    mkdir -p "$projectPath/output_data/scriptResults"
    mkdir "$projectPath/output_data/scriptResults/resampleAndExtractMocap"
    mkdir "$projectPath/output_data/scriptResults/crossCorrelation"
    mkdir "$projectPath/output_data/scriptResults/matchInitPose"
    
    touch "$projectPath/projectConfig.yaml"
    echo "EnabledBody: " >> "$projectPath/projectConfig.yaml"
    echo "EnabledRobot: " >> "$projectPath/projectConfig.yaml"
    if locate HartleyIEKF.so | grep install;then
        echo "Use_HartleyIEKF: " >> "$projectPath/projectConfig.yaml"
        echo -e "\n# predefined_sublengths: [1, 2, 3, 4, 5]" >> $projectPath/projectConfig.yaml

    fi
    echo "Project created. Please add the raw data of the mocap and mc_rtc's log into $projectPath/raw_data under the names mocapData.csv and controllerLog.bin, and fill in the configuration file $projectPath/projectConfig.yaml."
    exit
fi




############################ Relative paths initialization ############################


if ! $plot_results_only; then
    echo "Use the timestep defined in $mc_rtc_yaml ?"
select useMainConfRobot in "Yes" "No"; do
    case $useMainConfRobot in
        Yes ) 
            timeStep=$( grep 'Timestep:' $mc_rtc_yaml | grep -v '^#' | sed 's/Timestep: //'); break;;
        No ) 
            echo "Please enter the timestep of the controller in milliseconds: "
            read timeStep ; break;;
    esac
done
fi


RED='\033[0;31m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
RESET='\033[0m' # Reset color

if $plot_results_only; then
    source scripts/routine_scripts/plotMetrics.sh
    exit
fi

for projectName in "${selected_folders[@]}"; do
    echo -e "${YELLOW}Starting to work on project $projectName !${RESET}"
    
    projectPath="$cwd/Projects/$projectName"
    # main folders
    rawDataPath="$projectPath/raw_data"
    outputDataPath="$projectPath/output_data"
    scriptsPath="$cwd/scripts"

    # files of the resulting data after each step
    resampledMocapData="$outputDataPath/resampledMocapData.csv"
    lightData="$outputDataPath/lightData.csv"
    synchronizedObserversMocapData="$outputDataPath/synchronizedObserversMocapData.csv"

    # files of the replay
    logReplayCSV="$outputDataPath/logReplay.csv"
    logReplayBin="$outputDataPath/logReplay.bin"

    # configuration files
    projectConfig="$projectPath/projectConfig.yaml"

    if $compute_metrics_only; then
        source $scriptsPath/routine_scripts/computeMetrics.sh
    else
        source $scriptsPath/routine_scripts/mainRoutine.sh
    fi

done

# deactivate the virtual environment
deactivate

exit