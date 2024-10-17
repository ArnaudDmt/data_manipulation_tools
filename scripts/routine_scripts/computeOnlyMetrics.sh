#!/bin/zsh

onPress=true

# Change to your project directory
cd Projects

echo "WESH"
# Get folders containing the specific file, remove './' prefix, and sort them alphabetically
folders=($(find . -type d -exec test -e '{}/output_data/observerResultsCSV.csv' \; -print | sed 's|^\./||' | grep -v '^$' | sort))

# Check if no folders found
if [[ ${#folders[@]} -eq 0 ]]; then
  echo "No folders contain the specified file."
  exit 1
fi

# Function to select folders
select_folders() {
  local selected_folders=()
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
  for folder in "${selected_folders[@]}"; do
    echo "$folder"
  done
}

# Call the function
select_folders
