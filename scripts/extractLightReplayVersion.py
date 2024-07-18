import pandas as pd


# Define a list of patterns you want to match
partial_pattern = ['MocapAligner']  # Add more patterns as needed
exact_patterns = ['t']  # Add more column names as needed
input_csv_file_path = '../output_data/logReplay.csv'
output_csv_file_path = '../output_data/lightData.csv'

# Filter columns based on the predefined patterns
def filterColumns(dataframe, partial_pattern, exact_patterns):
    filtered_columns = []
    for col in dataframe.columns:
        if any(pattern in col for pattern in partial_pattern) or col in exact_patterns:
            filtered_columns.append(col)

    return filtered_columns

# Load the CSV files into pandas dataframes
replayData = pd.read_csv('../output_data/logReplay.csv', delimiter=';')
light_columns = filterColumns(replayData, partial_pattern, exact_patterns)
replayData_light = replayData[light_columns].copy()

#mocapData = pd.read_csv('resampledMocapData.csv', delimiter=',')

#extractedData = pd.concat([replayData_light, mocapData], axis=1)
extractedData = replayData_light

extractedData.to_csv(output_csv_file_path, index=False)
print("Output CSV file has been saved to", output_csv_file_path)
