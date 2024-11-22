import yaml


def to_camel_case_with_letters(snake_str):
    """
    Converts a snake_case string to CamelCase and replaces numbers with letters.
    Numbers are converted to their alphabetical equivalent: 1 -> A, 2 -> B, etc.
    """
    def number_to_letter(match):
        # Convert digit to corresponding letter (1 -> A, 2 -> B, etc.)
        return chr(int(match.group()) + 64)  # 1 -> 'A' (ASCII 65), 2 -> 'B', ...

    import re
    # Replace numbers with letters using regex
    string_with_letters = re.sub(r'\d', number_to_letter, snake_str)
    
    # Split by underscores and capitalize each component
    components = string_with_letters.split('_')
    return ''.join(x.capitalize() for x in components)




def format_value(value):
    """
    Formats a float value for LaTeX, ensuring:
    1. Negative zero (-0.0000) is replaced with 0.0000 if applicable.
    2. The value is rounded to four decimal places.
    """
    formatted_value = f"{float(value):.4f}"  # Format to 4 decimal places
    # Check if the value is effectively zero (e.g., "-0.0000" or "0.0000")
    #print(formatted_value.lstrip('-').replace('.', ''))
    if all(c in '0' for c in formatted_value.lstrip('-').replace('.', '')):  # Remove minus if all numeric characters are zero
        return "0.0000"
    return formatted_value




# Load the results YAML file
with open("/tmp/absolute_errors.yaml", "r") as absolute_errors_file:
    absolute_errors = yaml.safe_load(absolute_errors_file)

with open("/tmp/relative_errors.yaml", "r") as relative_errors_file:
    relative_errors = yaml.safe_load(relative_errors_file)

with open("/tmp/velocity_errors.yaml", "r") as velocity_errors_file:
    velocity_errors = yaml.safe_load(velocity_errors_file)


scenarioName = input("Please give the name of the experimental scenario:")

# desired_abs_errors = ['abs_e_trans_x_y', 'abs_e_trans_z', 'abs_e_tilt', 'abs_e_ypr_0']
# desired_rel_errors = ['rel_trans_x_y_norm', 'rel_trans_z', 'rel_tilt', 'rel_yaw']

absErrorsFilter = {'abs_e_trans_x_y': 'transXY', 'abs_e_trans_z': 'transZ', 'abs_e_tilt': 'tilt', 'abs_e_ypr_0': 'yaw'}
relErrorsFilter = {'rel_trans_x_y_norm': 'transXY', 'rel_trans_z': 'transZ', 'rel_tilt': 'tilt', 'rel_yaw': 'yaw'}
velErrorsFilter = {'llve_norm': 'llve', 'estimate_norm': 'estimate'}


desired_subdistances = [0.5]


# Create the LaTeX variables file
with open("/tmp/metrics_results.tex", "w") as latex_file:
    for estimator, cats in velocity_errors.items():
        for cat, metrics in cats.items():
            for var_name, metrics in metrics.items():
                if(var_name == 'norm'):
                    for metric, value in metrics.items():
                        if(metric == 'mean' or metric == 'std'):
                            # Construct the variable name
                            varName = f'{cat}_{var_name}'
                            snake_case_var = f"{scenarioName}_{estimator}_velError_{velErrorsFilter[varName]}_{metric}"
                            camel_case_var = to_camel_case_with_letters(snake_case_var)
                            
                            # Ensure the value is properly formatted as a float
                            formatted_value = format_value(value)                            
                            # Write the LaTeX definition with CamelCase
                            latex_file.write(f"\\newcommand{{\\{camel_case_var}}}{{{formatted_value}}}\n")
    for subdistance, estimators in relative_errors.items():
        if(subdistance in desired_subdistances):
            for estimator, variables in estimators.items():
                for var_name, metrics in variables.items():
                    if(var_name in relErrorsFilter.keys()):
                        for metric, value in metrics.items():
                            if(metric == 'mean' or metric == 'std'):
                                # Construct the variable name
                                snake_case_var = f"{scenarioName}_{estimator}_relError_{relErrorsFilter[var_name]}_{metric}"
                                camel_case_var = to_camel_case_with_letters(snake_case_var)
                                
                                # Ensure the value is properly formatted as a float
                                formatted_value = format_value(value)
                                
                                # Write the LaTeX definition with CamelCase
                                latex_file.write(f"\\newcommand{{\\{camel_case_var}}}{{{formatted_value}}}\n")

    # Write LaTeX commands for each desired variable
    for experiment, estimators in absolute_errors.items():
        for estimator, variables in estimators.items():
            for var_name, metrics in variables.items():
                if(var_name in absErrorsFilter.keys()):
                    for metric, value in metrics.items():
                        if(metric == 'mean' or metric == 'std'):
                            # Construct the variable name
                            snake_case_var = f"{scenarioName}_{estimator}_absError_{experiment}_{absErrorsFilter[var_name]}_{metric}"
                            camel_case_var = to_camel_case_with_letters(snake_case_var)
                            
                            # Ensure the value is properly formatted as a float
                            formatted_value = format_value(value)
                            
                            # Write the LaTeX definition with CamelCase
                            latex_file.write(f"\\newcommand{{\\{camel_case_var}}}{{{formatted_value}}}\n")

    


print("LaTeX variables saved to metrics_results.tex.")