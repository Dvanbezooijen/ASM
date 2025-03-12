import configparser
import pandas as pd
import os

def enschede_read_data(list_of_names):
    # Define the base directory where the files are stored
    base_dir = "Data_Enschede"

    # List of file names (without extensions)

    # Initialize a dictionary to store the DataFrames
    dataframes = {}

    # Function to clean and convert string lists to float lists
    def clean_float_list(data_string):
        return [float(x.strip().replace('"', '')) for x in data_string.split(",")]

    # Loop through the file names and process each INI file
    for name in list_of_names:
        file_path = os.path.join(base_dir, f"{name}.ini")
        
        # Initialize parser and read the INI file
        config = configparser.ConfigParser()
        config.read(file_path, encoding="utf-8")

        try:
            # Extract and clean values
            horizontal_displacement = clean_float_list(config["Shearing Stage"]["Horizontal Displacement"])
            force = clean_float_list(config["Shearing Stage"]["Force"])
            vertical_displacement = clean_float_list(config["Shearing Stage"]["Vertical Displacement"])
            stress = [(f / 3600) * 10e3  for f in force] #kPa   *10e6 for mm2 to m2 / 10e3 to go from pa to kpa
            # Create DataFrame and store in dictionary
            dataframes[name] = pd.DataFrame({
                "Stress (kPa)": stress,
                "Horizontal Displacement (mm)": horizontal_displacement,
                "Vertical Displacement (mm)": vertical_displacement
            })
            #print(f"Loaded {name}.ini successfully!")
        except KeyError as e:
            print(f"Missing key in {name}.ini: {e}")
        except ValueError as e:
            print(f"Error converting values in {name}.ini: {e}")

    return dataframes