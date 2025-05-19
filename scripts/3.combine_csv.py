import pandas as pd
import glob
import os

def concatenate_csv_files(file_list, output_file):
    """
    Concatenates a list of CSV files into a single Pandas DataFrame and saves it to a new CSV file.

    Args:
        file_list (list): A list of file paths to the CSV files.
        output_file (str): The file path for the output CSV file.
    """
    all_df = []
    for file in file_list:
        df = pd.read_csv(file)
        all_df.append(df)
    
    combined_df = pd.concat(all_df, ignore_index=True)
    combined_df.to_csv(output_file, index=False)

# Overall evaluation results
csv_files = glob.glob("pos_results/*results_new.txt") # Get a list of all files in a directory
output_csv = "1000_pos_results_new.txt"
concatenate_csv_files(csv_files, output_csv)	

# Overall evaluation results for individual tags
#csv_files = glob.glob("pos_results/*results_inidvidual_tag.txt") # Get a list of all files in a directory
#output_csv = "1000_pos_results_individual_tag.txt"
#concatenate_csv_files(csv_files, output_csv)
