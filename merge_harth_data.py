import pandas as pd
import os

def read_file(name):
    folder_path = './archive/merged_data/'
    df = pd.read_csv(os.path.join(folder_path, name))
    return df

def save_file(name, df):
    # Save the merged DataFrame to a new .csv file
    df.to_csv(name + '.csv', index=False)

def merge_files(folder_path):
    # Get all .csv files in the specified folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Initialize an empty DataFrame to store merged data
    merged_df = pd.DataFrame()

    for idx, file in enumerate(files):
        # Read each .csv file
        df = pd.read_csv(os.path.join(folder_path, file))
        
        # Add a new column 'user' with the file name
        df['user'] = file.split('.')[0]

        # Drop the 'time' column
        df = df.drop('timestamp', axis=1)
        
        # Append the data to the merged 
        if idx == 0:
            merged_df = df
        else: 
            merged_df = merged_df._append(df, ignore_index=True)

# Specify the folder path
folder_path = './archive/harth/'

