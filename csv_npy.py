import numpy as np
import pandas as pd
import os

# Load the CSV files
directory_path = './archive/harth/'
csv_file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith('.csv')]#+

for csv_file_path in csv_file_paths:
    df = pd.read_csv(csv_file_path)
    data = df.to_numpy()
    npy_file_path = os.path.splitext(csv_file_path)[0] + '.npy'  # Use the same name as the CSV file but with .npy extension#+
    np.save(npy_file_path, data)