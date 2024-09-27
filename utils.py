#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:29:53 2024

@author: anastassiakustenmacher
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.fftpack import fft
#from scipy.stats import entropy
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis, entropy


file_path = './archive/harth/S006.npy'
dataset = np.load(file_path, allow_pickle=True)
# Convert the NumPy array to a pandas DataFrame
df = pd.DataFrame(dataset[:,0], columns=['timestamp'])  # Assuming 'timestamp' is a column in the NumPy array
df['timestamp'] = pd.to_datetime(df['timestamp'])

df['time_diff'] = df['timestamp'].diff()
# Find elements where the difference is not constant
constant_difference = df['time_diff'].iloc[1]  # Get the constant difference

# Check where the difference is not equal to the constant difference
non_constant_diff = df[df['time_diff'] != constant_difference]

# Verify if all differences are the same
is_constant_difference = all(df['time_diff'].diff().isnull())

# Assuming the class labels are in a column named 'class'
# Replace 'class' with the actual name of the class label column if different
class_column_name = 'class'

# Function to filter samples of a specific class and convert to a NumPy array
def get_samples_of_class(df, class_label, class_column_name='class'):
    filtered_df = df[df[class_column_name] == class_label]
    return filtered_df.to_numpy()

def plot_allfeatures(dataset, fig_title):
    # 1. Plot the features over time
    # Set up the plot
    fig, axs = plt.subplots(6, 1, figsize=(15, 18), sharex=True)

    # Use index as x-axis
    dataset.reset_index(drop=True, inplace=True)

    # Plot each sensor data column
    axs[0].plot(dataset.index, dataset['back_x'], label='Back X', color='b')
    axs[0].set_title('Back X Acceleration over Time')
    axs[0].set_ylabel('Back X')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(dataset.index, dataset['back_y'], label='Back Y', color='g')
    axs[1].set_title('Back Y Acceleration over Time')
    axs[1].set_ylabel('Back Y')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(dataset.index, dataset['back_z'], label='Back Z', color='r')
    axs[2].set_title('Back Z Acceleration over Time')
    axs[2].set_ylabel('Back Z')
    axs[2].legend()
    axs[2].grid(True)

    axs[3].plot(dataset.index, dataset['thigh_x'], label='Thigh X', color='c')
    axs[3].set_title('Thigh X Acceleration over Time')
    axs[3].set_ylabel('Thigh X')
    axs[3].legend()
    axs[3].grid(True)

    axs[4].plot(dataset.index, dataset['thigh_y'], label='Thigh Y', color='m')
    axs[4].set_title('Thigh Y Acceleration over Time')
    axs[4].set_ylabel('Thigh Y')
    axs[4].legend()
    axs[4].grid(True)

    axs[5].plot(dataset.index, dataset['thigh_z'], label='Thigh Z', color='y')
    axs[5].set_title('Thigh Z Acceleration over Time')
    axs[5].set_ylabel('Thigh Z')
    axs[5].set_xlabel('Timestamp')
    axs[5].legend()
    axs[5].grid(True)

    # Adjust layout
    fig.suptitle(fig_title)
    plt.tight_layout()
    plt.show()
    
# Load the HARTH data from a CSV file

def load_df(single_file):
    if (single_file):
        # 1. Only take data from a signal csv-file
        # Load the dataset (replace 'path_to_your_csv_file' with the actual file path)
        file_path = './archive/harth/S006.csv'
        df = pd.read_csv(file_path)
    else: 
    
        # 2. Takes multiple files  
        # Define the path to the directory containing the CSV files
        directory_path = '/archive/harth/'
        
        # List to hold DataFrames
        df_list = []
        
        # Loop through all files in the directory
        for filename in os.listdir(directory_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(directory_path, filename)
                data = pd.read_csv(file_path)
                df_list.append(data)
        
        # Concatenate all DataFrames in the list into a single DataFrame
        df = pd.concat(df_list, ignore_index=True)
    return df
def plot_label(df, label_name):
     # Plotting the features for the selected label
     plt.figure(figsize=(12, 6))
     for column in df.columns:
         plt.plot(df[column], label=f'{column}')
         plt.title(f'Features for Label {df[column]}')
         plt.xlabel('Sample Index')
         plt.ylabel('Feature Value')
         plt.legend()
         plt.show()

def extract_statistical_features(df, columns):
    """Extract statistical features from the dataframe."""
    features = {}
    for column in columns: #df.select_dtypes(include=[np.number]).columns:
        features[f'{column}_mean'] = df[column].mean()
        features[f'{column}_std'] = df[column].std()
        #features[f'{column}_var'] = df[column].var()
        features[f'{column}_median'] = df[column].median()
        #features[f'{column}_max'] = df[column].max()
        #features[f'{column}_min'] = df[column].min()
        #features[f'{column}_range'] = df[column].max() - df[column].min()
        #features[f'{column}_skewness'] = skew(df[column]) # unwichtig wegen Feature analyse
        #features[f'{column}_kurtosis'] = kurtosis(df[column]) # unwichtig wegen Feature analyse
        features[f'{column}_iqr'] = np.percentile(df[column], 75) - np.percentile(df[column], 25)
        features[f'{column}_rms'] = np.sqrt(np.mean(df[column]**2))
        features[f'{column}_energy'] = np.sum(df[column]**2)
        #features[f'{column}_entropy'] = entropy(pd.Series(df[column]).value_counts(normalize=True), base=2)  # unwichtig wegen Feature analyse

    return features

def extract_frequency_features(df, columns):
    """Extract frequency-domain features from the dataframe."""
    features = {}
    for column in columns: #df.select_dtypes(include=[np.number]).columns:
        # Compute FFT
        fft_vals = np.abs(np.fft.fft(df[column].dropna()))
        #fft_vals = np.abs(fft(df[column].dropna()))
        fft_freqs = np.fft.fftfreq(len(fft_vals))
        
        # Filter positive frequencies
        positive_freqs = fft_freqs[:len(fft_vals)//2]
        positive_vals = fft_vals[:len(fft_vals)//2]

        features[f'{column}_fft_max'] = np.max(positive_vals)
        features[f'{column}_fft_mean'] = np.mean(positive_vals)
        features[f'{column}_fft_std'] = np.std(positive_vals)
        features[f'{column}_fft_energy'] = np.sum(positive_vals**2)
        #features[f'{column}_fft_entropy'] = entropy(positive_vals / np.sum(positive_vals))
        #features[f'{column}_fft_spectral_centroid'] = np.sum(positive_freqs * positive_vals) / np.sum(positive_vals)
        #features[f'{column}_fft_spectral_bandwidth'] = np.sqrt(np.sum(((positive_freqs - features[f'{column}_fft_spectral_centroid'])**2) * positive_vals) / np.sum(positive_vals))
        #features[f'{column}_fft_spectral_rolloff'] = np.sum(positive_vals * (positive_freqs <= np.percentile(positive_vals, 85))) / np.sum(positive_vals)

        # Peak frequencies (example: top 3 peaks)
        peak_indices = np.argsort(positive_vals)[-3:]
        for i, peak_idx in enumerate(peak_indices):
            features[f'{column}_fft_peak_{i+1}_freq'] = positive_freqs[peak_idx]
            features[f'{column}_fft_peak_{i+1}_amplitude'] = positive_vals[peak_idx]

    return features

# Function to extract features from each subsequence
def extract_features(df, columns):
    statistical_features = extract_statistical_features(df, columns)
    frequency_features = extract_frequency_features(df, columns)
    return {**statistical_features, **frequency_features}

# Features preprocessing Functions
def low_pass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Example usage:
# data: accelerometer data
# cutoff: desired cutoff frequency of the filter, e.g., 2 Hz
# fs: sampling rate, e.g., 50 Hz
#filtered_data = low_pass_filter(data, cutoff=2, fs=50)
def high_pass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Example usage:
#filtered_data = high_pass_filter(data, cutoff=0.1, fs=50)
def band_pass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Example usage:
#filtered_data = band_pass_filter(data, lowcut=0.1, highcut=2, fs=50)
def moving_average_filter(data, window_size):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
    ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    return ma_vec

# Example usage:
#filtered_data = moving_average_filter(data, window_size=5)
from scipy.signal import medfilt

def median_filter(data, kernel_size):
    filtered_data = medfilt(data, kernel_size=kernel_size)
    return filtered_data

# Example usage:
#filtered_data = median_filter(data, kernel_size=3)