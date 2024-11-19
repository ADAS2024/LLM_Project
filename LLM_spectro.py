import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def load_data(runs):
    df_dict = {}
    for run in runs:
        words = ["What", "Is", "A", "Spectrogram", "Ball"]
        for word in words:
            filepath = f"txt_files/{word}_{run}.txt"
            df = pd.read_csv(filepath, sep=',', header=None, 
                               names=["gyro_x", "gyro_y", "gyro_z"])
            df_dict[(word, run)] = df

    return df_dict

def parameters(df):
    df['timestamp'] = np.round(df['timestamp'] - df['timestamp'].min()).astype(int)
    N = len(df['timestamp'])
    T = np.mean(np.diff(df['timestamp']))  
    fs = 1 / T if T > 0 else 100  
    return N, T, fs, df['timestamp']

def spectro_analysis(data_dict):
    figures_path = "spectrograms"
    os.makedirs(figures_path, exist_ok=True)

    for (direction, run), df in data_dict.items():
        gyro_axes = {"gyro_x": df["gyro_x"], "gyro_y": df["gyro_y"], "gyro_z": df["gyro_z"]}

        N, T, fs, time_values = parameters(df)  

        
        for axis, data in gyro_axes.items():
            plt.figure(figsize=(12, 6))
            plt.specgram(data, NFFT=6, Fs=fs, noverlap=5, cmap='plasma', xextent=(0, time_values.iloc[-1]))  
            plt.colorbar(label='Power [dB]')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.title(f'Spectrogram of {axis} - Gyro Data - {direction} Run {run}')
            gyro_filename = f"gyro_spectro_{axis}_{direction}_{run}.png"
            plt.savefig(os.path.join(figures_path, gyro_filename), dpi=300, bbox_inches='tight')
            plt.close() 
            
def main():
    runs = ["00", "02"]
    df_dict = load_data(runs=runs)
    spectro_analysis(df_dict)

main()
