import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
import random

def load_data(runs):
    df_list = []
    for run in runs:
        words = ["What", "Is", "A", "Spectrogram", "Ball"]
        for word in words:
            filepath = f"txt_files/{word}_{run}.txt"
            df = pd.read_csv(filepath, sep=',', header=None, 
                               names=["gyro_x", "gyro_y", "gyro_z"])
            df_list.append((df, word, run))

    return df_list

def parameters(df):
    N = len(df['timestamp'])
    df['timestamp'] = df['timestamp'].astype(float)
    T = np.mean(np.diff(df['timestamp'])) 
    fs = 1 / T 
   
    return N, T, fs

def fft_analysis(data_list):
    figures_path = "FFTs"
    os.makedirs(figures_path, exist_ok=True)

    for df, direction, run in data_list:
        
        fft_gyro_x = np.fft.fft(df['gyro_x'])
        fft_gyro_y = np.fft.fft(df['gyro_y'])
        fft_gyro_z = np.fft.fft(df['gyro_z'])

        N, T, fs = parameters(df)
        freq = np.fft.fftfreq(N, T)

        
        positive_freq_indices = np.where(freq >= 0)
        freq = freq[positive_freq_indices]

        amplitude_x_gyro = np.abs(fft_gyro_x)[positive_freq_indices]
        amplitude_y_gyro = np.abs(fft_gyro_y)[positive_freq_indices]
        amplitude_z_gyro = np.abs(fft_gyro_z)[positive_freq_indices]

        
        plt.figure(figsize=(10, 6))
        plt.plot(freq, amplitude_x_gyro, label='FFT Gyro X', color='r')
        plt.plot(freq, amplitude_y_gyro, label='FFT Gyro Y', color='g')
        plt.plot(freq, amplitude_z_gyro, label='FFT Gyro Z', color='b')
        plt.xlim(0, fs / 2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title(f'FFT of Gyroscope Data for Direction {direction} - Run {run}')
        plt.legend()
        plt.grid()

        gyro_filename = f"gyro_{direction}_{run}.png"
        plt.savefig(os.path.join(figures_path, gyro_filename), dpi=300, bbox_inches='tight')



def main():
    runs = ["00", "02"]
    df_list = load_data(runs=runs)
    fft_analysis(df_list)

    
    
    


main()