import os
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def get_chunks(filename):
    # gets Gyro x, y, z datas from each chunk in the files
    chunks = []
    x, y, z = [], [], []

    with open(filename, "r") as f:
        for line in f:
            if "--END OF CHUNK--" in line:
                if x and y and z:  # finished reading the current chunk
                    chunks.append((np.array(x), np.array(y), np.array(z)))
                x, y, z = [], [], []  # Reset for the next chunk
            else:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    x.append(float(parts[0]))
                    y.append(float(parts[1]))
                    z.append(float(parts[2]))

    # Add the last chunk if any
    if x and y and z:
        chunks.append((np.array(x), np.array(y), np.array(z)))

    return chunks

def save_spectrogram(data, sampling_rate, output_path, title):
    # saving spectrogram graph at the specified plots

    nperseg = min(32, len(data))
    noverlap = max(0, nperseg // 2)

    if len(data) < 2:
        print(f"Skipping spectrogram for data with insufficient length: {len(data)}")
        return

    f, t, Sxx = spectrogram(data, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
    plt.colorbar(label="Power (dB)")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    data_dir = "txt_files"
    spectro_dir = "spectro_graphs"
    sampling_rate = 100  # Adjust as needed

    # Go through the data file for each word
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)

            # Get the word (e.g., "A") from the filename
            word = filename.split(".")[0]

            # Create subdirectories for x, y, z axes within the word's directory
            word_output_dir = os.path.join(spectro_dir, word)
            for axis in ['x', 'y', 'z']:
                axis_dir = os.path.join(word_output_dir, axis)
                if not os.path.exists(axis_dir):
                    os.makedirs(axis_dir)

            # Read chunks from the file
            chunks = get_chunks(filepath)

            # Process each chunk
            for i, (x, y, z) in enumerate(chunks):
                # Save spectrograms for x, y, z axes
                save_spectrogram(x, sampling_rate, os.path.join(word_output_dir, 'x', f"chunk_{i + 1}.png"), f"Gyro Spectrogram of {word} - Chunk {i + 1} (X-axis)")
                save_spectrogram(y, sampling_rate, os.path.join(word_output_dir, 'y', f"chunk_{i + 1}.png"), f"Gyro Spectrogram of {word} - Chunk {i + 1} (Y-axis)")
                save_spectrogram(z, sampling_rate, os.path.join(word_output_dir, 'z', f"chunk_{i + 1}.png"), f"Gyro Spectrogram of {word} - Chunk {i + 1} (Z-axis)")
