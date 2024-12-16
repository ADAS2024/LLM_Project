import os
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

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

def compute_fft(data, sampling_rate):
    N = len(data)
    T = 1.0 / sampling_rate
    yf = fft(data)
    xf = fftfreq(N, T)[:N // 2]
    return xf, 2.0 / N * np.abs(yf[:N // 2])

def save_fft_plot(xf, yf_x, yf_y, yf_z, output_path, title):
    # saving fft_plots at the specified path
    plt.figure(figsize=(10, 6))
    plt.plot(xf, yf_x, label="X-axis")
    plt.plot(xf, yf_y, label="Y-axis")
    plt.plot(xf, yf_z, label="Z-axis")
    plt.grid()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    data_dir = "txt_files"
    fft_dir = "fft_graphs"
    sampling_rate = 100  # Adjust as needed

    # Go through the data file for each word
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)

            # Get the word (e.g., "A") from the filename
            word = filename.split(".")[0]

            # Create a subdirectory for the word in the output directory
            word_output_dir = os.path.join(fft_dir, word)
            if not os.path.exists(word_output_dir):
                os.makedirs(word_output_dir)

            # Read chunks from the file
            chunks = get_chunks(filepath)

            # Process each chunk
            for i, (x, y, z) in enumerate(chunks):
                xf, yf_x = compute_fft(x, sampling_rate)
                _, yf_y = compute_fft(y, sampling_rate)
                _, yf_z = compute_fft(z, sampling_rate)

                # Save the FFT plot for the chunk
                output_path = os.path.join(word_output_dir, f"chunk_{i + 1}.png")
                title = f"Gyro FFT of {word} - Chunk {i + 1}"
                save_fft_plot(xf, yf_x, yf_y, yf_z, output_path, title)
