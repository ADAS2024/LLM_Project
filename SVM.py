import os
import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
import serial

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# from scipy.stats import mode
# from scipy.signal import resample

from gyro_funcs import *
from llm import llm_call
import json
import ast
# from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

def consolidate_into_one(word_file_path):
    files = ["What.csv", "Is.csv", "A.csv", "Ball.csv", "Spectrogram.csv"]
    dataframes = []

    for file in files:
        file_path = os.path.join(word_file_path, file)
        df = pd.read_csv(file_path)

        word = os.path.splitext(file)[0]
        df['Word'] = word
        print(word)
        dataframes.append(df)
    
    combined_df = pd.concat(dataframes, ignore_index=True)

    output_path = os.path.join(word_file_path, "combined_words.csv")
    combined_df.to_csv(output_path, index=False)

def preprocess(word):
    path = f"txt_files/{word}.txt" ## with more words we make this code loop
    chunks = process_file(path)
    put_into_csv(word, chunks)

def put_into_csv(word, chunks):
    txtfile_path = "word_csvs"
    file_name = f"{word}.csv"
    file_path = os.path.join(txtfile_path, file_name)

    os.makedirs(txtfile_path, exist_ok=True)

    column_names = ["Word", "Signal Data"]
    data = []
    for chunk in chunks:
        data.append([word, json.dumps(chunk)])
    df = pd.DataFrame(data, columns=column_names)
    df.to_csv(file_path, index=False, quotechar='"')


def process_file(path):
    with open(path, 'r') as file:
        chunks = []
        current_chunk = []

        for line in file:
            line = line.strip()
            if line == "--END OF CHUNK--":
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []

            else:
                if line:
                    # Convert line into a list of floats, assuming the data is comma-separated
                    float_values = [float(i) for i in line.split(',')]
                    current_chunk.append(float_values)
        
        if current_chunk:
            chunks.append(current_chunk)

    return chunks

## train and save and predict are from my HW3 will need to update for code TODO
def train_and_save_svm(X_train, y_train, model_path, scaler_path, maxlen):

    X_train_pad = pad_sequences(X_train, maxlen = maxlen, padding = 'post', dtype = 'float32')

    samples = X_train_pad.shape[0]
    features = X_train_pad.shape[1] * X_train_pad.shape[2]
    X_train_flat = X_train_pad.reshape(samples, -1)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train_scaled, y_train)

    joblib.dump(svm_classifier, model_path)
    joblib.dump(scaler, scaler_path)
    print("Model and Scaler dumped")

def real_time_predict(model_path, scaler_path, maxlen): ## plan: put gyro data in file, parse file, tokenize on newlines and pass each block of gyro signals to predict

    svm_classifier = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    ## note, can likely import gyro_func here for cleaner code TODO
    ser = serial.Serial('/dev/tty.SLAB_USBtoUART', 115200, timeout=1)
    time.sleep(2)

    gyro_x_offset, gyro_y_offset, gyro_z_offset = calibrate_gyro(ser)

    pred_words = []
    while True:
        # Reset results file
        filename = "result_files/result.txt"
        with open(filename, "w") as f:
            pass

        gyro_func(ser, gyro_x_offset, gyro_y_offset, gyro_z_offset, SVM_ongoing=True)

        ## TODO test if above code works properly and parse results.txt
    
        data = pd.read_csv(filename, header=None).values.astype(np.float32)
        print(data.shape)

        # Pad data
        padded_data = np.pad(data, ((0, maxlen-data.shape[0]), (0,0)))

        # Flatten data
        flattened_data = padded_data.flatten()

        # Predict
        pred = svm_classifier.predict(flattened_data.reshape(1, -1))[0]
        pred_words.append(pred)
        print(pred)

        t = np.arange(0, padded_data.shape[0])
        plt.figure()
        plt.plot(t, padded_data[:, 0], label="gyro x")
        plt.plot(t, padded_data[:, 1], label="gyro y")
        plt.plot(t, padded_data[:, 2], label="gyro z")
        plt.legend()
        plt.savefig("plot.png")
        plt.close()
        # input()

        if pred == "Ball" or pred == "Spectrogram":
            llm_message = " ".join(pred_words)
            print("User Prompt: " + llm_message)
            llm_call(llm_message)
            input("Press ENTER when ready for next prompt.")

        # time.sleep(3)


def main():
    ## note: will likely loop over all words
    ## preprocess("Spectrogram")
   
    ## consolidate_into_one("word_csvs")

    data = pd.read_csv("word_csvs/combined_words.csv")

    data["Signal Data"] = data["Signal Data"].apply(ast.literal_eval)

    maxlen = max(len(signal) for signal in data["Signal Data"])

    X = np.array(data["Signal Data"])
    y = data["Word"].values

    model_path = "svm_model.pkl"
    scaler_path = "scaler.pkl"

    # X_train, _, y_train, _ = train_test_split(X, y, test_size=None, random_state=42)
    # train_and_save_svm(X_train, y_train, model_path, scaler_path, maxlen)

    real_time_predict(model_path, scaler_path, maxlen)

main()