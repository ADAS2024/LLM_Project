import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import serial

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from gyro_funcs import *
from llm import llm_call
import json
import ast
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

    X_train_pad = np.pad(X_train, maxlen = maxlen, padding = 'post', dtype = 'float32')

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

def train_test_svm(X_train, y_train, X_test, y_test, maxlen):
    # print(X_train.shape)
    print(maxlen)
    for i in range(len(X_train)):
        X_train[i] = np.pad(X_train[i], ((0, maxlen-X_train[i].shape[0]), (0,0)))

    for i in range(len(X_test)):
        X_test[i] = np.pad(X_test[i], ((0, maxlen-X_test[i].shape[0]), (0,0)))

    X_train_pad = np.array(X_train)
    X_test_pad = np.array(X_test)

    print(X_train_pad.shape)

    n_samples, n_data_pts, n_dims = X_train_pad.shape
    n_samples_test = X_test_pad.shape[0]
    flattened_train = X_train_pad.reshape(n_samples, n_data_pts*n_dims)
    flattened_test = X_test_pad.reshape(n_samples_test, n_data_pts*n_dims)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(flattened_train)
    # X_test_scaled = scaler.fit_transform(flattened_test)

    svm_classifier = SVC(kernel='linear')
    print(X_train_scaled.shape)
    print(flattened_train.shape)
    print(y_train.shape)
    # svm_classifier.fit(X_train_scaled, y_train)
    svm_classifier.fit(flattened_train, y_train)

    # y_pred = svm_classifier.predict(X_test_scaled)
    y_pred = svm_classifier.predict(flattened_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'SVM accuracy: {accuracy:.3%}')

    # Plot the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, cmap="Blues")
    plt.title('SVM Prediction Accuracy')
    plt.xlabel('pred')
    plt.ylabel('actual')
    plt.savefig("conf_matrix.png")


def real_time_predict(model_path, scaler_path, maxlen): ## plan: put gyro data in file, parse file, tokenize on newlines and pass each block of gyro signals to predict

    svm_classifier = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    ## note, can likely import gyro_func here for cleaner code TODO
    ser = serial.Serial('/dev/tty.SLAB_USBtoUART', 115200, timeout=1)
    time.sleep(2)

    gyro_x_offset, gyro_y_offset, gyro_z_offset = calibrate_gyro(ser)
    ser.close()

    pred_words = []
    while True:
        # Reset results file
        filename = "result_files/result.txt"
        with open(filename, "w") as f:
            pass

        # Reset serial
        ser = serial.Serial('/dev/tty.SLAB_USBtoUART', 115200, timeout=1)

        gyro_func(ser, gyro_x_offset, gyro_y_offset, gyro_z_offset, SVM_ongoing=True)

        ## TODO test if above code works properly and parse results.txt
    
        data = pd.read_csv(filename, header=None).values.astype(np.float32)
        print(data.shape)

        # Pad data
        padded_data = np.pad(data, ((0, maxlen-data.shape[0]), (0,0)))

        # Flatten data
        flattened_data = padded_data.flatten()

        # Predict
        pred = svm_classifier.predict(flattened_data.reshape(1,-1))[0]
        pred_words.append(pred)
        print(pred)

        # t = np.arange(0, padded_data.shape[0])
        # plt.figure()
        # plt.plot(t, padded_data[:, 0], label="gyro x")
        # plt.plot(t, padded_data[:, 1], label="gyro y")
        # plt.plot(t, padded_data[:, 2], label="gyro z")
        # plt.legend()
        # plt.savefig("plot.png")
        # plt.close()
        # input()

        if pred == "Ball" or pred == "Spectrogram":
            llm_message = " ".join(pred_words)
            print("User Prompt: " + llm_message)
            llm_call(llm_message)
            input("Press ENTER when ready for next prompt.")
            pred_words = []

        ser.close()

        time.sleep(2)


def main():
    ## note: will likely loop over all words
    ## preprocess("Spectrogram")
   
    ## consolidate_into_one("word_csvs")

    data = pd.read_csv("word_csvs/combined_words.csv")

    data["Signal Data"] = data["Signal Data"].apply(ast.literal_eval)

    maxlen = max(len(signal) for signal in data["Signal Data"])

    X = np.array(data["Signal Data"])
    X = [np.array(x) for x in X]
    y = data["Word"].values

    model_path = "svm_model.pkl"
    scaler_path = "scaler.pkl"

    # print(X[1].shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
    # train_test_svm(X_train, y_train, X_test, y_test, maxlen)
    # train_and_save_svm(X_train, y_train, X_test, y_test, model_path, scaler_path, maxlen)

    real_time_predict(model_path, scaler_path, maxlen)

main()