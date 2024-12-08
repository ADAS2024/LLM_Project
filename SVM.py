import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


def preprocess(word):
    path = "txt_files/What.txt" ## with more words we make this code loop
    chunks = process_file(path)
    put_into_csv("What", chunks)

def put_into_csv(word, chunks):
    txtfile_path = "word_csvs"
    file_name = f"{word}.csv"
    file_path = os.path.join(txtfile_path, file_name)

    os.makedirs(txtfile_path, exist_ok=True)

    column_names = [word, "Signal Data"]
    data = []
    for chunk in chunks:
        data.append([word, chunk])
    df = pd.DataFrame(data, columns=column_names)
    df.to_csv(file_path)


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
                    current_chunk.append(line)
        
        if current_chunk:
            chunks.append(current_chunk)

    return chunks

## need to fix for later TODO
def merge_into_csvs(txt_file_path):
    input_path = txt_file_path  
    output_path = "word_csvs" 
    os.makedirs(output_path, exist_ok=True)

    word = ["What", "Is", "A", "Spectrogram", "Ball"]
    for word in words:

        word_data = []

        for i in range(21): ## goes from 00-20
            filename = f"{word}_{i:02d}.txt"
            file_path = os.path.join(input_path, filename)
            
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    values = line.strip().split(',')  
                    word_data.append([word] + values)  
            

        column_names = ["Word","Gyro_X", "Gyro_Y", "Gyro_Z"]
        df = pd.DataFrame(word_data, columns=column_names)
        output_file = os.path.join(output_path, f"{word}.csv")
        df.to_csv(output_file, index=False)
        print(f"{word} data saved to {output_file}")


def train_and_save_svm(X_train, y_train, model_path, scaler_path):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)

    joblib.dump(svm_classifier, model_path)
    joblib.dump(scaler, scaler_path)

    return scaler

def real_time_predict(model_path, scaler_path):

    svm_classifier = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    ## note, can likely import gyro_func here for cleaner code TODO
    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
    time.sleep(2)

    pattern = re.compile(
        r"Gyro_X=(-?\d+), Gyro_Y=(-?\d+), Gyro_Z=(-?\d+)"
    )

    print("Serial port opened. Listening for data... (CTRL+C to stop listening)")

    try:
        while True:
            line = ser.readline().decode('utf-8').strip()
            if not line:
                continue

            try:
                match = pattern.search(line)

                if match:

                    gyro_x = int(match.group(1))
                    gyro_y = int(match.group(2))
                    gyro_z = int(match.group(3))
                    

                    data = [gyro_x, gyro_y, gyro_z]
                    scaled_data = scaler.transform([data])
                    prediction = svm_classifier.predict(scaled_data)[0]
                    print("Predicted Direction of Gesture: {}".format(prediction))
                    print("Actual Direction: {}".format(direction))
                    print()
            
            except Exception as e:
                print("Error Processing Data: {}".format(e))

    except KeyboardInterrupt:
        print("\nClosing Serial Port...")
        ser.close()
        print("Exiting..")

def main():
    preprocess("What")
   
    ##consolidate_into_one("word_csvs")

    ##data = pd.read_csv("word_csvs/combined_words.csv")
    ##X = data[["Gyro_X", "Gyro_Y", "Gyro_Z"]].values.astype(np.float32)
    ##y = data["Word"]
    ##X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ##train_and_evaluate_svm(X_train, y_train, X_test, y_test)

main()