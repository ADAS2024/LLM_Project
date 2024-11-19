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

def consolidate_into_one(word_file_path):
    files = ["Up.csv", "Down.csv", "Left.csv", "Right.csv"]
    dataframes = []

    for file in files:
        file_path = os.path.join(word_file_path, file)
        df = pd.read_csv(file_path)

        word = os.path.splitext(file)[0]
        df['Word'] = word
        print(gesture_name)
        dataframes.append(df)
    
    combined_df = pd.concat(dataframes, ignore_index=True)

    output_path = os.path.join(gesture_file_path, "combined_gestures.csv")
    combined_df.to_csv(output_path, index=False)

    
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


def train_and_evaluate_svm(X_train, y_train, X_test, y_test):
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)

    y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'SVM accuracy: {accuracy:.3%}')

    class_labels = ["What", "Is", "A", "Spectrogram", "Ball"]

    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'SVM Confusion Matrix\nAccuracy: {accuracy:.3%}')
    plt.xlabel('pred')
    plt.ylabel('actual')
    plt.savefig("SVM_Confusion_Matrix.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def main():
    merge_into_csvs("txt_files")         ## these commands only need to be run once
    consolidate_into_one("word_csvs")

    data = pd.read_csv("word_csvs/combined_words.csv")
    X = data[["Gyro_X", "Gyro_Y", "Gyro_Z"]].values.astype(np.float32)
    y = data["Word"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_and_evaluate_svm(X_train, y_train, X_test, y_test)

main()