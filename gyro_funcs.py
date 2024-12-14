import serial
import re
import time
import os

def calibrate_gyro(ser, sample_duration=5):
    print("Calibrating gyroscope, please keep jaw steady...")
    pattern = re.compile(
        r"Gyro_X=(-?\d+), Gyro_Y=(-?\d+), Gyro_Z=(-?\d+)"
    )
    gyro_x_sum = 0
    gyro_y_sum = 0
    gyro_z_sum = 0
    count = 0

    start_time = time.time()
    while time.time() - start_time < sample_duration:
        line = ser.readline().decode('utf-8').strip()
        match = pattern.search(line)

        if match:
            gyro_x = int(match.group(1))
            gyro_y = int(match.group(2))
            gyro_z = int(match.group(3))

            gyro_x_sum += gyro_x
            gyro_y_sum += gyro_y
            gyro_z_sum += gyro_z
            count += 1

    if count == 0:
        raise ValueError("No data received during calibration!")

    gyro_x_offset = gyro_x_sum / count
    gyro_y_offset = gyro_y_sum / count
    gyro_z_offset = gyro_z_sum / count

    print(f"Calibration complete. Offsets - Gyro_X: {gyro_x_offset}, Gyro_Y: {gyro_y_offset}, Gyro_Z: {gyro_z_offset}")
    return gyro_x_offset, gyro_y_offset, gyro_z_offset

## SVM_ongoing to distinguish whether we are data collecting or testing on real time gyro data
def gyro_func(ser, gyro_x_offset, gyro_y_offset, gyro_z_offset, word = None, SVM_ongoing = False):
    try:
        pattern = re.compile(
            r"Gyro_X=(-?\d+), Gyro_Y=(-?\d+), Gyro_Z=(-?\d+)"
        )
        if SVM_ongoing:
            print("Serial port opened. Listening for data... (CTRL+C to stop listening)")
        else:
            print("Started data collection. Do CTRL+C to stop the process.")
        movement_threshold = 1500
        pause_threshold = 400
        data_chunk = []

        while True:
            line = ser.readline().decode('utf-8').strip()
            match = pattern.search(line)

            if match:
                gyro_x = int(match.group(1)) - gyro_x_offset
                gyro_y = int(match.group(2)) - gyro_y_offset
                gyro_z = int(match.group(3)) - gyro_z_offset

                if abs(gyro_x) > movement_threshold or abs(gyro_y) > movement_threshold or abs(gyro_z) > movement_threshold:
                    # print(f"Movement detected! Gyro_X={gyro_x}, Gyro_Y={gyro_y}, Gyro_Z={gyro_z}")
                    data_chunk.append((gyro_x, gyro_y, gyro_z))  

                elif abs(gyro_x) < pause_threshold and abs(gyro_y) < pause_threshold and abs(gyro_z) < pause_threshold:
                    if data_chunk:
                        print(f"Pause detected, logging chunk: {len(data_chunk)} samples.")
                        
                        for gyro_data in data_chunk:
                            log_data_to_file(*gyro_data, word=word, SVM_ongoing=SVM_ongoing)
                        # log_data_to_file("", "", "", word=word, is_pause=True, SVM_ongoing=SVM_ongoing)
                        data_chunk = []
                        break

            time.sleep(0.01)

    except Exception as e:
        print(f"Error: {e}")

    # finally:
    #     if 'ser' in locals() and ser.is_open:
    #         ser.close()
    #     print("Serial port closed")

def log_data_to_file(gyro_x, gyro_y, gyro_z, word = None, is_pause=False, SVM_ongoing = False):
    txt_file_path = ""
    file_name = ""
    file_path = ""

    ## logging live data to file for prediction
    if SVM_ongoing and word == None:
        txtfile_path = "result_files"
        file_name = "result.txt"
        file_path = os.path.join(txtfile_path, file_name)
    
    # gathering training data
    else:
        txtfile_path = "txt_files"
        file_name = f"{word}.txt"
        file_path = os.path.join(txtfile_path, file_name)

    os.makedirs(txtfile_path, exist_ok=True)

    with open(file_path, 'a') as f:
        f.write(f"{gyro_x},{gyro_y},{gyro_z}\n")
        # print(f"Data logged: {gyro_x},{gyro_y},{gyro_z}")