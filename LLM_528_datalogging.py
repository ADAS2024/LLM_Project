import serial
import re
import time
import os
from gyro_funcs import *

def log_data_to_file(gyro_x, gyro_y, gyro_z, word, is_pause=False):
    txtfile_path = "txt_files"
    file_name = f"{word}.txt"
    file_path = os.path.join(txtfile_path, file_name)

    os.makedirs(txtfile_path, exist_ok=True)

    with open(file_path, 'a') as f:
        if gyro_x == "" and gyro_y == "" and gyro_z == "":
            f.write("\n")  # Write a blank line for no movement
        else:
            f.write(f"{gyro_x},{gyro_y},{gyro_z}\n")
            print(f"Data logged: {gyro_x},{gyro_y},{gyro_z}")
        
        if is_pause:
            f.write("--END OF CHUNK--\n")
            print("End of chunk added.")


def main():
    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)  # Adjust to your serial port
    gyro_x_offset, gyro_y_offset, gyro_z_offset = calibrate_gyro(ser)
    gyro_func(ser)

if __name__ == "__main__":
    main()
