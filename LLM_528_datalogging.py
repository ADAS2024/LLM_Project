import serial
import re
import time
import os
from gyro_funcs import *


def main():
    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)  # Adjust to your serial port
    time.sleep(2)
    gyro_x_offset, gyro_y_offset, gyro_z_offset = calibrate_gyro(ser)
    gyro_func(ser, gyro_x_offset, gyro_y_offset, gyro_z_offset, word = "What") ## change word for each

if __name__ == "__main__":
    main()
