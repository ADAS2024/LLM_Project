import serial
import re
import time
import os




def log_data_to_file(gyro_x, gyro_y, gyro_z):
    txtfile_path = "txt_files"
    word = "Ball"  # Change word [What, Is, A, Spectrogram, Ball]
    file_name = f"{word}.txt"
    file_path = os.path.join(txtfile_path, file_name)

    os.makedirs(txtfile_path, exist_ok=True)

    with open(file_path, 'a') as f:
        if gyro_x == "" and gyro_y == "" and gyro_z == "":
            f.write("\n")  # Write a blank line for no movement
        else:
            f.write(f"{gyro_x},{gyro_y},{gyro_z}\n")
            print(f"Data logged to {file_name}: {gyro_x},{gyro_y},{gyro_z}")

def main():
    try:
        ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)  # Adjust to your serial port
        time.sleep(2)
        pattern = re.compile(
            r"Gyro_X=(-?\d+), Gyro_Y=(-?\d+), Gyro_Z=(-?\d+)"
        )
        print("Started data collection. Do CTRL+C to stop the process.")

        while True:
            line = ser.readline().decode('utf-8').strip()
            match = pattern.search(line)

            if match:
                gyro_x = int(match.group(1))
                gyro_y = int(match.group(2))
                gyro_z = int(match.group(3))

                if abs(gyro_x) > 1500 or abs(gyro_y) > 1500 or abs(gyro_z) > 1500:
                    print(f"Movement detected! Gyro_X={gyro_x}, Gyro_Y={gyro_y}, Gyro_Z={gyro_z}")
                    log_data_to_file(gyro_x, gyro_y, gyro_z)
                else:
                    print("No significant movement detected. Logging a blank line.")
                    log_data_to_file("", "", "")
                    
    except Exception as e:
        print(f"Error: {e}")

    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
        print("Serial port closed")

if __name__ == "__main__":
    main()