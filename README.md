| Supported Targets | ESP32 | ESP32-C3 | ESP32-C6 | ESP32-H2 | ESP32-P4 | ESP32-S2 | ESP32-S3 |
| ----------------- | ----- | -------- | -------- | -------- | -------- | -------- | -------- |

# I2C Simple Example

(See the README.md file in the upper level 'examples' directory for more information about examples.)

## Overview

This example demonstrates basic usage of I2C driver by reading and writing from a I2C connected sensor:

If you have a new I2C application to go (for example, read the temperature data from external sensor with I2C interface), try this as a basic template, then add your own code.

## How to use example

### Hardware Required

To run this example, you should have one ESP32, ESP32-S, ESP32-C or ESP32-H based development board as well as a MPU9250. MPU9250 is a inertial measurement unit, which contains a accelerometer, gyroscope as well as a magnetometer, for more information about it, you can read the [datasheet of the MPU9250 sensor](https://invensense.tdk.com/wp-content/uploads/2015/02/PS-MPU-9250A-01-v1.1.pdf).

#### Pin Assignment:

**Note:** The following pin assignments are used by default, you can change these in the `menuconfig` .

|                  | SDA             | SCL           |
| ---------------- | -------------- | -------------- |
| ESP I2C Master   | I2C_MASTER_SDA | I2C_MASTER_SCL |
| MPU9250 Sensor   | SDA            | SCL            |


For the actual default value of `I2C_MASTER_SDA` and `I2C_MASTER_SCL` see `Example Configuration` in `menuconfig`.

**Note:** There's no need to add an external pull-up resistors for SDA/SCL pin, because the driver will enable the internal pull-up resistors.

I used SCL as pin 0 and SDA as pin 1.

### Build and Flash

Enter `idf.py -p PORT flash` to build and flash the project.

HW1_528_datalogging.py can be used to monitor the IMU sensor and send data to txt files. HW1_FFT.py is used to graph the data.

## Example Output

```bash
I (328) i2c-simple-example: I2C initialized successfully
I (338) i2c-simple-example: WHO_AM_I = 71
I (338) i2c-simple-example: I2C de-initialized successfully
```

## Troubleshooting

(For any technical queries, please open an [issue](https://github.com/espressif/esp-idf/issues) on GitHub. We will get back to you as soon as possible.)
