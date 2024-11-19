    /* i2c - Simple example

   Simple I2C example that shows how to initialize I2C
   as well as reading and writing from and to registers for a sensor connected over I2C.

   The sensor used in this example is a MPU9250 inertial measurement unit.

   For other examples please check:
   https://github.com/espressif/esp-idf/tree/master/examples

   See README.md file to get detailed usage of this example.

   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "esp_log.h"
#include "driver/i2c.h"
#include "nvs_flash.h"

static const char *TAG = "LLM_Project";

#define I2C_MASTER_SCL_IO           01          /*!< GPIO number used for I2C master clock */
#define I2C_MASTER_SDA_IO           00          /*!< GPIO number used for I2C master data  */
#define I2C_MASTER_NUM              0           /*!< I2C master i2c port number, the number of i2c peripheral interfaces available will depend on the chip */
#define I2C_MASTER_FREQ_HZ          400000      /*!< I2C master clock frequency */
#define I2C_MASTER_TX_BUF_DISABLE   0           /*!< I2C master doesn't need buffer */
#define I2C_MASTER_RX_BUF_DISABLE   0           /*!< I2C master doesn't need buffer */
#define I2C_MASTER_TIMEOUT_MS       1000

#define MPU6050_SENSOR_ADDR         0x68        /*!< Slave address of the MPU9250 sensor */
#define MPU6050_PWR_MGMT_1_REG_ADDR 0x6B        /*!< Register addresses of the power managment register */
#define MPU6050_RESET_BIT           7

#define MPU6050_GYRO_XOUT_H         0x43        /*!< Register address for gyroscope data (X axis high byte) */
#define GYRO_THRESHOLD              250      /*!< Threshold to detect movement direction */






esp_err_t mpu6050_init();
void detect_movement(int16_t *gyro_data);
esp_err_t read_data(int16_t *gyro_data);


esp_err_t mpu6050_init() {
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (MPU6050_SENSOR_ADDR << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, MPU6050_PWR_MGMT_1_REG_ADDR, true);
    i2c_master_write_byte(cmd, 0x00, true);  
    i2c_master_stop(cmd);

    esp_err_t ret = i2c_master_cmd_begin(I2C_MASTER_NUM, cmd, I2C_MASTER_TIMEOUT_MS / portTICK_PERIOD_MS);
    i2c_cmd_link_delete(cmd);
    return ret;
}


esp_err_t read_data(int16_t *gyro_data) {
    uint8_t data_h[14];
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (MPU6050_SENSOR_ADDR << 1) | I2C_MASTER_READ, true);
    i2c_master_read(cmd, data_h, sizeof(data_h), I2C_MASTER_LAST_NACK);
    i2c_master_stop(cmd);

    esp_err_t ret = i2c_master_cmd_begin(I2C_MASTER_NUM, cmd, I2C_MASTER_TIMEOUT_MS / portTICK_PERIOD_MS);
    i2c_cmd_link_delete(cmd);

    if (ret == ESP_OK) {
        gyro_data[0] = (data_h[8] << 8) | data_h[9];
        gyro_data[1] = (data_h[10] << 8) | data_h[11];
        gyro_data[2] = (data_h[12] << 8) | data_h[13];
    }

    return ret;
}



void detect_movement(int16_t *gyro_data) {

    int16_t gyro_x = gyro_data[0];
    int16_t gyro_y = gyro_data[1];
    int16_t gyro_z = gyro_data[2];


    ESP_LOGI(TAG, "Gyro Data: Gyro_X=%d, Gyro_Y=%d, Gyro_Z=%d", gyro_x, gyro_y, gyro_z);
}

        
void app_main() {

    ESP_ERROR_CHECK(nvs_flash_init());
    

    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ,
    };

    ESP_ERROR_CHECK(i2c_param_config(I2C_MASTER_NUM, &conf));
    ESP_ERROR_CHECK(i2c_driver_install(I2C_MASTER_NUM, conf.mode, I2C_MASTER_RX_BUF_DISABLE, I2C_MASTER_TX_BUF_DISABLE, 0));

    ESP_ERROR_CHECK(mpu6050_init());

    int16_t gyro_data[3];

    while (1) {
        if (read_data(gyro_data) == ESP_OK) {
            detect_movement(gyro_data);
        } else {
            ESP_LOGE(TAG, "Failed to read gyroscope data");
        }
        vTaskDelay(1000 / portTICK_PERIOD_MS);  
    }
}