"""
IMU helper for odom+autonomous merged Pico runtime.

This file is intentionally standalone so the merged entrypoint can be uploaded
without requiring package-style imports.
"""

from machine import I2C, Pin
from math import pi
from utime import sleep


class MPU6050:
    """Minimal MPU6050 reader with one-time gyro bias calibration."""

    def __init__(self, pow_id=3, scl_id=5, sda_id=4, i2c_addr=0x68, dlpf_level=4):
        self.board_led = Pin(25, Pin.OUT)
        self.pow_pin = Pin(pow_id, Pin.OUT, value=1)
        sleep(0.5)

        self.i2c = I2C(0, scl=Pin(scl_id), sda=Pin(sda_id), freq=400_000)
        self.i2c_addr = i2c_addr

        # Wake up sensor.
        self.i2c.writeto_mem(self.i2c_addr, 0x6B, bytes([0x00]))

        # Configure digital low-pass filter to suppress high-frequency noise.
        assert 0 <= dlpf_level <= 6
        self.i2c.writeto_mem(self.i2c_addr, 0x1A, bytes([dlpf_level]))

        self.lin_acc_x = 0.0
        self.lin_acc_y = 0.0
        self.lin_acc_z = 0.0
        self.ang_vel_x = 0.0
        self.ang_vel_y = 0.0
        self.ang_vel_z = 0.0

        self.gyro_bias_x = 0.0
        self.gyro_bias_y = 0.0
        self.gyro_bias_z = 0.0

        self.calibrate_gyro()

    @staticmethod
    def _process_raw(value_raw, scale):
        if value_raw > 32768:
            return (value_raw - 65535) / scale
        return value_raw / scale

    def read_data(self):
        """
        Read and convert MPU6050 data.

        Returns angular rates in rad/s and linear acceleration in m/s^2.
        """
        words = self.i2c.readfrom_mem(self.i2c_addr, 0x3B, 14)
        data = [words[i] << 8 | words[i + 1] for i in range(0, len(words), 2)]

        self.lin_acc_x = self._process_raw(data[0], 16384) * 9.80665
        self.lin_acc_y = self._process_raw(data[1], 16384) * 9.80665
        self.lin_acc_z = self._process_raw(data[2], 16384) * 9.80665

        self.ang_vel_x = self._process_raw(data[4], 131) * pi / 180 - self.gyro_bias_x
        self.ang_vel_y = self._process_raw(data[5], 131) * pi / 180 - self.gyro_bias_y
        self.ang_vel_z = self._process_raw(data[6], 131) * pi / 180 - self.gyro_bias_z

        return {
            "acc_x": self.lin_acc_x,
            "acc_y": self.lin_acc_y,
            "acc_z": self.lin_acc_z,
            "omg_x": self.ang_vel_x,
            "omg_y": self.ang_vel_y,
            "omg_z": self.ang_vel_z,
        }

    def calibrate_gyro(self, num_samples=1000):
        """Average stationary gyro readings to estimate zero-rate bias."""
        dep_x = 0.0
        dep_y = 0.0
        dep_z = 0.0
        for i in range(num_samples):
            data = self.read_data()
            dep_x += data["omg_x"]
            dep_y += data["omg_y"]
            dep_z += data["omg_z"]
            sleep(0.005)
            if i % 50 == 0:
                self.board_led.toggle()

        self.gyro_bias_x = dep_x / num_samples
        self.gyro_bias_y = dep_y / num_samples
        self.gyro_bias_z = dep_z / num_samples
