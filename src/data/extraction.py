from enum import IntEnum
import pandas as pd
import numpy as np

class Data:
    def __init__(self):
        self.TimeCol = 0
        self.AccXCol = 1
        self.AccYCol = 2
        self.AccZCol = 3
        self.GyroXCol = 4
        self.GyroYCol = 5
        self.GyroZCol = 6
        self.data = pd.read_csv('IMU_data.csv')

    def get_imu_vectors(self, i):
        gyro = np.array([
            self.data.iloc[i, self.GyroXCol],
            self.data.iloc[i, self.GyroYCol],
            self.data.iloc[i, self.GyroZCol]
        ])
        acc = np.array([
            self.data.iloc[i, self.AccXCol],
            self.data.iloc[i, self.AccYCol],
            self.data.iloc[i, self.AccZCol],
        ])
        dt = self.data.iloc[i + 1, self.TimeCol] - self.data.iloc[i, self.TimeCol]
        
        return gyro, acc, dt
    
    @property
    def get_imu_data(self):
        """Return gyro, acc, dt for use in ESKF __init__"""
        # Return first sample data
        gyro, acc, dt = self.get_imu_vectors(0)
        return gyro, acc, dt