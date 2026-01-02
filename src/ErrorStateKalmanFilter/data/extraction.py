from enum import IntEnum
import pandas as pd
import numpy as np

class Data:
    def __init__(self):
        self.TimeCol = 0
        self.AccXCol = 1
        self.AccYCol = 2
        self.AccZCol = 3
        self.GryoXCol = 4
        self.GryoYCol = 5
        self.GryoZCol = 6

    def get_imu_vectors(self, data, i):
        gyro = np.array([
            data.iloc[i, self.GyroXCol],
            data.iloc[i, self.GyroYCol],
            data.iloc[i, self.GyroZCol]
        ])
        acc = np.array([
            data.iloc[i, self.AccXCol],
            data.iloc[i, self.AccYCol],
            data.iloc[i, self.AccZCol],
        ])
        dt = data.iloc[i + 1, self.TimeCol] - data.iloc[i, self.TimeCol]


    def get_data(self, path_name):
        return pd.read_csv(IMU_data.csv)