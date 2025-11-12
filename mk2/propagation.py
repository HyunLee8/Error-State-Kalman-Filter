"""
MY THOUGHT PROCESS FOR MK2

So the previous mk1 version was me still learning how to
get the state propagation and imu input functions and just attaining
and it was really fuckin messy and also my dumass didn't differntiate
the states from the IMU readings so i got really confused but the purpose
of this is to just organize all the variables first before trying to do 
any of the transformation matrices.

In the live filter module I'm going to set all the variables to 0 or initial position 
where the I'm assuming the IMU is in a perfect position (in this case its the states).
Then I put those values into the X nominal state vector

Get all the IMU Variables and put it into one single list so later I can
index through each one fast

Next I have to create the U-corrected by getting the actualy measured IMU
data from accelerometer and the gyroscope and subtract by the bias.

Initialize the Gaussian white noise (Need to look into this more)
"""


import numpy as np
import pandas as pd
imu_data = pd.read_csv('IMU_data.csv')

"""
INITIALIZATION OF STATE VARIABLES FOR PROPOGATION
p = position
v = velocity
q = orientation quaternion
ab = accelerometer bias
wb = gyroscope bias
gt = gravity vector
"""
def propogate_nominal_state(p, v, q, ab, wb, g):
    return x = [p, v, q, ab, wb, g]
    


"""
IMU INPUT VARIABLES && UK CORRECTED
dt = time step
ax, ay, az = accelerometer measurements
wx, wy, wz = gyroscope measurements
mag_x, mag_y, mag_z = magnetometer measurements
roll, pitch, yaw = euler angles from IMU
"""
def imu_inputs(am, wm, mag_m, euler_angles):
    return [am, wm, mag_m, euler_angles]

def get_u_corrected(am, wm, ab, wb):
    return np.array([[am - ab], [wm - wb]])


"""
NOISE VECTOR
sigma
"""
def get_sigmas(sigma_a_noise, sigma_w_noise, sigma_a_walk, sigma_w_walk):
    return w = [np.array([[sigma_a_noise], [sigma_w_noise], [sigma_a_walk], [sigma_w_walk]])]