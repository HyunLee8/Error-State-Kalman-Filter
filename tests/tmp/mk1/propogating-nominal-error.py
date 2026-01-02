"""WORK IN PROGRESS"""

"""Implementation of an Error State Kalman Filter (ESKF) using quaternions for orientation representation."""
import math
import numpy as np
import pandas as pd
from quaternion import Quaternion
from numpy.linalg import norm
df = pd.read_csv('IMU_data.csv')

def initialize_state(i):
    if i < 1:
        px = 0, py = 0, pz = 0
        vx = 0, vy = 0, vz = 0
        qw = 1, qx = 0, qy = 0, qz = 0

    if i >= 1:
        ax = df['AccX(m/s^2)'][i]
        ay = df['AccY(m/s^2)'][i]
        az = df['AccZ(m/s^2)'][i]
        wx = df['GyroX(rad/s)'][i]
        wy = df['GyroY(rad/s)'][i]
        wz = df['GyroZ(rad/s)'][i]
        mag_X = df['MagX(uT)'][i]
        mag_Y = df['MagY(uT)'][i]
        mag_Z = df['MagZ(uT)'][i]
        roll = df['Roll(deg)'][i]
        pitch = df['Pitch(deg)'][i]
        yaw = df['Yaw(deg)'][i]
        dt = df['Time(s)'][i+1] - df['Time(s)'][i]
    
    p = np.array([[px], [py], [pz]])
    v = np.array([[vx], [vy], [vz]])
    q = np.array([[qw], [qx], [qy], [qz]])
    a = np.array([[ax], [ay], [az]])
    w = np.array([[wx], [wy], [wz]])
    mag = np.array([[mag_X], [mag_Y], [mag_Z]])
    rpy = np.array([[roll], [pitch], [yaw]])

"""Nominal State Methods"""
def propagate_nominal_state():
    """
    p: position x, y, z
    v: velocity vx, vy, vz
    a: acceleration ax, ay, az
    q: orientation quaternion qw, qx, qy, qz
    wb: gyro bias wxb, wyb, wzb
    wm: measured angular velocity wxm, wym, wzm
    wn: gyro noise wxn, wyn, wzn
    ab: accelerometer bias axb, ayb, azb
    dt: time step
    """
    p = update_position(p, v, a, dt)
    v = update_velocity(v, a, dt)
    q = update_orientation(q, wb, wm, wn, dt)
    ab = update_accel_bias(ab, dt)
    wb = update_gyro_bias(wb, dt)
    g = update_gravity()

    #didn't because i want to index instead
    x = [p, v, q, ab, wb, g]
    return x

#i need xt= [pt, vt, qt, abt, wbt, gt]
def update_position(p, v, a, dt, R):
    p_next = p + v*dt + 0.5(R(a) + g)(dt**2)
    return p_next

def update_velocity(v, a, dt, R):
    v_next = v +(R @ a + g)dt
    return v_next

def update_orientation(q, wb, wm, wn, dt):
    q_dot = quaternion_derivative(q, wm - wb - wn)
    q_next = q + q_dot*dt
    #quaternion values should add up to 1 below
    q_next = q_next / np.linalg.norm(q_next)
    return q_next

def quaternion_derivative(q, omega):
    Omega = omega_matrix(omega)
    q_dot = 0.5 * (Omega @ q)
    return q_dot

def omega_matrix(omega):
    wx, wy, wz = omega.flatten()
    Omega = np.block([[0, -wx, -wy, -wz],
                      [wx,  0,  wz, -wy],
                      [wy, -wz,  0,  wx],
                      [wz,  wy, -wx,  0]])
    return Omega

def update_accel_bias(ab, dt):
    ab_next = ab + a_bias_derivative(update_a_bias()) * dt
    return ab_next

def a_bias_derivative(process_noise):
    return process_noise

def update_a_bias():
    axw = 1e-4
    ayw = 1.2e-4
    azw = 1.5e-4
    process_noise = np.array([[axw], [ayw], [azw]])
    return process_noise

def update_gyro_bias(wb, dt):
    wb_next = wb + w_bias_derivative(update_w_bias()) * dt
    return wb_next 

def w_bias_derivative(process_noise):
    return process_noise

def update_w_bias():
    wxw = 1e-5
    wyw = 2e-5
    wzw = 1.5e-5
    process_noise = np.array([[wxw], [wyw], [wzw]])
    return process_noise

def update_gravity():
    g = np.array([[0.0], [0.0], [9.81]])
    return g


"""Error State Methods"""
#(this is just to organize my thoughts)
#u = W_measured - W_bias -> u is a matrix with wx, wy, wz (I need to flatten this later in skew symmetric)
#skew_symmetric_matrix(input)

#OK this file will be just for propagating nominal states and not the error

#def skew_symmetric(phi):
#    vx, vy, vz = phi.flatten()
#    phi_skew = np.array([[0, -vz, vy],
#                         [vz, 0, -vx],
#                         [-vy, vx, 0]])
#    return phi_skew

#def compute_rotation_matrix(phi):
#    phi_skew = skew_symmetric(phi)
#    I = np.eye(3)
#    R = I + (math.sin(theta)/theta)*phi_skew  + (1-math.cos(theta))/(theta**2)*(phi_skew**2)
#    return R.T

#def propogate_error_state(wm, wn, am, an, dt):
#    Ua = am - an
#    Uw = wm - wn
#    Uk = np.vstack((Ua, Uw))
#    phi = Uk * dt
#    R_t = compute_rotation_matrix(phi)
#    I = np.eye(3)
#    zero = np.zeros((3,3))
#    Fk = np.array([[R_t, -dt*I],
#                   [zero, I]])


################################################################################################
"""Redoing propogating normal states"""
"""commented it out for now but if i fuck up the code up top im just gonna reuse this"""
#i need xt= [pt, vt, qt, abt, wbt, gt]
#def update_position(p, v, a, dt):
#   p_next = p + vk*dt + 0.5*ak*(dt**2)
#    return p_next

#def update_velocity(v, a, dt):
#    v_next = vk + ak*dt
#    return v_next

#def update_orientation(q, wb, wm, wn, dt):
#    q_dot = quaternion_derivative(q, wm - wb - wn)
#    q_next = q + q_dot*dt
#    #quaternion values should add up to 1 below
#    q_next = q_next / np.linalg.norm(q_next)
#    return q_next

#def update_accel_bias(ab, ):
#    ab_next = ab + 

#def propogate_nominal_state():
#    p = update_position()
#    v = update_velocity()
#    q = update_orientation()
#    ab = update_accel_bias()
#    wb = update_gyro_bias()
#    g = update_gravity()

#    xt_dot = np.vstack((p, v, q, ab, wb, g))
#    return xt_dot


if __name__ == "__main__":
    q = np.array([[1.0], [0.0],[0.0],[0.0]]) #default quaternion

    while count in range(len(df)-1):
        count = 1

        """INITIALIZAITION OF VARIABLES"""
        #angular velocities
        wx = df['GyroX(rad/s)'][count]
        wy = df['GyroY(rad/s)'][count]
        wz = df['GyroZ(rad/s)'][count]
        wbx = np.zeros((3,1))
        wby = np.zeros((3,1))
        wbz = np.zeros((3,1))
        wnx = np.zeros((3,1))
        wny = np.zeros((3,1))
        wnz = np.zeros((3,1))

        #accelerations
        ax = df['AccX(m/s^2)'][count]
        ay = df['AccY(m/s^2)'][count]
        az = df['AccZ(m/s^2)'][count]
        abx = np.zeros((3,1))
        aby = np.zeros((3,1))
        abz = np.zeros((3,1))
        anx = np.zeros((3,1))
        an_y = np.zeros((3,1))
        anz = np.zeros((3,1))

        wm = np.array([[wx], [wy], [wz]]) #measured angular velocity
        wb = np.array([[wbx], [wby], [wbz]]) #bias for angular velocity
        wn = np.array([[wnx], [wny], [wnz]]) #noise for angular velocity

        am = np.array([[ax], [ay], [az]]) #measured acceleration
        ab = np.array([[abx], [aby], [abz]]) #bias for acceleration
        an = np.array([[anx], [an_y], [anz]]) #bias for acceleration

        X_dot = propogate_nominal_state(q, wb, wm, wn)

##########################################################################################
        #Ua = am - an
        #Uw = wm - wn
        #Uk = np.vstack((Ua, Uw))
        #phi = Uk * dt

        #Now i need [[delta_theta_k], [delta_wb_k]]
        # delta_theta = 
        #delta_Xk = propogate_error_state(wm, wn, am, an, dt) """TODO"""