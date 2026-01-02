# Error-State Kalman Filter (ESKF) with Quaternion-Based Orientation
Current Implementation is in python but soon to be c++

A Python implementation of an Error-State Kalman Filter for fusing IMU (Inertial Measurement Unit) data with position/velocity measurements to estimate 3D orientation, velocity, position, and sensor biases.

## Data

<table>
  <tr>
    <td width="50%"><img src="https://github.com/HyunLee8/ErrorStateKalmanFilter/blob/c3cf1b0c57cff196f236105c246aefd366c0b9c2/results/orientation_animation.gif?raw=true" width="100%" /></td>
    <td width="50%"><img src="https://github.com/HyunLee8/Error-State-Kalman-Filter/blob/main/results/Screenshot%202026-01-02%20at%202.55.49%E2%80%AFPM.png?raw=true" width="100%" /></td>
  </tr>
  <tr>
    <td width="50%"><img src="https://github.com/HyunLee8/Error-State-Kalman-Filter/blob/main/results/Screenshot%202026-01-02%20at%202.56.02%E2%80%AFPM.png?raw=true" width="100%" /></td>
    <td width="50%"><img src="https://github.com/HyunLee8/Error-State-Kalman-Filter/blob/main/results/Screenshot%202026-01-02%20at%202.56.18%E2%80%AFPM.png?raw=true" width="100%" /></td>
  </tr>
</table>

## Overview

This ESKF implementation uses quaternion representation for robust orientation estimation, avoiding gimbal lock issues associated with Euler angles. The filter predicts state using gyroscope and accelerometer data, then corrects the estimates using position or velocity measurements.

**State Vector (16 elements):**
- Position (3D): `[x, y, z]`
- Orientation (quaternion): `[qw, qx, qy, qz]`
- Velocity (3D): `[vx, vy, vz]`
- Accelerometer bias: `[bax, bay, baz]`
- Gyroscope bias: `[bwx, bwy, bwz]`

**Error State (15 elements):**
- Position error (3D)
- Orientation error (3D rotation vector)
- Velocity error (3D)
- Accelerometer bias error (3D)
- Gyroscope bias error (3D)

## Theoretical Background

This implementation is based on the following resources:

1. **Basic Kalman Filter Intuition**: [How a Kalman Filter Works (in Pictures)](https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/)
2. **ESKF with Quaternions** (Sections 5.1-5.4.3): [Quaternion kinematics for the error-state Kalman filter](https://arxiv.org/pdf/1711.02508)
3. **ESKF Generalized Formulas**: [IEEE Paper on ESKF](https://ieeexplore.ieee.org/document/9206131)
4. **Quaternion Intuition**: [Visualizing Quaternions](https://www.youtube.com/watch?v=d4EgbgTm0Bg)
5. **Jacobian Matrix Intuition**: [Jacobian Matrix Explained](https://www.youtube.com/watch?v=6FesMicc844)
6. **Extended Kalman Filter Visualization**: [Extended Kalman Filter Visualized](https://www.youtube.com/watch?v=9X3jGGnbcvU&t=55s)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ErrorStateKalmanFilter.git
cd ErrorStateKalmanFilter
```

2. Install required dependencies:
```bash
pip install numpy pandas pyquaternion matplotlib
```

### Running the Filter

1. Prepare your sensor data files in the `src/` directory:
   - `IMU_data.csv` - IMU sensor measurements
   - `motion_data.csv` - Position/velocity measurements (optional for updates)

2. Run the filter:
```bash
cd src
python main.py
```
**Parameter Guidelines:**
- **`sig_a_noise`**: Set based on your accelerometer datasheet noise density
- **`sig_w_noise`**: Set based on your gyroscope datasheet noise density
- **`sig_a_walk`**: Accelerometer bias instability (how fast bias drifts)
- **`sig_w_walk`**: Gyroscope bias instability (how fast bias drifts)
- **`gravity`**: Local gravitational acceleration (typically 9.81 m/s²)

## Algorithm Details

### Prediction Step

1. **Remove biases** from IMU measurements
2. **Transform acceleration** to global frame using current orientation
3. **Integrate** position and velocity using kinematic equations
4. **Update orientation** via quaternion multiplication
5. **Propagate covariance** using linearized error-state dynamics

### Update Step

1. **Compute innovation** (measurement - prediction)
2. **Calculate Kalman gain** using current covariance and measurement noise
3. **Update error state** with innovation
4. **Inject error state** into nominal state:
   - Position: direct addition
   - Orientation: quaternion multiplication
   - Velocity and biases: direct addition
5. **Update covariance** using Joseph form (numerically stable)
6. **Reset error state** to zero
