# Extended Error State Kalman Filter w/ Quaternions

This project implements an Extended Error State Kalman Filter (ESKF) designed for systems that estimate 3D orientation, position, and velocity using quaternion-based attitude representation. The ESKF formulation decomposes the system state into a nominal state and a small error state, enabling numerically stable and computationally efficient filtering for inertial navigation and sensor fusion applications.

## Orientation vs. Time

![image alt](https://github.com/HyunLee8/Quaternion-Filter/blob/main/docs/orientation_time.png?raw=true)

## Euler vs. Time

![image alt](https://github.com/HyunLee8/Quaternion-Filter/blob/main/docs/euler_time.png?raw=true)

resources:

basic filter: 
https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/

ESKF with quaternions discretized Sections 5.1-5.4.3: 
https://arxiv.org/pdf/1711.02508

ESKF generalized formulas: 
https://ieeexplore.ieee.org/document/9206131

Quaternion intuition: 
https://www.youtube.com/watch?si=hcMDjBNQwnIs122w&v=d4EgbgTm0Bg&feature=youtu.be

Jacobian Matrix intuition: 
https://www.youtube.com/watch?v=6FesMicc844

Extended Kalman visualized: 
https://www.youtube.com/watch?v=9X3jGGnbcvU&t=55s



