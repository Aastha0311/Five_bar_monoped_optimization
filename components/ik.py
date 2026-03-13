import numpy as np
import mujoco as mj 
#import len_rad_id as lri
import time

# def inverse_kinematics(x, y, l1, l2):
#     """
#     Solves for the joint angles (theta1, theta2) of a 2-DOF planar manipulator given end-effector position (x, y)
#     relative to the base.
    
#     :param x: X-coordinate of end-effector
#     :param y: Y-coordinate of end-effector
#     :param l1: Length of first link
#     :param l2: Length of second link
#     :return: Tuple (theta1, theta2) in radians
#     """
    
#     # Compute inverse kinematics using the law of cosines
#     dist = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    
#     if abs(dist) > 1:
#         raise ValueError("No solution: Target is out of reach")
    
#     theta2 = np.arctan2(np.sqrt(1 - dist**2), dist)  # Elbow-up solution
#     theta1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
    
#     return theta1, theta2

def inverse_kinematics(x, z, l1, l2, branch=1):
    c2 = (x**2 + z**2 - l1**2 - l2**2) / (2 * l1 * l2)
    if abs(c2) > 1:
        raise ValueError("Target out of reach")
    if branch == 1:
        theta2 = np.arctan2(-np.sqrt(1 - c2**2), c2)
    else:   
        theta2 = np.arctan2(np.sqrt(1 - c2**2), c2)
    #theta1 = np.arctan2(z, x) - np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))
    A = l1 + l2 * c2
    B = l2 * np.sin(theta2)
    theta1 = np.arctan2(A*x + B*z, x*B - A*z)

    #print(f"theta1: {theta1}, theta2: {theta2}, theta1n: {theta1}")
    return theta1, theta2