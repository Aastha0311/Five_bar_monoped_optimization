import numpy as np


# -------------------------
# 2R IK
# -------------------------
def ik_2r(x, z, l1, l2, elbow=1):
    D = (x**2 + z**2 - l1**2 - l2**2) / (2 * l1 * l2)

    if abs(D) > 1:
        raise ValueError("Target out of reach")

    q2 = np.arctan2(elbow * np.sqrt(1 - D**2), D)

    q1 = np.arctan2(z, x) - np.arctan2(
        l2 * np.sin(q2),
        l1 + l2 * np.cos(q2)
    )

    return q1, q2


# -------------------------
# 2R FK
# -------------------------
def fk_2r(q1, q2, l1, l2):
    x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    z = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
    return np.array([x, z])


# -------------------------
# 5-bar IK
# -------------------------
def ik_5bar(x, z, l1, l2, hip_offset):

    # left hip at -offset
    q1_l, q2_l = ik_2r(
        x + hip_offset,
        z,
        l1,
        l2,
        elbow=1
    )

    # right hip at +offset
    q1_r, q2_r = ik_2r(
        x - hip_offset,
        z,
        l1,
        l2,
        elbow=-1
    )

    return q1_l, q2_l, q1_r, q2_r



# -------------------------
# 5-bar FK
# -------------------------
def fk_5bar(q1_l, q2_l, q1_r, q2_r, l1, l2, hip_offset):
    left_hip = np.array([hip_offset, 0.0])
    right_hip = np.array([-hip_offset, 0.0])

    left_tip_local = fk_2r(q1_l, q2_l, l1, l2)
    right_tip_local = fk_2r(q1_r, q2_r, l1, l2)

    left_tip_world = left_hip + left_tip_local
    right_tip_world = right_hip + right_tip_local

    return left_tip_world, right_tip_world


# x = 0.0
# z = -0.6

# q1_l, q2_l, q1_r, q2_r = ik_5bar(x, z, 0.4, 0.4, 0.05)

# L, R = fk_5bar(q1_l, q2_l, q1_r, q2_r, 0.4, 0.4, 0.05)

# print("Left:", L)
# print("Right:", R)

