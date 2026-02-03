import numpy as np
import matplotlib.pyplot as plt


def plot_simulation_results(time,
                            state,
                            foot_pos,
                            foot_force,
                            phase,
                            joint_angles,
                            joint_torques):

    time = np.array(time)
    state = np.array(state)
    foot_pos = np.array(foot_pos)
    foot_force = np.array(foot_force)
    phase = np.array(phase)
    joint_angles = np.array(joint_angles)
    joint_torques = np.array(joint_torques)

    hip = np.rad2deg(joint_angles[:, 0])
    knee = np.rad2deg(joint_angles[:, 1])

    fig, axs = plt.subplots(3, 2, figsize=(14, 10))

    axs[0, 0].plot(time, state[:, 0], label="x")
    axs[0, 0].plot(time, state[:, 1], label="y")
    axs[0, 0].set_title("COM Position")
    axs[0, 0].legend()

    axs[0, 1].plot(time, state[:, 3], label="xdot")
    axs[0, 1].plot(time, state[:, 4], label="ydot")
    axs[0, 1].set_title("COM Velocity")
    axs[0, 1].legend()

    axs[1, 0].plot(time, foot_pos[:, 0], label="Foot X")
    axs[1, 0].plot(time, foot_pos[:, 1], label="Foot Y")
    axs[1, 0].set_title("Foot Position")
    axs[1, 0].legend()

    axs[1, 1].plot(time, phase)
    axs[1, 1].set_title("Phase")

    axs[2, 0].plot(time, hip, label="Hip")
    axs[2, 0].plot(time, knee, label="Knee")
    axs[2, 0].set_title("Joint Angles (deg)")
    axs[2, 0].legend()

    axs[2, 1].plot(time, foot_force[:, 0], label="Fx")
    axs[2, 1].plot(time, foot_force[:, 1], label="Fy")
    axs[2, 1].set_title("Foot Forces")
    axs[2, 1].legend()

    plt.tight_layout()
    plt.show()
