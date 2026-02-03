import numpy as np

from components.robot_monoped import Robot_monoped_2D
from components.simulator_monoped import Simulator
from components.controller_monoped import Controller


if __name__ == "__main__":

    # -----------------------------
    # Robot
    # -----------------------------
    robot = Robot_monoped_2D(
        mass=5.0,
        moment_of_inertia=0.02,
        L_thigh=0.2,
        L_shank=0.2,
        friction_coeff=0.8,
        density=500
    )

    # -----------------------------
    # Controller
    # -----------------------------
    controller = Controller(
        robot=robot,
        control_dt=0.1,
        time_horizon=10,
        stepping_frequency=2.0,   # hop frequency (Hz)
        raibert_gain=0.08
    )

    # -----------------------------
    # Simulator
    # -----------------------------
    simulator = Simulator(
        robot=robot,
        controller=controller,
        sim_dt=0.01,              # smaller dt = smoother simulation
        visualize=True,
        stance_fraction=0.6    # 60% stance, 40% flight
    )

    # -----------------------------
    # Desired motion
    # -----------------------------
    desired_velocity = np.array([1, 0.0])  # forward velocity
    desired_ang_vel = 0.0

    print("Starting monoped simulation (Ctrl+C to stop)")

    try:
        while True:
            simulator.step(
                desired_velocity=desired_velocity,
                desired_ang_vel=desired_ang_vel
            )

    except KeyboardInterrupt:
        print("\nSimulation terminated.")
        simulator.plot_results()
