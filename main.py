import numpy as np

from components.robot import Robot_biped_2D
from components.simulator import Simulator
from components.controller import Controller


if __name__ == "__main__":
    robot = Robot_biped_2D(mass=5.0, moment_of_inertia=0.02, L_thigh=0.2, L_shank=0.2, friction_coeff=0.8, density=500)
    controller = Controller(robot, control_dt=0.1, time_horizon=10, stepping_frequency=2) 
    simulator = Simulator(robot, controller, sim_dt=0.1, visualize=True)

    desired_velocity = np.array([0.1, 0.0])  # desired forward velocity in x and y (reduced for testing)
    desired_ang_vel = 0.0  # desired angular velocity

    try:
        while True:
            if simulator.phase < 0.5:
                swing_foot = 'right'
            else:  
                swing_foot = 'left'
            foot_force, swing_foot_pos = controller.walk(simulator.state, simulator.foot_pos_world_frame, desired_velocity, desired_ang_vel, swing_foot=swing_foot)
            for _ in range(int(controller.control_dt/simulator.sim_dt)):
                simulator.simulate_the_robot(foot_force, swing_foot_pos, swing_foot)

    except KeyboardInterrupt:
        print("Simulation terminated.")
        simulator.plot_graphs()
