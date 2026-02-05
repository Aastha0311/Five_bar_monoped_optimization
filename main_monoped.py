# import numpy as np

# from components.robot_monoped import Robot_monoped_2D
# from components.simulator_monoped import Simulator
# from components.controller_monoped import Controller


# if __name__ == "__main__":

#     # -----------------------------
#     # Robot
#     # -----------------------------
#     robot = Robot_monoped_2D(
#         mass=5.0,
#         moment_of_inertia=0.02,
#         L_thigh=0.2,
#         L_shank=0.2,
#         friction_coeff=0.8,
#         density=500
#     )

#     # -----------------------------
#     # Controller
#     # -----------------------------
#     controller = Controller(
#         robot=robot,
#         control_dt=0.1,
#         time_horizon=10,
#         stepping_frequency=2.0,   # hop frequency (Hz)
#         raibert_gain=0.08
#     )

#     # -----------------------------
#     # Simulator
#     # -----------------------------
#     simulator = Simulator(
#         robot=robot,
#         controller=controller,
#         sim_dt=0.01,              # smaller dt = smoother simulation
#         visualize=True,
#         stance_fraction=0.6    # 60% stance, 40% flight
#     )

#     # -----------------------------
#     # Desired motion
#     # -----------------------------
#     desired_velocity = np.array([1, 0.0])  # forward velocity
#     desired_ang_vel = 0.0

#     print("Starting monoped simulation (Ctrl+C to stop)")

#     try:
#         while True:
#             simulator.step(
#                 desired_velocity=desired_velocity,
#                 desired_ang_vel=desired_ang_vel
#             )

#     except KeyboardInterrupt:
#         print("\nSimulation terminated.")
#         simulator.plot_results()

from components.robot_monoped import Robot_monoped_2D
from components.simulator_monoped import Simulator
from components.controller_monoped import Controller
import numpy as np


if __name__ == "__main__":
    robot = Robot_monoped_2D(mass=5.0, moment_of_inertia=0.02, L_thigh=0.2, L_shank=0.2, friction_coeff=0.8, density=500)
    controller = Controller(robot, control_dt=0.1, time_horizon=10, stepping_frequency=2, raibert_gain=0.08) 
    simulator = Simulator(robot, controller, sim_dt=0.01, visualize=True, stance_fraction=0.6)

    desired_velocity = np.array([0.1, 0.0])  # desired forward velocity in x and y (reduced for testing)
    desired_ang_vel = 0.0  # desired angular velocity

    # try:
    #     while True:
    #         if simulator.phase < simulator.stance_fraction:
    #             in_stance = 1
    #         else:  
    #             in_stance = 0
    #         foot_force, swing_foot_pos = controller.walk(simulator.state, simulator.foot_pos_world_frame, desired_velocity, desired_ang_vel, in_stance=in_stance)
    #         for _ in range(int(controller.control_dt/simulator.sim_dt)):
    #             simulator.simulate_the_robot(foot_force, swing_foot_pos, in_stance)
    #         simulator.plot_graphs()

    # except KeyboardInterrupt:
    #     print("Simulation terminated.")
    #     simulator.plot_graphs()

    # try:
    #     while True:
    #         old_phase = simulator.phase

    #         in_stance = simulator.phase < simulator.stance_fraction

    #         foot_force, swing_foot_pos = controller.walk(
    #             simulator.state,
    #             simulator.foot_pos_world_frame,
    #             desired_velocity,
    #             desired_ang_vel,
    #             in_stance=in_stance
    #         )

    #         simulator.simulate_the_robot(
    #             foot_force,
    #             swing_foot_pos,
    #             in_stance
    #         )

    #         # Phase update ONCE
    #         simulator.phase = (
    #             simulator.phase
    #             + simulator.sim_dt * controller.stepping_frequency
    #         ) % 1.0

    #         simulator.handle_phase_transition(old_phase)

    # except KeyboardInterrupt:
    #     simulator.plot_graphs()


import numpy as np
from components.mujoco_env import MuJoCoMonopedEnv
from components.controller import Controller
from components.robot_monoped import Robot_monoped_2D

robot = Robot_monoped_2D(
    mass=5.0,
    moment_of_inertia=0.02,
    L_thigh=0.4,
    L_shank=0.4,
    friction_coeff=0.8,
    density=500
)

controller = Controller(
    robot,
    time_horizon=10,
    stepping_frequency=2.0,
    control_dt=0.05,
    raibert_gain=0.08
)

env = MuJoCoMonopedEnv("hopper2d.xml")

desired_velocity = np.array([0.5, 0.0])
desired_ang_vel = 0.0

while True:
    # -----------------------------
    # Get MuJoCo feedback
    # -----------------------------
    state = env.get_state()
    foot_pos = env.get_foot_position()
    in_stance = env.in_stance()

    # -----------------------------
    # MPC + Raibert
    # -----------------------------
    foot_force, _ = controller.walk(
        state,
        foot_pos,
        desired_velocity,
        desired_ang_vel,
        in_stance
    )

    # -----------------------------
    # Apply to MuJoCo
    # -----------------------------
    env.apply_foot_force(foot_force)

    # Step MuJoCo for control_dt
    env.step(n=int(controller.control_dt / 0.001))


