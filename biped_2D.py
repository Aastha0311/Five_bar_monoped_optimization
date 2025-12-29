import numpy as np
import mpc_utils as mpc
import plotting_utils
class Robot_biped_2D:
    def __init__(self, mass=5.0, moment_of_inertia=0.02, L_thigh=0.2, L_shank=0.2, friction_coeff=0.8):
        self.mass    = mass
        self.moment_of_inertia = moment_of_inertia 
        self.L_thigh = L_thigh
        self.L_shank = L_shank
        self.friction_coeff = friction_coeff

        # com_pos: [x(t), y(t), theta(t)]
        # com_vel: [x_dot(t), y_dot(t), theta_dot(t)]
        # action:  [F_r_x, F_r_y, F_l_x, F_l_y]
        # foot_pos_body_frame: [p_r_x, p_r_y, p_l_x, p_l_y]

        self.com_pos = np.array([0,0,0])
        self.com_vel = np.array([0,0,0])
        self.action  = np.array([0,0,0,0])

    def A_hat(self):
        A_hat = np.array([[0,0,0,1,0,0,0],
                          [0,0,0,0,1,0,0],
                          [0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,-1],
                          [0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0]])
        return A_hat
    
    def B_hat(self, com_pos, foot_pos_world_frame, swing_foot='right'):
        m_inv = 1/self.mass
        I = self.moment_of_inertia

        p_c_x = com_pos[0]
        p_c_y = com_pos[1]

        p_r_x = foot_pos_world_frame[0]
        p_r_y = foot_pos_world_frame[1]
        p_l_x = foot_pos_world_frame[2]
        p_l_y = foot_pos_world_frame[3]

        if swing_foot == 'right':
            I_right = 0
            I_left  = 1
        else:
            I_right = 1
            I_left  = 0

        B_hat = np.array([[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [m_inv,0,m_inv,0],
                          [0,m_inv,0,m_inv],
                          [(-(p_r_y - p_c_y)/I)*I_right,(p_r_x - p_c_x)/I*I_right,(-(p_l_y - p_c_y)/I)*I_left,(p_l_x - p_c_x)/I*I_left],
                          [0,0,0,0]])
        return B_hat

    def dynamics(self, X, u, com_pos, foot_pos_world_frame):
        A = self.A_hat()
        B = self.B_hat(com_pos=com_pos, foot_pos_world_frame=foot_pos_world_frame)
        X_dot = np.dot(A,X) + np.dot(B,u)

        return X_dot

class Controller:
    def __init__(self, robot: Robot_biped_2D=None, time_horizon=10, stepping_frequency=0.1, control_dt=0.1):
        self.robot = robot
        self.time_horizon = time_horizon
        self.control_dt = control_dt
        self.stepping_frequency = stepping_frequency 
        self.Kp = 0.15
        self.swing_height = 0.1

    def stand(self, foot_pos_world_frame):
        g = 9.81
        weight_of_robot = self.robot.mass * g
        foot_force = np.array([0, weight_of_robot/2, 0, weight_of_robot/2])
        return foot_force, foot_pos_world_frame
    
    def walk(self, init_state, foot_pos_world_frame, desired_velocity, desired_ang_vel, swing_foot = 'right'):

        ## Stance foot
        A_mat = self.robot.A_hat()
        B_mat = self.robot.B_hat(com_pos=init_state[0:3], foot_pos_world_frame=foot_pos_world_frame, swing_foot=swing_foot)

        reference_trajectory = mpc.generate_reference_trajectory(init_state, desired_velocity, desired_ang_vel, self.time_horizon, self.control_dt)
        A_lifted, B_lifted = mpc.get_lifted_dynamics_matrices(A_mat, B_mat, dt=self.control_dt)
        force_constraints = mpc.get_force_constraints(self.robot.friction_coeff, self.robot.mass, self.time_horizon)
        optimal_foot_force = mpc.solve_qp(init_state, reference_trajectory, A_lifted, B_lifted, force_constraints)

        ## Swing foot
        expected_foot_x = init_state[0] + 0.5*init_state[3] * 1/self.stepping_frequency
        raiberts_correction = -(desired_velocity[0] - init_state[3])*self.Kp
        expected_foot_x += raiberts_correction
        if swing_foot == 'right':
            optimal_foot_pos = np.array([expected_foot_x, -0.3, foot_pos_world_frame[2], foot_pos_world_frame[3]])
        else:
            optimal_foot_pos = np.array([foot_pos_world_frame[0], foot_pos_world_frame[1], expected_foot_x, -0.3])

        return optimal_foot_force[:4], optimal_foot_pos


class Simulator:
    def __init__(self, robot, controller, sim_dt):
        self.robot = robot
        self.controller = controller
        self.sim_dt = sim_dt

        self.phase = 0  # Goes from 0 to 1. During 0-0.5, right foot swings. During 0.5-1, left foot swings.
        self.state = np.array([0,0,0,0,0,0,0])
        self.foot_pos_world_frame = np.array([0.1, -0.3, -0.1, -0.3])  # right foot (x,y), left foot (x,y)

        # Data recording for plotting
        self.time_history = []
        self.state_history = []
        self.foot_pos_history = []
        self.foot_force_history = []
        self.phase_history = []
        self.current_time = 0.0

    def simulate_the_robot(self, foot_force, swing_foot_pos):
        ## Foot force: moves the single rigid body
        # Extract current state
        com_pos = self.state[0:3]
        com_vel = self.state[3:6]

        # Record data before updating
        self.time_history.append(self.current_time)
        self.state_history.append(self.state.copy())
        self.foot_pos_history.append(self.foot_pos_world_frame.copy())
        self.foot_force_history.append(foot_force.copy())
        self.phase_history.append(self.phase)

        # Update dynamics using RK4 integration
        X_dot = self.robot.dynamics(self.state, foot_force, com_pos, self.foot_pos_world_frame)
        self.state = self.state + X_dot * self.sim_dt

        # Apply gravity to y-velocity
        self.state[4] += -9.81 * self.sim_dt

        # Update time
        self.current_time += self.sim_dt

        ## Swing foot position: updates the foot position target in world frame
        # and based on the phase, we calculate the new foot position. Simple curve extrapolation (taking swing height from the controller)

        # Update phase
        phase_increment = self.sim_dt * self.controller.stepping_frequency
        self.phase = (self.phase + phase_increment) % 1.0

        # Determine which foot is swinging
        if self.phase < 0.5:
            # Right foot swings (0 to 0.5)
            swing_phase = self.phase / 0.5  # Normalized swing phase (0 to 1)

            # Interpolate swing foot position with swing height
            target_x = swing_foot_pos[0]
            target_y = swing_foot_pos[1]
            current_x = self.foot_pos_world_frame[0]
            current_y = self.foot_pos_world_frame[1]

            # Parabolic trajectory for swing height
            swing_height_multiplier = 4 * swing_phase * (1 - swing_phase)  # Max at phase=0.5

            self.foot_pos_world_frame[0] = current_x + (target_x - current_x) * swing_phase
            self.foot_pos_world_frame[1] = current_y + (target_y - current_y) * swing_phase + self.controller.swing_height * swing_height_multiplier

        else:
            # Left foot swings (0.5 to 1.0)
            swing_phase = (self.phase - 0.5) / 0.5  # Normalized swing phase (0 to 1)

            # Interpolate swing foot position with swing height
            target_x = swing_foot_pos[2]
            target_y = swing_foot_pos[3]
            current_x = self.foot_pos_world_frame[2]
            current_y = self.foot_pos_world_frame[3]

            # Parabolic trajectory for swing height
            swing_height_multiplier = 4 * swing_phase * (1 - swing_phase)  # Max at phase=0.5

            self.foot_pos_world_frame[2] = current_x + (target_x - current_x) * swing_phase
            self.foot_pos_world_frame[3] = current_y + (target_y - current_y) * swing_phase + self.controller.swing_height * swing_height_multiplier

    def plot_graphs(self):
        """Generate comprehensive plots of the simulation results."""
        if len(self.time_history) == 0:
            print("No data to plot. Run the simulation first.")
            return

        plotting_utils.plot_simulation_results(
            time_data=self.time_history,
            state_data=self.state_history,
            foot_pos_data=self.foot_pos_history,
            foot_force_data=self.foot_force_history,
            phase_data=self.phase_history,
            L_thigh=self.robot.L_thigh,
            L_shank=self.robot.L_shank
        )



if __name__ == "__main__":
    robot = Robot_biped_2D(mass=5.0, moment_of_inertia=0.02, L_thigh=0.2, L_shank=0.2, friction_coeff=0.8)
    controller = Controller(robot, control_dt=0.1, time_horizon=10, stepping_frequency=0.5)
    simulator = Simulator(robot, controller, sim_dt=0.01)

    desired_velocity = np.array([0.5, 0.0])  # desired forward velocity in x and y
    desired_ang_vel = 0.0  # desired angular velocity

    try:
        while True:
            if simulator.phase < 0.5:
                swing_foot = 'right'
            else:  
                swing_foot = 'left'
            foot_force, swing_foot_pos = controller.walk(simulator.state, simulator.foot_pos_world_frame, desired_velocity, desired_ang_vel, swing_foot=swing_foot)
            for _ in range(int(controller.control_dt/simulator.sim_dt)):
                simulator.simulate_the_robot(foot_force, swing_foot_pos)

    except KeyboardInterrupt:
        print("Simulation terminated.")
        simulator.plot_graphs()
