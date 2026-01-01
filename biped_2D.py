import numpy as np
import mpc_utils as mpc
import plotting_utils
from scipy.linalg import expm

GRAVITY = 9.81
class Robot_biped_2D:
    def __init__(self, mass, moment_of_inertia, L_thigh, L_shank, friction_coeff):
        self.mass    = mass
        self.moment_of_inertia = moment_of_inertia
        self.L_thigh = L_thigh
        self.L_shank = L_shank
        self.friction_coeff = friction_coeff

        # Compute nominal COM height based on leg geometry
        # Use 85% of leg length to leave margin for leg extension
        self.leg_length = L_thigh + L_shank
        self.nominal_com_height = self.leg_length * 0.85
        self.ground_level = 0.0  # Ground at y=0

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
                          [0,0,0,0,0,0,1],
                          [0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0]])
        return A_hat
    
    def B_hat(self, com_pos, foot_pos_world_frame, swing_foot):
        m_inv = 1/self.mass
        I = self.moment_of_inertia

        p_c_x = com_pos[0]
        p_c_y = com_pos[1]

        p_r_x = foot_pos_world_frame[0]
        p_r_y = foot_pos_world_frame[1]
        p_l_x = foot_pos_world_frame[2]
        p_l_y = foot_pos_world_frame[3]

        if swing_foot == 'right':
            I_right = 0  # Right foot in swing, doesn't apply forces
            I_left  = 1  # Left foot in stance, applies forces
        else:
            I_right = 1  # Right foot in stance, applies forces
            I_left  = 0  # Left foot in swing, doesn't apply forces

        B_hat = np.array([[0,0,0,0],
                          [0,0,0,0],
                          [0,0,0,0],
                          [m_inv*I_right, 0, m_inv*I_left, 0],  # Only stance foot affects x acceleration
                          [0, m_inv*I_right, 0, m_inv*I_left],  # Only stance foot affects y acceleration
                          [(-(p_r_y - p_c_y)/I)*I_right, (p_r_x - p_c_x)/I*I_right, (-(p_l_y - p_c_y)/I)*I_left, (p_l_x - p_c_x)/I*I_left],
                          [0,0,0,0]])
        return B_hat

    def dynamics(self, X, u, com_pos, foot_pos_world_frame):
        A = self.A_hat()
        B = self.B_hat(com_pos=com_pos, foot_pos_world_frame=foot_pos_world_frame)
        X_dot = np.dot(A,X) + np.dot(B,u)

        return X_dot
    
    def step_lti(self, X, u, com_pos, foot_pos_world_frame, swing_foot, dt):
        A = self.A_hat()
        B = self.B_hat(com_pos=com_pos, foot_pos_world_frame=foot_pos_world_frame, swing_foot=swing_foot)

        n = A.shape[0]
        m = B.shape[1]

        M = np.zeros((n + m, n + m))
        M[:n, :n] = A
        M[:n, n:] = B

        Md = expm(M * dt)

        Ad = Md[:n, :n]
        Bd = Md[:n, n:]

        return Ad @ X + Bd @ u


class Controller:
    def __init__(self, robot: Robot_biped_2D, time_horizon, stepping_frequency, control_dt):
        self.robot = robot
        self.time_horizon = time_horizon
        self.control_dt = control_dt
        self.stepping_frequency = stepping_frequency 
        self.Kp = 0.15
        self.swing_height = 0.1

    def stand(self, state, foot_pos_world_frame, desired_height):
        g = GRAVITY
        weight_of_robot = self.robot.mass * g
        foot_force_per_foot = weight_of_robot / 2 + 0.3 * (desired_height - state[1])
        foot_force = np.array([0, foot_force_per_foot, 0, foot_force_per_foot])
        return foot_force, foot_pos_world_frame
    
    def walk(self, init_state, foot_pos_world_frame, desired_velocity, desired_ang_vel, swing_foot):

        ## Stance foot
        A_mat = self.robot.A_hat()
        B_mat = self.robot.B_hat(com_pos=init_state[0:3], foot_pos_world_frame=foot_pos_world_frame, swing_foot=swing_foot)

        reference_trajectory = mpc.generate_reference_trajectory(init_state, desired_velocity, desired_ang_vel, self.time_horizon, self.control_dt, desired_height=self.robot.nominal_com_height)
        A_lifted, B_lifted = mpc.get_lifted_dynamics_matrices(A_mat, B_mat, self.time_horizon, dt=self.control_dt)
        force_constraints = mpc.get_force_constraints(self.robot.friction_coeff, self.robot.mass, self.time_horizon, swing_foot=swing_foot)
        optimal_foot_force = mpc.solve_qp(init_state, reference_trajectory, A_lifted, B_lifted, force_constraints)

        # Debug output - print first 10 steps to see failure point
        if not hasattr(self, '_step_count'):
            self._step_count = 0
        self._step_count += 1

        if self._step_count <= 10:
            total_fy = optimal_foot_force[1] + optimal_foot_force[3]
            # Calculate stance foot distance
            if swing_foot == 'right':
                stance_x, stance_y = foot_pos_world_frame[2], foot_pos_world_frame[3]
            else:
                stance_x, stance_y = foot_pos_world_frame[0], foot_pos_world_frame[1]
            dist = np.sqrt((init_state[0] - stance_x)**2 + (init_state[1] - stance_y)**2)
            print(f"Step {self._step_count:3d} | Swing: {swing_foot:5s} | COM=({init_state[0]:5.2f},{init_state[1]:5.2f}) | Stance dist={dist:.3f}m | Fy={total_fy:5.1f}N")
        # forward_force = 0.3 * (desired_velocity[0] - init_state[3])
        # upward_force = self.robot.mass * GRAVITY + 0.3 * (0.3 - init_state[1])
        # if swing_foot == 'right':
        #     optimal_foot_force = np.array([forward_force, upward_force, 0, 0])
        # else:
        #     optimal_foot_force =  np.array([0, 0, forward_force, upward_force])

        ## Swing foot
        expected_foot_x = init_state[0] + 0.5*init_state[3] * 1/self.stepping_frequency
        raiberts_correction = -(desired_velocity[0] - init_state[3])*self.Kp
        expected_foot_x += raiberts_correction

        if swing_foot == 'right':
            optimal_foot_pos = np.array([expected_foot_x, self.robot.ground_level, foot_pos_world_frame[2], foot_pos_world_frame[3]])
        else:
            optimal_foot_pos = np.array([foot_pos_world_frame[0], foot_pos_world_frame[1], expected_foot_x, self.robot.ground_level])

        return optimal_foot_force[:4], optimal_foot_pos


class Simulator:
    def __init__(self, robot: Robot_biped_2D, controller: Controller, sim_dt):
        self.robot = robot
        self.controller = controller
        self.sim_dt = sim_dt

        self.phase = 0  # Goes from 0 to 1. During 0-0.5, right foot swings. During 0.5-1, left foot swings.

        # Initialize with COM at nominal height and feet on ground
        self.state = np.array([0, robot.nominal_com_height, 0, 0, 0, 0, -GRAVITY])

        # Feet on ground with narrow stance to stay within leg reach
        stance_width = 0.15
        self.foot_pos_world_frame = np.array([-stance_width, robot.ground_level, stance_width, robot.ground_level])

        # Data recording for plotting
        self.time_history = []
        self.state_history = []
        self.foot_pos_history = []
        self.foot_force_history = []
        self.phase_history = []
        self.current_time = 0.0

    def simulate_the_robot(self, foot_force, swing_foot_pos, swing_foot):
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
        # X_dot = self.robot.dynamics(self.state, foot_force, com_pos, self.foot_pos_world_frame)
        # self.state = self.state + X_dot * self.sim_dt
        self.state = self.robot.step_lti(self.state, foot_force, com_pos, self.foot_pos_world_frame, swing_foot, self.sim_dt)

        # Update time
        self.current_time += self.sim_dt

        ## Swing foot position: updates the foot position target in world frame
        # and based on the phase, we calculate the new foot position. Simple curve extrapolation (taking swing height from the controller)

        # Update phase and track phase transitions to capture initial foot positions
        old_phase = self.phase
        phase_increment = self.sim_dt * self.controller.stepping_frequency
        self.phase = (self.phase + phase_increment) % 1.0

        # Detect phase transitions and store initial foot positions for swing interpolation
        if not hasattr(self, '_swing_start_pos'):
            # Initialize storage for swing start positions
            self._swing_start_pos = {
                'right': np.array([self.foot_pos_world_frame[0], self.foot_pos_world_frame[1]]),
                'left': np.array([self.foot_pos_world_frame[2], self.foot_pos_world_frame[3]])
            }

        # Detect when right foot swing starts (phase crosses 0)
        # This means left foot has finished swinging - snap it to final position
        if old_phase > 0.5 and self.phase < 0.5:
            # Left foot just finished swinging - ensure it's at target position (ground)
            self.foot_pos_world_frame[2] = swing_foot_pos[2]
            self.foot_pos_world_frame[3] = swing_foot_pos[3]
            # Now capture right foot starting position for its swing
            self._swing_start_pos['right'] = np.array([self.foot_pos_world_frame[0], self.foot_pos_world_frame[1]])

        # Detect when left foot swing starts (phase crosses 0.5)
        # This means right foot has finished swinging - snap it to final position
        if old_phase < 0.5 and self.phase >= 0.5:
            # Right foot just finished swinging - ensure it's at target position (ground)
            self.foot_pos_world_frame[0] = swing_foot_pos[0]
            self.foot_pos_world_frame[1] = swing_foot_pos[1]
            # Now capture left foot starting position for its swing
            self._swing_start_pos['left'] = np.array([self.foot_pos_world_frame[2], self.foot_pos_world_frame[3]])

        # Determine which foot is swinging and interpolate
        if self.phase < 0.5:
            # Right foot swings (0 to 0.5)
            swing_phase = self.phase / 0.5  # Normalized swing phase (0 to 1)

            # Interpolate from initial position to target
            target_x = swing_foot_pos[0]
            target_y = swing_foot_pos[1]
            initial_x = self._swing_start_pos['right'][0]
            initial_y = self._swing_start_pos['right'][1]

            # Parabolic trajectory for swing height
            swing_height_multiplier = 4 * swing_phase * (1 - swing_phase)  # Max at phase=0.5

            self.foot_pos_world_frame[0] = initial_x + (target_x - initial_x) * swing_phase
            self.foot_pos_world_frame[1] = initial_y + (target_y - initial_y) * swing_phase + self.controller.swing_height * swing_height_multiplier

        else:
            # Left foot swings (0.5 to 1.0)
            swing_phase = (self.phase - 0.5) / 0.5  # Normalized swing phase (0 to 1)

            # Interpolate from initial position to target
            target_x = swing_foot_pos[2]
            target_y = swing_foot_pos[3]
            initial_x = self._swing_start_pos['left'][0]
            initial_y = self._swing_start_pos['left'][1]

            # Parabolic trajectory for swing height
            swing_height_multiplier = 4 * swing_phase * (1 - swing_phase)  # Max at phase=0.5

            self.foot_pos_world_frame[2] = initial_x + (target_x - initial_x) * swing_phase
            self.foot_pos_world_frame[3] = initial_y + (target_y - initial_y) * swing_phase + self.controller.swing_height * swing_height_multiplier

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
    controller = Controller(robot, control_dt=0.1, time_horizon=10, stepping_frequency=0.5)  # Much slower: 0.5 Hz = 2s per step
    simulator = Simulator(robot, controller, sim_dt=0.1)

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
