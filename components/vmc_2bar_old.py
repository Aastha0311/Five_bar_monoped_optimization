import mujoco as mj
import numpy as np
import time
import xml.etree.ElementTree as ET

class Controller:
    def __init__(self, xml_path, m, d, initial_hip_pos, initial_knee_pos, action, ori_l = 1.2, ori_theta=0):
        self.xml_path = xml_path
        self.initial_hip_pos = initial_hip_pos
        self.initial_knee_pos = initial_knee_pos
        self.m = m
        self.d = d
        self.action = action
        self.ori_l = ori_l
        self.ori_theta = ori_theta
        base_joint_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "slide_x")
        base_z_joint_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "slide_z")
        base_y_joint_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "hinge_y")
        self.base_y_joint_id = base_y_joint_id
        self.base_joint_id = base_joint_id
        self.base_z_joint_id = base_z_joint_id
        self.base_force_index = m.jnt_dofadr[base_joint_id]
        thigh_joint_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "hinge1")
        self.thigh_joint_id = thigh_joint_id
        self.hip_force_index = m.jnt_dofadr[thigh_joint_id]
        knee_joint_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "hinge2")
        self.knee_joint_id = knee_joint_id
        self.knee_force_index = m.jnt_dofadr[knee_joint_id]


    def get_ground_contact_forces(self):

        """
  Extracts ground contact forces from MuJoCo simulation data.

  Args:
    model: MuJoCo model object.
    data: MuJoCo data object.

  Returns:
    A list of contact forces, where each element is a tuple 
    containing the contact force vector and the corresponding 
    body index.
  """
        xml_path = self.xml_path
        #m = mj.MjModel.from_xml_path(xml_path)
        #d = mj.MjData(m)
        m = self.m
        d = self.d
        ground_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "floor")  # Assuming "slot" is the ground geom
        ground_contact_forces = []
        for i in range(d.ncon):  # ncon is the number of contacts
            con = d.contact[i]   # contact[i] is the i-th contact
            if con.geom1 == ground_id or con.geom2 == ground_id:
        # Calculate the contact force vector
                force = np.zeros(6, dtype=np.float64)
                mj.mj_contactForce(m, d, i, force) 

        # Extract force (first 3 elements)
                contact_force = force[:3]

        # Get the body index 
                body_id = con.geom1 if con.geom1 != ground_id else con.geom2

                ground_contact_forces.append((contact_force, body_id))

        return ground_contact_forces    
 
    
    def body_names(self):        
        num_bodies = self.m.nbody        
        body_names = [mj.mj_id2name(self.m, mj.mjtObj.mjOBJ_BODY, i) for i in range(num_bodies)]
        return body_names
    
    #to check for order of different bodies in array of bodies
    
    def body_dict(self):
        body_names = self.body_names()
        # get body index for bodies and make a dictionary
        body_id = {body_names[i]: i for i in range(len(body_names))}
        return body_id
    
    #following functions are to obtain velocity of the leg link in world frame 

    def body_cvel(self):  #COM velocity in local frame
        body_names = self.body_names()
        body_id = self.body_dict()
        # make a list of linear velocities of all bodies
        body_velocity = [self.d.cvel[body_id[body_name]]for body_name in body_names]
        return body_velocity
    
    def leg_cvel(self):
        #TODO: check the name for leg link using body_names    
        body_id = self.body_dict()
        # make a list of linear velocities of all bodies
        body_velocity = self.body_cvel()
        # get the index of the leg link from dictionary of body_id
        leg_index = body_id['link2']
        # get the linear velocity of the leg link
        leg_velocity = body_velocity[leg_index]
        return leg_velocity
    
    def thigh_joint_position(self):
        # get the index of thigh joint
        thigh_joint_id = self.thigh_joint_id
        # get the position of thigh joint
        thigh_joint_position = self.d.qpos[thigh_joint_id] 
        thigh_total_position = thigh_joint_position + self.initial_hip_pos #NOTE: Initial Thigh angle accounted for
        return thigh_total_position
    
    def rotmat_thigh(self):
        # get the position of thigh joint
        thigh_position = self.thigh_joint_position()
        # get the rotation matrix of thigh joint
        rotmat_thigh = np.array([[np.cos(thigh_position), -np.sin(thigh_position)], [np.sin(thigh_position), np.cos(thigh_position)]])
        return rotmat_thigh
    
    def leg_world_velocity(self):
        # get the linear velocity of the leg link
        leg_velocity = self.leg_cvel()
        leg_xz_velocity = np.array([leg_velocity[0], leg_velocity[2]])
        # get the rotation matrix of thigh joint
        rotmat_thigh = self.rotmat_thigh()
        leg_world_velocity = np.dot(rotmat_thigh, leg_xz_velocity)
        return leg_world_velocity  
    
    #function to obtain base linear velocity in world frame
    def base_cvel(self):
        # get the linear velocity of the base link
        body_id = self.body_dict()
        base_index = body_id['base']
        base_velocity = self.d.cvel[base_index]
        base_z_velocity = base_velocity[2]
        return base_z_velocity
    
    #function to obtain base jpint velocity (rootz)
    def rootz_velocity(self):
        # get the index of base joint
        base_joint_id = self.base_joint_id
        # get the position of base joint
        base_velocity= self.d.qvel[base_joint_id]
        return base_velocity
    
    #function to obtain number of joints in the model
    def num_joints(self):
        num_joints = self.m.njnt
        return num_joints

    def joint_names(self):
        num_joints = self.num_joints()
        #make a dictionary of joint names
        joint_names = {mj.mj_id2name(self.m, mj.mjtObj.mjOBJ_JOINT, i): i for i in range(num_joints)}
        return joint_names       
        
      
    def lin_sd_var(self, action):
        base_velocity = self.base_cvel()
        if base_velocity>= 0:
            lin_kp = action[0]
            lin_kd = action[1]
            #lin_kd = 0
        elif base_velocity<0:
            lin_kp = action[0]
            lin_kd = action[1]
            #lin_kd = 0
        return lin_kp, lin_kd
    
    def rot_sd_var(self,action):
        #TODO: CHECK FOR LOGIC AT REACHING GROUND AGAIN 
        base_velocity = self.base_cvel()
        if base_velocity>= 0:
            rot_kp = action[2]
            rot_kd = 0
        elif base_velocity<0:
            rot_kp = action[2]
            rot_kd = 50
        return rot_kp, rot_kd
    
    def rootz_position(self):
        # get the index of base joint
        base_joint_id = self.base_joint_id
        # get the position of base joint
        rootz_position = self.d.qpos[base_joint_id]
        return rootz_position

    def base_inertial_pos(self):
        body_id = self.body_dict()
        torso_index = body_id['base']
        base_inertial_pos = self.d.xipos[torso_index] #inertial position of base in WORLD coordinates
        return base_inertial_pos 

    def hip_position(self):
        thigh_joint_id = self.thigh_joint_id
        hip_position = self.d.qpos[thigh_joint_id] + self.initial_hip_pos
        return hip_position     

    def torso_bottom_position(self):
        rootz_position = self.rootz_position()
        torso_z_position = rootz_position 
        torso_x_position = 0
        torso_y_position = 0
        torso_bottom_position = np.array([torso_x_position, torso_y_position, torso_z_position])
        return torso_bottom_position
    
    def knee_position(self):
        knee_joint_id = self.knee_joint_id
        knee_position = self.d.qpos[knee_joint_id] + self.initial_knee_pos
        return knee_position
    
    def xml_parser(self):
        xml = self.xml_path
        tree = ET.parse(xml)
        root = tree.getroot()
        return root, tree  
    
    def thigh_length_xml(self):
            root, tree = self.xml_parser()
            # Get the index of the specified axis
            for link in root.findall('worldbody'):            
                for slot in link.findall('body'): 
                    if slot.get('name')== 'root': 
                        for new2 in slot.findall('body'):
                            if new2.get('name')== 'base':                    
                                for new in new2.findall('body'):
                                    if new.get('name') == 'link1':                           
                                        for element in new.findall('geom'):                                                                           
                                            fromto = element.get('fromto')                                            
                                            thigh_start_z = float(fromto.split()[2])                                    
                                            thigh_start_x = float(fromto.split()[0])
                                            
                                            thigh_start_y = float(fromto.split()[1])
                                                                                    
                                            thigh_end_z = float(fromto.split()[5])    
                                            
                                            thigh_end_x = float(fromto.split()[3])
                                            
                                            thigh_end_y = float(fromto.split()[4])
                                            
                                            thigh_length = np.linalg.norm(np.array([thigh_end_x, thigh_end_y, thigh_end_z]) - np.array([thigh_start_x, thigh_start_y, thigh_start_z]))
                                                                                    
                                            return thigh_length
                                                                 
                                        
            return thigh_length 


    def calf_length_xml(self):
            root, tree = self.xml_parser()
            for link in root.findall('worldbody'):                            
                for slot in link.findall('body'): 
                    if slot.get('name')== 'root':
                        for new2 in slot.findall('body'):
                            if new2.get('name')== 'base':                    
                                for new in new2.findall('body'):
                                    if new.get('name') == 'link1':                           
                                        for element in new.findall('body'):
                                            if element.get('name') == 'link2':
                                                for element in element.findall('geom'):
                                                    fromto = element.get('fromto')
                                                    knee_start_z = float(fromto.split()[2])
                                                    knee_start_x = float(fromto.split()[0])
                                                    knee_start_y = float(fromto.split()[1])
                                                                                        
                                                    knee_end_z = float(fromto.split()[5])    
                                                    
                                                    knee_end_x = float(fromto.split()[3])
                                                    
                                                    knee_end_y = float(fromto.split()[4])
                                                    
                                                    calf_length = np.linalg.norm(np.array([knee_end_x, knee_end_y, knee_end_z]) - np.array([knee_start_x, knee_start_y, knee_start_z]))
                                                                                    
                                                    return calf_length

    
    
    
    def calf_radius_xml(self):
        root, tree = self.xml_parser()
        
        # Initialize with a default value in case the element isn't found
        calf_radius = 0.025  # Default value based on your XML
        
        try:
            # Find all worldbody elements
            for worldbody in root.findall('worldbody'):
                # Find all body elements within worldbody
                for body in worldbody.findall('body'):
                    if body.get('name') == 'root':
                        # Find all body elements within root
                        for child_body in body.findall('body'):
                            if child_body.get('name') == 'base':
                                # Find all body elements within base
                                for base_child in child_body.findall('body'):
                                    if base_child.get('name') == 'link1':
                                        # Find all body elements within link1
                                        for link1_child in base_child.findall('body'):
                                            if link1_child.get('name') == 'link2':
                                                # Find geom elements within link2
                                                for geom in link1_child.findall('geom'):
                                                    if geom.get('name') == 'shank' or geom.get('name') is None:
                                                        radius = geom.get('size')
                                                        if radius:
                                                            calf_radius = float(radius)
                                                            return calf_radius
        except Exception as e:
            print(f"Error parsing XML for calf radius: {e}")
    
    # Return default value if not found
        return calf_radius
    
    
    
    def thigh_radius_xml(self):    
        root, tree = self.xml_parser()
        for link in root.findall('worldbody'):            
            for slot in link.findall('body'):  
                if slot.get('name')== 'root':
                    for new in slot.findall('body'):
                        if new.get('name')== 'base':
                            for new2 in new.findall('body'):
                                if new2.get('name') == 'link1':
                                    for element in new2.findall('geom'):                                                                           
                                        radius = element.get('size')                                            
                                        thigh_radius = float(radius)
                        
        return thigh_radius
    
    def torso_width_xml(self): #TODO: MODIFY FOR CORRECT BODY NAME - BASE
        root, tree = self.xml_parser()
        for link in root.findall('worldbody'):            
            for slot in link.findall('body'):                
                if slot.get('name')== 'slot':                   
                    for new in slot.findall('body'):
                        if new.get('name') == 'torso':                            
                            for element in new.findall('geom'):                                
                                torso_radius = element.get('size')                                                                                            
                                return torso_radius                                               
                                   
        return torso_radius

    def thigh_total_length(self):
        thigh_length = self.thigh_length_xml()
        thigh_radius = self.thigh_radius_xml()
        total_length = thigh_length
        return total_length

    def calf_total_length(self):
        calf_length = self.calf_length_xml()
        calf_radius = self.calf_radius_xml()
        total_length = calf_length 
        return total_length
    
    def ee_pos(self):
        base_inertial_pos = self.base_inertial_pos()
        base_z_pos = base_inertial_pos[2]
        base_x_pos = base_inertial_pos[0]
        base_y_pos = base_inertial_pos[1]        
        leg_length = self.calf_total_length()
        knee_pos = self.knee_position()
        hip_pos = self.hip_position()
        thigh_length = self.thigh_total_length()
        leg_width = float(self.calf_radius_xml())   
        #NOTE: REMOVED LEG WIDTH FROM CALCULATION OF BASE BOTTOM Z AS BASE BOTTOM AND LINK LENGTHS HAVE RIGHT VALUES 
        base_bottom_z = base_z_pos 
        base_bottom = np.array([base_x_pos, base_y_pos, base_bottom_z])
        ee_x_pos = base_bottom[0] + thigh_length*np.sin(hip_pos) + leg_length*np.sin(knee_pos+hip_pos)
        ee_z_pos = base_bottom[2] - thigh_length*np.cos(hip_pos) - leg_length*np.cos(knee_pos+hip_pos) 
        ee_y_pos = base_bottom[1]
        ee_pos = np.array([ee_x_pos, ee_y_pos, ee_z_pos])
        return ee_pos
    
    #distance calculated between INERTIAL pos of base and circular end of foot 
    
    def distance(self):
        ee_pos = self.ee_pos()
        base_inertial_pos = self.base_inertial_pos()
        distance = np.linalg.norm(ee_pos - base_inertial_pos)
        return distance
    
    def equivalent_orientation(self):
        ee_pos = self.ee_pos()
        base_inertial_pos = self.base_inertial_pos()
        distance = self.distance()
        z_axis_vector = np.array([0, 0, -1])
        base_to_foot_unit_vector = (ee_pos - base_inertial_pos) / distance
        dot_product = np.dot(base_to_foot_unit_vector, z_axis_vector)
        angle_with_z = np.arccos(np.clip(dot_product, -1.0, 1.0))
        return angle_with_z
    
    #CODE TO CALCULATE LINEAR DAMPER FORCE
    def dl_dxee(self):
        ee_pos = self.ee_pos()  
        ee_x_pos = ee_pos[0]
        base_inertial_pos = self.base_inertial_pos()
        base_x_pos = base_inertial_pos[0]
        distance = self.distance()
        dl_dxee = (ee_x_pos - base_x_pos)/distance
        return dl_dxee
    
    def dxee_dt(self):
        base_cvel = self.d.cvel[self.base_joint_id]
        base_x_cvel = base_cvel[0]
        leg_length = self.calf_total_length()
        knee_pos = self.knee_position()
        hip_pos = self.hip_position()        
        thigh_length = self.thigh_total_length()
        hip_velocity = self.d.qvel[self.thigh_joint_id]
        knee_velocity = self.d.qvel[self.knee_joint_id]
        dxee_dt = base_x_cvel + thigh_length*np.cos(hip_pos)*hip_velocity + leg_length*np.cos(knee_pos + hip_pos)*(knee_velocity+hip_velocity)
        return dxee_dt
    
    def dl_dxb(self):
        ee_pos = self.ee_pos()  
        ee_x_pos = ee_pos[0]
        base_inertial_pos = self.base_inertial_pos()
        base_x_pos = base_inertial_pos[0]
        distance = self.distance()
        dl_dxb = (base_x_pos - ee_x_pos)/distance
        return dl_dxb
    
    def dxb_dt(self):   
        base_cvel = self.d.cvel[self.base_joint_id]
        base_x_cvel = base_cvel[0]
        dxb_dt = base_x_cvel
        return dxb_dt
    
    def dl_dzee(self):
        ee_pos = self.ee_pos()  
        ee_z_pos = ee_pos[2]
        base_inertial_pos = self.base_inertial_pos()
        base_z_pos = base_inertial_pos[2]
        distance = self.distance()
        dl_dzee = (ee_z_pos - base_z_pos)/distance
        return dl_dzee
    
    def dzee_dt(self):
        base_cvel = self.d.cvel[self.base_z_joint_id]
        base_z_cvel = base_cvel[0]
        leg_length = self.calf_total_length()
        knee_pos = self.knee_position()
        hip_pos = self.hip_position()        
        thigh_length = self.thigh_total_length()
        hip_velocity = self.d.qvel[self.thigh_joint_id]
        knee_velocity = self.d.qvel[self.knee_joint_id]
        dzee_dt = base_z_cvel + thigh_length*np.sin(hip_pos)*hip_velocity + leg_length*np.sin(knee_pos + hip_pos)*(knee_velocity+hip_velocity)
        return dzee_dt
    
    def dl_dzb(self):
        ee_pos = self.ee_pos()  
        ee_z_pos = ee_pos[2]
        base_inertial_pos = self.base_inertial_pos()
        base_z_pos = base_inertial_pos[2]
        distance = self.distance()
        dl_dzb = (base_z_pos - ee_z_pos)/distance
        return dl_dzb
    
    def dzb_dt(self):
        base_cvel = self.d.cvel[self.base_z_joint_id]
        base_z_cvel = base_cvel[0]
        dzb_dt = base_z_cvel
        return dzb_dt
    
    def linear_damper_force(self, action):
        dl_dxee = self.dl_dxee()
        dxee_dt = self.dxee_dt()
        dl_dxb = self.dl_dxb()
        dxb_dt = self.dxb_dt()
        dl_dzee = self.dl_dzee()
        dzee_dt = self.dzee_dt()
        dl_dzb = self.dl_dzb()
        dzb_dt = self.dzb_dt()
        kp, kd = self.lin_sd_var(action)
        Fld = kd*(dl_dxee*dxee_dt + dl_dxb*dxb_dt + dl_dzee*dzee_dt + dl_dzb*dzb_dt)
        return Fld  
    

    
    def linear_spring_force (self,action):
        distance = self.distance()        
        kp, kd = self.lin_sd_var(action)        
        Fls =kp * (self.ori_l - distance)     
        return Fls
    
    def total_linear_force(self,action):
        linear_spring_force = self.linear_spring_force(action)
        linear_damper_force = self.linear_damper_force(action)
        total_linear_force = (linear_spring_force - linear_damper_force)
        return total_linear_force
    
    def torsional_spring_force(self,action):
        angle_with_z = self.equivalent_orientation()
        kp, kd = self.rot_sd_var(action)
        Fts = kp * (self.ori_theta - angle_with_z)
        return Fts
    
    def fz_by_ee(self,action):
        equivalent_orientation = self.equivalent_orientation()
        torsional_spring_force = self.torsional_spring_force(action)
        linear_spring_force = self.linear_spring_force(action)
        total_linear_force = self.total_linear_force(action)
        distance = self.distance()
        fz_wo_gravity = -total_linear_force*np.cos(equivalent_orientation) + (torsional_spring_force/distance)*np.sin(equivalent_orientation)
        mass = mj.mj_getTotalmass(self.m)
        weight = mass*9.81
        fz_by_ee = fz_wo_gravity - weight
        return fz_by_ee
    
    def fx_by_ee(self, action):
        equivalent_orientation = self.equivalent_orientation()
        torsional_spring_force = self.torsional_spring_force(action)
        linear_spring_force = self.linear_spring_force(action)
        total_linear_force = self.total_linear_force(action)
        distance = self.distance()
        fx_wo_gravity = -total_linear_force*np.sin(equivalent_orientation) - (torsional_spring_force/distance)*np.cos(equivalent_orientation)
        fx_by_ee = fx_wo_gravity
        return fx_by_ee
    
    def force_applied_ground(self,action):
        fz_by_ee = self.fz_by_ee(action)
        fx_by_ee = self.fx_by_ee(action)
        force_applied_ground = np.array([fx_by_ee, fz_by_ee])
        return force_applied_ground
    
    def jacobian(self):      
        leg_length = self.calf_total_length()
        knee_pos = self.knee_position()
        hip_pos = self.hip_position() 
        thigh_length = self.thigh_total_length()       
        J11 =  (thigh_length*np.cos(hip_pos)+leg_length*np.cos(knee_pos+hip_pos))
        J12 = (leg_length*np.cos(knee_pos+hip_pos))
        J21 = thigh_length*np.sin(hip_pos) + leg_length*np.sin(knee_pos + hip_pos)
        J22 = leg_length*np.sin(knee_pos + hip_pos)
        jacobian = np.array([[J11, J12], [J21, J22]])    
        return jacobian
    
    def mujoco_jacobian(self):
        J_pos = np.zeros((3, self.m.nv)) 
        foot_site_id = self.m.site("foot").id
        mj.mj_jacSite(self.m, self.d, J_pos, None, foot_site_id)
        J_2d = J_pos[:3, :3]
        return J_2d
    
    def jacobian_transpose(self):
        jacobian = self.jacobian()
        jacobian_transpose = np.transpose(jacobian)
        return jacobian_transpose
    
    def joint_torque(self,action):
        jacobian_transpose = self.jacobian_transpose()
        force_applied_ground = self.force_applied_ground(action)
        joint_torque = np.dot(jacobian_transpose, force_applied_ground)
        return joint_torque