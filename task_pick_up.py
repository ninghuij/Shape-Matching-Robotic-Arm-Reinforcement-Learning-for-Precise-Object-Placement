import numpy as np
from utils import *

def next_action(env,pos,quat):
    out_of_range = 0
    joint_positions = env._robot.arm.get_joint_positions()
    try:
      joint_positions = env._robot.arm.solve_ik_via_jacobian(pos, quaternion = quat)
    except:
      print("The position can't be reached!")
      out_of_range = 1
    return joint_positions,out_of_range

def actions(pos,gripper_s,type,step=0.01):
    x = pos[0]
    y = pos[1]
    z = pos[2]
    if type == 0: # up
      z = z + step
    if type == 1: # down
      z = z - step
    if type == 2: # left
      y = y + step
    if type == 3: # right
      y = y - step
    if type == 4: # forward
      x = x + step
    if type == 5: # back
      x = x - step
    if type == 6: # grip
      gripper_s = 0 
    if type == 7: # drop
      gripper_s = 1
    pos = [x,y,z]      
    return pos, gripper_s

class pick_up():
    def __init__(self):
        self.state_space = 4
        self.action_space = 7
        self.name = 'pick_up'
        self.min_dis = 10
        self.threshold = 0.13
    def modify_obs(self,env = None, obs = None,obj =None ):
            
        shape1, shape1_d = get_object(obs.wrist_mask,obs.wrist_depth,obj[1])
        gripper, gripper_d = get_object(obs.front_mask,obs.front_depth,31)
        if np.sum(shape1) == 0:
          shape1_d = 1
   
        #row_dis = np.sqrt(np.sum((shape1.squeeze()[1]-gripper[1])**2))
        #reward = - row_dis + (-shape1_d * 10)
        reward = -shape1_d * 10
        collision = env._scene.robot.arm.check_arm_collision()
        if collision == True: reward += -10

        obj0_pos = env._task.grasp_points[obj[0]].get_position()  # position of object 0
        des0_pos = env._task.drop_points[obj[0]].get_position()  # position of desitination 0
        end_eff_pos = env._robot.arm.get_tip().get_position()   # pos of end_effector
        
        dis = np.sqrt(np.sum((obj0_pos-end_eff_pos)**2))
        diff = obj0_pos - end_eff_pos
        #state = np.concatenate((diff,[shape1.squeeze()[1]-gripper[1]],[shape1_d]))
        state = np.concatenate((diff,[shape1_d]))*100
        reward += - dis*10

        if dis >= 0.05 and obs.gripper_open == 0:
            reward += -10
        terminated = 0
        if obs.gripper_open == 0: # check if the grab is succes every grap
            grap_succes = check_grasp(obs,value=obj[1],threshold=50)
            if grap_succes:
              terminated = 1
              reward = 200
            else:
              reward +=-5
              state = np.array([1,-2,-3,0]) # add noise

        return state.reshape(1,len(state)),np.array(reward).reshape(1,1),np.array(terminated).reshape(1,1)
    def discretized(self,env,act,step = 0.01):
    
        cur_pos, cur_quat = env._robot.arm.get_tip().get_position(), env._robot.arm.get_tip().get_quaternion()
        gripper_s = env._robot.gripper.get_open_amount()[1]*0 +1
        des_pos, gripper_s = actions(cur_pos,round(gripper_s),act,step)
        des_joint_poisitons,out_of_range = next_action(env,des_pos,cur_quat)
        action = np.concatenate((des_joint_poisitons,[gripper_s]))
        return action, out_of_range
     