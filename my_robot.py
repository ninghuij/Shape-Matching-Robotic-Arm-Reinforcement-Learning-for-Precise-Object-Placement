import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from utils import *
from task_reach import *
from task_pick_up import *
from task_put import *
def next_action(env,pos,quat):
    out_of_range = 0
    joint_positions = env._robot.arm.get_joint_positions()
    try:
      joint_positions = env._robot.arm.solve_ik_via_jacobian(pos, quaternion = quat)
    except:
      print("The position can't be reached!")
      out_of_range = 1
    return joint_positions,out_of_range

def discretized(env,act,step = 0.01):
    
    cur_pos, cur_quat = env._robot.arm.get_tip().get_position(), env._robot.arm.get_tip().get_quaternion()
    gripper_s = env._robot.gripper.get_open_amount()[1]
    des_pos, gripper_s = actions(cur_pos,round(gripper_s),act,step)
    des_joint_poisitons,out_of_range = next_action(env,des_pos,cur_quat)
    action = np.concatenate((des_joint_poisitons,[gripper_s]))
    return action, out_of_range
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
class Robot():
    def __init__(self):
        self.task_reach = reach()
        self.reach_model = create_neural_network(self.task_reach.state_space,self.task_reach.action_space)
        self.reach_model.load_state_dict(torch.load("./trained_models/parameters_reach.pkl"))
        self.max_timesteps = 150
        
        self.terminate = 0.008
        self.min_dis = 10
        self.threshold = 0.14

        self.task_pick_up = pick_up()
        self.pick_up_model = create_neural_network(self.task_pick_up.state_space,self.task_pick_up.action_space)
        self.pick_up_model.load_state_dict(torch.load("./trained_models/parameters_pick_up.pkl"))
        self.max_timesteps_pick = 60
        
        self.task_put = put()
        self.put_model = create_neural_network(self.task_put.state_space,self.task_put.action_space)
        self.put_model.load_state_dict(torch.load("./trained_models/parameters_put.pkl"))
        self.max_timesteps_put = 150
    def modify_obs(self,env,obs,obj):
      if obj[0]< 5:
        obj0_pos = env._task.grasp_points[obj[0]].get_position()  # position of object 0
      else:
        obj0_pos = obj[1:]
      end_eff_pos = env._robot.arm.get_tip().get_position()   # pos of end_effector
      dis = np.sqrt(np.sum((obj0_pos-end_eff_pos)**2))
      diff = obj0_pos - end_eff_pos
      state = diff
      reward = - dis

      terminated = 0
      if dis <= np.sqrt(0.02**2 * 3):
        terminated = 1
        reward = 200

      if (dis <=self.threshold) and abs(dis - self.min_dis < self.terminate): 
         terminated = 1
      if dis < self.min_dis: self.min_dis = dis   

      return state.reshape(1,len(state)),np.array(reward).reshape(1,1),np.array(terminated).reshape(1,1)

    def act(self, input):
      self.reach_model.eval()
      with torch.no_grad():
        pred = self.reach_model(torch.tensor(input.squeeze()).float()) # predict the Q-values
      q_values = pred.detach().data.numpy().squeeze() # transfer to NumPy array
      max_as = np.argwhere(q_values == np.max(q_values))
      if len( max_as)>0:
        action = max_as[0,0]  
      # action = torch.max(pred, 1)[1].data.numpy()               
      # action = action[0]
      return action 
    def reach(self,env,obs,obj):
      state, reward, terminated = self.modify_obs(env,obs,obj)
      GT = 0
      time_step = 0
      self.min_dis = 10
      step = 0.02
      while True:
        next_a = self.act(state)
        action, out_of_range = discretized(env,next_a,step)
        obs, _ , terminate = env.step(action)
        next_state, reward, terminated = self.modify_obs(env,obs,obj)
        state = next_state
        GT += reward # accumulate reward
        if abs(reward[0])<= self.threshold+0.05:
           step = 0.01
        time_step += 1
        if (terminated == True) or (time_step > self.max_timesteps):
          break
      return obs,GT
    def pick_up(self,env,obs,obj):
      state, reward, terminated = self.task_pick_up.modify_obs(env,obs,obj)
      GT = 0
      time_step = 0
      self.min_dis = 10
      step = 0.01
      pre_a = 0
      while True:
        self.pick_up_model.eval()
        with torch.no_grad():
          pred = self.pick_up_model(torch.tensor(state.squeeze()).float()) # predict the Q-values
        q_values = pred.detach().data.numpy().squeeze() # transfer to NumPy array
        max_as = np.argwhere(q_values == np.max(q_values))
        if len( max_as)>0:
          action = max_as[0,0]  
        next_a = action
        collision = env._scene.robot.arm.check_arm_collision()
        if collision == True: next_a = 0
        if (pre_a ==2 and next_a ==3) or (pre_a ==3 and next_a ==2): next_a = 1  # oscillation check
        pre_a = next_a
        action, out_of_range = self.task_pick_up.discretized(env,next_a,step)
        obs, _ , terminate = env.step(action)
        next_state, reward, terminated = self.task_pick_up.modify_obs(env,obs,obj)
        state = next_state
        GT += reward # accumulate reward
        if abs(reward[0])<= self.threshold+0.05:
           step = 0.01
        time_step += 1

        if (terminated[0,0] == 1) or (time_step > self.max_timesteps_pick):
          break
      return obs,GT
    def put(self,env,obs,obj):
      state, reward, terminated = self.task_put.modify_obs(env,obs,obj)
      GT = 0
      time_step = 0
      self.min_dis = 10
      step = 0.01
      while True:
        self.put_model.eval()
        with torch.no_grad():
          pred = self.put_model(torch.tensor(state.squeeze()).float()) # predict the Q-values
        q_values = pred.detach().data.numpy().squeeze() # transfer to NumPy array
        max_as = np.argwhere(q_values == np.max(q_values))
        if len( max_as)>0:
          action = max_as[0,0]  
        next_a = action
        action, out_of_range = self.task_put.discretized(env,next_a,step)
        obs, _ , terminate = env.step(action)
        next_state, reward, terminated = self.task_put.modify_obs(env,obs,obj)
        state = next_state
        GT = GT +reward # accumulate reward
        if abs(reward[0])<= self.threshold+0.05:
           step = 0.01
        time_step += 1
        if (terminated[0,0] == 1) or (time_step > self.max_timesteps_put):
          break
      return obs,GT

     