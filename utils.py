import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

def create_neural_network(input, classes):
    class net(nn.Module):
        def __init__(self):
            super().__init__()

            # Network Parameters
            n_input = input  # data input
            n_hidden_1 = 64  # 1st layer number of features
            n_hidden_2 = 64  # second layer
            n_classes = classes 

            # Create a model
            self.model = nn.Sequential(nn.Linear(n_input, n_hidden_1),
                                        nn.ReLU(),
                                        nn.Linear(n_hidden_1, n_hidden_2),
                                        nn.ReLU(),
                                        nn.Linear(n_hidden_2, n_classes))

        def forward(self, input):
            output = self.model(input)
            return output

    return net()
obj = np.array([0,103,1,105,2,92,3,95,4,101]).reshape(-1,2)
des_rect = np.array([0.224,0.278,-0.128,-0.075,0.843,0.92]).reshape(-1,2)
class hyperparameter():
    C_steps = 4
    Batch_size = 64
    Replay_capacity = 4000
    LR = 0.001
    Tau = 0.1
    Gamma = 0.95
    Max_timesteps = 150
    Performance = 100
    Window_size = 15
    Min_e = 0.02
    Max_e = 0.99
def plot_result(result,mean_window_size,color,xlabel,ylabel,titile,legend):
   plt.figure(figsize=(15, 10))
   plt.plot(result, color)
   if(mean_window_size > 0): # -1  no plot
      num = int(len(result)/mean_window_size)
      axis = np.linspace(0, len(result), num)
      mean = []
      for i in range(num):
         mean.append(np.average(result[i*mean_window_size:(i+1)*mean_window_size]))
         #mean = np.average(Total_r.reshape(int(len(result)/mean_window_size),mean_window_size), axis=1)
      plt.plot(axis,np.array(mean))
   plt.xlabel(xlabel, fontsize=28)
   plt.ylabel(ylabel, fontsize=28)
   plt.title(titile, fontsize=36)
   plt.legend(legend)
   plt.ylim(ymin = min(min(result),0), ymax= max(result)*1.1)
   plt.grid()
   plt.show()
def my_step(env,pos,quat):
    joint_positions = env._robot.arm.get_joint_positions()
    try:
      joint_positions = env._robot.arm.solve_ik_via_jacobian(pos, quaternion=quat)
      env._robot.arm.set_joint_target_positions(joint_positions)
      done = False
      prev_values = None
      # Move until reached target joint positions or until we stop moving
      # (e.g. when we collide wth something)
      while not done:
          env._scene.step()
          cur_positions = env._robot.arm.get_joint_positions()
          reached = np.allclose(cur_positions, joint_positions, atol=0.01)
          not_moving = False
          if prev_values is not None:
              not_moving = np.allclose(
                  cur_positions, prev_values, atol=0.001)
          prev_values = cur_positions
          done = reached or not_moving
    except:
      print("The position can't be reached!")
def virbration_check(env,pre_a,cur_a):
    flag = 0
    if (pre_a ==4 and cur_a ==5) or (pre_a ==5 and cur_a ==4):
      flag =1
    return flag
      
   #env._robot.arm.get_tip().get_position()  
   
def check_grasp(obs,left = 27,right = 95,up = 104,down = 127,value = 103,threshold = 320):
    succes = 0
    img = obs.wrist_mask
    rect = img[up:down,left:right]
    ob = np.argwhere(rect == value)
    g_l = np.argwhere(img == 34)
    g_r = np.argwhere(img == 31)
    cl = np.sum(g_l,axis =0)/len(g_l)
    cr = np.sum(g_r,axis =0)/len(g_r)

    n = np.argwhere(rect == 48)
    depth =  np.sum(obs.wrist_depth[n[:,0],n[:,1]])/len(n)
    #num = len(ob)+len(g_l)+ len(g_r)
    if len(ob)>threshold and (cr[1]-cl[1])>40 :
      succes =1
    return succes
    
def get_object(mask,depth,type):
    object = np.array([0,0]).reshape(1,2)
    object_d = 0 
    n = np.argwhere(mask == type)
    if len(n) > 0:
      object = np.round(np.sum(n,axis =0)/len(n))
      h, w = int(object[0]), int(object[1])
      object_d = depth[h,w]
    return object,object_d
def get_object_positions(mask,depth):
    shape1= np.array([0,0]).reshape(1,2)
    shape1_d = 0 
    n = np.argwhere(mask == 82)
    if len(n) > 0:
      shape_sorter = np.round(np.sum(n,axis =0)/len(n))
      h, w = int(shape_sorter[0]), int(shape_sorter[1])
      shape_sorter_d = depth[h,w]
    
    n = np.argwhere(mask == 105)
    if len(n) > 0:
      shape1 = np.round(np.sum(n,axis =0)/len(n))
      h, w = int(shape1[0]), int(shape1[1])
      shape1_d = depth[h,w]
    
    n = np.argwhere(mask == 103)
    if len(n) > 0:
      shape2 = np.round(np.sum(n,axis =0)/len(n))
      h, w = int(shape2[0]), int(shape2[1])
      shape2_d = depth[h,w]
    
    n = np.argwhere(mask == 101)
    if len(n) > 0:
      shape3 = np.round(np.sum(n,axis =0)/len(n))
      h, w = int(shape3[0]), int(shape3[1])
      shape3_d = depth[h,w]
    
    n = np.argwhere(mask == 95)
    if len(n) > 0:
      shape4 = np.round(np.sum(n,axis =0)/len(n))
      h, w = int(shape4[0]), int(shape4[1])
      shape4_d = depth[h,w]
   
    n = np.argwhere(mask == 92)
    if len(n) > 0:
      shape5 = np.round(np.sum(n,axis =0)/len(n))
      h, w = int(shape5[0]), int(shape5[1])
      shape5_d = depth[h,w]

    return shape1,shape1_d
def modify_obs_pick_up(obs):
    #shape1, shape1_d = get_object_positions(obs.wrist_mask,obs.wrist_depth)
    shape1, shape1_d = get_object(obs.wrist_mask,obs.wrist_depth,105)
    gripper, gripper_d = get_object(obs.front_mask,obs.front_depth,31)
    if np.sum(shape1) == 0:
       shape1_d = 1
    #state = np.hstack((np.array(shape1).reshape(1,2),np.array(shape1_d).reshape(1,1)))
    dis = np.sqrt(np.sum((shape1-gripper)**2) + 10000*(shape1_d - gripper_d)**2)
    state = np.concatenate((obs.joint_positions,[dis]))
    reward = - dis

    terminated = 0
    if dis <= 10:
      terminated = 1
      reward = 200

    return state.reshape(1,len(state)),np.array(reward).reshape(1,1),np.array(terminated).reshape(1,1)
def modify_obs_reach(env,obs):
    obj0_pos = env._task.grasp_points[0].get_position()  # position of object 0
    des0_pos = env._task.drop_points[0].get_position()  # position of desitination 0
    end_eff_pos = env._robot.arm.get_tip().get_position()   # pos of end_effector
    
    dis = np.sqrt(np.sum((obj0_pos-end_eff_pos)**2))
    #state = np.concatenate((obs.joint_positions,[dis]))
    #state = np.concatenate((end_eff_pos,[dis]))
    diff = obj0_pos - end_eff_pos
    state = diff
    #state = np.concatenate((end_eff_pos,obj0_pos,diff))
    reward = - dis

    terminated = 0
    if dis <= np.sqrt(0.02**2 * 3):
      terminated = 1
      reward = 200

    return state.reshape(1,len(state)),np.array(reward).reshape(1,1),np.array(terminated).reshape(1,1)
def next_action(env,pos,quat):
    out_of_range = 0
    joint_positions = env._robot.arm.get_joint_positions()
    try:
      joint_positions = env._robot.arm.solve_ik_via_jacobian(pos, quaternion = quat)
    except:
      print("The position can't be reached!")
      out_of_range = 1
    return joint_positions,out_of_range

def discretized0(env,act,step = 0.05):
    
    encoder = np.array([0, 0, 0, 0,0, 0, 0,0],dtype= float)
    encoder[act//2] = step * (-1)**(act%2)
    
    t = env._robot.arm.get_joint_positions()+ encoder[:7]
    action = np.concatenate((t,[1]))
    if act == 15: action = np.concatenate((t,[0]))

    return action

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

class Agent(object):
  """A simple random-action agent. """

  def __init__(self, action_shape):
      self.action_shape = action_shape

  def act(self, obs):
      arm = np.random.normal(0.0, 1, size=(self.action_shape[0] - 1,))
      gripper = [1.0]  # Always open
      return np.concatenate([arm, gripper], axis=-1)  
     