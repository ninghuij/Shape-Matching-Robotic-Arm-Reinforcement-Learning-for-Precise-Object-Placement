import numpy as np

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity,JointPosition,EndEffectorPoseViaPlanning,EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import PlaceShapeInShapeSorter
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape

import matplotlib.pyplot as plt
from utils import *
from task_reach import reach
from task_pick_up import pick_up
from task_put import put
from task_put_in import put_in
from my_robot import Robot

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def test(env,state_space,action_space,max_timesteps,path,episodes=500,min_e = 0,subtask = None):
  
    Q_act_v = create_neural_network(state_space,action_space)
    Q_act_v.load_state_dict(torch.load(path))
    agent = Robot()
    Total_r =np.zeros((episodes,1))
    for i in range(episodes):
          _, obs = env.reset()
          select = np.random.choice(5)
          des = obj[select]
          if subtask.name =='pick_up':
            obs,reward = agent.reach(env,obs,des)
          if subtask.name =='put':
            obs,reward = agent.reach(env,obs,des)
            obs,reward = agent.pick_up(env,obs,des)
          if subtask.name =='put_in':
            obs,reward = agent.reach(env,obs,des)
            obs,reward = agent.pick_up(env,obs,des)
            obs,reward = agent.put(env,obs,des)

          state, reward, terminated = subtask.modify_obs(env,obs,des)
          GT = 0
          time_step = 0
          pre_a = 0
          while True:
            next_a = choose_action(Q_act_v,state, 0 ,action_space)
            if virbration_check(env,pre_a,next_a):
                 next_a = 5
            action, out_of_range = subtask.discretized(env,next_a,0.02)
            pre_a = next_a
            obs, _ , terminate = env.step(action)
            next_state, reward, terminated = subtask.modify_obs(env,obs,des)
            state = next_state
            GT += reward # accumulate reward
            time_step += 1
            if (terminated == True) or (time_step > max_timesteps):
              break
          Total_r[i] = GT
          print('Episode: '+str(i)+'  '+'Total reward: '+str(GT))
   # Reward
    mean_window_size = 10
    num = int(len(Total_r)/mean_window_size)
    axis = np.linspace(0, len(Total_r), num)
    mean = np.average(Total_r.reshape(num,mean_window_size), axis=1)
    plt.figure(figsize=(15, 10))
    plt.plot(Total_r, 'ro')
    plt.plot(axis,mean)
    plt.xlabel('Episode', fontsize=28)
    plt.ylabel('Reward Value', fontsize=28)
    plt.title('Rewards Per Episode (test)', fontsize=36)
    plt.legend(['Q Actor-Critic'])
    plt.ylim(ymin = min(min(Total_r),0)[0], ymax= max(Total_r)[0]*1.1)
    plt.grid()
    plt.show()
    return Total_r
def test_all(env,state_space,action_space,max_timesteps,path,episodes=500,min_e = 0,subtask = None):
    writer = SummaryWriter()
    Q_act_v = create_neural_network(state_space,action_space)
    Q_act_v.load_state_dict(torch.load(path))
    agent = Robot()
    Total_r =np.zeros((episodes,1))
    for i in range(episodes):
          _, obs = env.reset()
          cur_pos, cur_quat = env._robot.arm.get_tip().get_position(), env._robot.arm.get_tip().get_quaternion()
          temp = obj.copy()
          left_num = 5

          while True:
            select = np.random.choice(left_num)
            des = temp[select]
            
            if subtask.name =='pick_up':
              obs,reward = agent.reach(env,obs,des)
            if subtask.name =='put':
              obs,reward = agent.reach(env,obs,des)
              obs,reward = agent.pick_up(env,obs,des)
            if subtask.name =='put_in':
              obs,reward = agent.reach(env,obs,des)
              obs,reward = agent.pick_up(env,obs,des)
              obs,reward = agent.put(env,obs,des)

            state, reward, terminated = subtask.modify_obs(env,obs,des)
            GT = 0
            time_step = 0
            pre_a = 0
            while True:
              next_a = choose_action(Q_act_v,state, 0 ,action_space)
              
              if virbration_check(env,pre_a,next_a):
                 next_a = 5
              action, out_of_range = subtask.discretized(env,next_a,0.02) 
              pre_a = next_a
              
              obs, _ , terminate = env.step(action)
              next_state, reward, terminated = subtask.modify_obs(env,obs,des)
              if (terminated == True) and (time_step < 5):
                 left_num = 0
                 break
                 
              state = next_state
              GT += reward # accumulate reward
              time_step += 1
              if (terminated == True) or (time_step > max_timesteps):
                action, out_of_range = subtask.discretized(env,6,0.02)
                env.step(action)
                cur_pos,_ = env._robot.arm.get_tip().get_position(), env._robot.arm.get_tip().get_quaternion()
                #my_step(env,[0.05,0,1.1],cur_quat)
                
                cur_pos = cur_pos + [-0.05,0,0.05]
                my_step(env,cur_pos,cur_quat)

                temp = np.delete(temp,select,0)
                left_num -= 1

                break

            if left_num == 0:
              break

          
          Total_r[i] = GT
          writer.add_scalars('Reward/', {'test':GT}, i)   
          print('Episode: '+str(i)+'  '+'Total reward: '+str(GT))
               
    writer.close()
    return Total_r


def choose_action(model,state, epsilon,action_space):
    # arm = np.random.normal(0.0, 1, size = 7)
    # gripper = [1.0]  # Always open
    # action = np.concatenate([arm, gripper], axis=-1)
    action = np.random.choice(action_space)
    if (np.random.rand() > epsilon):
      input = torch.tensor(state.squeeze())  # transfer to tensor
      model.eval()
      with torch.no_grad():
        pred = model(input.float()) # predict the Q-values
      model.train()
      #pred = pred.cpu()
      q_values = pred.detach().data.numpy().squeeze() # transfer to NumPy array
      max_as = np.argwhere(q_values == np.max(q_values))
      if len( max_as)>0:
        action = max_as[0,0]  
      # action = torch.max(pred, 1)[1].data.numpy()               
      # action = action[0]
    return action 

def Double_DQN(env,params,state_space,action_space,episodes = 400, subtask = None):
  
  C_steps = params.C_steps
  batch_size = params.Batch_size
  replay_capacity = params.Replay_capacity
  l_rate = params.LR
  gamma = params.Gamma
  max_timesteps = params.Max_timesteps

  replay_memory = np.zeros((replay_capacity,state_space * 2 + 1 + 2))
  mem_count = 0

  save_name = "parameters_"+ subtask.name + ".pkl"
  Q_act_v = create_neural_network(state_space,action_space)
  Q_t_act_v = create_neural_network(state_space,action_space)
  #Q_act_v.load_state_dict(torch.load(save_name))
  Q_t_act_v.load_state_dict(Q_act_v.state_dict())

  agent = Robot()
  
  Total_r = []
  Epsilons = []

  loss_fn = nn.MSELoss()
  optimizer = torch.optim.Adam(Q_act_v.parameters(), lr = l_rate)
  writer = SummaryWriter()
  average_sum = 0
  for i in range(episodes):
    _, obs = env.reset()

    if subtask.name =='pick_up':
      obs,reward = agent.reach(env,obs,obj[0])
    if subtask.name =='put':
      obs,reward = agent.reach(env,obs,obj[0])
      obs,reward = agent.pick_up(env,obs,obj[0])
    if subtask.name =='put_in':
      obs,reward = agent.reach(env,obs,obj[0])
      obs,reward = agent.pick_up(env,obs,obj[0])
      obs,reward = agent.put(env,obs,obj[0])


    state, reward, terminated = subtask.modify_obs(env,obs,obj[0])
    if terminated: 
       continue
    GT = 0
    time_step = 0
    epsilon = max(params.Min_e, params.Max_e**i )
    while True:
      next_a = choose_action(Q_act_v,state, epsilon,action_space) # with probability epsilon select a random action, otherwise select best action
      
      action, out_of_range = subtask.discretized(env,next_a,0.01)

      obs, _ , terminate = env.step(action)
      next_state, reward, terminated = subtask.modify_obs(env,obs,obj[0])
      if out_of_range:
         reward[0] = -100

      
      transition = np.hstack((state, np.array(next_a).reshape(-1,1), reward, next_state,terminated))  # store transition
      replay_memory[mem_count % replay_capacity,:] = transition
      mem_count = mem_count + 1   

      state = next_state 
      GT = GT +reward.squeeze() # accumulate reward

      update = mem_count % C_steps # 
      if (mem_count > batch_size) and (update == 0):  #  continue after the replay memory is full and update every C steps
        
        index = np.random.choice(min(replay_capacity,mem_count),batch_size)   # sample random batchsize transitions from replay memory D
        samples = replay_memory[index,:]

        q_state = torch.FloatTensor(samples[:, 0 : state_space]).view(batch_size,state_space) # change the shape
        q_action = torch.LongTensor(samples[:, state_space : state_space+1].astype(int)).view(batch_size,1)
        q_reward = torch.FloatTensor(samples[:, state_space+1 : state_space+2]).view(batch_size,1)
        q_next_s = torch.FloatTensor(samples[:, state_space+2 : state_space*2+2]).view(batch_size,state_space)
        q_terminated = 1- torch.FloatTensor(samples[:, state_space*2+2 : state_space*2+3]).view(batch_size,1)

        y = Q_act_v(q_state)
        y = y.gather(1,q_action) # get the action values
        y_t = Q_t_act_v(q_next_s) #.detach().max(1)[0].view(batch_size,1) # return the maximum value of each row
        max_action = torch.LongTensor(torch.max(Q_act_v(q_next_s).detach(), 1)[1]).view(batch_size,1)             
        y_t = y_t.gather(1,max_action).detach()

        y_t = q_reward + gamma * y_t *q_terminated
        
        loss = loss_fn(y_t, y) # Compute prediction error
        
        optimizer.zero_grad() # Backpropagation
        loss.backward()
        optimizer.step() # unpdate network parameters

        for target_param, local_param in zip(Q_t_act_v.parameters(), Q_act_v.parameters()):
            target_param.data.copy_(params.Tau * local_param.data + (1.0-params.Tau) * target_param.data)

      time_step += 1
      if (terminated == True) or ((time_step > max_timesteps) & (max_timesteps != -1)):  # -1 no max timesteps
            break

    Epsilons.append(epsilon)
    Total_r.append(GT)
    writer.add_scalars('Reward/', {'training':GT}, i)
    writer.add_scalars('Epsilon Decay/', {'training':epsilon}, i)
    print('Episode: '+str(i)+'  '+'Total reward: '+str(GT)+'  '+'Steps: '+str(time_step))
    if (i+1) % 50 ==0:
       name = str(i)+'.pkl'
       torch.save(Q_act_v.state_dict(),name)
    if (i >= params.Window_size):
      average_sum = np.sum(Total_r[i-params.Window_size:i])
      average = average_sum / params.Window_size
      if (average >= params.Performance) :   # stop trainning
            torch.save(Q_act_v.state_dict(), save_name)
            break     
  writer.close()
  torch.save(Q_act_v.state_dict(), save_name)
  return np.array(Total_r), np.array(Epsilons)  

if __name__ == "__main__":

    # Define the observations that we want to get at each timestep.
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    # Define the action mode of the arm. There are many to choose from.
    action_mode = MoveArmThenGripper(arm_action_mode=JointPosition(), gripper_action_mode=Discrete())
    # Create and launch the RLBench environment.
    env = Environment(action_mode, obs_config=obs_config, headless=False,static_positions= True)
    env.launch()

    self_task =  PlaceShapeInShapeSorter
    task = env.get_task(self_task)
    
    mode = 4  # 0 reach training, 1  pick up training, 2 put training, 3 put in training, 4 test single, 5 test all, 6 nothing

    if mode == 0: # reach training
      params = hyperparameter()
      params.Max_timesteps = 150
      params.Max_e = 0.8

      task1 = reach()
      state_space = task1.state_space
      action_space = task1.action_space
      Total_r, epsilon = Double_DQN(task,params,state_space, action_space, episodes = 500,subtask = task1)
      plot_result(Total_r.squeeze(),params.Window_size,color='ro',xlabel='Episode',ylabel='Reward Value',titile='Rewards Per Episode (Training)',legend=['DDQN','Mean'])
      #plt.savefig('training.png')
      plot_result(epsilon,-1,color='',xlabel='Episode',ylabel='Epsilon Values',titile='Epsilon Decay',legend=['DDQN'])
      #plt.savefig('epsilon_decay.png')

    if mode == 1: # pick up training
      params = hyperparameter()
      params.Max_timesteps = 150
      params.Max_e = 0.8
      task1 = pick_up() 
      state_space = task1.state_space
      action_space = task1.action_space
      Total_r, epsilon = Double_DQN(task,params,state_space, action_space, episodes = 500,subtask = task1)

    if mode == 2: # put training
      params = hyperparameter()
      params.Max_timesteps = 150
      params.Max_e = 0.9
      task1 = put()
      state_space = task1.state_space
      action_space = task1.action_space
      Total_r, epsilon = Double_DQN(task,params,state_space, action_space, episodes = 500,subtask = task1)

    if mode == 3: # put in training
      params = hyperparameter()
      params.Max_timesteps = 150
      params.Max_e = 0.8
      task1 = put_in()
      state_space = task1.state_space
      action_space = task1.action_space
      Total_r, epsilon = Double_DQN(task,params,state_space, action_space, episodes = 500,subtask = task1)

    if mode == 4:
      parameter_path = "./trained_models/parameters_put.pkl"
      task1 = put()
      state_space = task1.state_space
      action_space = task1.action_space
      test( task,state_space,action_space,max_timesteps = 150,path = parameter_path,
            episodes = 10,min_e=0.01,subtask = task1)
    if mode == 5:
      parameter_path = "./trained_models/parameters_put.pkl"
      task1 = put()
      state_space = task1.state_space
      action_space = task1.action_space
      test_all( task,state_space,action_space,max_timesteps = 150,path = parameter_path,
            episodes = 10,min_e=0.01,subtask = task1)
    if mode == 6:
      agent = Agent(env.action_shape)
      target = Shape.create(type=PrimitiveShape.SPHERE,
                      size=[0.02, 0.02, 0.02],
                      color=[1.0, 0.1, 0.1],
                      static=True, respondable=False)
      pos = list(np.array([ 0, -0.0 , 0.8]))
      target.set_position(pos)
      training_steps = 6000
      episode_length = 100
      for i in range(training_steps):
          if i % episode_length == 0:
              print('Reset Episode')
              
              _, obs = task.reset()

          action = agent.act(obs)
          starting_joint_positions = env._robot.arm.get_joint_positions()
          pos, quat = env._robot.arm.get_tip().get_position(), env._robot.arm.get_tip().get_quaternion()
          pos[0]= pos[0]-0.01
          my_step(env,pos,quat)
          #action = np.concatenate((task._robot.gripper.get_position(),np.array([1.0,0,0,0])))
          action = np.concatenate((task._robot.arm.get_joint_positions(),np.array([1.0])))
          # Step the task and obtain a new observation, reward and a terminate flag.
          obs, reward, terminate = task.step(action)
          # plt.imshow(obs.wrist_mask,cmap ='gray')
          # plt.show()
          # plt.imshow(obs.wrist_rgb)
          # plt.show()
          #plt.pause(0.5)
          #print(reward)
      print('Done')
    env.shutdown()
    