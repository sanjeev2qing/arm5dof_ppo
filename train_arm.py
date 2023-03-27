import os
import glob
import time
from datetime import datetime
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from arm5dof_hardware.arm5dof_uservo_server import Arm5DoFUServoServer
from arm5dof_interface.msg import JointAngleList, ToolPose
# 自定义服务
from arm5dof_interface.srv import GetArmState, GetJointAngle,\
	GetToolPose, SetDamping
# 自定义动作
from arm5dof_interface.action import GripperControl, \
	JointAngleControl, SetToolPose, MoveToWorkspace, GrabObject
import torch
import numpy as np

import gym

from PPO import PPO

################################### Training ###################################
def train(arm):
    print("============================================================================================")

    ####### initialize environment hyperparameters ######

    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 10                  # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################

    

    state_dim = 6
    action_dim = 6

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        # TODO MODIFY
        state = arm.arm.get_joint_angle_list(["joint1", "joint2", "joint3", "joint4", "joint5", "left_gripper_joint"])
        #state = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):

            # select action with policy
            action = ppo_agent.select_action(state)
            state = arm.arm.get_joint_angle_list(["joint1", "joint2", "joint3", "joint4", "joint5", "left_gripper_joint"])
            action_in = state + action*0.1
            #TODO MODIFY
            arm.arm.set_joint_angle_list_soft(joint_angle_list=action_in,joint_name_list = ["joint1", "joint2", "joint3", "joint4", "joint5", "left_gripper_joint"],T=0.5)
            #state, reward, done, _ = env.step(action)
            state = arm.arm.get_joint_angle_list(["joint1", "joint2", "joint3", "joint4", "joint5", "left_gripper_joint"])
            ee_pos = np.array(arm.arm.get_tool_pose()[:3])
            tar_pos = np.array([338.68807830688866, 10.948061519435399, 109.74647802644024])
            reward = -np.linalg.norm(ee_pos-tar_pos)
            done = np.linalg.norm(ee_pos-tar_pos)<10
            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)


            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

if __name__ == '__main__':
    rclpy.init()
    arm = Arm5DoFUServoServer()
    train(arm)
    
    
    
    
    
    
    
