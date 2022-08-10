# This Python file uses the following encoding: utf-8
# coding=utf-8
import os
import sys
import math
import time
import numpy as np
import pdb
import torch
import cProfile, pstats, io
from pstats import SortKey
from model import *
from envs import *
from state_action import *
from my_utils import *
from evaluation_ckpt import *


#-----------------------------------------------------------
# Training the DRL agent
#-----------------------------------------------------------
def training(env, actor_critic_net, ac_optimizer, scheduler,\
             batches, slots, iterations, \
             gamma, lambd, value_coeff, entropy_coeff, \
             clip_param, decaying_clip_param, clipping_critic,\
             eval_loops, eval_ites, device, clip_queues, Eval_At_Customized_Points, netw_topo_changing):
    # print option
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=2)
    
    # Evaluation results
    evolution_reward = [] # Evolution of reward of the whole training process 
    evolution_queue_length = [] # Evolution of the queue length of the whole training process
    evolution_rate_ckpt = [] # Evolution of average data rate at each checking point for Slots time slots
    evolution_delay_ckpt = [] # Evolution of average delay at each checking point for Slots time slots
    evolution_ratio_blockage_ckpt = [] # Evolution of ratio under blocakge at each checking point for Slots time slots
    
    # Init
    num_batch = np.ceil(slots/batches)
    if decaying_clip_param:
        clip_param_decay_amount = 1/(iterations*slots/batches)
    else:
        clip_param_decay_amount = 0
    
    # Detecting anomaly
    # torch.autograd.set_detect_anomaly(True)
    
    num_netw_topo_changing = 0;
    
    # Loop of episodes (also called epochs or iterations)
    for ite in range(iterations):
        
        # Code Profiler
        profiling_code = False
        if profiling_code and ite == 0:
            pr = cProfile.Profile()
            pr.enable()
        
        # Clear up the following lists
        ratios = []          # List of (pi/pi_old) in a batch (by default a batch is an episode)
        Vvals = []           # List of predicted value function in a batch (by default a batch is an episode)
        entropys = []           # List of predicted value function in a batch (by default a batch is an episode)
        rewards = []         # List of reward in a batch (by default a batch is an episode)
        rewards_episode = [] # List of reward in a batch (by default a batch is an episode)
        if ite == 0:
            state = env.reset()  # Initial state for an epiosde
        else:
            if netw_topo_changing and num_netw_topo_changing == 0 and ite>=iterations/4 and ite<iterations*2/4:
                env_parameter_new = env_init(11)
                env = envs(env_parameter_new,slots)
                num_netw_topo_changing = 1;
                print('Network Changed to network topo 11')
                state = env.reset()
            elif netw_topo_changing and num_netw_topo_changing == 1 and ite>=iterations*2/4 and ite<iterations*3/4:
                env_parameter_new = env_init(12)
                env = envs(env_parameter_new,slots)
                num_netw_topo_changing = 2;
                print('Network Changed to network topo 12')
                state = env.reset()
            elif netw_topo_changing and num_netw_topo_changing == 2 and ite>=iterations*3/4:
                env_parameter_new = env_init(13)
                env = envs(env_parameter_new,slots)
                num_netw_topo_changing = 3;
                print('Network Changed to network topo 13')
                state = env.reset()
            elif clip_queues:
                state = env.reset()
            else:
                env.reset()
                env.npkts_arrival = last_state.Queue_length
                env.npkts_arrival_evolution[env.ct,:] = env.npkts_arrival
                env.Queue[env.ct,:] = env.npkts_arrival
                env.current_Queue_dist_by_delay[1,:] = env.npkts_arrival # Initial packets have delay of 1 slot
                state = last_state                
        
        # Loop time slots (steps) within one episodes
        # Timer starts
        t1 = time.time()
        for slot in range(slots):   

            # Using actor-critic NNs to predict the value function Vval and a policy pi
            input_state = state.to_ndarray_normalized()
            Vval, pi = actor_critic_net.forward(input_state)
            
            # Choose an action from the generated policy
            pi_dist = pi.detach().cpu().numpy()
            action, action_ndarray, action_chosen_index = choose_action(pi_dist, env.env_parameter) 
            # Sanity check of action
            #action_sanity_check(slot, action, state)            
            
            # Calculate the ratio (prob(current pi) / prob(old pi))
            logProb = torch.log(pi.squeeze(0)[action_chosen_index])
            oldlogProb = np.log(pi_dist[action_chosen_index])
            ratio = torch.exp(logProb-oldlogProb)
            
            # Calulate the entropy
            pi_revise = pi+1e-10
            pi_revise = pi_revise/torch.sum(pi_revise)
            entropy = - torch.sum(pi_revise*torch.log(pi_revise))
            
            
            # Interaction with the enviroment to get a new state and a step-level reward   
            state, output, reward, done = env.step(state, action, env.channel_realization())
            
            # Cumulate the reward, value function, ratio and entropy
            rewards_episode.append(reward)
            rewards.append(reward)
            Vvals.append(Vval)
            ratios.append(ratio)
            entropys.append(entropy)
            
            # Save the last state of the current iteration (deepcopy is time consuming)
            if slot == slots-1 :
                last_state = copy.deepcopy(state)
                #last_action = copy.deepcopy(action)
            # Save queue length
            evolution_queue_length.append(np.mean(state.Queue_length))
            evolution_reward.append(reward)
                
            #------------------ Update policy with batch of samples or at the end of the episode ------------------#
            if done or ((slot+1) % batches == 0): 
                
                # Compute actor and predict next value function with actor-critic NN
                input_state = state.to_ndarray_normalized()
                Vval, pi = actor_critic_net.forward(input_state)
                
                Vvals.append(Vval)
                Vvals = torch.stack(Vvals)
                Vvals = Vvals.squeeze(1)
                
                # Compute discounted cumulated rewards, refered as return, label for critic network
                Returns = torch.zeros_like(Vvals, requires_grad = False)
                Returns[-1] = Vvals[-1]
                for t in reversed(range(len(rewards))):
                    Returns[t] = rewards[t] + gamma * Returns[t+1]
                
                # Compute advantage values
                advantages = torch.zeros_like(Vvals, requires_grad = False)
                for t in reversed(range(len(rewards))):
                    delta = rewards[t] + gamma * Vvals[t+1] - Vvals[t]
                    advantages[t] = delta + gamma * lambd * advantages[t+1]  
                advantages = advantages[0:-1]
                
                # Normalize the advantages
                if batches > 1:
                    Advs = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                else:
                    Advs = advantages
                
                #------------------------------------
                # Calculate loss
                #------------------------------------
                # Actor loss
                ratios = torch.stack(ratios)
                surr1 = ratios * Advs.detach()
                surr2 = torch.clamp(ratios, 1.0-clip_param, 1.0+clip_param) * Advs.detach()
                actor_loss = - torch.min(surr1, surr2).mean()
                # Critic loss
                #############################################################################################
                # Note: it is important to bounded return by 1 for example as we use clipped Vvals here.
                # If Vvals is in scale of 100, clip by 0.2 may slow down the learning speed
                #############################################################################################
                critic_losses_1 = (Returns - Vvals).pow(2)
                Vvals_clipped = Vvals.detach() + torch.clamp(Vvals-Vvals.detach(), -clip_param, clip_param)
                critic_losses_2 = (Returns - Vvals_clipped).pow(2)
                if clipping_critic:
                    critic_loss = value_coeff * torch.max(critic_losses_1,critic_losses_2).mean()
                else:
                    critic_loss = value_coeff * critic_losses_1.mean()
                    
                # Entropy loss
                pi_revise = pi+1e-10
                pi_revise = pi_revise/torch.sum(pi_revise)
                entropy = - torch.sum(pi_revise*torch.log(pi_revise))
                entropys.append(entropy)
                entropys = torch.stack(entropys)
                
                entropy_loss = - entropy_coeff * torch.mean(entropys)
                #entropy = - torch.sum(pi*torch.log(pi))
                #entropy_loss = - entropy_coeff * entropy
                
                # All loss
                ac_loss = actor_loss + critic_loss + entropy_loss

                # Backpropagate the gradient to update the NN
                ac_optimizer.zero_grad()
                ac_loss.backward()
                ac_optimizer.step()
                clip_param = max(clip_param - clip_param_decay_amount,0)
                
                # Prints results
                if done:
                    time_per_ite = time.time() - t1
                    sys.stdout.write("Ite: {:0>4d}, A_loss: {:+.5f}, C_loss: {:+.5f}, E_loss: {:+.5f}, Final queue: {:.2f}, Ave reward: {:.3f}, Time: {:.3f}\n".format(ite, actor_loss, critic_loss, entropy_loss, np.mean(state.Queue_length), np.mean(rewards_episode), time_per_ite))
                    t1 = time.time()
                
                # Clear up the following lists
                entropys = []       # List of H(pi)
                ratios = []         # List of (pi/pi_old) in a batch (by default a batch is an episode)
                Vvals = []          # List of predicted value function in a batch (by default a batch is an episode)
                rewards = []        # List of reward in a batch (by default a batch is an episode)  
                

        # Update learning rate with scheduler
        scheduler.step()
        
        # Evaluation at checkpoint
        To_Evaluate = False;
        if Eval_At_Customized_Points:
            if netw_topo_changing == 0:
                costomized_check_point = np.array([1,4,7,10,20,40,60,80,100,150,200,240])-1;
            else:
                costomized_check_point_base = np.array([1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,30,40,50,60,80,100])-1;
                #costomized_check_point_base = np.array([1,iterations/4])-1;
                costomized_check_point = costomized_check_point_base
                for i in range(3):
                    costomized_check_point = np.append(costomized_check_point, (costomized_check_point_base+iterations/4*(i+1)))
            if ite in costomized_check_point.tolist():
                To_Evaluate = True
        else:
            if ((ite+1) % eval_ites == 0 and ite > 0) or ite == iterations-1:
                To_Evaluate = True
                if ite == iterations-1: 
                    eval_loops = 200
        # To Evaluate the selected checkpoint
        if To_Evaluate == True:
            To_Evaluate = False;
            t_Eval_start = time.time()
            sys.stdout.write("\n----------------------------------------\n")
            sys.stdout.write("Evaluation starts\n")
            sys.stdout.write("----------------------------------------\n")
            Final_queue, Ave_npkts_dep_per_slot, Queue_Eval, Delay_dist_Eval,\
            Ave_num_using_relay, Ave_num_using_relay_detailed,\
            Ave_num_doing_tracking, Ave_num_doing_tracking_detailed,\
            Ave_num_bw_selection_detailed, Ave_ratio_under_blockage, Ave_ratio_under_blockage_detailed = \
            evaluation_ckpt(actor_critic_net,env.env_parameter,slots,eval_loops)
            sys.stdout.write("Iterations completed:{:4d}; Final queue: {:.2f}; Npkts: {:.6f} \n".format(ite+1,Final_queue,Ave_npkts_dep_per_slot))
            sys.stdout.write("Current learning rate:{:.5f}; Current clipping parameter: {:.3f} \n".format(scheduler.get_last_lr()[0],clip_param))
            sys.stdout.write("Within {:3d} slots; Average uses of relay:{:.2f}; Average uses of tracking: {:.3f} \n".format(slots, Ave_num_using_relay,Ave_num_doing_tracking))
            print('Detailed relay usages')
            print(Ave_num_using_relay_detailed)
            print('Detailed info on main link bw selection')
            print(Ave_num_bw_selection_detailed)
            print('Detailed ue tracking')
            print(Ave_num_doing_tracking_detailed)
            print('Detailed info on main link under blockage')
            print(Ave_ratio_under_blockage_detailed)
            ave_delay_in_slots_dist = np.sum(np.mean(Delay_dist_Eval,axis=0),axis=1)/\
            np.sum(np.mean(Delay_dist_Eval,axis=0))
            ave_delay_in_slots = \
            np.dot(np.transpose(ave_delay_in_slots_dist),np.asarray(range(len(ave_delay_in_slots_dist))))
            print('Average delay of packets in slots')
            print(ave_delay_in_slots)
            t_Eval = time.time() - t_Eval_start
            print('Time spent at this CKPT is %f seconds' %t_Eval)
            sys.stdout.write("----------------------------------------\n")
            sys.stdout.write("Evaluation ends\n")
            sys.stdout.write("----------------------------------------\n")
            # Save the parameter evolution            
            evolution_rate_ckpt.append(Ave_npkts_dep_per_slot*env.env_parameter.mean_packet_size/env.env_parameter.t_slot/1e9)
            evolution_delay_ckpt.append(ave_delay_in_slots)
            evolution_ratio_blockage_ckpt.append(Ave_ratio_under_blockage)     
            
        # Code Profiler
        if profiling_code and ite == 0:
            print('\n-------------------------------------------------------------------------------')
            print('cProfile output')
            print('-------------------------------------------------------------------------------')
            pr.disable()
            s = io.StringIO()
            #sortby = SortKey.CUMULATIVE
            sortby = 'tottime'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats(10)
            print(s.getvalue())
    
    # Return results
    return evolution_queue_length, evolution_reward, evolution_rate_ckpt, evolution_delay_ckpt, evolution_ratio_blockage_ckpt, Queue_Eval, Delay_dist_Eval, Ave_num_using_relay_detailed, Ave_num_bw_selection_detailed, Ave_num_doing_tracking_detailed, Ave_ratio_under_blockage_detailed, actor_critic_net
