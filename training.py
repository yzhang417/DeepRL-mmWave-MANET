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


#-----------------------------------------------------------
# Training the DRL agent
#-----------------------------------------------------------
def training(env, actor_critic_net, ac_optimizer, scheduler,\
             batches, slots, iterations, \
             gamma, lambd, value_coeff, entropy_coeff, \
             clip_param, decaying_clip_param, clipping_critic,\
             eval_loops, eval_ites, device, clip_queues):
    
    # Init
    evolution_reward_evaluated = []
    evolution_reward = []
    evolution_queue_length = []
    num_batch = np.ceil(slots/batches)
    if decaying_clip_param:
        clip_param_decay_amount = 1/(iterations*slots/batches)
    else:
        clip_param_decay_amount = 0
    
    # Detecting anomaly
    # torch.autograd.set_detect_anomaly(True)
    
    # Loop of episodes (also called epochs or iterations)
    for ite in range(iterations):
        
        # Code Profiler
        profiling_code = True
        if profiling_code and ite == 0:
            pr = cProfile.Profile()
            pr.enable()
        
        # Clear up the following lists
        ratios = []          # List of (pi/pi_old) in a batch (by default a batch is an episode)
        Vvals = []           # List of predicted value function in a batch (by default a batch is an episode)
        rewards = []         # List of reward in a batch (by default a batch is an episode)
        rewards_episode = [] # List of reward in a batch (by default a batch is an episode)
        if ite == 0:
            state = env.reset()  # Initial state for an epiosde
        else:
            if clip_queues:
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
            action_sanity_check(slot, action, state)            
            
            # Calculate the ratio (prob(current pi) / prob(old pi))
            logProb = torch.log(pi.squeeze(0)[action_chosen_index])
            oldlogProb = np.log(pi_dist[action_chosen_index])
            ratio = torch.exp(logProb-oldlogProb)
            
            # Interaction with the enviroment to get a new state and a step-level reward   
            state, output, reward, done = env.step(state, action, env.channel_realization())
            
            # Cumulate the reward, value function and ratio
            rewards_episode.append(reward)
            rewards.append(reward)
            Vvals.append(Vval)
            ratios.append(ratio)
            
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
                entropy = - torch.sum(pi*torch.log(pi))
                entropy_loss = - entropy_coeff * entropy
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
                    sys.stdout.write("Ite: {:0>4d}, A_loss: {:+.10f}, C_loss: {:+.5f}, E_loss: {:+.5f}, Ave final queue: {:.2f}, Ave reward: {:.3f}, Time: {:.3f}\n".format(ite, actor_loss, critic_loss, entropy_loss, np.mean(state.Queue_length), np.mean(rewards_episode), time_per_ite))
                    t1 = time.time()
                
                # Clear up the following lists
                entropys = []       # List of H(pi)
                ratios = []         # List of (pi/pi_old) in a batch (by default a batch is an episode)
                Vvals = []          # List of predicted value function in a batch (by default a batch is an episode)
                rewards = []        # List of reward in a batch (by default a batch is an episode)  
                

        # Update learning rate with scheduler
        scheduler.step()
        
        # Evaluation at checkpoint
        if ((ite+1) % eval_ites == 0 and ite > 0) or ite == iterations-1:
            # Evaluating the result
            if ite == iterations-1: eval_loops = eval_loops*10
            Final_queue, Ave_npkts_dep_per_slot, Queue_Eval, Delay_dist_Eval,\
            Ave_num_using_relay, Ave_num_using_relay_detailed,\
            Ave_num_doing_tracking, Ave_num_doing_tracking_detailed =\
            evaluation(actor_critic_net,env.env_parameter,slots,eval_loops)
            sys.stdout.write("\n----------------------------------------\n")
            sys.stdout.write("Evaluation starts\n")
            sys.stdout.write("----------------------------------------\n")
            sys.stdout.write("Iterations completed:{:4d}; Final_queue: {:.2f}; Npkts: {:.6f} \n".format(ite+1,Final_queue,Ave_npkts_dep_per_slot))
            sys.stdout.write("Current learning rate:{:.5f}; Current clipping parameter: {:.3f} \n".format(scheduler.get_last_lr()[0],clip_param))
            sys.stdout.write("Within {:3d} slots; Average uses of relay:{:.2f}; Average uses of tracking: {:.3f} \n".format(slots, Ave_num_using_relay,Ave_num_doing_tracking))
            print('Detailed relay usages')
            print(Ave_num_using_relay_detailed)
            print('Detailed ue tracking')
            print(Ave_num_doing_tracking_detailed)
            sys.stdout.write("\n----------------------------------------\n")
            sys.stdout.write("Evaluation ends\n")
            sys.stdout.write("----------------------------------------\n")
            evolution_reward_evaluated.append(Ave_npkts_dep_per_slot)
         
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
    return evolution_reward_evaluated, evolution_queue_length, evolution_reward, Queue_Eval, Delay_dist_Eval, actor_critic_net



#-----------------------------------------------------------
# Check the feasibility of the made action
#-----------------------------------------------------------
def action_sanity_check(slot, action, state):
    if slot>0 and action.UE_ID_BS2UE_Link<0:
        pdb.set_trace()
        sys.exit('Error: constraint violated: negative UE index occurs')
    violation_tracking = state.Is_Tracking and \
    ((action.UE_ID_BS2UE_Link != state.UE_ID_BS2UE_Link_Last_Slot) or state.Is_D2D_Link_Active)
    if violation_tracking:
        pdb.set_trace()
        sys.exit('Error: constraint violated: tracking order is not performed')
    violation_d2d_link = state.Is_D2D_Link_Active and\
    (action.UE_ID_BS2UE_Link == state.Tx_ID_D2D_Link or\
     action.UE_ID_BS2UE_Link == state.Rx_ID_D2D_Link or\
     action.Relay_ID == state.Tx_ID_D2D_Link or\
     action.Relay_ID == state.Rx_ID_D2D_Link or\
     state.Tx_ID_D2D_Link<0 or state.Rx_ID_D2D_Link<0)
    if violation_d2d_link:
        pdb.set_trace()
        sys.exit('Error: constraint violated: UEs in D2D link cannot be scheduled')
        

#-----------------------------------------------------------
# Draw an action with the current policy distribution
#-----------------------------------------------------------
def choose_action(pi_dist, env_parameter):
    action = def_action(env_parameter)
    num_type_of_action = int(action.num_type_of_action)
    num_action_per_type = action.num_action_per_type
    num_action = action.num_actions_valide
    threshold = 1e-8
    pi_dist[np.where(pi_dist<threshold)] = 0
    pi_dist = pi_dist/np.sum(pi_dist)
    if abs(np.sum(pi_dist) - 1) > 1e-5:
        sys.exit('Wrong distribution in training.py')
    else:
        pi_dist = pi_dist/np.sum(pi_dist)
        action_chosen_index = np.random.choice(np.squeeze(np.asarray(range(num_action))), size=1, p=np.squeeze(pi_dist))
    action_ndarray = action_index_to_action_ndarray(action_chosen_index,num_type_of_action,num_action_per_type)
    action.update_action_with_array(action_ndarray)
    return action, action_ndarray, np.squeeze(action_chosen_index)


#-----------------------------------------------------------
# Convert and index of action to a action ndarray
#-----------------------------------------------------------
def action_index_to_action_ndarray(action_chosen_index,num_type_of_action,num_action_per_type):
    action_ndarray = np.zeros(num_type_of_action,dtype=int)
    # Identify bw index
    num_actions_per_bw = num_action_per_type[0]*num_action_per_type[1]+num_action_per_type[0]
    action_ndarray[2] = action_chosen_index // num_actions_per_bw
    # Identify ue index
    action_chosen_index_new = action_chosen_index % num_actions_per_bw
    action_chosen_index_ue = action_chosen_index_new // num_action_per_type[1]
    action_chosen_index_relay = action_chosen_index_new % num_action_per_type[1]
    action_ndarray[1] = action_chosen_index_relay
    if action_chosen_index_ue < num_action_per_type[0]:
        action_ndarray[0] = action_chosen_index_ue
        action_ndarray[3] = 0
    else:
        action_ndarray[0] = action_chosen_index_relay
        action_ndarray[3] = 1
    for i in range(num_type_of_action):
        if action_ndarray[i] >= num_action_per_type[i]:
            sys.exit('Wrong transformation from action index to action array')
    return action_ndarray


#-----------------------------------------------------------
# Convert a action ndarray to a instance of def_action
#-----------------------------------------------------------
def action_ndarray_to_action_index(action_ndarray,num_type_of_action,num_action_per_type):
    action_index = int(0)
    num_actions_per_bw = num_action_per_type[0]*num_action_per_type[1]+num_action_per_type[0]
    action_index = action_index + action_ndarray[2] * num_actions_per_bw
    action_index = action_index + action_ndarray[1]
    if action_ndarray[3] == 0:
        action_index = action_index + action_ndarray[0]*num_action_per_type[1]
    else:
        action_index = action_index + num_action_per_type[0]*num_action_per_type[1]
    return action_index


#-----------------------------------------------------------
# Evaluate the checkpoint
#-----------------------------------------------------------
def evaluation(actor_critic_net, env_parameter, slots, LOOP):
    Queue = np.zeros((LOOP,slots,env_parameter.N_UE),dtype=int);
    Delay_Dist = np.zeros((LOOP,slots+1,env_parameter.N_UE),dtype=int);
    npkts_departure_evolution = np.zeros((LOOP,slots,env_parameter.N_UE),dtype=int);
    envEvaluation = envs(env_parameter,slots)
    num_using_relay = np.zeros((env_parameter.N_UE,env_parameter.N_UE),dtype=int)
    num_doing_tracking = np.zeros(env_parameter.N_UE,dtype=int)
    # Evaluation loop
    for loop in range(LOOP):  
        state = envEvaluation.reset();
        for slot in range(slots):            
            # Using actor-critic NNs to predict the value function Vval and a policy pi
            input_state = state.to_ndarray_normalized()
            Vval, pi = actor_critic_net.forward(input_state)
            # Choose an action from the generated policy
            pi_dist = pi.detach().cpu().numpy()
            action, action_chosen_ndarray, action_chosen_index= choose_action(pi_dist, env_parameter)
            action_sanity_check(slot, action, state)
            if state.Is_D2D_Link_Active:
                num_using_relay[state.Tx_ID_D2D_Link,state.Rx_ID_D2D_Link] += 1
            if state.Is_Tracking:
                num_doing_tracking[state.UE_ID_BS2UE_Link_Last_Slot] += 1
            # Interaction with the enviroment to get a new state and a step-level reward
            state, output, reward, done = envEvaluation.step(state, action, envEvaluation.channel_realization())
            Queue[loop,slot,:] = envEvaluation.Queue[slot,:]
            npkts_departure_evolution[loop,slot,:] = envEvaluation.npkts_departure_evolution[slot,:]
        Delay_Dist[loop,:,:] = envEvaluation.delay_dist
    # Output results
    Final_queue = np.mean(np.mean(Queue,axis=2),axis=0)[-1]
    Ave_npkts_dep_per_slot = np.mean(np.sum(npkts_departure_evolution,axis=2))
    Ave_num_using_relay = np.sum(num_using_relay)/LOOP
    Ave_num_using_relay_detailed = num_using_relay/LOOP
    Ave_num_doing_tracking = np.sum(num_doing_tracking)/LOOP
    Ave_num_doing_tracking_detailed = num_doing_tracking/LOOP
    return Final_queue, Ave_npkts_dep_per_slot, Queue, Delay_Dist, Ave_num_using_relay, Ave_num_using_relay_detailed,\
Ave_num_doing_tracking, Ave_num_doing_tracking_detailed
