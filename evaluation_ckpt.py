import numpy as np
import operator
from envs import *
from my_utils import *
from state_action import *
from decision_making import *
from bandit_function import *


#-----------------------------------------------------------
# Evaluate the checkpoint
#-----------------------------------------------------------
def evaluation_ckpt(actor_critic_net, env_parameter, slots, LOOP, \
    bandit_relay_para_list_trained = None, bandit_bw_para_list_trained = None, current_state_copy = None,\
    scheme_setting_list = None, scheme_id = None):
    
    # Check if it is for DRL evaluation
    if scheme_setting_list is None:
        N_schemes = 1
        scheme_id = 0;
    else:
        N_schemes = len(scheme_setting_list)
    
    # Result vector
    Queue = np.zeros((LOOP,slots,env_parameter.N_UE),dtype=int);
    Delay_dist = np.zeros((LOOP,slots+1,env_parameter.N_UE),dtype=int);
    npkts_departure_evolution = np.zeros((LOOP,slots,env_parameter.N_UE),dtype=int)
    num_using_relay = np.zeros((env_parameter.N_UE,env_parameter.N_UE),dtype=int)
    num_bw_selection = np.zeros((env_parameter.N_UE,env_parameter.K_bw),dtype=int)
    num_doing_tracking = np.zeros(env_parameter.N_UE,dtype=int)
    num_under_blockage = np.zeros(env_parameter.N_UE,dtype=int)
    
    # Evaluatio loop
    for loop in range(LOOP):
        state_list = list();
        action_list = list();
        output_list = list();
        lastOutput_list = list();
        bandit_bw_para_list = list();
        bandit_relay_para_list = list();
        env_list = list();
        for i in range(N_schemes):
            bandit_bw_para_list.append(def_bandit_bw_parameter(env_parameter));
            bandit_relay_para_list.append(def_bandit_relay_parameter(env_parameter));
            state_list.append(def_state(env_parameter));
            action_list.append(def_action(env_parameter));
            output_list.append(def_output(env_parameter)); 
            lastOutput_list.append(def_output(env_parameter)); 
            env_list.append(envs(env_parameter,slots));

        # Initialization of enviroment
        state_list[scheme_id] = env_list[scheme_id].reset()

        # Loop of slots in the training process
        for ct in range(slots):      
            # Last state
            last_state = copy.deepcopy(state_list[scheme_id]) 
            
            # Decisiion making
            if (scheme_setting_list is not None) and operator.not_(scheme_setting_list[scheme_id].Is_RL):         
                # Load traine information
                bandit_bw_para_list = copy.deepcopy(bandit_bw_para_list_trained)
                bandit_relay_para_list = copy.deepcopy(bandit_relay_para_list_trained)
                state_list[scheme_id].est_arrival = current_state_copy.est_arrival
                state_list[scheme_id].est_depart = current_state_copy.est_depart
                state_list[scheme_id].Reff_BS2UE_Estimated = current_state_copy.Reff_BS2UE_Estimated
                state_list[scheme_id].n_BS2UE = current_state_copy.n_BS2UE
                state_list[scheme_id].Reff_BS2UE_Tracking_Estimated = current_state_copy.Reff_BS2UE_Tracking_Estimated
                state_list[scheme_id].n_BS2UE_Tracking = current_state_copy.n_BS2UE_Tracking
                decision_making(ct, scheme_id, state_list, action_list, lastOutput_list,\
                                bandit_bw_para_list, bandit_relay_para_list, scheme_setting_list, \
                                env_list[scheme_id].env_parameter,False)
            else:
                input_state = state_list[scheme_id].to_ndarray_normalized()
                Vval, pi = actor_critic_net.forward(input_state)
                pi_dist = pi.detach().cpu().numpy()
                action_list[scheme_id], action_chosen_ndarray, action_chosen_index = \
                choose_action(pi_dist, env_list[scheme_id].env_parameter)

            # Sanity check of action
            action_sanity_check(ct, action_list[scheme_id], state_list[scheme_id])       

            # Interaction with enviroment
            state_list[scheme_id], output_list[scheme_id], reward, done = \
            env_list[scheme_id].step(state_list[scheme_id], action_list[scheme_id], env_list[scheme_id].channel_realization())

            # Save last output
            lastOutput_list[scheme_id] = output_list[scheme_id]; 

            # Statistics udpate
            num_bw_selection[action_list[scheme_id].Relay_ID,action_list[scheme_id].BW_ID_BS2UE_Link] += 1
            num_under_blockage[action_list[scheme_id].Relay_ID] = num_under_blockage[action_list[scheme_id].Relay_ID] + \
            env_list[scheme_id].is_in_blockage[action_list[scheme_id].Relay_ID,action_list[scheme_id].Relay_ID]
            if action_list[scheme_id].Relay_ID == action_list[scheme_id].UE_ID_BS2UE_Link and \
            action_list[scheme_id].UE_ID_BS2UE_Link >= 0:
                num_using_relay[action_list[scheme_id].UE_ID_BS2UE_Link,action_list[scheme_id].UE_ID_BS2UE_Link] += 1
            if last_state.Is_D2D_Link_Active:
                num_using_relay[last_state.Tx_ID_D2D_Link,last_state.Rx_ID_D2D_Link] += 1
            if last_state.Is_Tracking:
                num_doing_tracking[last_state.UE_ID_BS2UE_Link_Last_Slot] += 1

        # Statistics udpate
        Queue[loop,:,:] = env_list[scheme_id].Queue
        npkts_departure_evolution[loop,:,:] = env_list[scheme_id].npkts_departure_evolution[:,:]
        Delay_dist[loop,:,:] = env_list[scheme_id].delay_dist
        Delay_dist[loop,:,:] = env_list[scheme_id].get_delay_distribution()

    # Output results
    Final_queue = np.mean(np.mean(Queue,axis=2),axis=0)[-1]
    Ave_npkts_dep_per_slot = np.mean(np.sum(npkts_departure_evolution,axis=2))
    Ave_num_using_relay = (np.sum(num_using_relay)-np.sum(np.diag(num_using_relay)))/LOOP
    Ave_num_using_relay_detailed = num_using_relay/LOOP
    Ave_num_doing_tracking = np.sum(num_doing_tracking)/LOOP
    Ave_num_doing_tracking_detailed = num_doing_tracking/LOOP
    Ave_num_bw_selection_detailed = num_bw_selection/LOOP
    Ave_ratio_under_blockage = np.sum(num_under_blockage)/LOOP
    Ave_ratio_under_blockage_detailed = num_under_blockage/LOOP
    return Final_queue, Ave_npkts_dep_per_slot, Queue, Delay_dist,\
Ave_num_using_relay, Ave_num_using_relay_detailed,\
Ave_num_doing_tracking, Ave_num_doing_tracking_detailed,\
Ave_num_bw_selection_detailed, Ave_ratio_under_blockage, Ave_ratio_under_blockage_detailed


# #-----------------------------------------------------------
# # Evaluate the checkpoint deprecated
# #-----------------------------------------------------------
# def evaluation(actor_critic_net, env_parameter, slots, LOOP):
#     Queue = np.zeros((LOOP,slots,env_parameter.N_UE),dtype=int);
#     Delay_dist = np.zeros((LOOP,slots+1,env_parameter.N_UE),dtype=int);
#     npkts_departure_evolution = np.zeros((LOOP,slots,env_parameter.N_UE),dtype=int);
#     envEvaluation = envs(env_parameter,slots)
#     num_using_relay = np.zeros((env_parameter.N_UE,env_parameter.N_UE),dtype=int)
#     num_bw_selection = np.zeros((env_parameter.N_UE,env_parameter.K_bw),dtype=int)
#     num_doing_tracking = np.zeros(env_parameter.N_UE,dtype=int)
#     num_under_blockage = np.zeros(env_parameter.N_UE,dtype=int)
#     # Evaluation loop
#     for loop in range(LOOP):  
#         state = envEvaluation.reset();
#         for slot in range(slots):            
#             # Decision making
#             input_state = state.to_ndarray_normalized()
#             Vval, pi = actor_critic_net.forward(input_state)
#             pi_dist = pi.detach().cpu().numpy()
#             action, action_chosen_ndarray, action_chosen_index = choose_action(pi_dist, env_parameter)
            
#             # Sanity check of action
#             action_sanity_check(slot, action, state)

#             # Interaction with the enviroment to get a new state and a step-level reward
#             state, output, reward, done = envEvaluation.step(state, action, envEvaluation.channel_realization())
            
#             # Statistics udpate
#             if state.Is_D2D_Link_Active:
#                 num_using_relay[state.Tx_ID_D2D_Link,state.Rx_ID_D2D_Link] += 1
#             if state.Is_Tracking:
#                 num_doing_tracking[state.UE_ID_BS2UE_Link_Last_Slot] += 1
#             num_bw_selection[action.Relay_ID,action.BW_ID_BS2UE_Link] += 1
#             num_under_blockage[action.Relay_ID] = num_under_blockage[action.Relay_ID] + \
#             envEvaluation.is_in_blockage[action.Relay_ID,action.Relay_ID]
            
#         # Statistics udpate
#         Queue[loop,:,:] = envEvaluation.Queue
#         npkts_departure_evolution[loop,:,:] = envEvaluation.npkts_departure_evolution
#         Delay_dist[loop,:,:] = envEvaluation.delay_dist
#         Delay_dist[loop,:,:] = envEvaluation.get_delay_distribution()
                             
#     # Output results
#     Final_queue = np.mean(np.mean(Queue,axis=2),axis=0)[-1]
#     Ave_npkts_dep_per_slot = np.mean(np.sum(npkts_departure_evolution,axis=2))
#     Ave_num_using_relay = np.sum(num_using_relay)/LOOP
#     Ave_num_using_relay_detailed = num_using_relay/LOOP
#     Ave_num_doing_tracking = np.sum(num_doing_tracking)/LOOP
#     Ave_num_doing_tracking_detailed = num_doing_tracking/LOOP
#     Ave_num_bw_selection_detailed = num_bw_selection/LOOP
#     Ave_ratio_under_blockage = np.sum(num_under_blockage)/LOOP
#     Ave_ratio_under_blockage_detailed = num_under_blockage/LOOP
#     return Final_queue, Ave_npkts_dep_per_slot, Queue, Delay_dist,\
# Ave_num_using_relay, Ave_num_using_relay_detailed,\
# Ave_num_doing_tracking, Ave_num_doing_tracking_detailed,\
# Ave_num_bw_selection_detailed, Ave_ratio_under_blockage, Ave_ratio_under_blockage_detailed