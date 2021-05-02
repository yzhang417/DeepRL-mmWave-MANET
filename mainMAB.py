#----------------------------------------------
# Import packages
#----------------------------------------------
import argparse
import os
import os.path
from os import path
import sys
import math
import random
import time
import pdb
import operator 
import copy
import pickle
import platform
import numpy as np
import matplotlib.pyplot as plt
from numpy import matlib as mb
import torch
from envs import *
from my_utils import *
from state_action import *
from decision_making import *
from training import *
from bandit_function import *
from get_MAB_scheme_setting import *
from evaluation_ckpt import *
np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={'int': '{:2d}'.format})
np.set_printoptions(formatter={'float': '{:6.3f}'.format})

#-----------------------------------------------------------
# main function
#-----------------------------------------------------------
def main():
    print('\r\n------------------------------------')
    print('Enviroment Sumamry')
    print('------------------------------------')
    print('PyTorch ' + str(torch.__version__))
    print('Running with Python ' + str(platform.python_version()))    
    
    #-----------------------------------------------------------
    # Parse command line arguments
    #-----------------------------------------------------------
    print('\r\n------------------------------------')
    print('System Parameters')
    print('------------------------------------')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Mode
    parser.add_argument('--training', default=True, type=int, help='enable training phase')
    parser.add_argument('--testing', default=False, type=int, help='enable testing phase')
    parser.add_argument('--cuda', default=0, type=int, help='use to enable available CUDA')
    parser.add_argument('--Netw_topo_id', default=1, type=int, help='Id of network topology')
    parser.add_argument('--output', default=None, help='output folder of training results')
    # Training process
    parser.add_argument('--iterations', default=120, type=int, help='number of episodes')
    parser.add_argument('--slots', default=1500, type=int, help='number of slots in a single episode')
    parser.add_argument('--eval_loops', default=10, type=int, help='number of evaluations for a checkpoint')
    parser.add_argument('--eval_ites', default=1, type=int, help='number of iterations before each ckpt evaluation')
    parser.add_argument('--clip_queues', default=False, type=int, help='clip the queue at the end of each iteration')
    parser.add_argument('--loading_DRL', default=False, type=int, help='load the trained DRL')
    #Print args
    args = parser.parse_args()
    for item in vars(args):
        print(item + ': ' + str(getattr(args, item)))
    
    #----------------------------------------------
    # Initialization
    #----------------------------------------------
    Netw_topo_id = args.Netw_topo_id # Network topology             
    clip_queues = args.clip_queues  # Clip the queue for every iteration
    iterations = args.iterations # Number of iteration to train bandit
    eval_ites = args.eval_ites   # Number number of iterations before each ckpt evaluation
    eval_loops = args.eval_loops         # Number of testing MAB-based scheduler at each checkpoint
    slots_monitored = args.slots;         # Total time slots to be monitored in testing
    slots_training = iterations * slots_monitored # Total number of slots trained
    env_parameter = env_init(Netw_topo_id) # Enviroment parameters
    t_slot = env_parameter.t_slot          # Time duration for a single slot in seconds
    mean_packet_size = env_parameter.mean_packet_size # Mean packet size
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=2)
    
    #-----------------------------------------------------------
    # Check output folder
    #-----------------------------------------------------------
    if args.output is not None:
        output_folder = args.output+'/'
    else:
        output_folder = 'output_net'+str(args.Netw_topo_id)+'/'        
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print('Total time monitored is %0.2f seconds' %(slots_monitored*t_slot))
    print('Total time for MAB training is %0.2f minutes' %(slots_training*t_slot/60))
    print('Maximum activity range is %d meters' %env_parameter.max_activity_range)
    print('Average time in blockage is \r\n', env_parameter.target_prob_blockage)
    print('Probabbility of blockage is \r\n', env_parameter.prob_blockage)
    print('Proportion of arrival rate is \r\n', env_parameter.lambda_ratio)
    print('Workload is %0.2f Gbps\r\n' %(env_parameter.workload/1e9))

    #----------------------------------------------
    # All MAB algorithms settings
    #----------------------------------------------
    scheme_setting_list = get_MAB_scheme_setting()
    
    #----------------------------------------------
    # Result vectors
    #----------------------------------------------
    N_schemes = len(scheme_setting_list) - 1 # The last one is for DRL method
    evolution_reward_MAB = [list() for _ in range(N_schemes)] # Evolution of reward of the whole training process 
    evolution_queue_length_MAB = [list() for _ in range(N_schemes)] # Evolution of the queue length of the whole training process
    evolution_rate_ckpt_MAB = [list() for _ in range(N_schemes)] # Evolution of average data rate at each checking point for Slots time slots
    evolution_delay_ckpt_MAB = [list() for _ in range(N_schemes)] # Evolution of average delay at each checking point for Slots time slots
    evolution_ratio_blockage_ckpt_MAB = [list() for _ in range(N_schemes)] # Evolution of ratio under blocakge at each checking point for Slots time slots
    Queue_Eval_MAB = np.zeros((200,slots_monitored,env_parameter.N_UE,N_schemes));
    Delay_dist_Eval_MAB = np.zeros((200,slots_monitored+1,env_parameter.N_UE,N_schemes));
    Ave_num_using_relay_detailed_MAB = np.zeros((env_parameter.N_UE,env_parameter.N_UE,N_schemes));
    Ave_num_bw_selection_detailed_MAB = np.zeros((env_parameter.N_UE,env_parameter.K_bw,N_schemes));
    Ave_num_doing_tracking_detailed_MAB = np.zeros((env_parameter.N_UE,N_schemes));
    Ave_ratio_under_blockage_detailed_MAB = np.zeros((env_parameter.N_UE,N_schemes));
    
    #----------------------------------------------
    # Default state/action/output/banditBW/banditRelay list
    #----------------------------------------------
    state_list = list();
    action_list = list();
    output_list = list();
    lastOutput_list = list();
    bandit_bw_para_list = list();
    bandit_relay_para_list = list();
    env_list = list();
    for scheme_id in range(N_schemes):
        bandit_bw_para_list.append(def_bandit_bw_parameter(env_parameter));
        bandit_relay_para_list.append(def_bandit_relay_parameter(env_parameter));
        state_list.append(def_state(env_parameter));
        action_list.append(def_action(env_parameter));
        output_list.append(def_output(env_parameter)); 
        lastOutput_list.append(def_output(env_parameter)); 
        env_list.append(envs(env_parameter,slots_training));

    
    #-----------------------------------------------------------
    # Random seed
    #-----------------------------------------------------------
    random.seed(13579)     # random seeds for reproducation
    np.random.seed(246810) # random seeds for reproducation
        
    #----------------------------------------------
    # Training MAB-based scheduler
    #----------------------------------------------
    # Common enviroment for packet arrival and channel shadowing
    envCommon = envs(env_parameter,slots_training);
    
    # Load trained DRL-based controler
    if args.loading_DRL:
        loadpath = output_folder+'trained_model_DRL_netwTopo'+str(Netw_topo_id)+'.pt'
        checkpoint = torch.load(loadpath)
        DRL_Decision_Maker = checkpoint['model']
        DRL_Decision_Maker.load_state_dict(checkpoint['model_state_dict'])
        DRL_Decision_Maker.eval()
    else:
        DRL_Decision_Maker = None
    
    # Initialization of enviroment
    envCommon.reset();
    for scheme_id in range(N_schemes):
        env_list[scheme_id].is_external_packet_arrival_process = True
        env_list[scheme_id].external_npkts_arrival = envCommon.packet_arrival_process()
        state_list[scheme_id] = env_list[scheme_id].reset()

    # Timer start
    tStart = time.time()
    # Loop of slots in the training process
    for ct in range(slots_training):  
        # Channel realization (Maybe to consider the correlation between time slots)
        channel = envCommon.channel_realization()
        envCommon.ct += 1
        # Common packet arrival process
        common_npkts_arrival = envCommon.packet_arrival_process()        

        # Loop of schemes (one-step interaction with enviroment and update)
        for scheme_id in range(N_schemes):            
            # Decisiion making
            if scheme_setting_list[scheme_id].Is_RL:                
                break
            else:
                Last_BW_ID_BS2UE_Link = action_list[scheme_id].BW_ID_BS2UE_Link
                last_MCS_ID_BS2UE = lastOutput_list[scheme_id].MCS_ID_BS2UE
                decision_making(ct, scheme_id, state_list, action_list, lastOutput_list,\
                                bandit_bw_para_list, bandit_relay_para_list, scheme_setting_list, \
                                env_list[scheme_id].env_parameter)

            # Sanity check of action
            action_sanity_check(ct, action_list[scheme_id], state_list[scheme_id])       

            # Interaction with enviroment
            last_state = copy.deepcopy(state_list[scheme_id]) 
            env_list[scheme_id].external_npkts_arrival = common_npkts_arrival
            state_list[scheme_id], output_list[scheme_id], reward, done = \
            env_list[scheme_id].step(state_list[scheme_id], action_list[scheme_id], channel)
            evolution_queue_length_MAB[scheme_id].append(np.mean(state_list[scheme_id].Queue_length))
            evolution_reward_MAB[scheme_id].append(reward)

            # Save last output
            output_list[scheme_id].Last_BW_ID_BS2UE_Link = Last_BW_ID_BS2UE_Link
            output_list[scheme_id].last_MCS_ID_BS2UE = last_MCS_ID_BS2UE
            lastOutput_list[scheme_id] = output_list[scheme_id]; 

            # Step-wise bandit vector update
            bandit_relay_para_list[scheme_id] = \
            update_bandit_relay_para(bandit_relay_para_list[scheme_id],last_state,\
                                     action_list[scheme_id],output_list[scheme_id],\
                                     env_list[scheme_id].env_parameter);
            bandit_bw_para_list[scheme_id] = \
            update_bandit_bw_para(bandit_bw_para_list[scheme_id],\
                                  action_list[scheme_id],output_list[scheme_id],\
                                  env_list[scheme_id].env_parameter);
        
        # Evaluation at the check point
        if (((ct+1) % (eval_ites*slots_monitored) == 0) or ct == slots_training-1):
            ite = int((ct+1)/slots_monitored) - 1
            if ct == slots_training-1: 
                eval_loops = 200
            for scheme_id in range(N_schemes):
                sys.stdout.write("\n----------------------------------------\n")
                sys.stdout.write("Evaluation starts for scheme %d \n" % scheme_setting_list[scheme_id].scheme_id)
                sys.stdout.write("----------------------------------------\n")
                current_state_copy = copy.deepcopy(state_list[scheme_id])                
                Final_queue, Ave_npkts_dep_per_slot, Queue, Delay_dist,\
                Ave_num_using_relay, Ave_num_using_relay_detailed,\
                Ave_num_doing_tracking, Ave_num_doing_tracking_detailed,\
                Ave_num_bw_selection_detailed, Ave_ratio_under_blockage, Ave_ratio_under_blockage_detailed = \
                evaluation_ckpt(DRL_Decision_Maker, envCommon.env_parameter, slots_monitored, eval_loops,\
                                bandit_relay_para_list, bandit_bw_para_list,\
                                current_state_copy, scheme_setting_list, scheme_id)
                sys.stdout.write("Iterations completed:{:4d}; Final queue: {:.2f}; Npkts: {:.6f} \n".format(ite+1,Final_queue,Ave_npkts_dep_per_slot))
                #sys.stdout.write("Current learning rate:{:.5f}; Current clipping parameter: {:.3f} \n".format(scheduler.get_last_lr()[0],clip_param))
                sys.stdout.write("Within {:3d} slots; Average uses of relay:{:.2f}; Average uses of tracking: {:.3f} \n".format(slots_monitored, Ave_num_using_relay,Ave_num_doing_tracking))
                print('Detailed relay usages')
                print(Ave_num_using_relay_detailed)
                print('Detailed info on main link bw selection')
                print(Ave_num_bw_selection_detailed)
                print('Detailed ue tracking')
                print(Ave_num_doing_tracking_detailed)
                print('Detailed info on main link under blockage')
                print(Ave_ratio_under_blockage_detailed)
                ave_delay_in_slots_dist = np.sum(np.mean(Delay_dist,axis=0),axis=1)/\
                np.sum(np.mean(Delay_dist,axis=0))
                ave_delay_in_slots = \
                np.dot(np.transpose(ave_delay_in_slots_dist),np.asarray(range(len(ave_delay_in_slots_dist))))
                print('Average delay of packets in slots')
                print(ave_delay_in_slots)
                sys.stdout.write("----------------------------------------\n")
                sys.stdout.write("Evaluation ends for scheme %d \n" % scheme_setting_list[scheme_id].scheme_id)
                sys.stdout.write("----------------------------------------\n")
                # Save the parameter evolution
                evolution_rate_ckpt_MAB[scheme_id].append(Ave_npkts_dep_per_slot*mean_packet_size/t_slot/1e9)
                evolution_delay_ckpt_MAB[scheme_id].append(ave_delay_in_slots)
                evolution_ratio_blockage_ckpt_MAB[scheme_id].append(Ave_ratio_under_blockage)  
                if ct == slots_training-1:
                    Queue_Eval_MAB[:,:,:,scheme_id] = Queue      
                    Delay_dist_Eval_MAB[:,:,:,scheme_id] = Delay_dist
                    Ave_num_using_relay_detailed_MAB[:,:,scheme_id] = Ave_num_using_relay_detailed
                    Ave_num_bw_selection_detailed_MAB[:,:,scheme_id] = Ave_num_bw_selection_detailed
                    Ave_num_doing_tracking_detailed_MAB[:,scheme_id] = Ave_num_doing_tracking_detailed
                    Ave_ratio_under_blockage_detailed_MAB[:,scheme_id] = Ave_ratio_under_blockage_detailed
            time_per_ckpt = time.time() - tStart
            sys.stdout.write("\n----------------------------------------\n")
            sys.stdout.write("This CKPT takes %f seconds \n" % time_per_ckpt)
            sys.stdout.write("----------------------------------------\n")
            tStart = time.time()
            
    # Plot network topology    
    show_plot = False
    show_mobility_trace = False;
    plot_network_topology(env_list[0].env_parameter, output_folder, show_mobility_trace, show_plot) # plot network topology
    show_mobility_trace = True;
    plot_network_topology(env_list[0].env_parameter, output_folder, show_mobility_trace, show_plot) # plot network topology
            
    # Save all variable
    args_MAB = args
    training_results_filename = output_folder+'training_results_MAB_netwTopo'+str(Netw_topo_id)+'.pt'
    training_results_dict = {
        'args_MAB': args_MAB,
        'scheme_setting_list': scheme_setting_list,
        'env_parameter': env_parameter,
        'evolution_queue_length_MAB': evolution_queue_length_MAB,
        'evolution_reward_MAB': evolution_reward_MAB,
        'evolution_rate_ckpt_MAB': evolution_rate_ckpt_MAB,
        'evolution_delay_ckpt_MAB': evolution_delay_ckpt_MAB,
        'evolution_ratio_blockage_ckpt_MAB': evolution_ratio_blockage_ckpt_MAB,
        'Queue_Eval_MAB': Queue_Eval_MAB,
        'Delay_dist_Eval_MAB': Delay_dist_Eval_MAB,
        'Ave_num_using_relay_detailed_MAB': Ave_num_using_relay_detailed_MAB,
        'Ave_num_bw_selection_detailed_MAB': Ave_num_bw_selection_detailed_MAB,
        'Ave_num_doing_tracking_detailed_MAB': Ave_num_doing_tracking_detailed_MAB,
        'Ave_ratio_under_blockage_detailed_MAB': Ave_ratio_under_blockage_detailed_MAB,
        'bandit_relay_para_list': bandit_relay_para_list,
        'bandit_bw_para_list': bandit_bw_para_list
    }
    outfile = open(training_results_filename,'wb')
    pickle.dump(training_results_dict, outfile)
    outfile.close()

    
if __name__ == "__main__":
    main()


