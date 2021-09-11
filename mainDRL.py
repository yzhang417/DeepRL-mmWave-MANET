#!/bin/bash
# This Python file uses the following encoding: utf-8
# coding=utf-8
import argparse
import numpy as np
import torch
import torch.optim as optim
from envs import *
from model import *
from training import *
from my_utils import *
import pickle
import pdb
import cProfile
import platform

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
    parser.add_argument('--cudaid', default=0, type=int, help='id of CUDA')
    parser.add_argument('--Netw_topo_id', default=3, type=int, help='Id of network topology')
    parser.add_argument('--output', default=None, help='output folder of training results')
    # Training process
    parser.add_argument('--iterations', default=240, type=int, help='number of episodes')
    parser.add_argument('--slots', default=1500, type=int, help='number of slots in a single episode')
    parser.add_argument('--batches', default=5, type=int, help='number of slots in a single batch')
    parser.add_argument('--eval_loops', default=20, type=int, help='number of evaluations for a checkpoint')
    parser.add_argument('--eval_ites', default=1, type=int, help='number of iterations before each ckpt evaluation')
    parser.add_argument('--clip_queues', default=0, type=int, help='clip the queue at the end of each iteration')
    # Learning part
    parser.add_argument('--lr', default=0.001, type=float, help='actor/critic learning rate')
    parser.add_argument('--lr_decay_rate', default=0.9, type=float, help='decaying rate of learning rate')
    parser.add_argument('--lr_decay_steps', default=20, type=int, help='number of updates before decaying learning rate')
    # NN architecture
    parser.add_argument('--a_hid_dim', default=128, type=int, help='number of units for actor')
    parser.add_argument('--c_hid_dim', default=128, type=int, help='number of units for critic')
    # PPO parameter
    parser.add_argument('--clip_param', default=0.2, type=int, help='clip parameter')
    parser.add_argument('--gamma', default=0.999, type=float, help='gamma parameter for GAE')
    parser.add_argument('--lambd', default=1.00, type=float, help='lambda parameter for GAE')
    parser.add_argument('--value_coeff', default=0.5, type=float, help='value loss coeffecient')
    parser.add_argument('--entropy_coeff', default=0.05, type=float, help='entropy loss coefficient')
    parser.add_argument('--decaying_clip_param', default=0, type=int, help='Whether to decay the clipping parameter')
    parser.add_argument('--clipping_critic', default=1, type=int, help='Whether to clip the critic')    
    #Print args
    args = parser.parse_args()
    for item in vars(args):
        print(item + ': ' + str(getattr(args, item)))
    
    #-----------------------------------------------------------
    # Training device selection
    #-----------------------------------------------------------
    if args.cuda and torch.cuda.is_available(): 
        device_name = 'cuda:'+str(args.cudaid)
    else: 
        device_name = 'cpu'
    device = torch.device(device_name)

    #-----------------------------------------------------------
    # Check output folder
    #-----------------------------------------------------------
    if args.output is not None:
        output_folder = args.output+'/'
    else:
        output_folder = 'output_net'+str(args.Netw_topo_id)+'/'        
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    #-----------------------------------------------------------
    # Common argument
    #-----------------------------------------------------------
    Netw_topo_id = args.Netw_topo_id
    slots = args.slots
    
    #-----------------------------------------------------------
    # Random seed
    #-----------------------------------------------------------
    #random.seed(13579)     # random seeds for reproducation
    #np.random.seed(246810) # random seeds for reproducation
    random.seed()     # random seeds
    np.random.seed() # random seeds
    
    #-----------------------------------------------------------
    # Training
    #-----------------------------------------------------------
    if args.training:
        # Create an instance of enviroment and print key system parameters
        env_parameter = env_init(Netw_topo_id)
        env = envs(env_parameter,slots)
        print('Training with ' + device_name)
        print('Time slot duration is %f seconds' %env_parameter.t_slot)
        total_running_time = env_parameter.t_slot * args.iterations * args.slots / 3600
        print('Total running time is %f hours' %total_running_time)
        print('Maximum activity range is %d meters' %env_parameter.max_activity_range)
        print('Average time in blockage is \r\n', env_parameter.target_prob_blockage)
        print('Probabbility of blockage is \r\n', env_parameter.prob_blockage)
        print('Proportion of arrival rate is \r\n', env_parameter.lambda_ratio)
        print('Workload is %0.2f Gbps\r\n' %(env_parameter.workload/1e9))
        
        # Create learning framwork and optimizer, i.e. an actor_critic network
        a_hid_dim = args.a_hid_dim; 
        c_hid_dim = args.c_hid_dim; 
        actor_critic_net = model(a_hid_dim, c_hid_dim, env, device)
        actor_critic_net.to(device) # Move the model to desired device
        optimizer = optim.Adam(actor_critic_net.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_steps, args.lr_decay_rate)

        # Training with PPO algorithm
        iterations = args.iterations
        batches = args.batches; 
        gamma = args.gamma; 
        lambd = args.lambd; 
        clip_param = args.clip_param
        value_coeff = args.value_coeff; 
        entropy_coeff = args.entropy_coeff
        eval_loops = args.eval_loops; 
        eval_ites = args.eval_ites
        decaying_clip_param = args.decaying_clip_param
        clipping_critic = args.clipping_critic
        clip_queues = args.clip_queues
        
        # Training
        print('\n------------------------------------------------------')
        print('Starting Training')
        print('-------------------------------------------------------')                
        evolution_queue_length, evolution_reward, evolution_rate_ckpt, evolution_delay_ckpt, evolution_ratio_blockage_ckpt,\
        Queue_Eval, Delay_dist_Eval, Ave_num_using_relay_detailed, Ave_num_bw_selection_detailed,\
        Ave_num_doing_tracking_detailed, Ave_ratio_under_blockage_detailed, actor_critic_net = \
        training(env, actor_critic_net, optimizer, scheduler, batches, slots, iterations,\
                 gamma, lambd, value_coeff, entropy_coeff,\
                 clip_param, decaying_clip_param, clipping_critic,\
                 eval_loops, eval_ites, device, clip_queues)
        
        # Save the trained model result
        trained_model_filename = output_folder+'trained_model_DRL_netwTopo'+str(Netw_topo_id)+'.pt'
        torch.save({
            'model': actor_critic_net,
            'model_state_dict': actor_critic_net.state_dict(),
            }, trained_model_filename) 
        
        # Save all variable
        args_DRL = args
        training_results_filename = output_folder+'training_results_DRL_netwTopo'+str(Netw_topo_id)+'.pt'
        training_results_dict = {
            'args_DRL': args_DRL,
            'env_parameter': env_parameter,
            'evolution_queue_length': evolution_queue_length,
            'evolution_reward': evolution_reward,
            'evolution_rate_ckpt': evolution_rate_ckpt,
            'evolution_delay_ckpt': evolution_delay_ckpt,
            'evolution_ratio_blockage_ckpt': evolution_ratio_blockage_ckpt
            #'Queue_Eval': Queue_Eval, #Comment to save memory
            #'Delay_dist_Eval': Delay_dist_Eval,
            #'Ave_num_using_relay_detailed': Ave_num_using_relay_detailed,
            #'Ave_num_bw_selection_detailed': Ave_num_bw_selection_detailed,
            #'Ave_num_doing_tracking_detailed': Ave_num_doing_tracking_detailed,
            #'Ave_ratio_under_blockage_detailed': Ave_ratio_under_blockage_detailed,
            #'model': actor_critic_net,
            #'model_state_dict': actor_critic_net.state_dict()
        }
        outfile = open(training_results_filename,'wb')
        pickle.dump(training_results_dict, outfile)
        outfile.close()
        
        # Plot last evaluation result
#         show_plot = False
#         plot_training_testing_result(evolution_queue_length, evolution_reward, evolution_rate_ckpt, evolution_delay_ckpt,\
#                                      evolution_ratio_blockage_ckpt, Queue_Eval, Delay_dist_Eval, \
#                                      args.eval_ites, slots, Netw_topo_id, output_folder, show_plot)
    
    #-----------------------------------------------------------
    # Testing with and comparing with benchmarking algorithms
    #-----------------------------------------------------------
    if args.testing:
        print('\n------------------------------------------------------')
        print('Starting Testing')
        print('-------------------------------------------------------')
        # Load saved variable and print parameters
        training_results_filename = output_folder+'/training_results_DRL_netwTopo'+str(Netw_topo_id)+'.pt'
        infile = open(training_results_filename,'rb')
        training_results_dict = pickle.load(infile)
        infile.close()
        print(training_results_dict['args'])
        
        # Plot training/testing results
        evolution_queue_length = training_results_dict['evolution_queue_length']
        evolution_reward = training_results_dict['evolution_reward']
        evolution_rate_ckpt = training_results_dict['evolution_rate_ckpt']  
        evolution_delay_ckpt = training_results_dict['evolution_delay_ckpt']
        evolution_ratio_blockage_ckpt = training_results_dict['evolution_ratio_blockage_ckpt']
        Queue_Eval = training_results_dict['Queue_Eval']
        Delay_dist_Eval = training_results_dict['Delay_dist_Eval']
        show_plot = False
        plot_training_testing_result(evolution_queue_length, evolution_reward, evolution_rate_ckpt, evolution_delay_ckpt,\
                                     evolution_ratio_blockage_ckpt, Queue_Eval, Delay_dist_Eval, \
                                     args.eval_ites, slots, Netw_topo_id, output_folder, show_plot)

if __name__ == "__main__":
    main()