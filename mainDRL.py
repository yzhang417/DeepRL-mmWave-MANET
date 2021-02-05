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
    #Parse command line arguments
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
    parser.add_argument('--Netw_topo_id', default=1, type=int, help='Id of network topology')
    parser.add_argument('--output', default=None, help='output folder of training results')
    
    # Training process
    parser.add_argument('--iterations', default=500, type=int, help='number of episodes')
    parser.add_argument('--slots', default=2000, type=int, help='number of slots in a single episode')
    parser.add_argument('--batches', default=20, type=int, help='number of slots in a single batch')
    parser.add_argument('--eval_loops', default=5, type=int, help='number of evaluations for a checkpoint')
    parser.add_argument('--eval_ites', default=10, type=int, help='number of iterations before each ckpt evaluation')
    # Learning part
    parser.add_argument('--lr', default=0.001, type=float, help='actor/critic learning rate')
    parser.add_argument('--lr_decay_rate', default=0.9, type=float, help='decaying rate of learning rate')
    parser.add_argument('--lr_decay_steps', default=20, type=int, help='number of updates before decaying learning rate')
    # NN architecture
    parser.add_argument('--a_hid_dim', default=128, type=int, help='number of layers for actor')
    parser.add_argument('--c_hid_dim', default=128, type=int, help='number of units for critic')
    # PPO parameter
    parser.add_argument('--clip_param', default=0.2, type=int, help='clip parameter')
    parser.add_argument('--gamma', default=0.999, type=float, help='gamma parameter for GAE')
    parser.add_argument('--lambd', default=1.00, type=float, help='lambda parameter for GAE')
    parser.add_argument('--value_coeff', default=0.5, type=float, help='value loss coeffecient')
    parser.add_argument('--entropy_coeff', default=0.05, type=float, help='entropy loss coeffecient')
    parser.add_argument('--decaying_clip_param', default=0, type=int, help='Whether to decay the clipping parameter')
    parser.add_argument('--clipping_critic', default=1, type=int, help='Whether to clip the critic')    

    #-----------------------------------------------------------
    #Print args
    #-----------------------------------------------------------
    args = parser.parse_args()
    for item in vars(args):
        print(item + ': ' + str(getattr(args, item)))
    
    #-----------------------------------------------------------
    #Training device selection
    #-----------------------------------------------------------
    if args.cuda and torch.cuda.is_available(): 
        device_name = 'cuda:'+str(args.cudaid)
    else: 
        device_name = 'cpu'
    device = torch.device(device_name)

    #-----------------------------------------------------------
    #Check output folder
    #-----------------------------------------------------------
    if args.output is not None:
        output_folder = args.output+'/'
    else:
        output_folder = 'output_net'+str(args.Netw_topo_id)+'/'        
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    #-----------------------------------------------------------
    #Training
    #-----------------------------------------------------------
    if args.training:
        # Create an instance of enviroment and print key system parameters
        slots = args.slots
        Netw_topo_id = args.Netw_topo_id
        env_parameter = env_init(Netw_topo_id)
        env = envs(env_parameter,slots)
        print('Training with ' + device_name)
        print('Time slot duration is %f seconds' %env_parameter.t_slot)
        print('Maximum activity range is %d meters' %env_parameter.max_activity_range)
        print('Probabbility of blockage is:', env_parameter.target_prob_blockage)

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
        
        # Training
        print('\n------------------------------------------------------')
        print('Starting Training')
        print('-------------------------------------------------------')
        all_rewards, Queue_Eval, Delay_dist_Eval, actor_critic_net = \
        training(env, actor_critic_net, optimizer, scheduler, batches, slots, iterations,\
                 gamma, lambd, value_coeff, entropy_coeff,\
                 clip_param, decaying_clip_param, clipping_critic,\
                 eval_loops, eval_ites, device)
        
        # Save the trained model result
        trained_model_filename = output_folder+'trained_model_netwTopo'+str(Netw_topo_id)+'.pt'
        torch.save({
            'model': actor_critic_net,
            'model_state_dict': actor_critic_net.state_dict(),
            }, trained_model_filename) 
        
        # Save all variable
        training_results_filename = output_folder+'training_results_netwTopo'+str(Netw_topo_id)+'.pt'
        training_results_dict = {
            'args': args,
            'env_parameter': env_parameter,
            'device_name': device_name,
            'all_rewards': all_rewards,
            'Queue_Eval': Queue_Eval,
            'Delay_dist_Eval': Delay_dist_Eval,
            'model': actor_critic_net,
            'model_state_dict': actor_critic_net.state_dict()
        }
        outfile = open(training_results_filename,'wb')
        pickle.dump(training_results_dict, outfile)
        outfile.close()
        
        # Plot last evaluation result
        plot_last_evaluation_result(all_rewards, Queue_Eval, Delay_dist_Eval, slots, Netw_topo_id, output_folder)

    
    #-----------------------------------------------------------
    # Testing with and comparing with benchmarking algorithms
    #-----------------------------------------------------------
    if args.testing:
        print('\n------------------------------------------------------')
        print('Starting Testing')
        print('-------------------------------------------------------')
        # Load saved variable and print parameters
        training_results_filename = output_folder+'/training_results_netwTopo'+str(Netw_topo_id)+'.pt'
        infile = open(training_results_filename,'rb')
        training_results_dict = pickle.load(infile)
        infile.close()
        print(training_results_dict['args'])
        
        # Load saved nn models for inference
        trained_model_filename = output_folder+'/trained_model_netwTopo'+str(args.Netw_topo_id)+'.pt'
        checkpoint = torch.load(trained_model_filename)
        DRL_Decision_Maker = checkpoint['model']
        DRL_Decision_Maker.load_state_dict(checkpoint['model_state_dict'])
        DRL_Decision_Maker.eval()
        
        # Plot trained results
        plot_last_evaluation_result(all_rewards, Queue_Eval, Delay_dist_Eval, slots, Netw_topo_id, output_folder)
        pass


if __name__ == "__main__":
    main()