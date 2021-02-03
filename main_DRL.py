import argparse
import numpy as np
import torch
import torch.optim as optim
from envs import *
from model_single_pi import *
from training_single_pi import *
from my_utils import *
import shelve
import pdb
import cProfile
import platform

def main():
    print('\r\n------------------------------------')
    print('Enviroment Sumamry')
    print('------------------------------------')
    print('PyTorch ' + str(torch.__version__))
    print('Running with Python ' + str(platform.python_version()))    
    
    ############---------Parse command line arguments---------############
    print('\r\n------------------------------------')
    print('System Parameters')
    print('------------------------------------')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batches', default=20, type=int, help='number of slots in a single batch')
    parser.add_argument('--slots', default=2000, type=int, help='number of slots in a single episode')
    parser.add_argument('--iterations', default=500, type=int, help='number of episodes')
    parser.add_argument('--lr', default=0.001, type=float, help='actor/critic learning rate')
    parser.add_argument('--a_hid_dim', default=128, type=int, help='number of layers for actor')
    parser.add_argument('--c_hid_dim', default=128, type=int, help='number of units for critic')
    parser.add_argument('--clip_param', default=0.2, type=int, help='clip parameter')
    parser.add_argument('--training', default=True, type=int, help='enable training phase')
    parser.add_argument('--testing', default=False, type=int, help='enable testing phase')
    parser.add_argument('--cudaid', default=0, type=int, help='id of CUDA')
    parser.add_argument('--cuda', default=0, type=int, help='use to enable available CUDA')
    parser.add_argument('--gamma', default=0.999, type=float, help='gamma parameter for GAE')
    parser.add_argument('--lambd', default=1.00, type=float, help='lambda parameter for GAE')
    parser.add_argument('--value_coeff', default=0.5, type=float, help='value loss coeffecient')
    parser.add_argument('--entropy_coeff', default=0.05, type=float, help='entropy loss coeffecient')
    parser.add_argument('--eval_loops', default=5, type=int, help='number of evaluations for a checkpoint')
    parser.add_argument('--eval_ites', default=10, type=int, help='number of iterations before each ckpt evaluation')
    parser.add_argument('--save_path', default='trained_model_netwTopo', help='path to save the trained DRL agent')
    parser.add_argument('--load_path', default=os.getcwd()+'/trained_model_netwTopo', help='path to load the nn models')
    parser.add_argument('--all_var_path', default='all_vars_out', help='path to save all variable')
    parser.add_argument('--Netw_topo_id', default=1, type=int, help='Id of network topology')
    args = parser.parse_args()
    for item in vars(args):
        print(item + ': ' + str(getattr(args, item)))
    
    ############---------Training device---------############
    if args.cuda and torch.cuda.is_available(): 
        device_name = 'cuda:'+str(args.cudaid)
    else: 
        device_name = 'cpu'
    device = torch.device(device_name)

    ############---------Training---------############
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
        lr_decay_rate = 0.9
        lr_decay_steps = 20
        scheduler = optim.lr_scheduler.StepLR(optimizer, lr_decay_steps, lr_decay_rate)

        # Training with PPO algorithm
        batches = args.batches; 
        iterations = args.iterations
        gamma = args.gamma; 
        lambd = args.lambd; 
        clip_param = args.clip_param
        value_coeff = args.value_coeff; 
        entropy_coeff = args.entropy_coeff
        eval_loops = args.eval_loops; 
        eval_ites = args.eval_ites
        decaying_clip_param = False
        clipping_critic = True
        save_path = os.getcwd()+'/'+args.save_path+str(Netw_topo_id)+'.pt'
        
        # Training
        print('\n------------------------------------------------------')
        print('Starting Training')
        print('-------------------------------------------------------')
        all_rewards, Queue_Eval, Delay_dist_Eval, actor_critic_net = \
        training(env, actor_critic_net, optimizer, scheduler, batches, slots, iterations,\
                 gamma, lambd, value_coeff, entropy_coeff,\
                 clip_param, decaying_clip_param, clipping_critic,\
                 eval_loops, eval_ites, save_path, device)
        
        # Save training result
        torch.save({
            'model': actor_critic_net,
            'model_state_dict': actor_critic_net.state_dict(),
            }, save_path) 
        
        # Plot last evaluation result
        plot_last_evaluation_result(all_rewards,Queue_Eval,Delay_dist_Eval,slots)
        
        # Save all variable
        filename = os.getcwd() + '/' + args.all_var_path
        # To be done
    
    
    # Doing a complete evaluation with benchmarking algorithms
    if args.testing:
        print('-------------------------------------------------------')
        print('-------------------Starting Testing--------------------')
        print('-------------------------------------------------------')
        # Load saved variable and print parameters
        
        # To be done
        
        # Load saved nn models for inference
        checkpoint = torch.load(args.load_path)
        DRL_Decision_Maker = checkpoint['model']
        DRL_Decision_Maker.load_state_dict(checkpoint['model_state_dict'])
        DRL_Decision_Maker.eval()
        pdb.set_trace()
        # Plot trained results
        plot_last_evaluation_result(all_rewards,Queue_Eval,slots)
        pass
    
if __name__ == "__main__":
    main()