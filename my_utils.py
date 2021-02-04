import numpy as np
import sys
import operator 
import math
import pdb
import matplotlib.pyplot as plt

        
#-------------------------------------------------------------------------
# parameter for bw bandit 
#-------------------------------------------------------------------------   
class def_bandit_bw_parameter():
    def __init__(self, env_parameter):
        K_bw = env_parameter.K_bw;
        M = env_parameter.M;
        N_UE = env_parameter.N_UE;
        self.alpha_uwmts = np.ones((K_bw,M+1,N_UE));  
        self.UWMTS_CountLeader = np.zeros((K_bw,N_UE));
        self.UWMTS_Num_Codebook_Use = np.zeros((K_bw,N_UE));
        self.UWMTS_Mean_Codebook = np.zeros((K_bw,N_UE));
        self.UWMTS_Mean_Codebook[:] = math.inf;
        
        
#-------------------------------------------------------------------------
# parameter for relay bandit 
#-------------------------------------------------------------------------        
class def_bandit_relay_parameter():
    def __init__(self,env_parameter):
        K_relay = env_parameter.K_relay;
        M = env_parameter.M;
        N_UE = env_parameter.N_UE;        
        self.alpha_wmts = np.ones((K_relay,M+1,N_UE));

        
#-------------------------------------------------------------------------
# Transform cartesian coordinates to polar coordiantes 
#-------------------------------------------------------------------------
def pol2cart(angle,radius):
    x = radius*np.cos(angle)
    y = radius*np.sin(angle)
    return x,y


#-------------------------------------------------------------------------
# Calculate CDF of queue length
#-------------------------------------------------------------------------
def cdf_dist_P_vs_b(queue_length,n_linspace):
    max_v = np.amax(queue_length);
    b = np.linspace(0,max_v,n_linspace);
    P = np.zeros(n_linspace);
    for i in range(n_linspace):
        P[i] = len(np.where(queue_length>=b[i])[0])/len(queue_length);
    return P, b


#-------------------------------------------------------------------------
# Plot network topology
#-------------------------------------------------------------------------
def plot_network_topology(env_parameter):
    ue_color = ['b','g','c','m','k']
    number_dots_border = range(360);
    Xcoor_border = env_parameter.max_activity_range * np.cos(np.radians(number_dots_border))
    Ycoor_border = env_parameter.max_activity_range * np.sin(np.radians(number_dots_border))            
    fig = plt.figure(figsize=(8,4),dpi=100);   
    ax_netw_topo = fig.add_subplot(111)
    ax_netw_topo.axis('equal')
    ax_netw_topo.grid(b=True, which='major', color='#666666', linestyle='-')
    ax_netw_topo.plot(0,0,'s',label='BS',c='r');  
    for u in range(env_parameter.N_UE):
        ax_netw_topo.plot(env_parameter.Xcoor_init[u],env_parameter.Ycoor_init[u],'*',\
                          label='Mobile UE '+str(u),c=ue_color[u])
        ax_netw_topo.plot(env_parameter.Xcoor_init[u] + Xcoor_border,\
                          env_parameter.Ycoor_init[u] + Ycoor_border,'b-',\
                          label='Border of Mobile UE '+str(u),c=ue_color[u])
    ax_netw_topo.set_xlabel('X-axis (meters)')
    ax_netw_topo.set_ylabel('Y-axis (meters)')
    ax_netw_topo.set_title('Network topology')
    ax_netw_topo.legend()
    for u in range(env_parameter.N_UE):
        ax_netw_topo.plot(env_parameter.Xcoor_list[u],env_parameter.Ycoor_list[u],'-',\
                          c=ue_color[u],linewidth=0.1, markersize=0.1)
    plt.savefig('output/Network_topology'+str(env_parameter.Netw_topo_id)+'.png',format='png')
    plt.show()
    

#-------------------------------------------------------------------------
# Plot last evaluation result
#-------------------------------------------------------------------------
def plot_last_evaluation_result(all_rewards,Queue_Eval,Delay_dist_Eval,slots,Netw_topo_id):
    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'axes.labelsize': 18})
    plt.rcParams.update({'legend.fontsize': 12})
    plt.rcParams.update({'legend.fontsize': 12})
    plt.rcParams.update({'legend.loc': 'lower right'})
    
    plt.figure(num=2,figsize=(12,7),dpi=100)
    plt.title('Training Evolution');
    plt.plot(range(len(all_rewards)),all_rewards);
    plt.xlabel('Training Iteration');
    plt.ylabel('Rewards');
    plt.savefig('output_net'+str(Netw_topo_id)+'Training_evolution.png',format='png')

    mean_queue_length = np.mean(Queue_Eval,axis=2)
    mean_queue_length = np.mean(mean_queue_length,axis=0)
    plt.figure(num=3,figsize=(7,7),dpi=100)
    plt.title('Distribution of queue length')
    n_linspace = 100
    [P, b] = cdf_dist_P_vs_b(mean_queue_length,n_linspace)
    plt.plot(b,P)
    plt.xlabel('Averaged queue length q')
    plt.ylabel('Prob (queue length > q)')
    plt.savefig('output_net'+str(Netw_topo_id)+'CDF_queue_length_last_evaluation.png',format='png')

    plt.figure(num=4,figsize=(7,7),dpi=100)
    plt.title('Evolution of average queue length')
    plt.plot(range(slots),mean_queue_length[0:slots])
    plt.xlabel('Time slot index')
    plt.ylabel('Average queue length')
    plt.savefig('output_net'+str(Netw_topo_id)+'Evolution_queue_length_last_evaluation.png',format='png')
    
    ave_Delay_dist_Eval = np.mean(np.mean(Delay_dist_Eval,axis=2),axis=0)
    ave_Delay_dist = ave_Delay_dist_Eval/np.sum(ave_Delay_dist_Eval)
    ave_Delay_CDF = np.cumsum(ave_Delay_dist)
    plt.figure(num=5,figsize=(7,7),dpi=100)
    plt.title('CDF of delay (slots)')
    max_delay_to_show = slots+1
    max_delay_to_show = min(max_delay_to_show,slots+1)
    plt.plot(range(max_delay_to_show),ave_Delay_CDF[0:max_delay_to_show+1])
    plt.xlabel('Averaged delay in slots t')
    plt.ylabel('Prob (delay <= t)')
    plt.savefig('output_net'+str(Netw_topo_id)+'CDF_delay_last_evaluation.png',format='png')
    plt.show()