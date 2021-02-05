import numpy as np
import sys
import operator 
import math
import pdb
import matplotlib.pyplot as plt


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
def plot_network_topology(env_parameter,output_folder):
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
    Netw_topo_id = env_parameter.Netw_topo_id
    plt.savefig(output_folder+'Network_topology'+str(env_parameter.Netw_topo_id)+'.png',format='png', facecolor='w', transparent=False)
    plt.show()
    

#-------------------------------------------------------------------------
# Plot last evaluation result
#-------------------------------------------------------------------------
def plot_last_evaluation_result(all_rewards, Queue_Eval, Delay_dist_Eval, slots, Netw_topo_id, output_folder):
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
    plt.savefig(output_folder+'Training_evolution.png',format='png', facecolor='w', transparent=False)

    mean_queue_length = np.mean(Queue_Eval,axis=2)
    mean_queue_length = np.mean(mean_queue_length,axis=0)
    plt.figure(num=3,figsize=(7,7),dpi=100)
    plt.title('Distribution of queue length')
    n_linspace = 100
    [P, b] = cdf_dist_P_vs_b(mean_queue_length,n_linspace)
    plt.plot(b,P)
    plt.xlabel('Averaged queue length q')
    plt.ylabel('Prob (queue length > q)')
    plt.savefig(output_folder+'CDF_queue_length_last_evaluation.png',format='png', facecolor='w', transparent=False)

    plt.figure(num=4,figsize=(7,7),dpi=100)
    plt.title('Evolution of average queue length')
    plt.plot(range(slots),mean_queue_length[0:slots])
    plt.xlabel('Time slot index')
    plt.ylabel('Average queue length')
    plt.savefig(output_folder+'Evolution_queue_length_last_evaluation.png',format='png', facecolor='w', transparent=False)
    
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
    plt.savefig(output_folder+'CDF_delay_last_evaluation.png',format='png', facecolor='w', transparent=False)
    plt.show()