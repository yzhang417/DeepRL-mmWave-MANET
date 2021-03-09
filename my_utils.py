# This Python file uses the following encoding: utf-8
# coding=utf-8
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
def plot_network_topology(env_parameter, output_folder, show_mobility_trace, show_plot):
    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'axes.labelsize': 16})
    plt.rcParams.update({'legend.fontsize': 12})
    plt.rcParams.update({'legend.fontsize': 12})
    plt.rcParams.update({'legend.loc': 'lower right'})
    plt.rcParams.update({'figure.autolayout': True})
    plt.rcParams['lines.linewidth'] = 1.5
    ue_color = ['b','g','c','m','k','r']
    number_dots_border = range(360);
    Xcoor_border = env_parameter.max_activity_range * np.cos(np.radians(number_dots_border))
    Ycoor_border = env_parameter.max_activity_range * np.sin(np.radians(number_dots_border))            
    fig = plt.figure(figsize=(10,5),dpi=1200);   
    ax_netw_topo = fig.add_subplot(111)
    ax_netw_topo.axis('equal')
    ax_netw_topo.grid(b=True, which='major', color='#666666', linestyle='-')
    for u in range(env_parameter.N_UE):
        ax_netw_topo.plot(env_parameter.Xcoor_init[u],env_parameter.Ycoor_init[u],\
                          '*', label='UE '+str(u),c=ue_color[u])
        ax_netw_topo.plot(env_parameter.Xcoor_init[u] + Xcoor_border,\
                          env_parameter.Ycoor_init[u] + Ycoor_border,\
                          '-', label='Border of UE '+str(u),c=ue_color[u])
    #ax_netw_topo.plot(0,0,'s',label='AP',c='r');  
    ax_netw_topo.plot(env_parameter.Xcoor_init[-1],env_parameter.Ycoor_init[-1],\
                      's',label='AP',c=ue_color[-1])
    ax_netw_topo.plot(env_parameter.Xcoor_init[-1] + Xcoor_border,\
                      env_parameter.Ycoor_init[-1] + Ycoor_border,\
                      '-', label='Border of AP',c=ue_color[-1])
    ax_netw_topo.set_xlabel('X-axis (meters)')
    ax_netw_topo.set_ylabel('Y-axis (meters)')
    ax_netw_topo.set_title('Network topology')
    ax_netw_topo.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    if show_mobility_trace:
        for u in range(env_parameter.N_UE+1):
            ax_netw_topo.plot(env_parameter.Xcoor_list[u],env_parameter.Ycoor_list[u],'-',\
                              c=ue_color[u],linewidth=0.3, markersize=0.15)
        plt.savefig(output_folder+'Network_topology_trace_'+str(env_parameter.Netw_topo_id)+'.eps',\
                    format='eps', facecolor='w', transparent=False, dpi=1200)
    else:
        plt.savefig(output_folder+'Network_topology_'+str(env_parameter.Netw_topo_id)+'.eps',\
                    format='eps', facecolor='w', transparent=False, dpi=1200)
    if show_plot:
        plt.show()
    else:
        plt.close()
        

#-------------------------------------------------------------------------
# Plot training results and last evaluation result
#-------------------------------------------------------------------------
def plot_training_testing_result(evolution_queue_length, evolution_reward, evolution_rate_ckpt, evolution_delay_ckpt,\
                                 evolution_ratio_blockage_ckpt, Queue_Eval, Delay_dist_Eval, \
                                 eval_ites, slots, Netw_topo_id, output_folder, show_plot):
    # Plot settings
    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'axes.labelsize': 16})
    plt.rcParams.update({'legend.fontsize': 12})
    plt.rcParams.update({'legend.fontsize': 12})
    plt.rcParams.update({'legend.loc': 'lower right'})
    plt.rcParams.update({'figure.autolayout': True})
    plt.rcParams['lines.linewidth'] = 2.5
    
    #----------------------------------------------------------------
    # Results from training phase
    #----------------------------------------------------------------
    # Queue length evolution of a single shot training realization
    figID = 2
    plt.figure(num=figID,figsize=(8,6),dpi=1200)
    plt.title('Evolution of queue length during a single realization of training process');
    plt.plot(range(len(evolution_queue_length)),evolution_queue_length);
    plt.xlabel('Time slot index');
    plt.ylabel('Average queue length');
    plt.savefig(output_folder+'Training_phase_evolution_queue_length.eps',\
                format='eps', facecolor='w', transparent=False)
    
    # Reward evolution of a single shot training realization
    figID += 1
    plt.figure(num=figID,figsize=(8,6),dpi=1200)
    plt.title('Evolution of reward during a single realization of training process');
    plt.plot(range(len(evolution_reward)),evolution_reward);
    plt.xlabel('Time slot index');
    plt.ylabel('Reward');
    plt.savefig(output_folder+'Training_phase_evolution_reward.eps',\
                format='eps', facecolor='w', transparent=False)
    
    # Average data rate at each check point
    figID += 1
    plt.figure(num=figID,figsize=(8,6),dpi=1200)
    plt.title('Evolution of data rate evaluated at each checkpoint');
    plt.plot(range(1*eval_ites,len(evolution_rate_ckpt)*eval_ites+1,eval_ites),evolution_rate_ckpt);
    plt.xlabel('Training iteration (one iteration is one minute)');
    plt.ylabel('Data rate (Gbits/s)');
    plt.savefig(output_folder+'Training_phase_evolution_rate_ckpt.eps',format='eps', facecolor='w', transparent=False)
    
    # Average delay at each check point
    figID += 1
    plt.figure(num=figID,figsize=(8,6),dpi=1200)
    plt.title('Evolution of average delay evaluated at each checkpoint');
    plt.plot(range(1*eval_ites,len(evolution_delay_ckpt)*eval_ites+1,eval_ites),evolution_delay_ckpt);
    plt.xlabel('Training iteration (one iteration is one minute)');
    plt.ylabel('Delay in slots');
    plt.savefig(output_folder+'Training_phase_evolution_delay_ckpt.eps',format='eps', facecolor='w', transparent=False)
    
    # Average blockage probability at each check point
    figID += 1
    plt.figure(num=figID,figsize=(8,6),dpi=1200)
    plt.title('Evolution of percentage of time slots under blockage evaluated at each checkpoint');
    plt.plot(range(1*eval_ites,len(evolution_ratio_blockage_ckpt)*eval_ites+1,eval_ites),evolution_ratio_blockage_ckpt);
    plt.xlabel('Training iteration (one iteration is one minute)');
    plt.ylabel('Percentage of time slots under blockage');
    plt.savefig(output_folder+'Training_phase_evolution_ratio_blockage_ckpt.eps',format='eps', facecolor='w', transparent=False)
    

    #----------------------------------------------------------------
    # Results from testing phase, namely running limited number of slots
    #----------------------------------------------------------------
    # CDF of queue length
    mean_queue_length = np.mean(Queue_Eval,axis=2)
    mean_queue_length = np.mean(mean_queue_length,axis=0)
    figID += 1
    plt.figure(num=figID,figsize=(8,6),dpi=1200)
    plt.title('Distribution of queue length')
    n_linspace = 100
    [P, b] = cdf_dist_P_vs_b(mean_queue_length,n_linspace)
    plt.plot(b,P)
    plt.xlabel('Average queue length q')
    plt.ylabel('Prob (queue length > q)')
    plt.savefig(output_folder+'Testing_phase_CDF_queue_length.eps',format='eps', facecolor='w', transparent=False)

    # CDF of delay
    ave_Delay_dist_Eval = np.sum(np.mean(Delay_dist_Eval,axis=0),axis=1)
    ave_Delay_dist = ave_Delay_dist_Eval/np.sum(ave_Delay_dist_Eval)
    ave_Delay_CDF = np.cumsum(ave_Delay_dist)
    figID += 1
    plt.figure(num=figID,figsize=(8,6),dpi=1200)
    plt.title('CDF of delay (slots)')
    max_delay_to_show = slots+1
    max_delay_to_show = 20
    max_delay_to_show = min(max_delay_to_show,slots+1)
    plt.plot(range(max_delay_to_show),ave_Delay_CDF[0:max_delay_to_show])
    plt.xlabel('Average delay (in slots)')
    plt.ylabel('Prob (delay <= t)')
    plt.savefig(output_folder+'Testing_phase_CDF_delay.eps',format='eps', facecolor='w', transparent=False)
    
    # Evolution of queue length
    figID += 1
    plt.figure(num=figID,figsize=(8,6),dpi=1200)
    plt.title('Averaged evolution of queue length')
    plt.plot(range(slots),mean_queue_length[0:slots])
    plt.xlabel('Time slot index')
    plt.ylabel('Average queue length')
    plt.savefig(output_folder+'Testing_phase_evolution_queue_length.eps',format='eps', facecolor='w', transparent=False)
    
    if show_plot:
        plt.show()