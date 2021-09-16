# This Python file uses the following encoding: utf-8
# coding=utf-8
import os
import argparse
import pickle
import numpy as np
import sys
import operator 
import math
import pdb
import time
import matplotlib.pyplot as plt
from get_MAB_scheme_setting import *

#-------------------------------------------------------------------------
# Plot training results and testing result
#-------------------------------------------------------------------------
# Arg parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--Netw_topo_id', default=1, type=int, help='Id of network topology')
parser.add_argument('--output', default=None, help='output folder of training results')
args = parser.parse_args()
if args.output is not None:
    output_folder = args.output+'/'
else:
    output_folder = 'output_net'+str(args.Netw_topo_id)+'/'      
if not os.path.exists(output_folder):
    sys.exit('No such directory %s' %output_folder);
    output_folder = 'output_net1/'
Netw_topo_id = args.Netw_topo_id

# Plot settings
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'axes.labelsize': 18})
plt.rcParams.update({'legend.fontsize': 12})
plt.rcParams.update({'legend.loc': 'best'})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams['lines.linewidth'] = 2.5


#-------------------------------------------------------------------------
# Load saved variables and print parameters
#-------------------------------------------------------------------------
# DRL results
training_results_DRL_filename = output_folder+'/training_results_DRL_netwTopo'+str(Netw_topo_id)+'.pt'
infile_DRL = open(training_results_DRL_filename,'rb')
training_results_DRL_dict = pickle.load(infile_DRL)
infile_DRL.close()
locals().update(training_results_DRL_dict)
print(args_DRL)

# MAB results
training_results_MAB_filename = output_folder+'/training_results_MAB_netwTopo'+str(Netw_topo_id)+'.pt'
infile_MAB = open(training_results_MAB_filename,'rb')
training_results_MAB_dict = pickle.load(infile_MAB)
infile_MAB.close()
locals().update(training_results_MAB_dict)
print(args_MAB)

if args_DRL.slots != args_MAB.slots:
    print('\n !!!WARNING!!!: MAB and DRL have different number of slots in an iteration\n')
if args_DRL.eval_ites != args_MAB.eval_ites:
    print('\n !!!WARNING!!!: MAB and DRL have different frequencies of CKPTs\n')
    
#----------------------------------------------------------------
# Results from training phase
#----------------------------------------------------------------
# Complementary parameters
scheme_setting_list = get_MAB_scheme_setting()
N_schemes = len(scheme_setting_list)    
eval_ites_MAB = args_MAB.eval_ites
eval_ites_DRL = args_DRL.eval_ites

# Queue length evolution of a single shot training realization
figID = 2
plt.figure(num=figID,figsize=(10,6),dpi=1200)
#plt.title('Evolution of queue length during the training process')
evolution_queue_length_MAB.append(evolution_queue_length)
evolution_queue_length = evolution_queue_length_MAB
slots_to_show = 1500 * 240
for scheme_id in range(N_schemes):
    plt.plot(range(len(evolution_queue_length[scheme_id][0:slots_to_show])),evolution_queue_length[scheme_id][0:slots_to_show],\
             label=scheme_setting_list[scheme_id].legend,c=scheme_setting_list[scheme_id].color)
plt.rcParams.update({'legend.loc': 'best'})
plt.legend()
plt.xlabel('Time slot index')
plt.ylabel('Average queue length')
ymax = 10*max(evolution_queue_length[6])
plt.ylim(0, ymax)
plt.savefig(output_folder+'Training_phase_evolution_queue_length_zoom.eps',\
            format='eps', facecolor='w', transparent=False)

figID += 1
plt.figure(num=figID,figsize=(10,6),dpi=1200)
#plt.title('Evolution of queue length during the training process')
evolution_queue_length_MAB.append(evolution_queue_length)
evolution_queue_length = evolution_queue_length_MAB
for scheme_id in range(N_schemes):
    plt.plot(range(len(evolution_queue_length[scheme_id])),evolution_queue_length[scheme_id],\
             label=scheme_setting_list[scheme_id].legend,c=scheme_setting_list[scheme_id].color)
plt.rcParams.update({'legend.loc': 'best'})
plt.legend()
plt.xlabel('Time slot index')
plt.ylabel('Average queue length')
ymax = 10*max(evolution_queue_length[7])
plt.ylim(0, ymax)
plt.savefig(output_folder+'Training_phase_evolution_queue_length.eps',\
            format='eps', facecolor='w', transparent=False)

# Reward evolution of a single shot training realization
# figID += 1
# plt.figure(num=figID,figsize=(10,6),dpi=1200)
# #plt.title('Evolution of reward during training process')
# evolution_reward_MAB.append(evolution_reward)
# evolution_reward = evolution_reward_MAB
# for scheme_id in range(N_schemes):
#     plt.plot(range(len(evolution_reward[scheme_id])),evolution_reward[scheme_id],\
#              label=scheme_setting_list[scheme_id].legend,c=scheme_setting_list[scheme_id].color)
# plt.rcParams.update({'legend.loc': 'lower right'})
# plt.legend()
# plt.xlabel('Time slot index');
# plt.ylabel('Reward');
# plt.savefig(output_folder+'Training_phase_evolution_reward.eps',\
#             format='eps', facecolor='w', transparent=False)

# Average data rate at each check point
ite_duration = args_DRL.slots * env_parameter.t_slot
figID += 1
plt.figure(num=figID,figsize=(10,6),dpi=1200)
#plt.title('Evolution of data rate evaluated at each checkpoint')
evolution_rate_ckpt_MAB.append(evolution_rate_ckpt)
evolution_rate_ckpt = evolution_rate_ckpt_MAB
for scheme_id in range(N_schemes):
    if scheme_id == N_schemes - 1:
        eval_ites = eval_ites_DRL
    else:
        eval_ites = eval_ites_MAB
    plt.plot(range(1*eval_ites,len(evolution_rate_ckpt[scheme_id])*eval_ites+1,eval_ites),\
             evolution_rate_ckpt[scheme_id],\
             label=scheme_setting_list[scheme_id].legend,\
             c=scheme_setting_list[scheme_id].color)
plt.rcParams.update({'legend.loc': 'lower right'})
plt.legend()
plt.xlabel('Training iteration (one iteration is %d seconds)' %ite_duration)
plt.ylabel('Data rate (Gbits/s)');
plt.savefig(output_folder+'Training_phase_evolution_rate_ckpt.eps',format='eps', facecolor='w', transparent=False)

# Average delay at each check point
figID += 1
plt.figure(num=figID,figsize=(10,6),dpi=1200)
#plt.title('Evolution of average delay evaluated at each checkpoint')
evolution_delay_ckpt_MAB.append(evolution_delay_ckpt)
evolution_delay_ckpt = evolution_delay_ckpt_MAB
#for scheme_id in range(N_schemes):
for scheme_id in [4,6,7]:
    if scheme_id == N_schemes - 1:
        eval_ites = eval_ites_DRL
    else:
        eval_ites = eval_ites_MAB
    plt.plot(range(1*eval_ites,len(evolution_delay_ckpt[scheme_id])*eval_ites+1,eval_ites),\
             evolution_delay_ckpt[scheme_id],\
             label=scheme_setting_list[scheme_id].legend,\
             c=scheme_setting_list[scheme_id].color)
plt.rcParams.update({'legend.loc': 'upper right'})
plt.legend()
plt.xlabel('Training iteration (one iteration is %d seconds)' %ite_duration)
plt.ylabel('Delay in slots');
plt.savefig(output_folder+'Training_phase_evolution_delay_ckpt.eps',format='eps', facecolor='w', transparent=False)

# Average blockage probability at each check point
figID += 1
plt.figure(num=figID,figsize=(10,6),dpi=1200)
#plt.title('Evolution of percentage of time under blockage evaluated at each checkpoint')
evolution_ratio_blockage_ckpt_MAB.append(evolution_ratio_blockage_ckpt)
evolution_ratio_blockage_ckpt = evolution_ratio_blockage_ckpt_MAB

#for scheme_id in range(N_schemes):
for scheme_id in [4,6,7]:
    if scheme_id == N_schemes - 1:
        eval_ites = eval_ites_DRL
    else:
        eval_ites = eval_ites_MAB
    plt.plot(range(1*eval_ites,len(evolution_ratio_blockage_ckpt[scheme_id])*eval_ites+1,eval_ites),\
             np.divide(evolution_ratio_blockage_ckpt[scheme_id],args_DRL.slots),\
             label=scheme_setting_list[scheme_id].legend,\
             c=scheme_setting_list[scheme_id].color)

plt.rcParams.update({'legend.loc': 'best'})
plt.legend()
plt.xlabel('Training iteration (one iteration is %d seconds)' %ite_duration)
plt.ylabel('Percentage of time under blockage')
plt.savefig(output_folder+'Training_phase_evolution_ratio_blockage_ckpt.eps',format='eps', facecolor='w', transparent=False)

#----------------------------------------------------------------------------
# Testing results of final evaluation, namely running limited number of slots
#----------------------------------------------------------------------------
# CDF of delay
ave_Delay_dist_Eval = np.sum(np.mean(Delay_dist_Eval,axis=0),axis=1);
ave_Delay_dist = ave_Delay_dist_Eval/np.sum(ave_Delay_dist_Eval);
ave_Delay_dist = np.expand_dims(ave_Delay_dist, axis=1);
ave_Delay_dist_Eval_MAB = np.sum(np.mean(Delay_dist_Eval_MAB,axis=0),axis=1);
ave_Delay_dist_MAB = ave_Delay_dist_Eval_MAB/np.sum(ave_Delay_dist_Eval_MAB, axis=0);
ave_Delay_dist = np.append(ave_Delay_dist_MAB, ave_Delay_dist, axis = 1);
ave_Delay_CDF = np.cumsum(ave_Delay_dist,axis=0);
figID += 1
plt.figure(num=figID,figsize=(10,6),dpi=1200)
#plt.title('CDF of delay (slots)')
slots = args_DRL.slots
max_delay_to_show = slots+1
max_delay_to_show = 100
max_delay_to_show = min(max_delay_to_show,slots+1)
for scheme_id in [6,7]:
    #if N_schemes - scheme_id <= 3: 
    plt.plot(range(max_delay_to_show),ave_Delay_CDF[0:max_delay_to_show,scheme_id],\
             label=scheme_setting_list[scheme_id].legend,\
             c=scheme_setting_list[scheme_id].color)
plt.rcParams.update({'legend.loc': 'lower right'})
plt.legend()
plt.xlabel('Average delay (in slots)')
plt.ylabel('Prob (delay <= t)')
plt.savefig(output_folder+'Testing_phase_CDF_delay.eps',format='eps', facecolor='w', transparent=False)

# Evolution of queue length distribution
mean_queue_length = np.mean(np.mean(Queue_Eval,axis=2),axis=0);
mean_queue_length_MAB = np.mean(np.mean(Queue_Eval_MAB,axis=2),axis=0);
mean_queue_length = np.expand_dims(mean_queue_length, axis=1);
mean_queue_length = np.append(mean_queue_length_MAB, mean_queue_length, axis = 1);
figID += 1
plt.figure(num=figID,figsize=(10,6),dpi=1200)
#plt.title('Averaged evolution of queue length')
for scheme_id in [6,7]:
    if N_schemes - scheme_id <= 3: 
        plt.plot(range(slots),mean_queue_length[:,scheme_id],\
                 label=scheme_setting_list[scheme_id].legend,\
                 c=scheme_setting_list[scheme_id].color)
plt.rcParams.update({'legend.loc': 'lower right'})
plt.legend()
plt.xlabel('Time slot index')
plt.ylabel('Average queue length')
plt.savefig(output_folder+'Testing_phase_evolution_queue_length.eps',format='eps', facecolor='w', transparent=False)

# CDF of queue length
# figID += 1
# plt.figure(num=figID,figsize=(8,6),dpi=1200)
# #plt.title('Distribution of queue length')
# n_linspace = 100
# mean_queue_length = np.mean(Queue_Eval,axis=2)
# mean_queue_length = np.mean(mean_queue_length,axis=0)
# [P, b] = cdf_dist_P_vs_b(mean_queue_length,n_linspace)
# plt.plot(b,P,label=scheme_setting_list[scheme_id].legend,c=scheme_setting_list[scheme_id].color)
# plt.xlabel('Average queue length q')
# plt.ylabel('Prob (queue length > q)')
# plt.rcParams.update({'legend.loc': 'lower right'})
# plt.legend()
# plt.savefig(output_folder+'Testing_phase_CDF_queue_length.eps',format='eps', facecolor='w', transparent=False)

#sys.exit()
# Bar plot of relay use per UE (3D)
Ave_num_using_relay_detailed = np.expand_dims(Ave_num_using_relay_detailed, axis=2);
Ave_num_using_relay_detailed = np.append(Ave_num_using_relay_detailed_MAB, Ave_num_using_relay_detailed, axis = 2);
figID += 1
plt.figure(num=figID,figsize=(10,6),dpi=1200)
#plt.title('Average number of use of relay')
labels = ['UE 1', 'UE 2', 'UE 3', 'UE 4', 'UE 5']
x = np.arange(len(labels))  # the label locations
width = 0.1  # the width of the bars
ue_color = ['b','g','c','m','k']
xpos = np.array([-2,-1,0,1,2])
for scheme_id in [7]:
    for ue_id in range(5):
        plt.bar(x + xpos[ue_id]*width, Ave_num_using_relay_detailed[ue_id,:,scheme_id], width, \
                color = ue_color[ue_id], label='Relay '+ str(ue_id+1))
plt.rcParams.update({'legend.loc': 'upper right'})
plt.legend()
plt.xticks(x, labels)
plt.xlabel('UE index')
plt.ylabel('Average number of use')
plt.savefig(output_folder+'Analysis_relay_usages.eps',format='eps', facecolor='w', transparent=False)

# Bar plot of bw selection per UE (3D)
Ave_num_bw_selection_detailed = np.expand_dims(Ave_num_bw_selection_detailed, axis=2);
Ave_num_bw_selection_detailed = np.append(Ave_num_bw_selection_detailed_MAB, Ave_num_bw_selection_detailed, axis = 2);
figID += 1
plt.figure(num=figID,figsize=(10,6),dpi=1200)
#plt.title('Average number of use of different codebooks')
cb_color = ['b','g','c','m','k','r']
xpos = np.array([-2,-1,0,1,2,3])
for scheme_id in [6]:
    for cb_id in range(6):
        plt.bar(x + xpos[cb_id]*width, Ave_num_bw_selection_detailed[:,cb_id,scheme_id], width, \
                color = cb_color[cb_id], label='Codebook '+ str(cb_id+1))
plt.rcParams.update({'legend.loc': 'upper right'})
plt.legend()
plt.xticks(x, labels)
plt.xlabel('UE index')
plt.ylabel('Average number of use')
plt.savefig(output_folder+'Analysis_codebook_usages.eps',format='eps', facecolor='w', transparent=False)

# Bar plot of tracking use per UE (2D)
#breakpoint()
Ave_num_doing_tracking_detailed = np.expand_dims(Ave_num_doing_tracking_detailed, axis=1);
Ave_num_doing_tracking_detailed = np.append(Ave_num_doing_tracking_detailed_MAB, Ave_num_doing_tracking_detailed, axis = 1);
# figID += 1
# plt.figure(num=figID,figsize=(10,6),dpi=1200)
# #plt.title('Average number of use of different codebooks')
# cb_color = ['b','g','c','m','k','r']
# xpos = np.array([-2,-1,0,1,2,3])
# for scheme_id in [6]:
#     for cb_id in range(6):
#         plt.bar(x + xpos[cb_id]*width, Ave_num_bw_selection_detailed[:,cb_id,scheme_id], width, \
#                 color = cb_color[cb_id], label='Codebook '+ str(cb_id+1))
# plt.rcParams.update({'legend.loc': 'upper right'})
# plt.legend()
# plt.xticks(x, labels)
# plt.xlabel('UE index')
# plt.ylabel('Average number of use')
# plt.savefig(output_folder+'Analysis_codebook_usages.eps',format='eps', facecolor='w', transparent=False)

# # Bar plot of under blocakge (2D)
# Ave_ratio_under_blockage_detailed = np.expand_dims(Ave_ratio_under_blockage_detailed, axis=2);
# Ave_ratio_under_blockage_detailed = np.append(Ave_ratio_under_blockage_detailed_MAB, Ave_ratio_under_blockage_detailed, axis = 2);

