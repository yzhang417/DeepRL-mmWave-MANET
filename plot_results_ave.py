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
parser.add_argument('--Netw_topo_id', default=3, type=int, help='Id of network topology')
parser.add_argument('--output', default=None, help='output folder of training results')
parser.add_argument('--num_realization', default=60, type=int, help='number of training realizations to be averaged')
args = parser.parse_args()
if args.output is not None:
    output_folder = args.output+'/'
else:
    output_folder = 'train_process_net'+str(args.Netw_topo_id)+'/'      
if not os.path.exists(output_folder):
    sys.exit('No such directory %s' %output_folder);
Netw_topo_id = args.Netw_topo_id
num_realization = args.num_realization

# Plot settings
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'lines.markersize': 10})
plt.rcParams.update({'axes.labelsize': 18})
plt.rcParams.update({'legend.fontsize': 14})
plt.rcParams.update({'legend.loc': 'best'})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams['lines.linewidth'] = 2.5


#-------------------------------------------------------------------------
# Load saved variables and print parameters
#-------------------------------------------------------------------------
ave_evolution_rate_ckpt_DRL = 0;
ave_evolution_delay_ckpt_DRL = 0;
ave_evolution_ratio_blockage_ckpt_DRL = 0;
ave_evolution_rate_ckpt_MAB = 0;
ave_evolution_delay_ckpt_MAB = 0;
ave_evolution_ratio_blockage_ckpt_MAB = 0;
min_evolution_delay_ckpt_DRL = np.inf;
max_evolution_delay_ckpt_DRL = 0;
min_evolution_delay_ckpt_MAB = np.inf;
max_evolution_delay_ckpt_MAB = 0;

nMAB = 0;
nDRL = 0;

for i in range(num_realization):
    if i!=-1:
        # DRL results
        try:
            training_results_DRL_filename = output_folder+'realization_'+str(i)+'/training_results_DRL_netwTopo'+str(Netw_topo_id)+'.pt'
            infile_DRL = open(training_results_DRL_filename,'rb')
            training_results_DRL_dict = pickle.load(infile_DRL)
            infile_DRL.close()
            locals().update(training_results_DRL_dict)
            ave_evolution_rate_ckpt_DRL = (ave_evolution_rate_ckpt_DRL*nDRL + np.array(evolution_rate_ckpt))/(nDRL+1)
            ave_evolution_delay_ckpt_DRL = (ave_evolution_delay_ckpt_DRL*nDRL + np.array(evolution_delay_ckpt))/(nDRL+1)
            ave_evolution_ratio_blockage_ckpt_DRL = (ave_evolution_ratio_blockage_ckpt_DRL*nDRL + np.array(evolution_ratio_blockage_ckpt))/(nDRL+1)
            min_evolution_delay_ckpt_DRL = np.minimum(min_evolution_delay_ckpt_DRL,evolution_delay_ckpt)
            max_evolution_delay_ckpt_DRL = np.maximum(max_evolution_delay_ckpt_DRL,evolution_delay_ckpt)
            nDRL = nDRL + 1;            
        except:
            print('No file %d for DRL\\' %i);

        # MAB results
        try:
            training_results_MAB_filename = output_folder+'realization_'+str(i)+'/training_results_MAB_netwTopo'+str(Netw_topo_id)+'.pt'
            infile_MAB = open(training_results_MAB_filename,'rb')
            training_results_MAB_dict = pickle.load(infile_MAB)
            infile_MAB.close()
            locals().update(training_results_MAB_dict)
            ave_evolution_rate_ckpt_MAB = (ave_evolution_rate_ckpt_MAB*nMAB + np.array(evolution_rate_ckpt_MAB))/(nMAB+1)
            ave_evolution_delay_ckpt_MAB = (ave_evolution_delay_ckpt_MAB*nMAB + np.array(evolution_delay_ckpt_MAB))/(nMAB+1)
            ave_evolution_ratio_blockage_ckpt_MAB = (ave_evolution_ratio_blockage_ckpt_MAB*nMAB + np.array(evolution_ratio_blockage_ckpt_MAB))/(nMAB+1)
            min_evolution_delay_ckpt_MAB = np.minimum(min_evolution_delay_ckpt_MAB,evolution_delay_ckpt_MAB)
            max_evolution_delay_ckpt_MAB = np.maximum(max_evolution_delay_ckpt_MAB,evolution_delay_ckpt_MAB)
            nMAB = nMAB + 1;
        except:
            print('No file %d for MAB\\' %i)

        print('Read file %d\\' %i)
ave_evolution_rate_ckpt = np.append(ave_evolution_rate_ckpt_MAB,ave_evolution_rate_ckpt_DRL.reshape((1,12)),axis=0)
ave_evolution_delay_ckpt = np.append(ave_evolution_delay_ckpt_MAB,ave_evolution_delay_ckpt_DRL.reshape((1,12)),axis=0)
ave_evolution_ratio_blockage_ckpt = np.append(ave_evolution_ratio_blockage_ckpt_MAB,ave_evolution_ratio_blockage_ckpt_DRL.reshape((1,12)),axis=0)
min_evolution_delay_ckpt = np.append(min_evolution_delay_ckpt_MAB,min_evolution_delay_ckpt_DRL.reshape((1,12)),axis=0)
max_evolution_delay_ckpt = np.append(max_evolution_delay_ckpt_MAB,max_evolution_delay_ckpt_DRL.reshape((1,12)),axis=0)

#----------------------------------------------------------------
# Results from training phase
#----------------------------------------------------------------
# Complementary parameters
scheme_setting_list = get_MAB_scheme_setting()
N_schemes = len(scheme_setting_list)    
eval_ites_MAB = args_MAB.eval_ites
eval_ites_DRL = args_DRL.eval_ites
costomized_check_point = np.array([1,4,7,10,20,40,60,80,100,150,200,240])
figID = 100



############################# Average data rate at each check point
ite_duration = args_DRL.slots * env_parameter.t_slot
figID += 1
plt.figure(num=figID,figsize=(10,6),dpi=1200)
#plt.title('Evolution of data rate evaluated at each checkpoint')
for scheme_id in range(N_schemes):
    if scheme_id == N_schemes - 1:
        eval_ites = eval_ites_DRL
    else:
        eval_ites = eval_ites_MAB
    plt.plot(costomized_check_point,\
             ave_evolution_rate_ckpt[scheme_id],'*-',\
             label=scheme_setting_list[scheme_id].legend,\
             c=scheme_setting_list[scheme_id].color)
    
plt.rcParams.update({'legend.loc': 'lower right'})
plt.legend()
plt.xlabel('Training iteration (one iteration is %d seconds)' %ite_duration)
plt.ylabel('Data rate (Gbits/s)');
plt.savefig(output_folder+'Ave_Training_phase_evolution_rate_ckpt.eps',format='eps', facecolor='w', transparent=False)



############################# Average delay at each check point
figID += 1
plt.figure(num=figID,figsize=(10,6),dpi=1200)
#plt.title('Evolution of average delay evaluated at each checkpoint')
for scheme_id in range(N_schemes):
    if scheme_id == N_schemes - 1:
        eval_ites = eval_ites_DRL
        plt.fill_between(costomized_check_point, max_evolution_delay_ckpt[scheme_id], min_evolution_delay_ckpt[scheme_id], interpolate=True, alpha=0.3)
    else:
        eval_ites = eval_ites_MAB
        plt.fill_between(costomized_check_point, max_evolution_delay_ckpt[scheme_id], min_evolution_delay_ckpt[scheme_id], interpolate=True, alpha=0.3)
    plt.plot(costomized_check_point,\
             ave_evolution_delay_ckpt[scheme_id],'*-',\
             label=scheme_setting_list[scheme_id].legend,\
             c=scheme_setting_list[scheme_id].color)
    ckpt_with_text = np.array([1,10,40,60,100,240])
    
    i_x = costomized_check_point
    if scheme_id == 0:
        i_y = ave_evolution_delay_ckpt[0]
        plt.text(1+5,i_y[0]+1.5, '({}, {:.2f})'.format(costomized_check_point[0], i_y[0]), c=scheme_setting_list[0].color, size = 'x-small')
        plt.text(10-12,2.3, '({}, {:.2f})'.format(costomized_check_point[3], i_y[3]), c=scheme_setting_list[0].color, size = 'x-small')
        plt.text(40-15,2.3, '({}, {:.2f})'.format(costomized_check_point[5], i_y[5]), c=scheme_setting_list[0].color, size = 'x-small')
        plt.text(60-3,i_y[11]+1.25, '({}, {:.2f})'.format(costomized_check_point[6], i_y[6]), c=scheme_setting_list[0].color, size = 'x-small')
        plt.text(100-10,i_y[11]+1.25, '({}, {:.2f})'.format(costomized_check_point[8], i_y[8]), c=scheme_setting_list[0].color, size = 'x-small')
        plt.text(240-20,i_y[11]+1.25, '({}, {:.2f})'.format(costomized_check_point[11], i_y[11]), c=scheme_setting_list[0].color, size = 'x-small')
    else:
        i_y = ave_evolution_delay_ckpt[1]
        plt.text(1-3,i_y[0]+1.5, '({}, {:.2f})'.format(costomized_check_point[0], i_y[0]), c=scheme_setting_list[1].color, size = 'x-small')
        plt.text(10+5,i_y[3], '({}, {:.2f})'.format(costomized_check_point[3], i_y[3]), c=scheme_setting_list[1].color, size = 'x-small')
        plt.text(40+1,i_y[5]+2, '({}, {:.2f})'.format(costomized_check_point[5], i_y[5]), c=scheme_setting_list[1].color, size = 'x-small')
        plt.text(60-3,i_y[11]-1.85, '({}, {:.2f})'.format(costomized_check_point[6], i_y[6]), c=scheme_setting_list[1].color, size = 'x-small')
        plt.text(100-10,i_y[11]-1.85, '({}, {:.2f})'.format(costomized_check_point[8], i_y[8]), c=scheme_setting_list[1].color, size = 'x-small')
        plt.text(240-20,i_y[11]-1.85, '({}, {:.2f})'.format(costomized_check_point[11], i_y[11]), c=scheme_setting_list[1].color, size = 'x-small')   
        
plt.rcParams.update({'legend.loc': 'upper right'})
plt.legend()
plt.xlabel('Training iteration (one iteration is %d seconds)' %ite_duration)
plt.ylabel('Delay in slots');
plt.savefig(output_folder+'Ave_Training_phase_evolution_delay_ckpt.eps',format='eps', facecolor='w', transparent=True)
plt.savefig(output_folder+'Ave_Training_phase_evolution_delay_ckpt.pdf',format='pdf', facecolor='w', transparent=True)



############################# Average blockage probability at each check point
figID += 1
plt.figure(num=figID,figsize=(10,6),dpi=1200)
#plt.title('Evolution of percentage of time under blockage evaluated at each checkpoint')
for scheme_id in range(N_schemes):
    if scheme_id == N_schemes - 1:
        eval_ites = eval_ites_DRL
    else:
        eval_ites = eval_ites_MAB
    plt.plot(costomized_check_point,\
             np.divide(ave_evolution_ratio_blockage_ckpt[scheme_id],args_DRL.slots),'*-',\
             label=scheme_setting_list[scheme_id].legend,\
             c=scheme_setting_list[scheme_id].color)

plt.rcParams.update({'legend.loc': 'best'})
plt.legend()
plt.xlabel('Training iteration (one iteration is %d seconds)' %ite_duration)
plt.ylabel('Percentage of time under blockage')
plt.savefig(output_folder+'Ave_Training_phase_evolution_ratio_blockage_ckpt.eps',format='eps', facecolor='w', transparent=False)