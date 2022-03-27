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
args = parser.parse_args()
if args.output is not None:
    output_folder = args.output+'/'
else:
    output_folder = 'study_batch/'      
if not os.path.exists(output_folder):
    sys.exit('No such directory %s' %output_folder);
Netw_topo_id = args.Netw_topo_id

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
target_size_vec = [1,5,10,20,50];
len_target_size_vec = len(target_size_vec)
for i in target_size_vec:
    ave_evolution_delay_ckpt_DRL = 0;
    nDRL = 0;
    if i == 5:
        num_realizations = 60;
    else:
        num_realizations = 5;
    for j in range(num_realizations):
        try:
            if i != 5:
                training_results_DRL_filename = output_folder+'batch_size_'+str(i)+'/realization_'+str(j)+'/training_results_DRL_netwTopo'+str(Netw_topo_id)+'.pt'
            else:
                training_results_DRL_filename = 'train_process_net3/realization_'+str(j)+'/training_results_DRL_netwTopo'+str(Netw_topo_id)+'.pt'
            infile_DRL = open(training_results_DRL_filename,'rb')
            training_results_DRL_dict = pickle.load(infile_DRL)
            infile_DRL.close()
            locals().update(training_results_DRL_dict)                    
            ave_evolution_delay_ckpt_DRL = (ave_evolution_delay_ckpt_DRL*nDRL + np.array(evolution_delay_ckpt).reshape((1,12)))/(nDRL+1);
            nDRL = nDRL + 1;   
            print('PASS: Read file of realization %d for batch size %d \\' %(j,i));
        except:
            print('ERROR: No realization %d for batch size %d or Error \\' %(j,i));
    if i == 1:
        ave_evolution_delay_ckpt = ave_evolution_delay_ckpt_DRL;
    else:
        ave_evolution_delay_ckpt = np.append(ave_evolution_delay_ckpt,ave_evolution_delay_ckpt_DRL,axis=0)

#----------------------------------------------------------------
# Results from training phase
#----------------------------------------------------------------
# Complementary parameters
scheme_setting_list = get_MAB_scheme_setting()
N_schemes = len(scheme_setting_list)
eval_ites_DRL = args_DRL.eval_ites
costomized_check_point = np.array([1,4,7,10,20,40,60,80,100,150,200,240])
figID = 100


#############################
ite_duration = args_DRL.slots * env_parameter.t_slot
figID += 1
plt.figure(num=figID,figsize=(10,6),dpi=1200)
for i in range(len(target_size_vec)):
    plt.plot(costomized_check_point,\
             (ave_evolution_delay_ckpt[i]),'*-',\
             label='Number of time steps in one batch: '+str(target_size_vec[i]))
plt.legend(loc='upper left',bbox_to_anchor=(0.4,0.78))
plt.xlabel('Training iteration (one iteration is %d seconds)' %ite_duration)
plt.ylabel('Delay in slots');
plt.savefig(output_folder+'Batch_Training_phase_evolution_delay_ckpt.pdf',format='pdf', facecolor='w', transparent=True)
figID += 1
plt.figure(num=figID,figsize=(10,6),dpi=1200)
for i in range(5):
    plt.plot(costomized_check_point,\
             ave_evolution_delay_ckpt[i],'*-',\
             label='Number of time steps in one batch: '+str(target_size_vec[i]))
plt.yscale('log')
plt.rcParams.update({'legend.loc': 'upper right'})
plt.legend()
plt.xlabel('Training iteration (one iteration is %d seconds)' %ite_duration)
plt.ylabel('Delay in slots');
plt.yticks([2, 3, 4, 5, 7, 10, 20, 40], ['2','3','4','5','7','10','20', '40'])
plt.savefig(output_folder+'Batch_Training_phase_evolution_delay_ckpt_log.pdf',format='pdf', facecolor='w', transparent=True)