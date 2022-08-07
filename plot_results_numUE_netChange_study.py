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
parser.add_argument('--all_nets', default=1, type=int, help='all nets in one figures')
parser.add_argument('--customized_check_point', default=1, type=int, help='all nets in one figures')
parser.add_argument('--netw_changed', default=0, type=int, help='all nets in one figures')
args = parser.parse_args()

# Plot settings
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'lines.markersize': 10})
plt.rcParams.update({'axes.labelsize': 18})
plt.rcParams.update({'legend.fontsize': 14})
plt.rcParams.update({'legend.loc': 'best'})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams['lines.linewidth'] = 2.5

if args.all_nets == 1:
    Netw_topo_id_set = [33,4,5,6]
    Legend_set = [3,4,5,6]
else:
    Netw_topo_id_set = [args.Netw_topo_id]
    Legend_set = [args.Netw_topo_id]
    
ind = 0
for Netw_topo_id in Netw_topo_id_set:
    print('Loading file of network topo id' + str(Netw_topo_id))
    legend_id = Legend_set[ind]
    ind = ind+1;
    if args.output is not None:
        output_folder = args.output+'/'
    else:
        output_folder = 'train_process_net'+str(Netw_topo_id)+'/'      
    if not os.path.exists(output_folder):
        sys.exit('No such directory %s' %output_folder);
    num_realization = args.num_realization

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

    # Complementary parameters
    scheme_setting_list = get_MAB_scheme_setting()
    N_schemes = len(scheme_setting_list)
    eval_ites_DRL = args_DRL.eval_ites
    try:
        eval_ites_MAB = args_MAB.eval_ites
    except:
        eval_ites_MAB = eval_ites_DRL
    if args.customized_check_point == 1:
        customized_check_point = np.array([1,4,7,10,20,40,60,80,100,150,200,240])
    else:
        customized_check_point = range(len(ave_evolution_rate_ckpt_DRL))
        
    num_check_points = len(customized_check_point)
    try:
        ave_evolution_rate_ckpt = np.append(ave_evolution_rate_ckpt_MAB,ave_evolution_rate_ckpt_DRL.reshape((1,num_check_points)),axis=0)
        ave_evolution_delay_ckpt = np.append(ave_evolution_delay_ckpt_MAB,ave_evolution_delay_ckpt_DRL.reshape((1,num_check_points)),axis=0)
        ave_evolution_ratio_blockage_ckpt = np.append(ave_evolution_ratio_blockage_ckpt_MAB,ave_evolution_ratio_blockage_ckpt_DRL.reshape((1,num_check_points)),axis=0)
        min_evolution_delay_ckpt = np.append(min_evolution_delay_ckpt_MAB,min_evolution_delay_ckpt_DRL.reshape((1,num_check_points)),axis=0)
        max_evolution_delay_ckpt = np.append(max_evolution_delay_ckpt_MAB,max_evolution_delay_ckpt_DRL.reshape((1,num_check_points)),axis=0)
        target_scheme = range(N_schemes)
    except:
        ave_evolution_rate_ckpt = np.tile(ave_evolution_rate_ckpt_DRL.reshape((1,num_check_points)),(N_schemes,1))
        ave_evolution_delay_ckpt = np.tile(ave_evolution_delay_ckpt_DRL.reshape((1,num_check_points)),(N_schemes,1))
        ave_evolution_ratio_blockage_ckpt = np.tile(ave_evolution_ratio_blockage_ckpt_DRL.reshape((1,num_check_points)),(N_schemes,1))
        min_evolution_delay_ckpt = np.tile(min_evolution_delay_ckpt_DRL.reshape((1,num_check_points)),(N_schemes,1))
        max_evolution_delay_ckpt = np.tile(max_evolution_delay_ckpt_DRL.reshape((1,num_check_points)),(N_schemes,1))
        target_scheme = [7]

    #----------------------------------------------------------------
    # Results from training phase
    #----------------------------------------------------------------

    ############################# Average delay at each check point
    ite_duration = args_DRL.slots * env_parameter.t_slot
    plt.figure(num=102,figsize=(10,6),dpi=1200)
    #plt.title('Evolution of average delay evaluated at each checkpoint')
    for scheme_id in target_scheme:
        plt.plot(customized_check_point,\
                 ave_evolution_delay_ckpt[scheme_id],'*-',\
                 label='Number of UEs: '+str(legend_id)) 

    if Netw_topo_id == Netw_topo_id_set[-1]:
        plt.rcParams.update({'legend.loc': 'upper right'})
        plt.legend()
        plt.xlabel('Training iteration (one iteration is %d seconds)' %ite_duration)
        plt.ylabel('Delay in slots');
        plt.savefig(output_folder+'Ave_Training_phase_evolution_delay_ckpt.eps',format='eps', facecolor='w', transparent=True)
        plt.savefig(output_folder+'Ave_Training_phase_evolution_delay_ckpt.pdf',format='pdf', facecolor='w', transparent=True)
        
    if args.netw_changed == 1:
        for scheme_id in target_scheme:
            change_id = 0;
            plt.plot(customized_check_point[0+change_id*100:99+change_id*100],\
                         ave_evolution_delay_ckpt[scheme_id][0+change_id*100:99+change_id*100],'-',\
                         label='Initial network topology') 
            change_id = 1;
            plt.plot(customized_check_point[0+change_id*100:99+change_id*100],\
                         ave_evolution_delay_ckpt[scheme_id][0+change_id*100:99+change_id*100],'-',\
                         label='First network topology change (small variation)') 
            change_id = 2;
            plt.plot(customized_check_point[0+change_id*100:99+change_id*100],\
                         ave_evolution_delay_ckpt[scheme_id][0+change_id*100:99+change_id*100],'-',\
                         label='Second network topology change (medium variation)') 
            change_id = 3;
            plt.plot(customized_check_point[0+change_id*100:99+change_id*100],\
                         ave_evolution_delay_ckpt[scheme_id][0+change_id*100:99+change_id*100],'-',\
                         label='Third network topology change (big variation)') 
            plt.rcParams.update({'legend.loc': 'upper right'})
            plt.legend()
            plt.xlabel('Training iteration (one iteration is %d seconds)' %ite_duration)
            plt.ylabel('Delay in slots');
            plt.savefig(output_folder+'Ave_Training_phase_evolution_delay_ckpt.eps',format='eps', facecolor='w', transparent=True)
            plt.savefig(output_folder+'Ave_Training_phase_evolution_delay_ckpt.pdf',format='pdf', facecolor='w', transparent=True)
