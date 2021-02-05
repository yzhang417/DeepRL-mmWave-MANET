# This Python file uses the following encoding: utf-8
# coding=utf-8
import numpy as np
import os
import sys
import random
import operator 
import math
import pdb
import matplotlib.pyplot as plt
from envs import *
from my_utils import *

#----------------------------------------------------
# Calculate optimal beamwidth for each UE
#----------------------------------------------------
def calculate_ground_truth_optimal_beamwidth(slots, LOOP):
    Netw_topo_id = 1
    env_parameter = env_init(Netw_topo_id)
    envEvaluation = envs(env_parameter,slots)
    reff_statistic = np.zeros((LOOP,env_parameter.N_UE,len(env_parameter.BeamWidth_TX_vec)))
    final_eff_coeff = np.zeros((LOOP,env_parameter.N_UE,len(env_parameter.BeamWidth_TX_vec)))
    # Evaluation loop
    for loop in range(LOOP):  
        for ue_id in range(env_parameter.N_UE):
            for bw_id in range(len(env_parameter.BeamWidth_TX_vec)):
                state = envEvaluation.reset()
                for slot in range(slots):    
                    # Choose an action
                    action = def_action(env_parameter)
                    action.UE_ID_BS2UE_Link = ue_id
                    action.Relay_ID = ue_id
                    action.BW_ID_BS2UE_Link = bw_id
                    action.Is_Tracking = False
                    # Interaction with the enviroment to get a new state and a step-level reward
                    channel = envEvaluation.channel_realization()
                    state, output, reward, done = envEvaluation.step(state, action,channel)
                    # SNR bs2ue Calculation
                    SNR_BS2UE_dB = envEvaluation.env_parameter.Ptx_BS_dBm + \
                    envEvaluation.env_parameter.GtGr_dBi_mat[bw_id] - \
                    envEvaluation.env_parameter.Pathloss[ue_id,ue_id] - \
                    envEvaluation.env_parameter.W - \
                    envEvaluation.env_parameter.Pn_dBm + channel[ue_id,ue_id]
                    # Coeff_BS2UE
                    Coeff_BS2UE = envEvaluation.env_parameter.Coeff_BS2UE
                    Coeff = Coeff_BS2UE[bw_id]*(1 - envEvaluation.outage_coeff[ue_id,ue_id])
                    final_eff_coeff[loop,ue_id,bw_id] = Coeff
                    MCS_ID_BS2UE = len(np.where(envEvaluation.env_parameter.SNR_Value<=SNR_BS2UE_dB)[0])-1;
                    Reff_BS2UE_Link = Coeff*envEvaluation.env_parameter.Rate_Value[MCS_ID_BS2UE];
                    reff_statistic[loop,ue_id,bw_id] = Reff_BS2UE_Link
    outage_coeff = final_eff_coeff/Coeff
    return reff_statistic, final_eff_coeff, outage_coeff, env_parameter


# ---------------------------------------------------
# Plot effective rate and print Final effective coeff
# ---------------------------------------------------
def plot_reff_statistic(reff_statistic, final_eff_coeff, outage_coeff, env_parameter):
    mean_final_eff_coeff = np.mean(final_eff_coeff,axis=0)
    for u in range(env_parameter.N_UE):
        print('Final effective coeff for UE ' + str(u) +' is: ', mean_final_eff_coeff[u,:])
        print('Outage coeff for UE ' + str(u) +' is: ', outage_coeff[u])
    mean_reff_statistic = np.mean(reff_statistic,axis=0)
    fig = plt.figure(figsize=(8,6),dpi=100)
    ax = fig.add_subplot(111)
    for u in range(env_parameter.N_UE):
        ax.plot(range(len(env_parameter.BeamWidth_TX_vec)),mean_reff_statistic[u,:],label='UE'+str(u))
    ax.legend()
    ax.grid(b=True, which='major', color='#666666', linestyle='-')
    ax.set_xticks(range(len(env_parameter.BeamWidth_TX_vec)))
    ax.set_xticklabels(env_parameter.BeamWidth_TX_vec)
    ax.set_xlabel('Beamwidth (degrees)')
    ax.set_ylabel('Effective rate')
    ax.set_title('Average effective rate for different BW selection')
    plt.show();
    
    
# ---------------------------------------------------------------
# Independent function to calculate statistics of outage
# under different location, velocity, beamwidth and self-rotation
# ---------------------------------------------------------------
def calculate_outage_statistic(slots, LOOP):
    Netw_topo_id = 1
    env_parameter = env_init(Netw_topo_id)
    env_parameter.max_activity_range = 5
    env_parameter.max_number_last_rotate = 0
    env_parameter.max_number_last_direction = 0
    envEvaluation = envs(env_parameter,slots)
    ue_velocity_vec = np.array([0,2,4,6,8,10])
    ue_selfrotate_vec = np.array([0,2,4,6,8])
    outage = np.zeros((LOOP,\
                       env_parameter.N_UE,\
                       len(ue_velocity_vec),\
                       len(ue_selfrotate_vec),\
                       len(env_parameter.BeamWidth_TX_vec)))
    # Evaluation loop
    for loop in range(LOOP):  
        for ue_id in range(env_parameter.N_UE):
            for v_id in range(len(ue_velocity_vec)):
                for bw_id in range(len(env_parameter.BeamWidth_TX_vec)):
                    for r_id in range(len(ue_selfrotate_vec)):
                        state = envEvaluation.reset()
                        envEvaluation.env_parameter.v_min = ue_velocity_vec[v_id]
                        envEvaluation.env_parameter.v_max = ue_velocity_vec[v_id]
                        envEvaluation.env_parameter.v_self_rotate_min = ue_selfrotate_vec[r_id]
                        envEvaluation.env_parameter.v_self_rotate_max = ue_selfrotate_vec[r_id]
                        for slot in range(slots):    
                            # Choose an action
                            action = def_action(env_parameter)
                            action.UE_ID_BS2UE_Link = ue_id
                            action.Relay_ID = ue_id
                            action.BW_ID_BS2UE_Link = bw_id
                            action.Is_Tracking = False
                            # Interaction with the enviroment to get a new state and a step-level reward
                            state, output, reward, done = envEvaluation.step(state, action, envEvaluation.channel_realization())
                            outage[loop,ue_id,v_id,r_id,bw_id] = envEvaluation.outage_coeff[ue_id,ue_id]
                            envEvaluation.env_parameter.Xcoor = envEvaluation.env_parameter.Xcoor_init
                            envEvaluation.env_parameter.Ycoor = envEvaluation.env_parameter.Ycoor_init
    return outage, ue_velocity_vec, ue_selfrotate_vec, envEvaluation.env_parameter


# -------------------------------------
# Independent function to show outage trend
# -------------------------------------
def plot_outage_trend(outage, ue_velocity_vec, ue_selfrotate_vec, env_parameter):
    outage_mean = np.mean(outage,axis=0)
    
    # Effect of UE velocity
    fig1 = plt.figure(figsize=(16,12),dpi=100)
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    axD = []
    axD.append(fig1.add_subplot(321))
    axD.append(fig1.add_subplot(322))
    axD.append(fig1.add_subplot(323))
    axD.append(fig1.add_subplot(324))
    axD.append(fig1.add_subplot(325))
    for ue_id in range(5):
        for bw_id in range(len(env_parameter.BeamWidth_TX_vec)):
            axD[ue_id].plot(ue_velocity_vec,outage_mean[ue_id,:,0,bw_id],\
                            label='BW: '+str(env_parameter.BeamWidth_TX_vec[bw_id])+' degrees')
            axD[ue_id].set_title('Effect of UE velocity for UE id '+str(ue_id))
            axD[ue_id].grid(b=True, which='major', color='#666666', linestyle='-')
            axD[ue_id].legend()
            axD[ue_id].set_xlabel('UE velocity (m/s)')
            axD[ue_id].set_ylabel('Outage coefficent')

    # Effect of UE self-rotation rate
    fig2 = plt.figure(figsize=(16,12),dpi=100)
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    axR = []
    axR.append(fig2.add_subplot(321))
    axR.append(fig2.add_subplot(322))
    axR.append(fig2.add_subplot(323))
    axR.append(fig2.add_subplot(324))
    axR.append(fig2.add_subplot(325))
    for ue_id in range(5):
        for bw_id in range(len(env_parameter.BeamWidth_TX_vec)):
            axR[ue_id].plot(ue_selfrotate_vec,outage_mean[ue_id,0,:,bw_id],\
                            label='BW: '+str(env_parameter.BeamWidth_TX_vec[bw_id])+' degrees')
            axR[ue_id].set_title('Effect of UE self-rotation for UE id '+str(ue_id))
            axR[ue_id].grid(b=True, which='major', color='#666666', linestyle='-')
            axR[ue_id].legend()
            axR[ue_id].set_xlabel('UE self-rotation (degrees/s)')
            axR[ue_id].set_ylabel('Outage coefficent')
            
    # Effect of UE distance
    fig3 = plt.figure(figsize=(16,12),dpi=100)
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    axV = []
    axV.append(fig3.add_subplot(321))
    axV.append(fig3.add_subplot(322))
    axV.append(fig3.add_subplot(323))
    axV.append(fig3.add_subplot(324))
    axV.append(fig3.add_subplot(325))
    for v_id in range(5): 
        for bw_id in range(len(env_parameter.BeamWidth_TX_vec)):
            axV[v_id].plot(env_parameter.radius,outage_mean[:,v_id,0,bw_id],\
                           label='BW: '+str(env_parameter.BeamWidth_TX_vec[bw_id])+' degrees');
            axV[v_id].set_title('Effect of UE distance for velocity = '+str(ue_velocity_vec[v_id])+' m/s')
            axV[v_id].grid(b=True, which='major', color='#666666', linestyle='-')
            axV[v_id].legend()
            axV[v_id].set_xlabel('UE distance (meters)')
            axV[v_id].set_ylabel('Outage coefficent')

    # Overall outage for each UE over velocity (0-10) and self-rotation (0-10)
    fig4 = plt.figure(figsize=(7,7),dpi=100)
    ax = fig4.add_subplot(111)
    for u in range(env_parameter.N_UE):
        ax.plot(range(len(env_parameter.BeamWidth_TX_vec)),\
                np.mean(np.mean(outage_mean[u,:,:,:],axis=0),axis=0),label='UE'+str(u))
    ax.legend()
    ax.grid(b=True, which='major', color='#666666', linestyle='-')
    ax.set_xticks(range(len(env_parameter.BeamWidth_TX_vec)))
    ax.set_xticklabels(env_parameter.BeamWidth_TX_vec)
    ax.set_xlabel('Beamwidth (degrees)')
    ax.set_ylabel('Outage coefficent')
    ax.set_title('Average outage coefficent for different BW selection')
    plt.show();
    return np.mean(np.mean(outage_mean[:,:,:,:],axis=1),axis=1)

    