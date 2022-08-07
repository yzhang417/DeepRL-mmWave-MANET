#!/usr/bin/env python
# coding: utf-8
import sys
import numpy as np
import operator
import os
import math
import random
import pdb
from my_utils import *
from state_action import *
import copy
# from scipy.ndimage.interpolation import shift
# numpy.pad is much more efficient than shift


#-----------------------------------------------------------------------
# Class of parameters configuring the wireless system
#-----------------------------------------------------------------------
class def_env_para():
    def __init__(self):
        pass
    

#-----------------------------------------------------------------------
# Following function allows configuring the studied wirelss system and 
# will return an instance of def_env_para consisting of necesary parameters
#-----------------------------------------------------------------------
def env_init(Netw_topo_id):    
    
    # -------------------------------------
    # Network topology
    # -------------------------------------    
    if Netw_topo_id == 10 or Netw_topo_id == 3:  # Scenario presented in the paper
        workload = 1.0 * 1e9;                              # Total downlink data stream rate
        N_UE = 5; # Number of users 
        radius = np.array([10, 10, 15, 25, 30]);  # Distance between Tx and Rx
        angle =  np.array([5, 85, 45, 10, 80]);   # Angle between Tx and Rx
        lambda_ratio = np.array([1, 3, 1, 1, 1]); # Ratio of arrival rate
        target_prob_blockage_to_AP = np.array([0.05, 0.05, 0.8, 0.05, 0.05]); # Average percentage of slots in blockage
        target_prob_blockage_D2D = 0.05
    if Netw_topo_id == 11:  # Added for changinig enviroment from scenario 3
        workload = 1.0 * 1e9;                              # Total downlink data stream rate
        N_UE = 5; # Number of users 
        radius = np.array([10+5, 10+5, 15, 25-5, 30+5]); # Distance between Tx and Rx after enviromental change
        angle = np.array([5-10, 85-5, 45-5, 10+5, 80-5]);  # Angle between Tx and Rx after enviromental change
        lambda_ratio = np.array([1, 3, 1, 1, 1]); # Ratio of arrival rate
        target_prob_blockage_to_AP = np.array([0.05, 0.05, 0.8, 0.05, 0.05]); # Average percentage of slots in blockage
        target_prob_blockage_D2D = 0.05
    if Netw_topo_id == 12:  # Added for changinig enviroment from scenario 3
        workload = 1.0 * 1e9;                              # Total downlink data stream rate
        N_UE = 5; # Number of users 
        radius = np.array([25-5, 30+5, 15, 10+5, 10+5]); # Distance between Tx and Rx after enviromental change
        angle = np.array([10+5, 80-5, 45-5, 5-10, 85-5]);  # Angle between Tx and Rx after enviromental change
        lambda_ratio = np.array([1, 3, 1, 1.5, 1]); # Ratio of arrival rate
        target_prob_blockage_to_AP = np.array([0.05, 0.05, 0.8, 0.05, 0.05]); # Average percentage of slots in blockage
        target_prob_blockage_D2D = 0.05
    if Netw_topo_id == 13:  # Added for changinig enviroment from scenario 3
        workload = 1.0 * 1e9;                              # Total downlink data stream rate
        N_UE = 5; # Number of users 
        radius = np.array([25-5, 30+5, 15, 10+5, 10+5]); # Distance between Tx and Rx after enviromental change
        angle = np.array([10+5, 80-5, 45-5, 5-10, 85-5]);  # Angle between Tx and Rx after enviromental change
        lambda_ratio = np.array([1, 2, 1, 1, 1]); # Ratio of arrival rate
        target_prob_blockage_to_AP = np.array([0.7, 0.5, 0.1, 0.05, 0.05]); # Average percentage of slots in blockage
        target_prob_blockage_D2D = 0.05
    if Netw_topo_id == 14: 
        workload = 1.0 * 1e9;                              # Total downlink data stream rate
        N_UE = 5; # Number of users 
        radius = np.array([10, 20, 30, 40, 50]);  # Distance between Tx and Rx
        angle =  np.array([0, 45, 35, 5, 20]);    # Angle between Tx and Rx
        lambda_ratio = np.array([1, 3, 1, 1, 1]); # Ratio of arrival rate
        target_prob_blockage_to_AP = np.array([0.05, 0.05, 0.8, 0.05, 0.05]); # Average percentage of slots in blockage
        target_prob_blockage_D2D = 0.05
    if Netw_topo_id == 2:
        workload = 1.0 * 1e9;                              # Total downlink data stream rate
        N_UE = 5; # Number of users 
        radius = np.array([10, 10, 15, 25, 30]);  # Distance between Tx and Rx
        angle =  np.array([5, 85, 45, 10, 80]);   # Angle between Tx and Rx
        lambda_ratio = np.array([1, 100, 1, 1, 1]); # Ratio of arrival rate
        target_prob_blockage_to_AP = np.array([0.05, 0.05, 0.8, 0.05, 0.05]); # Average percentage of slots in blockage
        target_prob_blockage_D2D = 0.05
    if Netw_topo_id == 33:  
        workload = 1.0 * 1e9;                              # Total downlink data stream rate
        N_UE = 3; # Number of users 
        radius = np.array([10, 10, 15]);  # Distance between Tx and Rx
        angle =  np.array([5, 85, 45]);   # Angle between Tx and Rx
        lambda_ratio = np.array([1, 3, 1]); # Ratio of arrival rate
        target_prob_blockage_to_AP = np.array([0.05, 0.05, 0.8]); # Average percentage of slots in blockage
        target_prob_blockage_D2D = 0.05
    if Netw_topo_id == 4: 
        workload = 1.0 * 1e9;                              # Total downlink data stream rate
        N_UE = 4; # Number of users 
        radius = np.array([10, 10, 15, 25]);  # Distance between Tx and Rx
        angle =  np.array([5, 85, 45, 10]);   # Angle between Tx and Rx
        lambda_ratio = np.array([1, 3, 1, 1]); # Ratio of arrival rate
        target_prob_blockage_to_AP = np.array([0.05, 0.05, 0.8, 0.05]); # Average percentage of slots in blockage
        target_prob_blockage_D2D = 0.05
    if Netw_topo_id == 5:  
        workload = 1.0 * 1e9;                              # Total downlink data stream rate
        N_UE = 5; # Number of users 
        radius = np.array([10, 10, 15, 25, 30]);  # Distance between Tx and Rx
        angle =  np.array([5, 85, 45, 10, 80]);   # Angle between Tx and Rx
        lambda_ratio = np.array([1, 3, 1, 1, 1]); # Ratio of arrival rate
        target_prob_blockage_to_AP = np.array([0.05, 0.05, 0.8, 0.05, 0.05]); # Average percentage of slots in blockage
        target_prob_blockage_D2D = 0.05
    if Netw_topo_id == 6: 
        workload = 1.0 * 1e9;                              # Total downlink data stream rate
        N_UE = 6; # Number of users 
        radius = np.array([10, 10, 15, 25, 30, 20]);  # Distance between Tx and Rx
        angle =  np.array([5, 85, 45, 10, 80, 28]);   # Angle between Tx and Rx
        lambda_ratio = np.array([1, 3, 1, 1, 1, 1]); # Ratio of arrival rate
        target_prob_blockage_to_AP = np.array([0.05, 0.05, 0.8, 0.05, 0.05, 0.05]); # Average percentage of slots in blockage
        target_prob_blockage_D2D = 0.05
    if Netw_topo_id == 7: 
        workload = 1.0 * 1e9;                              # Total downlink data stream rate
        N_UE = 7; # Number of users 
        radius = np.array([10, 10, 15, 25, 30, 20, 15]);  # Distance between Tx and Rx
        angle =  np.array([5, 85, 45, 10, 80, 28, 30]);   # Angle between Tx and Rx
        lambda_ratio = np.array([1, 3, 1, 1, 1, 1, 1]); # Ratio of arrival rate
        target_prob_blockage_to_AP = np.array([0.05, 0.05, 0.8, 0.05, 0.05, 0.05, 0.05]); # Average percentage of slots in blockage
        target_prob_blockage_D2D = 0.05
    target_prob_blockage = np.ones((N_UE,N_UE)) - np.diag(np.ones(N_UE))
    target_prob_blockage = target_prob_blockage * target_prob_blockage_D2D + np.diag(target_prob_blockage_to_AP)
    Xcoor_init, Ycoor_init = pol2cart(np.deg2rad(angle),radius);
    Xcoor_init = np.append(Xcoor_init,0) # Last index for AP X
    Ycoor_init = np.append(Ycoor_init,0) # Last index for AP Y

    # -----------------------------------------------
    # Beam training setting (realignment periodicity)
    # -----------------------------------------------
    t_slot = 10 * 1e-3;       # Time duration for a single slot in seconds
    t_SSW = 10 * 1e-6;        # Time of SSW frame in seconds (time duration for per measurement)
    Num_arr_AP = 4;           # Number of antenna arrays equipped to cover 360 degrees
    Num_arr_UE = 4;           # Number of antenna arrays equipped to cover 360 degrees
    Beampair_Repetition = 1;  # Repetion of each beam pair sounding
    BeamWidth_vertical = 75;  # Elevation beamwidth
    single_side_beam_training = False;    # Double side beam training    
    
    
    # -------------------------------------
    # Mobility model
    # -------------------------------------
    max_activity_range = 5; # maximum distance in meters from the initial position
    v_min = 0;  # minimum speed in m/s
    v_max = 10; # maximum speed in m/s
    last_direction = np.zeros(N_UE+1) # last index is for the mobility of AP
    last_velocity = np.zeros(N_UE+1) # last index is for the mobility of AP
    number_last_direction = np.zeros(N_UE+1) # last index is for the mobility of AP
    v_self_rotate_min = 0  # minimum ratation speed in degrees/s
    v_self_rotate_max = 10  # maximum ratation speed in degrees/s
    number_last_rotate = np.zeros(N_UE+1) # last index is for the mobility of AP
    max_number_last_rotate = 20
    max_number_last_direction = 20
    
    
    # -------------------------------------
    # Blockage model
    # -------------------------------------
    blockage_loss_dB = 20;     # Blockage loss
    min_blockage_duration = 2; # Number of minimum slots that blockage exists
    max_blockage_duration = 6; # Number of maximum slots that blockage exists
    min_blockage_duration_guess = 1;   # Guess of min_blockage_duration
    max_blockage_duration_guess = 10;  # Guess of max_blockage_duration
    prob_blockage = target_prob_blockage/\
    (target_prob_blockage+(min_blockage_duration+max_blockage_duration)/2*(1-target_prob_blockage))  

    
    # -------------------------------------
    # UE packet arrival model
    # -------------------------------------
    mean_packet_size = 2312*8;                         # Mean size of a single packet in bit
    lambda_ratio = lambda_ratio/np.sum(lambda_ratio);  
    total_arrival_rate = workload/mean_packet_size;    # Total packet arrival rate
    lambda_vec = lambda_ratio * total_arrival_rate;    # Packet arrival rate for each user
    
    
    # -------------------------------------
    # Channel correlation and fading model
    # -------------------------------------
    #channel_corr_coeff = np.array([0.8,0.2]); # Channel time correlation
    channel_corr_coeff = np.array([1]);
    mean_X_coeff = 0;
    sigma_X_coeff = np.sqrt(4);
    mean_X = mean_X_coeff*np.zeros((N_UE,N_UE));
    sigma_X = sigma_X_coeff*np.ones((N_UE,N_UE));

    
    # -------------------------------------
    # PHY/MAC parameters
    # -------------------------------------
    gamma_UWMTS = 3;    # Parameter for Unimodal Weighted Multinomial TS algorithm
    W1 = 5;             # Link marginal budget
    W2 = 5;             # Implementation Loss
    W = W1 + W2;        # Total link budget
    B = 2160 * 1e6;                # Bandwith in Hz
    tracking_angular_space_d = 30; # Beam sweeping area in tracking slot
    tracking_t_SSW = t_SSW;        # Time of SSW frame in tracking slot in seconds 
    tracking_t_slot = t_slot;      # Time duration for a single tracking slot in seconds
    Using_MCS = 1;      # Using MCS and sensibility of IEEE 802.11 ad standard (Can be updated to 5G NR)
    MCS = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.1, 10, 11, 12, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6]);
    RS_Value = np.array([-1000, -78, -68, -66, -65, -64, -63, -62, -61, -60, \
                         -59, -57, -55, -54, -53, -51, -50, -48, -46, -44, -42]);
    Rate_Value = np.array([0, 27.5, 385, 770, 962.5, 1155, 1251.25, 1540, 1925, 2310, \
                           2502.5, 2695, 3080, 3850, 4620, 5005, 5390, 5775, 6390, 7507.5, 8085])*1e6;
    maxRateValue = np.amax(Rate_Value);    # Maximum rate supported (Highest MCS)
    RateNor = Rate_Value/maxRateValue;     # Normalized rate within [0,1] so that the reward is bounded
    M = len(Rate_Value)-1;                 # Number of MCS, aka size of supports without 0 bps rate option
    Pn_dBm = -174 + 10*np.log10(B) + 10;   # Power of noise in dBm
    SNR_Value = RS_Value - Pn_dBm;         # Corresponding SNR table to RSS


    # -------------------------------------
    # Antenna setting at BS and UE
    # -------------------------------------
    #N_SSW_BS_vec = np.array([24, 32, 64, 128, 256, 512]); # Tx total number of sectors to cover 2D space
    N_SSW_BS_vec = np.array([24, 32, 64, 128, 256, 512]); # Tx total number of sectors to cover 2D space
    BeamWidth_TX_vec = 360./N_SSW_BS_vec;   # Tx antenna beamwidth
    N_SSW_UE_vec = 24;                      # Rx total number of sectors to cover 2D space
    BeamWidth_RX_vec = 360/N_SSW_UE_vec;    # Rx antenna beamwidth
    fc = 60;                                # carrier frequency in GHz
    c = 3e8;                                # speed of light
    l = c/(fc*1e9);                         # wavelength
    d = l/2;                                # antenna spacing
    
    
    # -------------------------------------
    # Antenna gain between BS and UEs
    # -------------------------------------
    Ptx_BS_dBm = 15;   # Power of transmitting signal from BS in dBm
    Gt_vec = 16*np.pi/(6.67*np.deg2rad(BeamWidth_TX_vec)*np.deg2rad(BeamWidth_vertical)); # Transmitting antenna gain
    Gt_dBi_vec = 10*np.log10(Gt_vec);                      # Transmitting antenna gain in dBi
    Gr_vec = 16*np.pi/(6.67*np.deg2rad(BeamWidth_RX_vec)*np.deg2rad(BeamWidth_vertical)); # Receiving antenna gain
    Gr_dBi_vec = 10*np.log10(Gr_vec);                      # Receiving antenna gain in dBi
    EIRP = 43;                                             # Limitation of EIRP in USA by FCC
    EIRP_real_max = np.amax(Gt_dBi_vec) + Ptx_BS_dBm;
    if EIRP_real_max > EIRP:
        print('Error in EIRP for BS');     # Validate that the EIRP meets the FCC requirement
    GtGr_dBi_mat = Gt_dBi_vec + Gr_dBi_vec;

    
    # -------------------------------------
    # Antenna gain between UEs
    # -------------------------------------
    Ptx_UE_dBm = 10;   # Power of transmitting signal from UE in dBm
    Gt_D2D = 16*np.pi/(6.67*np.deg2rad(BeamWidth_RX_vec)*np.deg2rad(BeamWidth_vertical)); 
    Gt_D2D_dBi = 10*np.log10(Gt_D2D);                      
    Gr_D2D = Gt_D2D;   
    Gr_D2D_dBi = 10*np.log10(Gr_D2D);
    EIRP_real_max = np.amax(Gr_D2D_dBi) + Ptx_UE_dBm;
    if EIRP_real_max > EIRP:
        print('Error in EIRP for D2D');    # Validate that the EIRP meets the FCC requirement

    
    # -------------------------------------
    # Saved enviroment parameter
    # -------------------------------------
    env_parameter = def_env_para(); 
    env_parameter.Netw_topo_id = Netw_topo_id;
    env_parameter.workload = workload;
    env_parameter.lambda_ratio = lambda_ratio;
    env_parameter.N_UE = N_UE;
    env_parameter.gamma_UWMTS = gamma_UWMTS;
    env_parameter.Rate_Value = Rate_Value;
    env_parameter.maxRateValue = maxRateValue;
    env_parameter.RateNor = RateNor;
    env_parameter.M = M;
    env_parameter.SNR_Value = SNR_Value;
    env_parameter.N_SSW_BS_vec = N_SSW_BS_vec;
    env_parameter.N_SSW_UE_vec = N_SSW_UE_vec;
    env_parameter.mean_packet_size = mean_packet_size;
    env_parameter.W = W;
    env_parameter.B = B;
    env_parameter.fc = fc;
    env_parameter.t_slot = t_slot;
    env_parameter.t_SSW = t_SSW;
    env_parameter.tracking_angular_space_d = tracking_angular_space_d;
    env_parameter.tracking_t_SSW = tracking_t_SSW;
    env_parameter.tracking_t_slot = tracking_t_slot;
    env_parameter.BeamWidth_TX_vec = BeamWidth_TX_vec;
    env_parameter.BeamWidth_RX_vec = BeamWidth_RX_vec;
    env_parameter.Ptx_BS_dBm = Ptx_BS_dBm;    
    env_parameter.GtGr_dBi_mat = GtGr_dBi_mat;
    env_parameter.Ptx_UE_dBm = Ptx_UE_dBm;
    env_parameter.Gt_D2D_dBi = Gt_D2D_dBi;
    env_parameter.Gr_D2D_dBi = Gr_D2D_dBi;
    env_parameter.Pn_dBm = Pn_dBm;
    env_parameter.Using_MCS = Using_MCS;
    env_parameter.Num_arr_AP = Num_arr_AP;    
    env_parameter.Num_arr_UE = Num_arr_UE;           
    env_parameter.Beampair_Repetition = Beampair_Repetition;
    env_parameter.BeamWidth_vertical = BeamWidth_vertical;
    env_parameter.single_side_beam_training = single_side_beam_training;
    
    if single_side_beam_training:
        Coeff_BS2UE = 1 - t_SSW / t_slot * (N_SSW_BS_vec+N_SSW_UE_vec)*Beampair_Repetition/(Num_arr_AP*Num_arr_UE)
    else:
        Coeff_BS2UE = 1 - t_SSW / t_slot * (N_SSW_BS_vec*N_SSW_UE_vec)*Beampair_Repetition/(Num_arr_AP*Num_arr_UE)
    Coeff_BS2UE[np.where(Coeff_BS2UE<0)[0]] = 0;
    env_parameter.Coeff_BS2UE = Coeff_BS2UE;
    
    if single_side_beam_training:
        Coeff_BS2UE_Tracking = 1 - tracking_t_SSW / tracking_t_slot * \
        (np.ceil(tracking_angular_space_d/360*N_SSW_BS_vec)+np.ceil(tracking_angular_space_d/360*N_SSW_UE_vec))*\
        Beampair_Repetition
    else:
        Coeff_BS2UE_Tracking = 1 - tracking_t_SSW / tracking_t_slot * \
        (np.ceil(tracking_angular_space_d/360*N_SSW_BS_vec)*np.ceil(tracking_angular_space_d/360*N_SSW_UE_vec))*\
        Beampair_Repetition
    Coeff_BS2UE_Tracking[np.where(Coeff_BS2UE_Tracking<0)[0]] = 0;
    env_parameter.Coeff_BS2UE_Tracking = Coeff_BS2UE_Tracking;
    if single_side_beam_training:
        Coeff_D2D = (t_slot - t_SSW * (N_SSW_UE_vec + N_SSW_UE_vec) * Beampair_Repetition / (Num_arr_UE*Num_arr_UE)) / t_slot;
    else:
        Coeff_D2D = (t_slot - t_SSW * (N_SSW_UE_vec * N_SSW_UE_vec) / (Num_arr_UE*Num_arr_UE)) / t_slot;
    env_parameter.Coeff_D2D = Coeff_D2D;
    K_bw = len(N_SSW_BS_vec);
    env_parameter.K_bw = K_bw;
    K_relay = N_UE;
    env_parameter.K_relay = K_relay;
    env_parameter.lambda_vec = lambda_vec;
    env_parameter.channel_corr_coeff = channel_corr_coeff;
    env_parameter.mean_X = mean_X;
    env_parameter.sigma_X = sigma_X;
    env_parameter.min_use_per_cb_guaranteed = 5;
    env_parameter.min_use_per_relay_guaranteed = 5;
    env_parameter.min_service_guaranteed = N_UE*3;

    
    # -------------------------------------
    # Saved Mobility model
    # -------------------------------------
    # Position
    env_parameter.radius = radius
    env_parameter.Xcoor_init = Xcoor_init;
    env_parameter.Ycoor_init = Ycoor_init;
    env_parameter.Xcoor_list = [list() for _ in range(len(Xcoor_init))]
    env_parameter.Ycoor_list = [list() for _ in range(len(Xcoor_init))]
    for u in range(len(Xcoor_init)):
        env_parameter.Xcoor_list[u].append(Xcoor_init[u]);
        env_parameter.Ycoor_list[u].append(Ycoor_init[u]);
    env_parameter.Xcoor = Xcoor_init;
    env_parameter.Ycoor = Ycoor_init;
    # Velocity
    env_parameter.max_activity_range = max_activity_range
    env_parameter.v_min = v_min
    env_parameter.v_max = v_max
    env_parameter.number_last_direction = number_last_direction
    env_parameter.last_direction = last_direction
    env_parameter.last_velocity = last_velocity
    # Self-rotation
    env_parameter.v_self_rotate_min = v_self_rotate_min
    env_parameter.v_self_rotate_max = v_self_rotate_max
    env_parameter.number_last_rotate = number_last_rotate
    env_parameter.max_number_last_rotate = max_number_last_rotate
    env_parameter.max_number_last_direction = max_number_last_direction
    
    
    # -------------------------------------
    # Saved Blockage model
    # -------------------------------------s
    env_parameter.blockage_loss_dB = blockage_loss_dB;    # Blockage
    env_parameter.target_prob_blockage = target_prob_blockage;
    env_parameter.prob_blockage = prob_blockage;    # Blockage
    env_parameter.min_blockage_duration = min_blockage_duration; # Number of minimum slots that blockage exists
    env_parameter.max_blockage_duration = max_blockage_duration; # Number of maximum slots that blockage exists
    env_parameter.min_blockage_duration_guess = min_blockage_duration_guess;
    env_parameter.max_blockage_duration_guess = max_blockage_duration_guess;
    env_parameter.outage_coeff = np.zeros((env_parameter.N_UE,env_parameter.N_UE))
    env_parameter.last_outage_coeff = np.zeros((env_parameter.N_UE,env_parameter.N_UE))
    return env_parameter


#-----------------------------------------------------------------------
# In the following class, we create a enviroment
#-----------------------------------------------------------------------
class envs():
    # -------------------------------------
    # Initialization
    # -------------------------------------
    def __init__(self,env_parameter,slots_monitored):
        self.env_parameter = copy.deepcopy(env_parameter)
        self.slots_monitored = slots_monitored
        self.is_external_packet_arrival_process = False
        # Following would be reset when reset() called
        self.ct = 0
        self.Reff_BS2UE_Link_Last_Slot = 0
        self.remain_slots_in_blockage = np.zeros((env_parameter.N_UE,env_parameter.N_UE),dtype=int)
        self.num_slots_to_last_blockage_starts = np.ones(env_parameter.N_UE,dtype=int) * 100
        self.pathloss_history = np.zeros((slots_monitored,env_parameter.N_UE,env_parameter.N_UE))
        self.outage_coeff = np.zeros((env_parameter.N_UE,env_parameter.N_UE))
        self.position_update()
        self.pathloss_update()
        self.channel_X_history = np.zeros((slots_monitored,env_parameter.N_UE,env_parameter.N_UE))
        self.last_arrival_time = np.zeros(env_parameter.N_UE) - np.mean(env_parameter.t_slot)
        self.npkts_arrival = np.zeros((env_parameter.N_UE),dtype=int)
        self.npkts_arrival_evolution = np.zeros((slots_monitored,env_parameter.N_UE),dtype=int)
        self.npkts_departure_evolution = np.zeros((slots_monitored,env_parameter.N_UE),dtype=int)
        self.throughput = np.zeros((slots_monitored,env_parameter.N_UE));
        self.Queue = np.zeros((slots_monitored,env_parameter.N_UE),dtype=int)
        self.external_npkts_arrival = np.zeros((env_parameter.N_UE),dtype=int)
        self.current_Queue_dist_by_delay = np.zeros((slots_monitored+1,env_parameter.N_UE),dtype=int)
        self.delay_dist = np.zeros((slots_monitored+1,env_parameter.N_UE),dtype=int)
        self.is_in_blockage = np.zeros((env_parameter.N_UE,env_parameter.N_UE),dtype=int)
        self.couting_tracking_slots = 0

    # -------------------------------------    
    # Reset enviroment
    # -------------------------------------
    def reset(self):
        self.ct = 0
        self.Reff_BS2UE_Link_Last_Slot = 0
        self.remain_slots_in_blockage = np.zeros((self.env_parameter.N_UE,self.env_parameter.N_UE),dtype=int)
        self.num_slots_to_last_blockage_starts = np.ones(self.env_parameter.N_UE,dtype=int) * 100
        self.pathloss_history = np.zeros((self.slots_monitored,self.env_parameter.N_UE,self.env_parameter.N_UE))
        self.outage_coeff = np.zeros((self.env_parameter.N_UE,self.env_parameter.N_UE))
        self.position_update()
        self.pathloss_update()
        self.channel_X_history = np.zeros((self.slots_monitored,self.env_parameter.N_UE,self.env_parameter.N_UE))
        self.last_arrival_time = np.zeros(self.env_parameter.N_UE) - np.mean(self.env_parameter.t_slot)
        self.npkts_arrival = np.zeros((self.env_parameter.N_UE),dtype=int)
        self.npkts_arrival_evolution = np.zeros((self.slots_monitored,self.env_parameter.N_UE),dtype=int)
        self.npkts_departure_evolution = np.zeros((self.slots_monitored,self.env_parameter.N_UE),dtype=int)
        self.throughput = np.zeros((self.slots_monitored,self.env_parameter.N_UE))
        self.Queue = np.zeros((self.slots_monitored,self.env_parameter.N_UE),dtype=int)
        self.external_npkts_arrival = np.zeros((self.env_parameter.N_UE),dtype=int)
        self.current_Queue_dist_by_delay = np.zeros((self.slots_monitored+1,self.env_parameter.N_UE),dtype=int)
        self.delay_dist = np.zeros((self.slots_monitored+1,self.env_parameter.N_UE),dtype=int)
        self.is_in_blockage = np.zeros((self.env_parameter.N_UE,self.env_parameter.N_UE),dtype=int)
        # Above is the similar to as __init__
        if self.is_external_packet_arrival_process:
            self.npkts_arrival = self.external_npkts_arrival
        else:
            self.npkts_arrival = self.packet_arrival_process()
        # Reset would generate a random initial state with one-slot packets 
        self.npkts_arrival_evolution[self.ct,:] = self.npkts_arrival
        self.Queue[self.ct,:] = self.npkts_arrival
        self.current_Queue_dist_by_delay[1,:] = self.npkts_arrival # Initial packets have delay of 1 slot
        initial_state = def_state(self.env_parameter)
        initial_state.Queue_length = self.Queue[self.ct,:]
        self.couting_tracking_slots = 0
        return initial_state
        
    # -------------------------------------
    # Run the enviroment for one time slot
    # -------------------------------------
    def step(self, state, action, channel):
        output = self.enviroment(state, action, channel)   
        output.Reff_BS2UE_Estimated = state.Reff_BS2UE_Estimated
        output.n_BS2UE = state.n_BS2UE 
        output.Reff_BS2UE_Tracking_Estimated = state.Reff_BS2UE_Tracking_Estimated
        output.n_BS2UE_Tracking = state.n_BS2UE_Tracking
        output.est_depart = state.est_depart
        output.est_arrival = state.est_arrival
        
        # Departing packets and update queue statistics
        Reff_BS2UE_Estimated = state.Reff_BS2UE_Estimated
        n_BS2UE = state.n_BS2UE
        Reff_BS2UE_Tracking_Estimated = state.Reff_BS2UE_Tracking_Estimated
        n_BS2UE_Tracking = state.n_BS2UE_Tracking
        UE_ID_BS2UE_link_Current = action.UE_ID_BS2UE_Link  
        
        # Update statistics caused by active D2D link
        if state.Is_D2D_Link_Active:
            # Update number of packets delivered in D2D link
            npkts_departed = min(output.npkts_departed_D2D_Link,self.Queue[self.ct,state.Rx_ID_D2D_Link])
            npkts_departed_D2D = npkts_departed
            self.npkts_departure_evolution[self.ct,state.Rx_ID_D2D_Link] = npkts_departed
            # Update queues of D2D link
            self.Queue[self.ct,state.Rx_ID_D2D_Link] = self.Queue[self.ct,state.Rx_ID_D2D_Link]-npkts_departed;
            # Update statistics for delay
            if npkts_departed > 0:
                this_delay = np.max(np.where(np.squeeze(self.current_Queue_dist_by_delay[:,state.Rx_ID_D2D_Link])>0)[0])
            while npkts_departed > 0:
                npkts_departed_with_this_dalay = \
                min(self.current_Queue_dist_by_delay[this_delay,state.Rx_ID_D2D_Link],npkts_departed)
                self.current_Queue_dist_by_delay[this_delay,state.Rx_ID_D2D_Link] = \
                self.current_Queue_dist_by_delay[this_delay,state.Rx_ID_D2D_Link] - npkts_departed_with_this_dalay
                self.delay_dist[this_delay,state.Rx_ID_D2D_Link] = \
                self.delay_dist[this_delay,state.Rx_ID_D2D_Link] + npkts_departed_with_this_dalay
                npkts_departed = npkts_departed - npkts_departed_with_this_dalay
                this_delay -= 1
            # Update estimate of link quality
            Reff_BS2UE_Estimated[state.Rx_ID_D2D_Link] = \
            (Reff_BS2UE_Estimated[state.Rx_ID_D2D_Link]*n_BS2UE[state.Rx_ID_D2D_Link] + \
             1/2*output.Reff_D2D_Link)/(n_BS2UE[state.Rx_ID_D2D_Link] + 1);       
            n_BS2UE[state.Rx_ID_D2D_Link] += 1;
            
        # Update statistics caused by active main link
        if operator.not_(action.Is_D2D_Link_Activated_For_Next_Slot):
            # Update number of packets delivered in main link
            npkts_departed = min(output.npkts_departed_BS2UE_Link,self.Queue[self.ct,action.UE_ID_BS2UE_Link])
            npkts_departed_Main = npkts_departed
            self.npkts_departure_evolution[self.ct,action.UE_ID_BS2UE_Link] = npkts_departed
            # Update queues of main link
            self.Queue[self.ct,action.UE_ID_BS2UE_Link] = self.Queue[self.ct,action.UE_ID_BS2UE_Link] - npkts_departed
            # Update statistics for delay
            if npkts_departed > 0:
                this_delay = np.max(np.where(np.squeeze(self.current_Queue_dist_by_delay[:,action.UE_ID_BS2UE_Link])>0)[0])
            while npkts_departed > 0:
                npkts_departed_with_this_dalay = \
                min(self.current_Queue_dist_by_delay[this_delay,action.UE_ID_BS2UE_Link],npkts_departed)
                self.current_Queue_dist_by_delay[this_delay,action.UE_ID_BS2UE_Link] = \
                self.current_Queue_dist_by_delay[this_delay,action.UE_ID_BS2UE_Link] - npkts_departed_with_this_dalay
                self.delay_dist[this_delay,action.UE_ID_BS2UE_Link] = \
                self.delay_dist[this_delay,action.UE_ID_BS2UE_Link] + npkts_departed_with_this_dalay
                npkts_departed = npkts_departed - npkts_departed_with_this_dalay
                this_delay -= 1
            # Update estimate of link quality
            Reff_BS2UE_Estimated[UE_ID_BS2UE_link_Current] =\
            (Reff_BS2UE_Estimated[UE_ID_BS2UE_link_Current]*n_BS2UE[UE_ID_BS2UE_link_Current] +\
            output.Reff_BS2UE_Link)/(n_BS2UE[UE_ID_BS2UE_link_Current] + 1);
            n_BS2UE[UE_ID_BS2UE_link_Current] += 1;    
            # Update estimate of link quality for heuristic tracking decision
            Reff_BS2UE_Tracking_Estimated[UE_ID_BS2UE_link_Current] =\
            (Reff_BS2UE_Tracking_Estimated[UE_ID_BS2UE_link_Current]*n_BS2UE_Tracking[UE_ID_BS2UE_link_Current] +\
            npkts_departed_Main*self.env_parameter.mean_packet_size)/(n_BS2UE_Tracking[UE_ID_BS2UE_link_Current] + 1);
            n_BS2UE_Tracking[UE_ID_BS2UE_link_Current] += 1;

        
        # Reward (requiring post processing to reduce variance)
        # reward = - np.linalg.norm(state.Queue_length,ord=2)
        # reward = - np.mean(state.Queue_length)
        # fairness_index = np.sum(state.Queue_length)**2/(np.sum(state.Queue_length**2)*len(state.Queue_length))
        npkt_10gbps = 2 * 1e9/self.env_parameter.mean_packet_size * self.env_parameter.t_slot
        reward = np.sum(self.npkts_departure_evolution[self.ct,:]) / npkt_10gbps  
            
        # Estimate the departure rate and arrival rate
        est_depart = Reff_BS2UE_Tracking_Estimated/self.env_parameter.mean_packet_size 
        #est_depart = Reff_BS2UE_Estimated/self.env_parameter.mean_packet_size 
        est_arrival = (state.est_arrival*self.ct + self.npkts_arrival)/(self.ct+1)
        
        # Check whether the episode ends
        if self.ct == self.slots_monitored-1:
            done = True
        else:
            done = False
            self.ct += 1
            
        # New packets arrived
        if self.is_external_packet_arrival_process:
            self.npkts_arrival = self.external_npkts_arrival
        else:
            self.npkts_arrival = self.packet_arrival_process()                
        if operator.not_(done):
            if self.ct > 0:
                self.npkts_arrival_evolution[self.ct,:] = self.npkts_arrival
                #self.current_Queue_dist_by_delay = shift(self.current_Queue_dist_by_delay, [1,0], cval=0)
                self.current_Queue_dist_by_delay = \
                np.pad(self.current_Queue_dist_by_delay,((1,0),(0,0)), mode='constant')[:-1, :]
                self.current_Queue_dist_by_delay[0,:] = self.npkts_arrival
                self.Queue[self.ct,:] = self.Queue[self.ct-1,:] + self.npkts_arrival
                state.Queue_length = self.Queue[self.ct,:]
            else:
                sys.exit('Env step function should not get here!!!')
        else:
            state.Queue_length = self.Queue[self.ct,:] + self.npkts_arrival
        
        # Sanity check of queue consitancy and npkts consitancy
        if any(np.squeeze(np.sum(self.npkts_departure_evolution,axis=0)) != np.squeeze(np.sum(self.delay_dist,axis=0))):
            #pdb.set_trace()
            #sys.exit('Number of packets delivered are not consistant!')
            print('Number of packets delivered are not consistant!')
        if any(np.sum(self.current_Queue_dist_by_delay,axis=0) != np.squeeze(self.Queue[self.ct,:])):
            #pdb.set_trace()
            #sys.exit('Queue lengths are not consistant!')
            print('Queue lengths are not consistant!')
        
        # Update the state
        state.est_depart = est_depart
        state.est_arrival = est_arrival
        state.Reff_BS2UE_Estimated = Reff_BS2UE_Estimated + 1e-8  # Avoid -inf * 0 when Reff_BS2UE_Estimated = 0 
        state.n_BS2UE = n_BS2UE
        state.Reff_BS2UE_Tracking_Estimated = Reff_BS2UE_Tracking_Estimated
        state.n_BS2UE_Tracking = n_BS2UE_Tracking
        state.Reff_BS2UE_Link_Last_Slot = output.Reff_BS2UE_Link
        state.Is_D2D_Link_Active = action.Is_D2D_Link_Activated_For_Next_Slot
        state.Tx_ID_D2D_Link = action.Relay_ID
        state.Rx_ID_D2D_Link = action.UE_ID_BS2UE_Link
        state.Is_Tracking = action.Is_Tracking_Required_For_Next_Slot
        state.UE_ID_BS2UE_Link_Last_Slot = action.UE_ID_BS2UE_Link
#         # Avoid continuing tracking
#         if state.Is_Tracking == True:
#             self.couting_tracking_slots += 1
#         else:
#             self.couting_tracking_slots = 0
#         if self.couting_tracking_slots == 10 and reward < 0.01:
#             state.Is_Tracking == False;
        
        # Update the blocakge scenario        
        self.num_slots_to_last_blockage_starts += 1        
        state.prob_still_in_blockage = ((self.env_parameter.max_blockage_duration_guess-\
                                        self.num_slots_to_last_blockage_starts)/\
                                       (self.env_parameter.max_blockage_duration_guess-\
                                        self.env_parameter.min_blockage_duration_guess+1))
        state.prob_still_in_blockage[np.where(state.prob_still_in_blockage<0)[0]] = 0
        
        # Save similar esimates in output
        output.est_depart = est_depart
        output.est_arrival = est_arrival
        output.Reff_BS2UE_Estimated = Reff_BS2UE_Estimated
        output.n_BS2UE = n_BS2UE
        
        # Return variables
        return state, output, reward, done

    # -------------------------------------
    # Packet arrival process
    # -------------------------------------
    def packet_arrival_process(self,external_ct=False):
        if type(external_ct)==bool:
            ct = self.ct
        else:
            ct = external_ct
        # Arrival with poission distribution
        npkts_arrival = np.squeeze(np.random.poisson(lam = self.env_parameter.lambda_vec*self.env_parameter.t_slot,\
                                                     size=(1,self.env_parameter.N_UE)))
        self.npkts_arrival = npkts_arrival;
        return npkts_arrival
    
    # -------------------------------------
    # Channel shadowing realization
    # -------------------------------------
    def channel_realization(self,external_ct=False):
        if type(external_ct)==bool:
            ct = self.ct
        else:
            ct = external_ct
        channel_corr_coeff = self.env_parameter.channel_corr_coeff
        channel_corr_coeff_temp = channel_corr_coeff/ \
        (np.sum(channel_corr_coeff[0:min(ct+1,len(channel_corr_coeff))])/np.sum(channel_corr_coeff))
        for u1 in range(self.env_parameter.N_UE):
            for u2 in range(u1,self.env_parameter.N_UE):
                self.channel_X_history[ct,u1,u2] = np.random.normal(self.env_parameter.mean_X[u1,u2],\
                                                                    self.env_parameter.sigma_X[u1,u2])
                self.channel_X_history[ct,u2,u1] = self.channel_X_history[ct,u1,u2]             
        for i in range(min(len(channel_corr_coeff)-1,self.ct-1)):
            self.channel_X_history[self.ct,:,:] = self.channel_X_history[self.ct,:,:]+\
            channel_corr_coeff_temp[i+1]*self.channel_X_history[self.ct-i,:,:]    
            #print('Channel correlation included')
        channel = np.squeeze(self.channel_X_history[ct,:,:])
        return channel
    
    # -------------------------------------
    # Position update with user mobility
    # -------------------------------------
    def position_update(self):
        # Compute new position
        if self.ct == 0:
            self.env_parameter.Xcoor = self.env_parameter.Xcoor_init;
            self.env_parameter.Ycoor = self.env_parameter.Ycoor_init;
            self.env_parameter.Xcoor_list = [list() for _ in range(self.env_parameter.N_UE+1)]
            self.env_parameter.Ycoor_list = [list() for _ in range(self.env_parameter.N_UE+1)]
            for u in range(self.env_parameter.N_UE+1):
                self.env_parameter.Xcoor_list[u].append(self.env_parameter.Xcoor_init[u]);
                self.env_parameter.Ycoor_list[u].append(self.env_parameter.Ycoor_init[u]);
            self.env_parameter.number_last_direction = np.zeros(self.env_parameter.N_UE+1)
            self.env_parameter.direction_new = np.zeros(self.env_parameter.N_UE+1)
            self.env_parameter.velocity_new = np.zeros(self.env_parameter.N_UE+1)
        else:
            # Fetch initial position
            Xcoor_init = self.env_parameter.Xcoor_init
            Ycoor_init = self.env_parameter.Ycoor_init
            # Fetech current position
            Xcoor = self.env_parameter.Xcoor
            Ycoor = self.env_parameter.Ycoor
            # Compute the new position
            Xcoor_new = np.zeros(self.env_parameter.N_UE+1)
            Ycoor_new = np.zeros(self.env_parameter.N_UE+1)
            # Current distance to the initial position
            current_dist_to_original = np.sqrt((Xcoor-Xcoor_init)**2 + (Ycoor-Ycoor_init)**2)  
            new_dist_to_original = np.zeros(self.env_parameter.N_UE+1)
            direction_new = np.zeros(self.env_parameter.N_UE+1)
            velocity_new = np.zeros(self.env_parameter.N_UE+1)
            for u in range(self.env_parameter.N_UE+1):
                # Case UE is currently on border
                if abs(current_dist_to_original[u]-self.env_parameter.max_activity_range) <= 1e-6:
                    # Bouncing back to the original position
                    direction_new[u] = np.arctan2(Ycoor[u]-Ycoor_init[u],Xcoor[u]-Xcoor_init[u]) + np.pi 
                    velocity_new[u] = self.env_parameter.last_velocity[u]
                    #print('Bouncing back')
                else:
                    if self.env_parameter.number_last_direction[u] >= self.env_parameter.max_number_last_direction:
                        direction_new[u] = np.random.uniform(low=-np.pi, high=np.pi,size=1)
                        velocity_new[u] = np.random.uniform(low=self.env_parameter.v_min, \
                                                            high=self.env_parameter.v_max, size=1)
                        self.env_parameter.number_last_direction[u] = 0
                        #print('New random direction')
                    else:
                        direction_new[u] = self.env_parameter.last_direction[u]
                        velocity_new[u] = self.env_parameter.last_velocity[u]
                        self.env_parameter.number_last_direction[u] += 1
                        #print('Continue the previous direction')
                self.env_parameter.last_direction[u] = direction_new[u]
                self.env_parameter.last_velocity[u] = velocity_new[u]
                Xcoor_new[u] = Xcoor[u] + np.cos(direction_new[u]) * velocity_new[u] * self.env_parameter.t_slot
                Ycoor_new[u] = Ycoor[u] + np.sin(direction_new[u]) * velocity_new[u] * self.env_parameter.t_slot
                new_dist_to_original[u] = np.sqrt((Xcoor_new[u]-Xcoor_init[u])**2 + (Ycoor_new[u]-Ycoor_init[u])**2)
                if  new_dist_to_original[u] > self.env_parameter.max_activity_range:
                    Xcoor_new[u] = Xcoor_init[u] + self.env_parameter.max_activity_range * 0.9 *\
                    np.cos(np.arctan2(Ycoor_new[u]-Ycoor_init[u],Xcoor_new[u]-Xcoor_init[u]))
                    Ycoor_new[u] = Ycoor_init[u] + self.env_parameter.max_activity_range * 0.9 *\
                    np.sin(np.arctan2(Ycoor_new[u]-Ycoor_init[u],Xcoor_new[u]-Xcoor_init[u]))
                # Save historical position
                self.env_parameter.Xcoor_list[u].append(Xcoor_new[u]);
                self.env_parameter.Ycoor_list[u].append(Ycoor_new[u]);
            # Update new coordinates and speed
            self.env_parameter.direction_new = direction_new
            self.env_parameter.velocity_new = velocity_new
            self.env_parameter.Xcoor = Xcoor_new
            self.env_parameter.Ycoor = Ycoor_new    
            
            # Sanity check of new coordiantes
            new_dist_to_original_check = np.sqrt((self.env_parameter.Xcoor_init - self.env_parameter.Xcoor)**2 + \
                                           (self.env_parameter.Ycoor_init - self.env_parameter.Ycoor)**2)   
            if any((new_dist_to_original_check-self.env_parameter.max_activity_range) > 1e-1):
                pdb.set_trace()
                sys.exit('Ue moves out of the activity region')
    
    # -------------------------------------
    # Pathloss update in dB with UMa model and blockage model
    # -------------------------------------
    def pathloss_update(self):
        # Compute new distance
        dist_D2D = np.ones((self.env_parameter.N_UE,self.env_parameter.N_UE));
        for u1 in range(self.env_parameter.N_UE):
            dist_D2D[u1,u1] = np.sqrt((self.env_parameter.Xcoor[u1]-self.env_parameter.Xcoor[-1])**2 +\
                                      (self.env_parameter.Ycoor[u1]-self.env_parameter.Ycoor[-1])**2);
            for u2 in range(u1+1,self.env_parameter.N_UE,1):
                dist_D2D[u1,u2] = np.sqrt((self.env_parameter.Xcoor[u1]-self.env_parameter.Xcoor[u2])**2 +\
                                          (self.env_parameter.Ycoor[u1]-self.env_parameter.Ycoor[u2])**2);
                dist_D2D[u2,u1] = dist_D2D[u1,u2];
        # Blocakge
        if self.ct ==0:
            Pathloss = 28.0 + 22*np.log10(dist_D2D) + 20*np.log10(self.env_parameter.fc);
            self.env_parameter.Pathloss = Pathloss
        else:
            new_blockage_duration = \
            np.random.randint(self.env_parameter.min_blockage_duration,self.env_parameter.max_blockage_duration+1,\
                             (self.env_parameter.N_UE,self.env_parameter.N_UE))
            new_blockage_status = np.random.binomial(1,self.env_parameter.prob_blockage) 
            for uu1 in range(self.env_parameter.N_UE):
                for uu2 in range(uu1+1,self.env_parameter.N_UE):
                    new_blockage_duration[uu2,uu1] = new_blockage_duration[uu1,uu2];
                    new_blockage_status[uu2,uu1] = new_blockage_status[uu1,uu2];
                    
            self.remain_slots_in_blockage -= 1;
            self.remain_slots_in_blockage.clip(min=0,out=self.remain_slots_in_blockage)            
            is_in_blockage = np.zeros_like(self.remain_slots_in_blockage)
            is_in_blockage.clip(max=1,out=is_in_blockage)           
            self.remain_slots_in_blockage = self.remain_slots_in_blockage + \
            (1 - is_in_blockage) * new_blockage_duration * new_blockage_status
            is_in_blockage = np.zeros_like(self.remain_slots_in_blockage)
            self.remain_slots_in_blockage.clip(max=1,out=is_in_blockage)     
            self.is_in_blockage = is_in_blockage
            ##blockage_loss_dB = is_in_blockage * self.env_parameter.blockage_loss_dB
            blockage_loss_dB = is_in_blockage * np.random.uniform(low=10,high=30,size=1);
            Pathloss = 28.0 + 22*np.log10(dist_D2D) + 20*np.log10(self.env_parameter.fc) + blockage_loss_dB; 
            self.env_parameter.Pathloss = Pathloss
            self.num_slots_to_last_blockage_starts = self.num_slots_to_last_blockage_starts * \
            (1-np.diag(new_blockage_status))
        
        # Save path loss history
        self.pathloss_history[self.ct,:,:] = Pathloss

    # -------------------------------------    
    # Outage duration calculatio for a successfully connected
    # The event outage is a function of UE distance, UE velocity (value and direction), 
    # UE self-rotation, time slot duration, and beamwidth
    # -------------------------------------
    def outage_coeff_update(self, state, action):
        self.env_parameter.last_outage_coeff = self.env_parameter.outage_coeff
        outage_coeff = 0
        if self.ct == 0:
            return outage_coeff
        # Fetch mobility parameters
        direction_new = self.env_parameter.direction_new
        v_tmp = self.env_parameter.velocity_new
        v_self_rotate_min = self.env_parameter.v_self_rotate_min
        v_self_rotate_max = self.env_parameter.v_self_rotate_max
        self.env_parameter.v_self_rotate = \
        np.random.uniform(low=v_self_rotate_min,high=v_self_rotate_max,size=self.env_parameter.N_UE)
        if self.env_parameter.number_last_rotate[0] <= self.env_parameter.max_number_last_rotate:
            self.env_parameter.number_last_rotate += 1
        else:
            self.env_parameter.v_self_rotate = self.env_parameter.v_self_rotate *\
            (2*np.random.binomial(1,0.5,self.env_parameter.N_UE)-1)
            self.env_parameter.number_last_rotate = np.zeros_like(self.env_parameter.number_last_rotate);
        v_self_rotate = self.env_parameter.v_self_rotate
        num_seg_slot = 100
        t_seg_vec = np.asarray(range(1,num_seg_slot+1))*self.env_parameter.t_slot/num_seg_slot
        
        # Outage in main link for case l=0 and in d2d link for case l=1
        for l in [0,1]:
            # Evolution of Tx mobility
            if l == 0: 
                uTx = action.Relay_ID
                uRx = action.Relay_ID
                v_tmp_Tx = v_tmp[-1]
                Xcoor_Tx_last = self.env_parameter.Xcoor_list[-1][-2]
                Ycoor_Tx_last = self.env_parameter.Ycoor_list[-1][-2]
                Xcoor_Tx_evo = Xcoor_Tx_last + np.cos(direction_new[-1]) * v_tmp_Tx * t_seg_vec
                Ycoor_Tx_evo = Ycoor_Tx_last + np.sin(direction_new[-1]) * v_tmp_Tx * t_seg_vec
                theta_Tx_self_rotate = v_self_rotate[-1] * t_seg_vec
                theta_Rx_self_rotate = v_self_rotate[uRx] * t_seg_vec
            elif  l == 1 and state.Is_D2D_Link_Active:
                uTx = state.Tx_ID_D2D_Link
                uRx = state.Rx_ID_D2D_Link
                v_tmp_Tx = v_tmp[uTx]
                Xcoor_Tx_last = self.env_parameter.Xcoor_list[uTx][-2]
                Ycoor_Tx_last = self.env_parameter.Ycoor_list[uTx][-2]
                Xcoor_Tx_evo = Xcoor_Tx_last + np.cos(direction_new[uTx]) * v_tmp_Tx * t_seg_vec
                Ycoor_Tx_evo = Ycoor_Tx_last + np.sin(direction_new[uTx]) * v_tmp_Tx * t_seg_vec
                theta_Tx_self_rotate = v_self_rotate[uTx] * t_seg_vec
                theta_Rx_self_rotate = v_self_rotate[uRx] * t_seg_vec
            else:
                return outage_coeff
            # Evolution of Rx mobility
            v_tmp_Rx = v_tmp[uRx]
            Xcoor_Rx_last = self.env_parameter.Xcoor_list[uRx][-2]
            Ycoor_Rx_last = self.env_parameter.Ycoor_list[uRx][-2]
            Xcoor_Rx_evo = Xcoor_Rx_last + np.cos(direction_new[uRx]) * v_tmp_Rx * t_seg_vec
            Ycoor_Rx_evo = Ycoor_Rx_last + np.sin(direction_new[uRx]) * v_tmp_Rx * t_seg_vec
            # Calculate the angle due the the Rx self-rotation
            vector_Tx_point_to_Rx_Xcoor =  Xcoor_Rx_last - Xcoor_Tx_last
            vector_Tx_point_to_Rx_Ycoor =  Ycoor_Rx_last - Ycoor_Tx_last
            vector_Txevo_point_to_Rxevo_Xcoor = Xcoor_Rx_evo - Xcoor_Tx_evo
            vector_Txevo_point_to_Rxevo_Ycoor = Ycoor_Rx_evo - Ycoor_Tx_evo
            tmp_cos_value = (vector_Tx_point_to_Rx_Xcoor * vector_Txevo_point_to_Rxevo_Xcoor+\
                             vector_Tx_point_to_Rx_Ycoor * vector_Txevo_point_to_Rxevo_Ycoor)/\
            (np.sqrt(vector_Tx_point_to_Rx_Xcoor**2+vector_Tx_point_to_Rx_Ycoor**2)*\
             np.sqrt(vector_Txevo_point_to_Rxevo_Xcoor**2+vector_Txevo_point_to_Rxevo_Ycoor**2))
            tmp_cos_value[np.where(tmp_cos_value>1)[0]] = 1
            intersection_angle_point_to_new_Rx = np.degrees(np.arccos(tmp_cos_value))
            intersection_angle_point_to_new_Tx = intersection_angle_point_to_new_Rx
            # Check outage and calculate outage coefficent
            if l==0:
                bw_Tx = self.env_parameter.BeamWidth_TX_vec[action.BW_ID_BS2UE_Link]
            else:
                bw_Tx = self.env_parameter.BeamWidth_RX_vec
            bw_Rx = self.env_parameter.BeamWidth_RX_vec
            outage_in_Tx_beam = bw_Tx/2 - intersection_angle_point_to_new_Rx - theta_Tx_self_rotate
            outage_in_Rx_beam = bw_Rx/2 - intersection_angle_point_to_new_Tx - theta_Rx_self_rotate
            outage_in_Tx_beam[np.where(outage_in_Tx_beam>=0)] = 1
            outage_in_Tx_beam[np.where(outage_in_Tx_beam<0)] = 0
            outage_in_Rx_beam[np.where(outage_in_Rx_beam>=0)] = 1
            outage_in_Rx_beam[np.where(outage_in_Rx_beam<0)] = 0
            violation_status = outage_in_Tx_beam * outage_in_Rx_beam
            violation_slots = np.where(violation_status==0)[0]
            if len(violation_slots) >=1:
                outage_coeff = 1 - violation_slots[0]/num_seg_slot
            else:
                outage_coeff = 0
            self.outage_coeff[uTx,uRx] = outage_coeff
            self.env_parameter.outage_coeff = self.outage_coeff
        
        return outage_coeff

    # -------------------------------------        
    # Interaction with enviroment, called within step function
    # -------------------------------------
    def enviroment(self, state, action, channel):
        # Get state
        Is_D2D_Link_Active = state.Is_D2D_Link_Active;
        Tx_ID_D2D_Link = state.Tx_ID_D2D_Link;   
        Rx_ID_D2D_Link = state.Rx_ID_D2D_Link;
        Is_Tracking = state.Is_Tracking;
        Reff_BS2UE_Link_Last_Slot = state.Reff_BS2UE_Link_Last_Slot;

        # Get action
        UE_ID_BS2UE_Link = action.UE_ID_BS2UE_Link;
        BW_ID_BS2UE_Link = action.BW_ID_BS2UE_Link;
        Is_D2D_Link_Activated_For_Next_Slot = action.Is_D2D_Link_Activated_For_Next_Slot;
        Relay_ID = action.Relay_ID;

        # Enviroment parameter
        N_SSW_BS_vec = self.env_parameter.N_SSW_BS_vec;
        N_SSW_UE_vec = self.env_parameter.N_SSW_UE_vec;
        mean_packet_size = self.env_parameter.mean_packet_size;
        W = self.env_parameter.W;
        B = self.env_parameter.B;
        t_slot = self.env_parameter.t_slot;
        t_SSW = self.env_parameter.t_SSW;
        tracking_angular_space_d = self.env_parameter.tracking_angular_space_d;
        tracking_t_SSW = self.env_parameter.tracking_t_SSW;
        tracking_t_slot = self.env_parameter.tracking_t_slot;
        Ptx_BS_dBm = self.env_parameter.Ptx_BS_dBm;    
        GtGr_dBi_mat = self.env_parameter.GtGr_dBi_mat;
        Ptx_UE_dBm = self.env_parameter.Ptx_UE_dBm;
        Gt_D2D_dBi = self.env_parameter.Gt_D2D_dBi;
        Gr_D2D_dBi = self.env_parameter.Gr_D2D_dBi;
        Pn_dBm = self.env_parameter.Pn_dBm;
        SNR_Value = self.env_parameter.SNR_Value;
        Rate_Value = self.env_parameter.Rate_Value;
        Using_MCS = self.env_parameter.Using_MCS;
        Pathloss = self.env_parameter.Pathloss; 

        # Get Channel
        if Is_D2D_Link_Activated_For_Next_Slot:
            X_BS2UE = channel[Relay_ID,Relay_ID];
        else:
            X_BS2UE = channel[UE_ID_BS2UE_Link,UE_ID_BS2UE_Link];
        if Is_D2D_Link_Active:
            X_D2D = channel[Tx_ID_D2D_Link,Rx_ID_D2D_Link];

        # Update the UE position (Mobility modeling)
        self.position_update()
        
        # Update the pathloss (Blockage modeling)
        self.pathloss_update()
        
        # Update outage event (Outage modeling)
        self.outage_coeff_update(state, action)
        
        # Beam training and data transmission for main link
        if Is_D2D_Link_Activated_For_Next_Slot:
            SNR_BS2UE_dB = \
            Ptx_BS_dBm + GtGr_dBi_mat[BW_ID_BS2UE_Link] -\
            Pathloss[Relay_ID,Relay_ID] - W - Pn_dBm + X_BS2UE;
        else:
            SNR_BS2UE_dB = \
            Ptx_BS_dBm + GtGr_dBi_mat[BW_ID_BS2UE_Link] -\
            Pathloss[UE_ID_BS2UE_Link,UE_ID_BS2UE_Link] - W - Pn_dBm + X_BS2UE;
        if Is_Tracking:
            Coeff_BS2UE = self.env_parameter.Coeff_BS2UE_Tracking[BW_ID_BS2UE_Link];
        else:
            Coeff_BS2UE = self.env_parameter.Coeff_BS2UE[BW_ID_BS2UE_Link];
        Coeff_BS2UE = Coeff_BS2UE * (1-self.outage_coeff[action.Relay_ID,action.Relay_ID]);
        if Using_MCS:
            MCS_ID_BS2UE = len(np.where(SNR_Value<=SNR_BS2UE_dB)[0])-1;
            Reff_BS2UE_Link = Coeff_BS2UE*Rate_Value[MCS_ID_BS2UE];
        else:
            Reff_BS2UE_Link = Coeff_BS2UE*B*np.log2(1+10^(SNR_BS2UE_dB/10));
        Reff_BS2UE_Link = max(Reff_BS2UE_Link,0);
        npkts_departed_BS2UE_Link = np.floor(Reff_BS2UE_Link*t_slot/mean_packet_size);

        # Beam training and data transmission for d2d link
        if Is_D2D_Link_Active:
            SNR_D2D_dB = Ptx_UE_dBm + Gt_D2D_dBi + Gr_D2D_dBi\
            - Pathloss[Tx_ID_D2D_Link,Rx_ID_D2D_Link] - W - Pn_dBm + X_D2D;
            Coeff_D2D = self.env_parameter.Coeff_D2D;
            Coeff_D2D = Coeff_D2D * (1-self.outage_coeff[Tx_ID_D2D_Link,Rx_ID_D2D_Link]);
            if Using_MCS:
                MCS_ID_D2D = len(np.where(SNR_Value<SNR_D2D_dB)[0])-1;
                Reff_D2D_Link = Coeff_D2D*Rate_Value[MCS_ID_D2D];
            else:
                Reff_D2D_Link = Coeff_D2D*B*np.log2(1+10^(SNR_D2D_dB/10));
        else:
            MCS_ID_D2D = 0;
            Reff_D2D_Link = 0;
        Reff_D2D_Link_real = min(Reff_BS2UE_Link_Last_Slot,Reff_D2D_Link);
        Reff_D2D_Link_real = max(Reff_D2D_Link_real,0);
        if Is_D2D_Link_Active:
            MCS_ID_D2D_real = np.argmin(np.abs(Reff_D2D_Link_real-Rate_Value));    
        else:
            MCS_ID_D2D_real = 0;
        npkts_departed_D2D_Link = np.floor(Reff_D2D_Link_real*t_slot/mean_packet_size);

        # Output
        output = def_output(self.env_parameter);
        output.npkts_departed_D2D_Link = npkts_departed_D2D_Link;
        output.npkts_departed_BS2UE_Link = npkts_departed_BS2UE_Link;
        output.Reff_BS2UE_Link = Reff_BS2UE_Link;
        output.Reff_D2D_Link = Reff_D2D_Link_real;
        output.action = action;
        output.MCS_ID_BS2UE = MCS_ID_BS2UE;
        output.MCS_ID_D2D = MCS_ID_D2D;
        output.MCS_ID_D2D_real = MCS_ID_D2D_real;
        output.D2D_Link_Smaller = (Reff_D2D_Link < Reff_BS2UE_Link_Last_Slot)

        # Return object
        return output
    
    # -------------------------------------
    # Calculate the delay distribution
    # This function is called after the step function, 
    # namely that the newly arrived packets are included
    # -------------------------------------
    def get_delay_distribution(self):        
        # Sanitity check of number of packets
        total_num_npkts_arrived_per_ue = np.squeeze(np.sum(self.npkts_arrival_evolution,axis=0))
        total_num_npkts_sent_per_ue = np.squeeze(np.sum(self.delay_dist,axis=0))
        total_num_npkts_queued_per_ue = np.squeeze(self.Queue[self.ct,:])
        
        # Maximum delay
        num_extra_episodes = np.ceil(total_num_npkts_queued_per_ue/total_num_npkts_sent_per_ue);
        est_max_delay = (np.max(num_extra_episodes) + 1) * self.slots_monitored
        
        # Estimated average delay (including delay occured by emptying the queues)
        est_delay_distribution = np.zeros_like(self.delay_dist)
        est_delay_distribution = self.delay_dist
        est_delay_distribution = self.delay_dist + self.current_Queue_dist_by_delay
        
        return est_delay_distribution