# This Python file uses the following encoding: utf-8
# coding=utf-8
import numpy as np
import sys
import operator 
import math
import pdb


#----------------------------------------------------
# class of state
#----------------------------------------------------
class def_state():
    def __init__(self,env_parameter):
        self.Queue_length = np.zeros(env_parameter.N_UE);
        self.N_UE = env_parameter.N_UE;
        self.Is_D2D_Link_Active = False;
        self.Tx_ID_D2D_Link = -1;
        self.Rx_ID_D2D_Link = -1;
        self.Is_Tracking = False;
        self.UE_ID_BS2UE_Link_Last_Slot = -1;
        self.Reff_BS2UE_Estimated = np.zeros(env_parameter.N_UE);
        self.Reff_BS2UE_Estimated[:] = 0;
        self.n_BS2UE = np.zeros(env_parameter.N_UE);
        self.Reff_BS2UE_Tracking_Estimated = np.zeros(env_parameter.N_UE);
        self.Reff_BS2UE_Tracking_Estimated[:] = 0;
        self.n_BS2UE_Tracking = np.zeros(env_parameter.N_UE);
        self.est_arrival = np.zeros(env_parameter.N_UE);
        self.est_depart = np.floor(self.Reff_BS2UE_Estimated/env_parameter.mean_packet_size);
        self.Reff_BS2UE_Link_Last_Slot = 0;
        self.prob_still_in_blockage = np.zeros(env_parameter.N_UE);
        self.ct = -1;
   
    def to_ndarray(self):
        # One hot encoding of active D2D link
        Is_D2D_Link_Active = np.zeros(self.N_UE);
        if self.Is_D2D_Link_Active:
            Is_D2D_Link_Active[self.Tx_ID_D2D_Link] = 1.0
            Is_D2D_Link_Active[self.Rx_ID_D2D_Link] = 1.0
        # One-hot encoding of tracking user
        if self.Is_Tracking:
            Is_Tracking = np.zeros(self.N_UE);
            Is_Tracking[self.UE_ID_BS2UE_Link_Last_Slot] = np.asarray([self.Is_Tracking],float)
        else:
            Is_Tracking = np.ones(self.N_UE);
        # Other information
        est_arrival = self.est_arrival;
        est_depart = self.est_depart;     
        Queue_length = self.Queue_length;
        prob_still_in_blockage = self.prob_still_in_blockage;
        #prob_still_in_blockage = np.zeros_like(self.prob_still_in_blockage)
        #state_ndarray = np.concatenate([Is_D2D_Link_Active, Is_Tracking, est_arrival, est_depart, Queue_length])
        state_ndarray = np.concatenate([Is_D2D_Link_Active, Is_Tracking, Queue_length, prob_still_in_blockage])
        return state_ndarray

    def to_ndarray_normalized(self):
        # One hot encoding of active D2D link
        Is_D2D_Link_Active = np.zeros(self.N_UE);
        if self.Is_D2D_Link_Active:
            Is_D2D_Link_Active[self.Tx_ID_D2D_Link] = 1.0
            Is_D2D_Link_Active[self.Rx_ID_D2D_Link] = 1.0
        # One-hot encoding of tracking user
        if self.Is_Tracking:
            Is_Tracking = np.zeros(self.N_UE);
            Is_Tracking[self.UE_ID_BS2UE_Link_Last_Slot] = np.asarray([self.Is_Tracking],float)
        else:
            Is_Tracking = np.ones(self.N_UE);
        # Other information
        est_arrival = self.est_arrival;
        if np.max(est_arrival) <= 1:
            est_arrival = np.ones(self.N_UE)
        else:
            est_arrival = est_arrival/np.max(est_arrival)
        est_depart = self.est_depart; 
        if np.max(est_depart) <= 1:
            est_depart = np.ones(self.N_UE)
        else:
            est_depart = est_depart/np.max(est_depart)
        Queue_length = self.Queue_length;
        if np.max(Queue_length) <= 1:
            Queue_length = np.ones(self.N_UE)
        else:
            Queue_length = Queue_length/np.max(Queue_length)
        prob_still_in_blockage = self.prob_still_in_blockage;
        #state_ndarray = np.concatenate([Is_D2D_Link_Active, Is_Tracking, est_arrival, est_depart, Queue_length])
        state_ndarray = np.concatenate([Is_D2D_Link_Active, Is_Tracking, Queue_length, prob_still_in_blockage])
    
        return state_ndarray
    
    def _ndarray_length(self):
        return len(self.to_ndarray_normalized())
    

#----------------------------------------------------
# class of action
#----------------------------------------------------
class def_action():
    def __init__(self, env_parameter, action_array = False):
        self.num_UE = int(env_parameter.N_UE)
        self.num_Relay = int(env_parameter.K_relay)
        self.num_BW = int(env_parameter.K_bw)
        self.num_Tracking = 2
        self.num_type_of_action = 4
        self.num_action_per_type = np.zeros(self.num_type_of_action,dtype=int)
        self.num_action_per_type[0] = self.num_UE
        self.num_action_per_type[1] = self.num_Relay
        self.num_action_per_type[2] = self.num_BW
        self.num_action_per_type[3] = self.num_Tracking
        self.num_actions = np.prod(self.num_action_per_type)
        self.num_actions_valide = np.prod(self.num_action_per_type) - self.num_UE*(self.num_UE-1)*self.num_BW
        
        if type(action_array)==bool:
            self.UE_ID_BS2UE_Link = -1
            self.Relay_ID = -2
            self.BW_ID_BS2UE_Link = -1
            self.Is_Tracking_Required_For_Next_Slot = False 
        else:
            self.UE_ID_BS2UE_Link = action_array[0]
            self.Relay_ID = action_array[1]
            self.BW_ID_BS2UE_Link = action_array[2]
            self.Is_Tracking_Required_For_Next_Slot = action_array[3]>0  
        self.Is_D2D_Link_Activated_For_Next_Slot = (self.UE_ID_BS2UE_Link != self.Relay_ID) and\
        (self.UE_ID_BS2UE_Link>=0) and (self.Relay_ID>=0)
        
    def update_action_with_array(self,action_ndarray):
        self.UE_ID_BS2UE_Link = int(action_ndarray[0])
        self.Relay_ID = int(action_ndarray[1])
        self.BW_ID_BS2UE_Link = int(action_ndarray[2])
        self.Is_Tracking_Required_For_Next_Slot = action_ndarray[3]>0  
        self.Is_D2D_Link_Activated_For_Next_Slot = (self.UE_ID_BS2UE_Link != self.Relay_ID) and\
        (self.UE_ID_BS2UE_Link>=0) and (self.Relay_ID>=0)
        
    def to_ndarray(self):
        UE_ID_BS2UE_Link = np.asarray([self.UE_ID_BS2UE_Link],int)
        Relay_ID = np.asarray([self.Relay_ID],int)
        BW_ID_BS2UE_Link = np.asarray([self.BW_ID_BS2UE_Link],int)
        Is_Tracking_Required_For_Next_Slot = np.asarray([self.Is_Tracking_Required_For_Next_Slot],int)
        action_ndarray = np.concatenate([UE_ID_BS2UE_Link,Relay_ID,BW_ID_BS2UE_Link,\
                                         Is_Tracking_Required_For_Next_Slot])
        return action_ndarray
    
    def _ndarray_length(self):
        return len(self.to_ndarray())

    
#-----------------------------------------------------------
# Draw an action with the current policy distribution
#-----------------------------------------------------------
def choose_action(pi_dist, env_parameter):
    action = def_action(env_parameter)
    num_type_of_action = int(action.num_type_of_action)
    num_action_per_type = action.num_action_per_type
    num_action = action.num_actions_valide
    threshold = 1e-8
    pi_dist[np.where(pi_dist<threshold)] = 0
    pi_dist = pi_dist/np.sum(pi_dist)
    if abs(np.sum(pi_dist) - 1) > 1e-5:
        sys.exit('Wrong distribution in training.py')
    else:
        pi_dist = pi_dist/np.sum(pi_dist)
        action_chosen_index = np.random.choice(np.squeeze(np.asarray(range(num_action))), size=1, p=np.squeeze(pi_dist))
    action_ndarray = action_index_to_action_ndarray(action_chosen_index,num_type_of_action,num_action_per_type)
    action.update_action_with_array(action_ndarray)
    return action, action_ndarray, np.squeeze(action_chosen_index)


#-----------------------------------------------------------
# Convert and index of action to a action ndarray
#-----------------------------------------------------------
def action_index_to_action_ndarray(action_chosen_index,num_type_of_action,num_action_per_type):
    action_ndarray = np.zeros(num_type_of_action,dtype=int)
    # Identify bw index
    num_actions_per_bw = num_action_per_type[0]*num_action_per_type[1]+num_action_per_type[0]
    action_ndarray[2] = action_chosen_index // num_actions_per_bw
    # Identify ue index
    action_chosen_index_new = action_chosen_index % num_actions_per_bw
    action_chosen_index_ue = action_chosen_index_new // num_action_per_type[1]
    action_chosen_index_relay = action_chosen_index_new % num_action_per_type[1]
    action_ndarray[1] = action_chosen_index_relay
    if action_chosen_index_ue < num_action_per_type[0]:
        action_ndarray[0] = action_chosen_index_ue
        action_ndarray[3] = 0
    else:
        action_ndarray[0] = action_chosen_index_relay
        action_ndarray[3] = 1
    for i in range(num_type_of_action):
        if action_ndarray[i] >= num_action_per_type[i]:
            sys.exit('Wrong transformation from action index to action array')
    return action_ndarray


#-----------------------------------------------------------
# Convert a action ndarray to a instance of def_action
#-----------------------------------------------------------
def action_ndarray_to_action_index(action_ndarray,num_type_of_action,num_action_per_type):
    action_index = int(0)
    num_actions_per_bw = num_action_per_type[0]*num_action_per_type[1]+num_action_per_type[0]
    action_index = action_index + action_ndarray[2] * num_actions_per_bw
    action_index = action_index + action_ndarray[1]
    if action_ndarray[3] == 0:
        action_index = action_index + action_ndarray[0]*num_action_per_type[1]
    else:
        action_index = action_index + num_action_per_type[0]*num_action_per_type[1]
    return action_index


#-----------------------------------------------------------
# Check the feasibility of the made action
#-----------------------------------------------------------
def action_sanity_check(slot, action, state):
    if slot>0 and action.UE_ID_BS2UE_Link<0:
        #pdb.set_trace()
        #sys.exit('Error: constraint violated: negative UE index occurs')
        print('Error: constraint violated: negative UE index occurs')
    violation_tracking = state.Is_Tracking and \
    ((action.UE_ID_BS2UE_Link != state.UE_ID_BS2UE_Link_Last_Slot) or state.Is_D2D_Link_Active)
    if violation_tracking:
        #pdb.set_trace()
        #sys.exit('Error: constraint violated: tracking order is not performed')
        print('Error: constraint violated: tracking order is not performed')
    violation_d2d_link = state.Is_D2D_Link_Active and\
    (action.UE_ID_BS2UE_Link == state.Tx_ID_D2D_Link or\
     action.UE_ID_BS2UE_Link == state.Rx_ID_D2D_Link or\
     action.Relay_ID == state.Tx_ID_D2D_Link or\
     action.Relay_ID == state.Rx_ID_D2D_Link or\
     state.Tx_ID_D2D_Link<0 or state.Rx_ID_D2D_Link<0)
    if violation_d2d_link:
        pdb.set_trace()
        sys.exit('Error: constraint violated: UEs in D2D link cannot be scheduled')
        

#----------------------------------------------------
# class of output of enviroment
#----------------------------------------------------
class def_output():
    def __init__(self, env_parameter):
        self.npkts_departed_D2D_Link = 0;
        self.npkts_departed_BS2UE_Link = 0;
        self.Reff_BS2UE_Link = 0;
        self.Reff_D2D_Link = 0;
        self.action = def_action(env_parameter);
        self.MCS_ID_BS2UE = 0;
        self.MCS_ID_D2D = 0;
        self.MCS_ID_D2D_real = 0;
        self.Reff_BS2UE_Estimated = np.zeros(env_parameter.N_UE);
        self.Reff_BS2UE_Estimated[:] = 100e9;
        self.n_BS2UE = np.zeros(env_parameter.N_UE);
        self.est_arrival = np.zeros(env_parameter.N_UE);
        self.est_depart = np.floor(self.Reff_BS2UE_Estimated/env_parameter.mean_packet_size);