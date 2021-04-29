# This Python file uses the following encoding: utf-8
# coding=utf-8
import numpy as np
import copy
import pdb

def heuristic_tracking_selection(state,action,env_parameter):
    # Observable information
    est_arrival = state.est_arrival;
    est_depart = state.est_depart;
    Queue_length = state.Queue_length;
    N_UE = len(state.Queue_length);
    UE_ID_BS2UE_Link = action.UE_ID_BS2UE_Link;
    Reff_BS2UE_Estimated = state.Reff_BS2UE_Estimated;
    
    # Estimate average queue length for next time
    Queue_length_est_next_slot = np.zeros(N_UE);
    for u in range(N_UE):
        active_ue = 0;
        if u == UE_ID_BS2UE_Link:
            active_ue = 1;
        Queue_length_est_next_slot[u] = max(Queue_length[u] - est_depart[u]*active_ue,0) + est_arrival[u];
    
    # Estimated UE_ID_BS2UE_Link next time slot
    UE_ID_BS2UE_Link_est_next_slot = np.argmax(Queue_length_est_next_slot*Reff_BS2UE_Estimated);
    
    # Make decision
    # Decide to track if the UE to be served is the same as the current UE
    if UE_ID_BS2UE_Link_est_next_slot == UE_ID_BS2UE_Link:
        Is_Tracking_Required_For_Next_Slot = True
    else: 
        Is_Tracking_Required_For_Next_Slot = False
        
    # Return decision
    return Is_Tracking_Required_For_Next_Slot