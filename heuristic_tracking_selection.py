import numpy as np
import copy
import pdb

def heuristic_tracking_selection(state,action):
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
        Queue_length_est_next_slot[u] = Queue_length[u] + \
        max(Queue_length[u] + est_arrival[u] - est_depart[u]*active_ue,0);
    
    # Estimated UE_ID_BS2UE_Link next time slot
    UE_ID_BS2UE_Link_est_next_slot = np.argmax(Queue_length_est_next_slot*Reff_BS2UE_Estimated);
    
    # Estimate average queue length for next two time
    Queue_length_est_next_next_slot_with_tracking = np.zeros(N_UE);
    Queue_length_est_next_next_slot_without_tracking = np.zeros(N_UE);
    for u in range(N_UE):
        active_ue_with_tracking = 0;
        active_ue_without_tracking = 0;
        # Estimate average queue length for next two time slots without tracking
        if u == UE_ID_BS2UE_Link_est_next_slot:
            active_ue_without_tracking = 1;
        Queue_length_est_next_next_slot_without_tracking[u] = \
        max(Queue_length_est_next_slot[u] + est_arrival[u] - est_depart[u]*active_ue_without_tracking,0);
        # Estimate average queue length for next two time slots with tracking
        if u == UE_ID_BS2UE_Link:
            active_ue_with_tracking = 1;
        Queue_length_est_next_next_slot_with_tracking[u] = \
            + max(Queue_length_est_next_slot[u] + est_arrival[u] - est_depart[u]*active_ue_with_tracking,0);
    
    # Make decision
    if np.sum(Queue_length_est_next_next_slot_with_tracking) < \
    np.sum(Queue_length_est_next_next_slot_without_tracking):
        Is_Tracking_Required_For_Next_Slot = True;
    else: 
        Is_Tracking_Required_For_Next_Slot = False;
    
    return Is_Tracking_Required_For_Next_Slot