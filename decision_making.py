import numpy as np
import sys
import operator 
import math
import pdb
import copy
import bandit_function as bf
from heuristic_tracking_selection import heuristic_tracking_selection

def decision_making(loop, ct, scheme_id, state_list, action_list, lastOutput_list,\
                    bandit_bw_para_list, bandit_relay_para_list, scheme_setting_list, env_parameter):
    
    # Queue length
    Queue_length_copy = copy.deepcopy(state_list[scheme_id].Queue_length).astype(np.float64);
    n_BS2UE_copy = copy.deepcopy(state_list[scheme_id].n_BS2UE);
    
    # Last activated user
    state_list[scheme_id].UE_ID_BS2UE_Link_Last_Slot = lastOutput_list[scheme_id].action.UE_ID_BS2UE_Link;
    
    # Check whether there is D2D link from last slot behavior
    state_list[scheme_id].Is_D2D_Link_Active = lastOutput_list[scheme_id].action.Is_D2D_Link_Activated_For_Next_Slot;
    if state_list[scheme_id].Is_D2D_Link_Active:
        state_list[scheme_id].Tx_ID_D2D_Link = lastOutput_list[scheme_id].action.Relay_ID;
        state_list[scheme_id].Rx_ID_D2D_Link = lastOutput_list[scheme_id].action.UE_ID_BS2UE_Link;
        Queue_length_copy[state_list[scheme_id].Tx_ID_D2D_Link] = -np.inf;
        Queue_length_copy[state_list[scheme_id].Rx_ID_D2D_Link] = -np.inf;
        n_BS2UE_copy[state_list[scheme_id].Tx_ID_D2D_Link] = np.inf;
        n_BS2UE_copy[state_list[scheme_id].Rx_ID_D2D_Link] = np.inf;
    
    # Check whether this is a tracking slot
    state_list[scheme_id].Is_Tracking = lastOutput_list[scheme_id].action.Is_Tracking_Required_For_Next_Slot;

    ##---------------------------- ACTION DECICSION ----------------------------
    
    # Action on UE selection for BS2UE link
    if state_list[scheme_id].Is_Tracking and operator.not_(state_list[scheme_id].Is_D2D_Link_Active): 
        UE_ID_BS2UE_link_Current = state_list[scheme_id].UE_ID_BS2UE_Link_Last_Slot;
    else:
        v = np.amin(n_BS2UE_copy);
        if v < env_parameter.min_service_guaranteed:
            UE_ID_BS2UE_link_Current = np.argmin(n_BS2UE_copy);
        else:
            UE_ID_BS2UE_link_Current = np.argmax(Queue_length_copy*state_list[scheme_id].Reff_BS2UE_Estimated);
        if state_list[scheme_id].Is_D2D_Link_Active and \
        (UE_ID_BS2UE_link_Current == state_list[scheme_id].Tx_ID_D2D_Link or \
         UE_ID_BS2UE_link_Current == state_list[scheme_id].Rx_ID_D2D_Link):
            pdb.set_trace()
            sys.exit('Cannot serve the user who is in the current D2D link!');
    action_list[scheme_id].UE_ID_BS2UE_Link = UE_ID_BS2UE_link_Current; 
    
    # Action on relay selection
    if scheme_setting_list[scheme_id].Is_bandit_relay:
        if state_list[scheme_id].Is_Tracking:
            action_list[scheme_id].Relay_ID = action_list[scheme_id].UE_ID_BS2UE_Link;
        else:
            action_list[scheme_id].Relay_ID = \
            bf.bandit_relay_selection(bandit_relay_para_list[scheme_id],\
                                      state_list[scheme_id],\
                                      action_list[scheme_id],\
                                      env_parameter);
        if action_list[scheme_id].Relay_ID == action_list[scheme_id].UE_ID_BS2UE_Link:
            action_list[scheme_id].Is_D2D_Link_Activated_For_Next_Slot = False;
        else:
            action_list[scheme_id].Is_D2D_Link_Activated_For_Next_Slot = True; 
    else:
        action_list[scheme_id].Relay_ID = action_list[scheme_id].UE_ID_BS2UE_Link;
        action_list[scheme_id].Is_D2D_Link_Activated_For_Next_Slot = False;
    
    # Action on beamwidth selection
    if scheme_setting_list[scheme_id].Is_bandit_bw:
        action_list[scheme_id].BW_ID_BS2UE_Link, bandit_bw_para_list[scheme_id] = \
        bf.bandit_bw_selection(bandit_bw_para_list[scheme_id],action_list[scheme_id],env_parameter);
    else:
        action_list[scheme_id].BW_ID_BS2UE_Link = (ct % env_parameter.K_bw);

    # Action on tracking selection
    if scheme_setting_list[scheme_id].Is_heuristic_tracking and \
    operator.not_(action_list[scheme_id].Is_D2D_Link_Activated_For_Next_Slot) and \
    np.amin(state_list[scheme_id].n_BS2UE) >= 1:
        action_list[scheme_id].Is_Tracking_Required_For_Next_Slot = \
        heuristic_tracking_selection(state_list[scheme_id],action_list[scheme_id]);
    else: 
        action_list[scheme_id].Is_Tracking_Required_For_Next_Slot = False;