# This Python file uses the following encoding: utf-8
# coding=utf-8
import numpy as np
import sys
import operator 
import math
import pdb

#-------------------------------------------------------------------------
# parameter for bw bandit 
#-------------------------------------------------------------------------   
class def_bandit_bw_parameter():
    def __init__(self, env_parameter):
        K_bw = env_parameter.K_bw;
        M = env_parameter.M;
        N_UE = env_parameter.N_UE;
        self.alpha_uwmts = np.ones((K_bw,M+1,N_UE));  
        self.UWMTS_CountLeader = np.zeros((K_bw,N_UE));
        self.UWMTS_Num_Codebook_Use = np.zeros((K_bw,N_UE));
        self.UWMTS_Mean_Codebook = np.zeros((K_bw,N_UE));
        self.UWMTS_Mean_Codebook[:] = math.inf;
        
        
#-------------------------------------------------------------------------
# parameter for relay bandit 
#-------------------------------------------------------------------------        
class def_bandit_relay_parameter():
    def __init__(self,env_parameter):
        K_relay = env_parameter.K_relay;
        M = env_parameter.M;
        N_UE = env_parameter.N_UE;        
        self.alpha_wmts = np.ones((K_relay,M+1,N_UE));
        self.WMTS_Num_Relay_Use = np.sum((self.alpha_wmts-1),axis=1);

        
#-------------------------------------------------------------------------
# Bandit BW selection
#------------------------------------------------------------------------- 
def bandit_bw_selection(bandit_bw_para, action, env_parameter):
    is_unimodal = False;
    K = env_parameter.K_bw;
    UE_ID = action.Relay_ID;
    Coeff_BS2UE = env_parameter.Coeff_BS2UE;
    UWMTS_CountLeader = bandit_bw_para.UWMTS_CountLeader[:,UE_ID];
    alpha_uwmts = bandit_bw_para.alpha_uwmts[:,:,UE_ID];
    UWMTS_Mean_Codebook = bandit_bw_para.UWMTS_Mean_Codebook[:,UE_ID];
    UWMTS_Num_Codebook_Use = bandit_bw_para.UWMTS_Num_Codebook_Use[:,UE_ID];
    RateNor = env_parameter.RateNor;
    if is_unimodal:
        gamma_UWMTS = env_parameter.gamma_UWMTS;
    else:
        gamma_UWMTS = 10e9;
    if np.isinf(np.amax(UWMTS_Mean_Codebook)):
        It = np.argmax(UWMTS_Mean_Codebook)
        UWMTS_CountLeader[It] = 0;
    elif np.amin(UWMTS_Num_Codebook_Use) < env_parameter.min_use_per_cb_guaranteed:
        It = np.argmin(UWMTS_Num_Codebook_Use)
    else:
        TSIndex = np.zeros(K);
        Leadert = np.argmax(UWMTS_Mean_Codebook); # Decide leader at index t
        UWMTS_CountLeader[Leadert] += 1
        if (UWMTS_CountLeader[Leadert] % gamma_UWMTS) == 0:
            It = Leadert; # Decide the arm to play It
        else:
            if is_unimodal:
                neighbor = np.array([Leadert-1, Leadert, Leadert+1]);
            else:
                neighbor = np.array(range(0,K));
            for k in np.intersect1d(neighbor,np.array(range(0,K))):
                Lk = np.random.dirichlet(alpha_uwmts[k,:]);                          
                #TSIndex[k] = np.dot(RateNor,Lk)*Coeff_BS2UE[k];
                TSIndex[k] = np.dot(RateNor,Lk);
            It = np.argmax(TSIndex); # Decide the arm to play It within the neighbors
            if len(np.intersect1d(It,neighbor)) == 0:
                sys.exit('Error in not choosing arm within neighbor of leaders in UWMTS algorithm');
    bandit_bw_para.UWMTS_CountLeader[:,UE_ID] = UWMTS_CountLeader;
    bandit_bw_para.alpha_uwmts[:,:,UE_ID] = alpha_uwmts;
    bandit_bw_para.UWMTS_Mean_Codebook[:,UE_ID] = UWMTS_Mean_Codebook;
    bandit_bw_para.UWMTS_Num_Codebook_Use[:,UE_ID] = UWMTS_Num_Codebook_Use;
    return It, bandit_bw_para


#-------------------------------------------------------------------------
# Bandit relay selection
#------------------------------------------------------------------------- 
def bandit_relay_selection(bandit_relay_para, state, action, env_parameter):
    K = env_parameter.K_relay;
    RateNor = env_parameter.RateNor;
    UE_ID = action.UE_ID_BS2UE_Link;
    alpha_wmts = bandit_relay_para.alpha_wmts[:,:,UE_ID];
    WMTS_Num_Relay_Use = bandit_relay_para.WMTS_Num_Relay_Use[:,UE_ID];
    legible_arms = np.asarray(range(K));
    if state.Is_D2D_Link_Active:
        legible_arms[state.Tx_ID_D2D_Link] = -5
        legible_arms[state.Rx_ID_D2D_Link] = -5
        legible_arms = np.squeeze(np.where(legible_arms>=0))
    if np.amin(WMTS_Num_Relay_Use[legible_arms]) < env_parameter.min_use_per_relay_guaranteed:
        It = legible_arms[np.argmin(WMTS_Num_Relay_Use[legible_arms])]
        if state.Is_D2D_Link_Active:
            if It != state.Tx_ID_D2D_Link and It != state.Rx_ID_D2D_Link:
                legible_It = True;
        else:
            legible_It = True;
        if operator.not_(legible_It):
            print('Something wrong in bandit_relay_selection() part 1')
    else:
        legible_It = False;        
        while operator.not_(legible_It):
            TSIndex = np.zeros(len(legible_arms));
            for k in range(len(legible_arms)):
                Lk = np.random.dirichlet(alpha_wmts[k,:]);
                TSIndex[k] = np.dot(RateNor,Lk);
            It = legible_arms[np.argmax(TSIndex)]; # Decide the arm to play It
            # UE in D2D link cannot be relay
            if state.Is_D2D_Link_Active:
                if It != state.Tx_ID_D2D_Link and It != state.Rx_ID_D2D_Link and \
                WMTS_Num_Relay_Use[It] >= env_parameter.min_use_per_relay_guaranteed:
                    legible_It = True;
            else:
                if WMTS_Num_Relay_Use[It] >= env_parameter.min_use_per_relay_guaranteed:
                    legible_It = True; 
            if operator.not_(legible_It):
                print('Something wrong in bandit_relay_selection() part 2')
    return It


#-------------------------------------------------------------------------
# Bandit parameter update for bw selection
#------------------------------------------------------------------------- 
def update_bandit_bw_para(bandit_bw_para, action, output, env_parameter):
    RateNor = env_parameter.RateNor;
    Coeff_BS2UE = env_parameter.Coeff_BS2UE;
    mt = output.MCS_ID_BS2UE;
    UE_ID = action.UE_ID_BS2UE_Link;
    It = action.BW_ID_BS2UE_Link;
    #-----Randomization-----
    Coeff_Eff = env_parameter.Coeff_BS2UE[It] * (1-env_parameter.outage_coeff[UE_ID,UE_ID]);
    #Coeff_Eff = (1-env_parameter.outage_coeff[UE_ID,UE_ID]);
    if np.random.binomial(1,Coeff_Eff) == 0:
        mt = 0; 
    #-----Randomization-----
    alpha_uwmts = bandit_bw_para.alpha_uwmts[:,:,UE_ID];
    UWMTS_Mean_Codebook = bandit_bw_para.UWMTS_Mean_Codebook[:,UE_ID];
    UWMTS_Num_Codebook_Use = bandit_bw_para.UWMTS_Num_Codebook_Use[:,UE_ID];
    alpha_uwmts[It,mt] = alpha_uwmts[It,mt] + 1;
    RateTmp = RateNor[mt]*Coeff_Eff;
    if np.isinf(UWMTS_Mean_Codebook[It]):
        UWMTS_Mean_Codebook[It] = RateTmp;
    else:
        UWMTS_Mean_Codebook[It] = \
        (UWMTS_Mean_Codebook[It]*UWMTS_Num_Codebook_Use[It] + RateTmp)/(UWMTS_Num_Codebook_Use[It]+1);
    UWMTS_Num_Codebook_Use[It] += 1
    bandit_bw_para.alpha_uwmts[:,:,UE_ID] = alpha_uwmts;
    bandit_bw_para.UWMTS_Mean_Codebook[:,UE_ID] = UWMTS_Mean_Codebook;
    bandit_bw_para.UWMTS_Num_Codebook_Use[:,UE_ID] = UWMTS_Num_Codebook_Use;
    return bandit_bw_para


#-------------------------------------------------------------------------
# Bandit parameter update for relay selection
#------------------------------------------------------------------------- 
def update_bandit_relay_para(bandit_relay_para,state,action,output,env_parameter):
    UE_ID = output.action.UE_ID_BS2UE_Link;
    mMain = output.MCS_ID_BS2UE;
    D2DlinkSmaller = True
    #-----Randomization-----  
    Coeff_Eff_Main = \
    env_parameter.Coeff_BS2UE[action.BW_ID_BS2UE_Link] * (1-env_parameter.outage_coeff[UE_ID,UE_ID]);
    if np.random.binomial(1,Coeff_Eff_Main) == 0:
        mMain = 0;
    if output.D2D_Link_Smaller:
        Coeff_Eff_D2D = \
        env_parameter.Coeff_D2D * (1-env_parameter.outage_coeff[state.Tx_ID_D2D_Link,state.Rx_ID_D2D_Link]);
        #mt2 = output.MCS_ID_D2D_real;
        mD2D = output.MCS_ID_D2D;
        if np.random.binomial(1,Coeff_Eff_D2D*0.5) == 0:
            mD2D = 0;
    else:
        Last_BW_ID_BS2UE_Link = output.Last_BW_ID_BS2UE_Link
        Coeff_Eff_Main_Last = \
        env_parameter.Coeff_BS2UE[Last_BW_ID_BS2UE_Link] * \
        (1-env_parameter.last_outage_coeff[state.Tx_ID_D2D_Link,state.Tx_ID_D2D_Link]);
        mD2D = output.last_MCS_ID_BS2UE
        if np.random.binomial(1,Coeff_Eff_Main_Last*0.5) == 0:
            mD2D = 0;
    #-----Randomization-----    
    if state.Is_D2D_Link_Active:
        bandit_relay_para.alpha_wmts[state.Tx_ID_D2D_Link,mD2D,state.Rx_ID_D2D_Link] += 1
    if operator.not_(action.Is_D2D_Link_Activated_For_Next_Slot):
        if UE_ID != action.Relay_ID:
            sys.exit('UE Id should be identical to the relay ID when no D2D link to be activated for the next slot');
        bandit_relay_para.alpha_wmts[UE_ID,mMain,UE_ID] += 1
    bandit_relay_para.WMTS_Num_Relay_Use = np.sum((bandit_relay_para.alpha_wmts-1),axis=1);
    return bandit_relay_para