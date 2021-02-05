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
                TSIndex[k] = np.dot(RateNor,Lk)*Coeff_BS2UE[k];
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
    legible_arms = np.asarray(range(K));
    if state.Is_D2D_Link_Active:
        legible_arms[state.Tx_ID_D2D_Link] = -5
        legible_arms[state.Rx_ID_D2D_Link] = -5
        legible_arms = np.squeeze(np.where(legible_arms>=0))
    legible_It = False;        
    while operator.not_(legible_It):
        TSIndex = np.zeros(len(legible_arms));
        for k in range(len(legible_arms)):
            Lk = np.random.dirichlet(alpha_wmts[k,:]);
            TSIndex[k] = np.dot(RateNor,Lk);
        It = legible_arms[np.argmax(TSIndex)]; # Decide the arm to play It
        # UE in D2D link cannot be relay
        if state.Is_D2D_Link_Active:
            if It != state.Tx_ID_D2D_Link and It != state.Rx_ID_D2D_Link:
                legible_It = True;
        else:
            legible_It = True; 
        if operator.not_(legible_It):
            print('Something wrong in bandit_relay_selection')
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
    alpha_uwmts = bandit_bw_para.alpha_uwmts[:,:,UE_ID];
    UWMTS_Mean_Codebook = bandit_bw_para.UWMTS_Mean_Codebook[:,UE_ID];
    UWMTS_Num_Codebook_Use = bandit_bw_para.UWMTS_Num_Codebook_Use[:,UE_ID];
    alpha_uwmts[It,mt] = alpha_uwmts[It,mt] + 1;
    RateTmp = RateNor[mt]*Coeff_BS2UE[It];
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
    mt1 = output.MCS_ID_BS2UE;
    mt2 = output.MCS_ID_D2D_real;
    Coeff_BS2UE = env_parameter.Coeff_BS2UE[action.BW_ID_BS2UE_Link];
    Coeff_D2D = env_parameter.Coeff_D2D;        
    if np.random.binomial(1,Coeff_BS2UE) == 0:
        mt1 = 0;
    if np.random.binomial(1,Coeff_D2D*0.5) == 0:
        mt2 = 0;
    if state.Is_D2D_Link_Active:
        bandit_relay_para.alpha_wmts[state.Tx_ID_D2D_Link,mt2,state.Rx_ID_D2D_Link] += 1
    if operator.not_(action.Is_D2D_Link_Activated_For_Next_Slot):
        if UE_ID != action.Relay_ID:
            sys.exit('UE Id should be identical to the relay ID when no D2D link to be activated for the next slot');
        bandit_relay_para.alpha_wmts[UE_ID,mt1,UE_ID] += 1
    return bandit_relay_para