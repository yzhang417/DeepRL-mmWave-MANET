class def_scheme_setting:
    def __init__(self):
        self.scheme_id = 0;
        self.Is_bandit_relay = False;
        self.Is_bandit_bw = False;
        self.Is_heuristic_tracking = False;
        self.Is_RL = False;
        self.legend = 'Max-weight + no tracking + no relay + random cb';
        self.color = 'r';
        return
        
def get_MAB_scheme_setting():
    scheme_setting_list = list();

    # Scheme 0: Max-weight + no tracking + no relay + random codebook
    scheme_setting = def_scheme_setting();
    scheme_setting.scheme_id = 0;
    scheme_setting.Is_bandit_relay = False;
    scheme_setting.Is_bandit_bw = False;
    scheme_setting.Is_heuristic_tracking = False;
    scheme_setting.Is_RL = False;
    scheme_setting.legend = 'Max-weight UE scheduling without link configurations';
    scheme_setting.color = 'g';
    scheme_setting_list.append(scheme_setting);

    # Scheme 1: Max-weight + no tracking + no relay + bandit codebook
    scheme_setting = def_scheme_setting();
    scheme_setting.scheme_id = 1;
    scheme_setting.Is_bandit_relay = False;
    scheme_setting.Is_bandit_bw = True;
    scheme_setting.Is_heuristic_tracking = False;
    scheme_setting.Is_RL = False;
    scheme_setting.legend = 'Empirical MAB-based scheduler (relay selection and tracking are disabled)';
    scheme_setting.color = 'm';
    scheme_setting_list.append(scheme_setting);

    # Scheme 2: Max-weight + no tracking + bandit relay + random codebook
    scheme_setting = def_scheme_setting();
    scheme_setting.scheme_id = 2;
    scheme_setting.Is_bandit_relay = True;
    scheme_setting.Is_bandit_bw = False;
    scheme_setting.Is_heuristic_tracking = False;
    scheme_setting.Is_RL = False;
    scheme_setting.legend = 'Empirical MAB-based scheduler (codebook selection and tracking are disabled)';
    scheme_setting.color = 'y';
    scheme_setting_list.append(scheme_setting);

    # Scheme 3: Max-weight + heuristic tracking + no relay + random codebook
    scheme_setting = def_scheme_setting();
    scheme_setting.scheme_id = 3;
    scheme_setting.Is_bandit_relay = False;
    scheme_setting.Is_bandit_bw = False;
    scheme_setting.Is_heuristic_tracking = True;
    scheme_setting.Is_RL = False;
    scheme_setting.legend = 'Empirical MAB-based scheduler (relay/codebook selections are disabled)';
    scheme_setting.color = 'c';
    scheme_setting_list.append(scheme_setting);

    # Scheme 4: Max-weight + heuristic tracking + no relay + bandit codebook
    scheme_setting = def_scheme_setting();
    scheme_setting.scheme_id = 4;
    scheme_setting.Is_bandit_relay = False;
    scheme_setting.Is_bandit_bw = True;
    scheme_setting.Is_heuristic_tracking = True;
    scheme_setting.Is_RL = False;
    scheme_setting.legend = 'Empirical MAB-based scheduler (relay selection is disabled)';
    scheme_setting.color = 'darkorange';
    scheme_setting_list.append(scheme_setting);
    
    # Scheme 5: Max-weight + no tracking + bandit relay + bandit codebook
    scheme_setting = def_scheme_setting();
    scheme_setting.scheme_id = 5;
    scheme_setting.Is_bandit_relay = True;
    scheme_setting.Is_bandit_bw = True;
    scheme_setting.Is_heuristic_tracking = False;
    scheme_setting.Is_RL = False;
    scheme_setting.legend = 'Empirical MAB-based scheduler (tracking is disabled)';
    scheme_setting.color = 'k';
    scheme_setting_list.append(scheme_setting);
    
    
    #Comment all the other scheme to only test the scheme 6
    
    #Scheme 6: Max-weight + heuristic tracking + bandit relay + bandit bw (Our first proposal)
    scheme_setting = def_scheme_setting();
    scheme_setting.scheme_id = 6;
    scheme_setting.Is_bandit_relay = True;
    scheme_setting.Is_bandit_bw = True;
    scheme_setting.Is_heuristic_tracking = True;
    scheme_setting.Is_RL = False;
    scheme_setting.legend = 'Empirical MAB-based scheduler';
    scheme_setting.color = 'b';
    scheme_setting_list.append(scheme_setting);

    # Scheme 7: Deep Reinforcement Learning (PPO)
    scheme_setting = def_scheme_setting();
    scheme_setting.scheme_id = 7;
    scheme_setting.Is_bandit_relay = False;
    scheme_setting.Is_bandit_bw = False;
    scheme_setting.Is_heuristic_tracking = False;
    scheme_setting.Is_RL = True;
    scheme_setting.legend = 'DRL-based scheduler (PPO with A2C)';
    scheme_setting.color = 'r';
    scheme_setting_list.append(scheme_setting);
    
    return scheme_setting_list