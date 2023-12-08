# coding: utf-8

"""
Definition of variables.
"""

import order as od

from columnflow.columnar_util import EMPTY_FLOAT


def add_variables(config: od.Config) -> None:
    """
    Adds all variables to a *config*.
    """
    config.add_variable(
        name="event",
        expression="event",
        binning=(1, 0.0, 1.0e9),
        x_title="Event number",
        discrete_x=True,
    )
    config.add_variable(
        name="run",
        expression="run",
        binning=(1, 100000.0, 500000.0),
        x_title="Run number",
        discrete_x=True,
    )
    config.add_variable(
        name="lumi",
        expression="luminosityBlock",
        binning=(1, 0.0, 5000.0),
        x_title="Luminosity block",
        discrete_x=True,
    )
    config.add_variable(
        name="n_jet",
        expression="n_jet",
        binning=(11, -0.5, 10.5),
        x_title="Number of jets",
        discrete_x=True,
    )
    config.add_variable(
        name="n_hhbtag",
        expression="n_hhbtag",
        binning=(4, -0.5, 3.5),
        x_title="Number of HH b-tags",
        discrete_x=True,
    )
    config.add_variable(
        name="ht",
        binning=[0, 80, 120, 160, 200, 240, 280, 320, 400, 500, 600, 800],
        unit="GeV",
        x_title="HT",
    )
    config.add_variable(
        name="jet1_pt",
        expression="Jet.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )
    config.add_variable(
        name="jet1_eta",
        expression="Jet.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Jet 1 $\eta$",
    )
    config.add_variable(
        name="jet2_pt",
        expression="Jet.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 2 $p_{T}$",
    )
    config.add_variable(
        name="met_phi",
        expression="MET.phi",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        x_title=r"MET $\phi$",
    )
    config.add_variable(
        name="tau1_pt",
        expression="Tau.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        x_title=r"$\tau$ $p_{T}$",
    )
    config.add_variable(
        name="tau2_pt",
        expression="Tau.pt[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Tau 2 $p_{T}$",
    )
    config.add_variable(
        name="tau1_eta",
        expression="Tau.eta[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"Tau 1 $\eta$",
    )
    config.add_variable(
        name="tau1_phi",
        expression="Tau.phi",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        x_title=r"Tau $\phi$",
    )
    config.add_variable(
        name="tau1_mass",
        expression="Tau.mass[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(10, 0.0, 5.0),
        unit="GeV",
        x_title=r"Tau 1 mass ",
    )
    config.add_variable(
        name="tau2_mass",
        expression="Tau.mass[:,1]",
        null_value=EMPTY_FLOAT,
        binning=(10, 0.0, 5.0),
        unit="GeV",
        x_title=r"Tau 2 mass ",
    )
    config.add_variable(
        name="tau_VS_e",
        expression="Tau.idDeepTau2017v2p1VSe",
        null_value=EMPTY_FLOAT,
        binning=(10, 0.0, 20),
        x_title=r"Tau VS e ",
    )
    config.add_variable(
        name="tau_VS_mu",
        expression="Tau.idDeepTau2017v2p1VSmu",
        null_value=EMPTY_FLOAT,
        binning=(10, 0.0, 20),
        x_title=r"Tau VS mu ",
    )
    config.add_variable(
        name="tau_VS_jet",
        expression="Tau.idDeepTau2017v2p1VSjet",
        null_value=EMPTY_FLOAT,
        binning=(10, 0.0, 20),
        x_title=r"Tau VS Jet ",
    )
    config.add_variable(
        name="tau_flav",
        expression="Tau.genPartFlav",
        null_value=EMPTY_FLOAT,
        binning=(10, 0.0, 20),
        x_title=r"Tau Flavor ",
    )
    config.add_variable(
        name="tau_decay",
        expression="Tau.decayMode",
        null_value=EMPTY_FLOAT,
        binning=(10, 0.0, 20),
        x_title=r"Tau decay Mode ",
    )
    config.add_variable(
        name="met_pt",
        expression="MET.pt",
        null_value=EMPTY_FLOAT,
        binning=(20, 0, 200),
        unit="GeV",
        x_title=r"MET $p_{T}$",
    )
    config.add_variable(
        name="tau1_pos_pt",
        expression="Tau_pos.pt[:,0]",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$hardest \tau^{+}$ $p_{T}$",
    )
    config.add_variable(
        name="tau_pos_pt",
        expression="Tau_pos.pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$\tau^{+}$ $p_{T}$",
    )
    config.add_variable(
        name="tau_neg_pt",
        expression="Tau_neg.pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$\tau^{-}$ $p_{T}$",
        
    )   
    
    config.add_variable(
        name="gen_tau_ele_pt",
        expression="GenTauEle.pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$\tau^{gen}_{e}$ $p_{T}$",
    )   
    
    config.add_variable(
        name="gen_tau_ele_eta",
        expression="GenTauEle.eta",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$\tau^{gen}_{e}$ $\eta$",
    )   
    
    config.add_variable(
        name="gen_tau_ele_phi",
        expression="GenTauEle.phi",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        x_title=r"$\tau^{gen}_{e}$ $\phi$",
    )   
    
    config.add_variable(
        name="gen_tau_ele_mass",
        expression="GenTauEle.mass",
        null_value=EMPTY_FLOAT,
        binning=(10, 0.0, 5.0),
        unit="GeV",
        x_title=r"$\tau^{gen}_{e}$ mass",
    ) 
    
    config.add_variable(
        name="gen_ele_from_tau_pt",
        expression="GenEleFromTau.pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$e^{gen}_{\tau}$ $p_{T}$",
    ) 

    config.add_variable(
        name="gen_ele_from_tau_eta",
        expression="GenEleFromTau.eta",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$e^{gen}_{\tau}$ $\eta$",
    ) 
    
    config.add_variable(
        name="gen_ele_from_tau_phi",
        expression="GenEleFromTau.phi",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        x_title=r"$e^{gen}_{\tau}$ $\phi$",
    ) 
    
    config.add_variable(
        name="gen_ele_from_tau_mass",
        expression="GenEleFromTau.mass",
        null_value=EMPTY_FLOAT,
        binning=(10, 0.0, 5.0),
        unit="GeV",
        x_title=r"$e^{gen}_{\tau}$ mass",
    ) 
    
    config.add_variable(
        name="det_ele_from_tau_pt",
        expression="ElectronFromTau.pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$e^{det}_{\tau}$ $p_{T}$",
    ) 
    
    config.add_variable(
        name="det_ele_from_tau_eta",
        expression="ElectronFromTau.eta",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$e^{det}_{\tau}$ $\eta$",
    ) 
    
    config.add_variable(
        name="det_ele_from_tau_phi",
        expression="ElectronFromTau.phi",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        x_title=r"$e^{det}_{\tau}$ $\phi$",
    ) 
    
    config.add_variable(
        name="det_ele_from_tau_mass",
        expression="ElectronFromTau.mass",
        null_value=EMPTY_FLOAT,
        binning=(10, 0.0, 5.0),
        unit="GeV",
        x_title=r"$e^{det}_{\tau}$ mass",
    ) 
    config.add_variable(
        name="gen_tau_muon_pt",
        expression="GenTauMuon.pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$\tau^{gen}_{\mu}$ $p_{T}$",
    )   
    
    config.add_variable(
        name="gen_tau_muon_eta",
        expression="GenTauMuon.eta",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$\tau^{gen}_{\mu}$ $\eta$",
    ) 
    
    config.add_variable(
        name="gen_tau_muon_phi",
        expression="GenTauMuon.phi",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        x_title=r"$\tau^{gen}_{\mu}$ $\phi$",
    ) 
    
    config.add_variable(
        name="gen_tau_muon_mass",
        expression="GenTauMuon.mass",
        null_value=EMPTY_FLOAT,
        binning=(10, 0.0, 5.0),
        unit="GeV",
        x_title=r"$\tau^{gen}_{\mu}$ mass",
    )  
        
    config.add_variable(
        name="gen_muon_from_tau_pt",
        expression="GenMuonFromTau.pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$\mu^{gen}_{\tau}$ $p_{T}$",
    ) 
    
    config.add_variable(
        name="gen_muon_from_tau_eta",
        expression="GenMuonFromTau.eta",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$\mu^{gen}_{\tau}$ $\eta$",
    ) 
    
    config.add_variable(
        name="gen_muon_from_tau_phi",
        expression="GenMuonFromTau.phi",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        x_title=r"$\mu^{gen}_{\tau}$ $\phi$",
    )
    
    config.add_variable(
        name="gen_muon_from_tau_mass",
        expression="GenMuonFromTau.mass",
        null_value=EMPTY_FLOAT,
        binning=(10, 0.0, 5.0),
        unit="GeV",
        x_title=r"$\mu^{gen}_{\tau}$ mass",
    ) 

    config.add_variable(
        name="det_muon_from_tau_pt",
        expression="MuonFromTau.pt",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$\mu^{det}_{\tau}$ $p_{T}$",
    ) 
    
    config.add_variable(
        name="det_muon_from_tau_eta",
        expression="MuonFromTau.eta",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$\mu^{det}_{\tau}$ $\eta$",
    ) 
    
    config.add_variable(
        name="det_muon_from_tau_phi",
        expression="MuonFromTau.phi",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        x_title=r"$\mu^{det}_{\tau}$ $\phi$",
    )
    
    config.add_variable(
        name="det_muon_from_tau_mass",
        expression="MuonFromTau.mass",
        null_value=EMPTY_FLOAT,
        binning=(10, 0.0, 5.0),
        unit="GeV",
        x_title=r"$\mu^{det}_{\tau}$ mass",
    ) 
    
    config.add_variable(
        name="gen_z_e",
        expression="gen_z_e",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 1),
        x_title=r"$z_{e}^{gen}$",
    ) 
    config.add_variable(
        name="gen_z_pos_e",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 1),
        x_title=r"$z_{e}^{gen, +}$",
    ) 
    config.add_variable(
        name="gen_z_neg_e",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 1),
        x_title=r"$z_{e}^{gen, -}$",
    ) 
    
    config.add_variable(
        name="gen_z_m",
        expression="gen_z_m",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 1),
        x_title=r"$z_{m}^{gen}$",
    ) 
    config.add_variable(
        name="gen_z_pos_m",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 1),
        x_title=r"$z_{m}^{gen, +}$",
    ) 
    config.add_variable(
        name="gen_z_neg_m",
        null_value=EMPTY_FLOAT,
        binning=(40, 0.0, 1),
        x_title=r"$z_{m}^{gen, -}$",
    )
    
    config.add_variable(
        name="gen_cos_theta_hat_e",
        null_value=EMPTY_FLOAT,
        binning=(40, -1, 1),
        x_title=r"$cos\theta_e^{gen}$",
    )

    config.add_variable(
        name="gen_cos_theta_hat_m",
        null_value=EMPTY_FLOAT,
        binning=(40, -1, 1),
        x_title=r"$cos\theta_mu^{gen}$",
    )
    
    config.add_variable(
        name="gen_cos_omega_e",
        null_value=EMPTY_FLOAT,
        binning=(40, -1, 1),
        x_title=r"$cos\omega_e^{gen}$",
    )
    
    config.add_variable(
        name="gen_cos_omega_m",
        null_value=EMPTY_FLOAT,
        binning=(40, -1, 1),
        x_title=r"$cos\omega_m^{gen}$",
    )

    config.add_variable(
        name="gen_cos_omega_e_new",
        null_value=EMPTY_FLOAT,
        binning=(40, -1, 1),
        x_title=r"$cos\omega_e^{gen}$ new",
    )
    
    config.add_variable(
        name="gen_cos_omega_m_new",
        null_value=EMPTY_FLOAT,
        binning=(40, -1, 1),
        x_title=r"$cos\omega_m^{gen}$ new",
    )

    config.add_variable(
        name="tau_pos_eta",
        expression="Tau_pos.eta",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$\tau^{+}$ $\eta$",
    )
    config.add_variable(
        name="tau_neg_eta",
        expression="Tau_neg.eta",
        null_value=EMPTY_FLOAT,
        binning=(30, -3.0, 3.0),
        x_title=r"$\tau^{-}$ $\eta$",
    )
    config.add_variable(
        name="tau_pos_phi",
        expression="Tau_pos.phi",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        x_title=r"$\tau^{+}$ $\phi$",
    )
    config.add_variable(
        name="tau_neg_phi",
        expression="Tau_neg.phi",
        null_value=EMPTY_FLOAT,
        binning=(33, -3.3, 3.3),
        x_title=r"$\tau^{-}$ $\phi$",
    )
    config.add_variable(
        name="tau_pos_mass",
        expression="Tau_pos.mass",
        null_value=EMPTY_FLOAT,
        binning=(10, 0.0, 5.0),
        unit="GeV",
        x_title=r"$\tau^{+}$ mass ",
    )
    config.add_variable(
        name="tau_neg_mass",
        expression="Tau_neg.mass",
        null_value=EMPTY_FLOAT,
        binning=(10, 0.0, 5.0),
        unit="GeV",
        x_title=r"$\tau^{-}$ mass ",
    )
    config.add_variable(
        name="tau_pos_decay",
        expression="Tau_pos.decayMode",
        null_value=EMPTY_FLOAT,
        binning=(13, -0.5, 12.5),
        x_title=r"$\tau^{+}$ decay Mode ",
    )
    config.add_variable(
        name="tau_neg_decay",
        expression="Tau_neg.decayMode",
        null_value=EMPTY_FLOAT,
        binning=(13, -0.5, 12.5),
        x_title=r"$\tau^{-}$ decay Mode ",
    )
    config.add_variable(
        name="tau_neg_deepTauVse",
        expression="Tau_neg.rawDeepTau2017v2p1VSe",
        null_value=EMPTY_FLOAT,
        binning=(20, 0.0, 1.6),
        x_title=r"$\tau^{-}$ DeepTau output (vs e)",
    )
    config.add_variable(
        name="tau_pos_deepTauVse",
        expression="Tau_pos.rawDeepTau2017v2p1VSe",
        null_value=EMPTY_FLOAT,
        binning=(20, 0.0, 1.6),
        x_title=r"$\tau^{+}$ DeepTau output (vs e)",
    )
    config.add_variable(
        name="tau_pos_deepTauVsmu",
        expression="Tau_pos.rawDeepTau2017v2p1VSmu",
        null_value=EMPTY_FLOAT,
        binning=(20, 0.0, 1.6),
        x_title=r"$\tau^{+}$ DeepTau output (vs mu)",
    )  
    config.add_variable(
        name="tau_neg_deepTauVsmu",
        expression="Tau_neg.rawDeepTau2017v2p1VSmu",
        null_value=EMPTY_FLOAT,
        binning=(20, 0.0, 1.6),
        x_title=r"$\tau^{-}$ DeepTau output (vs mu)",
    )  
    config.add_variable(
        name="tau_neg_deepTauVsjet",
        expression="Tau_neg.rawDeepTau2017v2p1VSjet",
        null_value=EMPTY_FLOAT,
        binning=(20, 0.0, 1.6),
        x_title=r"$\tau^{-}$ DeepTau output (vs jet)",
    ) 
    config.add_variable(
        name="tau_pos_deepTauVsjet",
        expression="Tau_pos.rawDeepTau2017v2p1VSjet",
        null_value=EMPTY_FLOAT,
        binning=(20, 0.0, 1.6),
        x_title=r"$\tau^{+}$ DeepTau output (vs jet)",
    ) 
    
    # weights
    config.add_variable(
        name="mc_weight",
        expression="mc_weight",
        binning=(200, -10, 10),
        x_title="MC weight",
    )
    config.add_variable(
        name="pu_weight",
        expression="pu_weight",
        binning=(40, 0, 2),
        x_title="Pileup weight",
    )
    config.add_variable(
        name="normalized_pu_weight",
        expression="normalized_pu_weight",
        binning=(40, 0, 2),
        x_title="Normalized pileup weight",
    )
    #config.add_variable(
    #    name="btag_weight",
    #    expression="btag_weight",
    #    binning=(60, 0, 3),
    #    x_title="b-tag weight",
    #)
    #config.add_variable(
    #    name="normalized_btag_weight",
    #    expression="normalized_btag_weight",
    #    binning=(60, 0, 3),
    #    x_title="Normalized b-tag weight",
    #)
    #config.add_variable(
    #    name="normalized_njet_btag_weight",
    #    expression="normalized_njet_btag_weight",
    #    binning=(60, 0, 3),
    #    x_title="$N_{jet}$ normalized b-tag weight",
    #)

    # cutflow variables
    config.add_variable(
        name="cf_njet",
        expression="cutflow.n_jet",
        binning=(17, -0.5, 16.5),
        x_title="Jet multiplicity",
        discrete_x=True,
    )
    config.add_variable(
        name="cf_ht",
        expression="cutflow.ht",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"$H_{T}$",
    )
    config.add_variable(
        name="cf_jet1_pt",
        expression="cutflow.jet1_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 1 $p_{T}$",
    )
    config.add_variable(
        name="cf_jet1_eta",
        expression="cutflow.jet1_eta",
        binning=(40, -5.0, 5.0),
        x_title=r"Jet 1 $\eta$",
    )
    config.add_variable(
        name="cf_jet1_phi",
        expression="cutflow.jet1_phi",
        binning=(32, -3.2, 3.2),
        x_title=r"Jet 1 $\phi$",
    )
    config.add_variable(
        name="cf_jet2_pt",
        expression="cutflow.jet2_pt",
        binning=(40, 0.0, 400.0),
        unit="GeV",
        x_title=r"Jet 2 $p_{T}$",
    )
