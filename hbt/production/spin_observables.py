# coding: utf-8

"""
Column production methods related to higher-level features.
"""

import numpy as np

import functools

from columnflow.production import Producer, producer
from columnflow.production.util import attach_coffea_behavior

from columnflow.production.categories import category_ids
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column

from hbt.production.gen_lep_tau import gen_lep_tau
np = maybe_import("numpy")
ak = maybe_import("awkward")

# helpers
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)
set_ak_column_i32 = functools.partial(set_ak_column, value_type=np.int32)

@producer(
    uses=({
        "Electron.*", "Muon.*", gen_lep_tau.PRODUCES, attach_coffea_behavior
    }),
    produces=({
        f"gen_z{version}_{flavor}"
        for version in ("", "_pos", "_neg")
        for flavor in ("e","m")
    } | {
        "gen_cos_theta_hat_e",
        "gen_cos_omega_m",
        "gen_cos_omega_e",
        "gen_cos_theta_hat_m",
        "gen_cos_omega_m_new",
        "gen_cos_omega_e_new",
    }
    ),
    exposed=True,
)
def spin_observables(self: Producer, events: ak.Array, **kwargs):
    
    
    collections = {
        f: {
            "type_name": "GenParticle",
        }
        for f in ("GenTauEle", "GenEleFromTau", "GenTauMuon", "GenMuonFromTau")
    }
    
    collections.update({
        f: {
            "type_name": "Electron"
        }
        for f in ("Electron", "ElectronFromTau")
    })
    
    collections.update({
        f: {
            "type_name": "Muon"
        }
        for f in ("Muon", "MuonFromTau")
    })

    for f in collections.keys():
        collections[f].update({
                "check_attr": "metric_table",
                # "skip_fields": "*Idx*G",
            })
    events = self[attach_coffea_behavior](
        events,
        collections=collections
    )

    positive_mask = events.ElectronFromTau.charge > 0
    negative_mask = events.ElectronFromTau.charge < 0
    gen_matched_ele = events.GenEleFromTau
    gen_matched_ele_tau = events.GenTauEle
    
    positive_mask_mu = events.MuonFromTau.charge > 0
    negative_mask_mu = events.MuonFromTau.charge < 0
    
    gen_matched_muon = events.GenMuonFromTau
    gen_matched_muon_tau = events.GenTauMuon
    
    gen_z_e = gen_matched_ele.energy / gen_matched_ele_tau.energy
    beta_e_tau = gen_matched_ele_tau.pvec.absolute() / gen_matched_ele_tau.energy
    
    beta_m_tau = gen_matched_muon_tau.pvec.absolute() / gen_matched_muon_tau.energy
    gen_z_m = gen_matched_muon.energy / gen_matched_muon_tau.energy
    
    is_real_gen_e_tau = gen_matched_ele_tau.pt != EMPTY_FLOAT
    is_real_gen_muon_tau = gen_matched_muon_tau.pt != EMPTY_FLOAT
    
    
    
    #pos_mask=ak.fill_none(ak.flatten(events.Electron.mass, axis=1), EMPTY_FLOAT)>=-1 & <=1
    #pos_ele_mass=(ak.fill_none(ak.flatten(events.Electron.mass, axis=1), EMPTY_FLOAT))[pos_mask]
    
    pdg_mass_electron = 0.00051099895
    pdg_mass_tau = 1.77686
    gen_a_e = pdg_mass_electron/pdg_mass_tau
    gen_cos_theta_hat_e = (2*gen_z_e - 1 - gen_a_e**2)/(beta_e_tau*(1-gen_a_e**2))
    gen_cos_omega_e = (1 - gen_a_e**2 + (1 + gen_a_e**2)*gen_cos_theta_hat_e)/(1 + gen_a_e**2 + (1 - gen_a_e**2)*gen_cos_theta_hat_e)
    
    
    pdg_mass_muon = 0.1056583755
    gen_a_m = pdg_mass_muon/pdg_mass_tau
    gen_cos_theta_hat_m = (2*gen_z_m - 1 - gen_a_m**2)/(beta_m_tau*(1-gen_a_m**2))
    gen_cos_omega_m = (1 - gen_a_m**2 + (1 + gen_a_m**2)*gen_cos_theta_hat_m)/(1 + gen_a_m**2 + (1 - gen_a_m**2)*gen_cos_theta_hat_m)

    gamma_e =1/np.sqrt(1-beta_e_tau**2)
    sin_theta_e = np.sqrt(1-gen_cos_theta_hat_e**2)
    gen_cos_omega_e_new = (1 - gen_a_e**2 + (1 + gen_a_e**2)*beta_e_tau*gen_cos_theta_hat_e)/np.sqrt((gen_cos_theta_hat_e**2 + gamma_e**(-2)*sin_theta_e**2)*(1-gen_a_e**2)**2+2*(1-gen_a_e**4)*beta_e_tau*gen_cos_theta_hat_e+beta_e_tau**2*(1+gen_a_e**2)**2)
    gamma_m=1/np.sqrt(1-beta_m_tau**2)
    sin_theta_m = np.sqrt(1-gen_cos_theta_hat_m**2)
    gen_cos_omega_m_new = (1 - gen_a_m**2 + (1 + gen_a_m**2)*beta_m_tau*gen_cos_theta_hat_m)/np.sqrt((gen_cos_theta_hat_m**2 + gamma_m**(-2)*sin_theta_m**2)*(1-gen_a_m**2)**2+2*(1-gen_a_m**4)*beta_m_tau*gen_cos_theta_hat_m+beta_m_tau**2*(1+gen_a_m**2)**2)
    

    
    def fill_real_finite_values_only(
        events: ak.Array,
        col_name: str,
        default_mask: ak.Array[bool],
        col_values: ak.Array,
    ):
        # check which values are actually finite
        finite_mask = np.isfinite(col_values)
        
        # the final mask consists of the decisions in 'default_mask' and the finite_mask
        final_mask = default_mask & finite_mask
        
        return set_ak_column_f32(events, col_name, ak.where(final_mask, col_values, EMPTY_FLOAT))
    
    events = fill_real_finite_values_only(events, "gen_z_e", is_real_gen_e_tau, gen_z_e)
    events = fill_real_finite_values_only(events, "gen_z_m", is_real_gen_muon_tau, gen_z_m)

    events = fill_real_finite_values_only(events, "gen_cos_theta_hat_e", is_real_gen_e_tau, gen_cos_theta_hat_e)
    events = fill_real_finite_values_only(events, "gen_cos_theta_hat_m", is_real_gen_muon_tau, gen_cos_theta_hat_m)
    
    events = fill_real_finite_values_only(events, "gen_cos_omega_m", is_real_gen_muon_tau, gen_cos_omega_m)
    events = fill_real_finite_values_only(events, "gen_cos_omega_e", is_real_gen_e_tau, gen_cos_omega_e)
    
    events = fill_real_finite_values_only(events, "gen_cos_omega_m_new", is_real_gen_muon_tau, gen_cos_omega_m_new)
    events = fill_real_finite_values_only(events, "gen_cos_omega_e_new", is_real_gen_e_tau, gen_cos_omega_e_new)

    events = set_ak_column_f32(events, "gen_z_pos_e", ak.where((is_real_gen_e_tau)&positive_mask, gen_z_e, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "gen_z_neg_e", ak.where((is_real_gen_e_tau)&negative_mask, gen_z_e, EMPTY_FLOAT))

    events = set_ak_column_f32(events, "gen_z_pos_m", ak.where((is_real_gen_muon_tau)&positive_mask_mu, gen_z_m, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "gen_z_neg_m", ak.where((is_real_gen_muon_tau)&negative_mask_mu, gen_z_m, EMPTY_FLOAT))    
    return events
    
