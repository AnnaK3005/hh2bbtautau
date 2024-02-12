# coding: utf-8

"""
Column production methods related to higher-level features.
"""
from __future__ import annotations
import numpy as np

import functools

from columnflow.production import Producer, producer
from columnflow.production.util import attach_coffea_behavior

from columnflow.production.categories import category_ids
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column
from columnflow.types import Union

from hbt.production.gen_lep_tau import gen_lep_tau
from hbt.production.neutrino_energies import ideal_neutrino_energies
np = maybe_import("numpy")
ak = maybe_import("awkward")

# helpers
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)
set_ak_column_i32 = functools.partial(set_ak_column, value_type=np.int32)

@producer(
    uses=({
        "Electron.*", "Muon.*", gen_lep_tau.PRODUCES, attach_coffea_behavior,
        ideal_neutrino_energies,
    }),
    produces = ({
     ideal_neutrino_energies,   
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
    events = self[ideal_neutrino_energies](events, **kwargs)
    positive_mask = events.ElectronFromTau.charge > 0
    negative_mask = events.ElectronFromTau.charge < 0
    gen_matched_ele = events.GenEleFromTau
    gen_matched_ele_tau = events.GenTauEle
    
    positive_mask_mu = events.MuonFromTau.charge > 0
    negative_mask_mu = events.MuonFromTau.charge < 0
    
    gen_matched_muon = events.GenMuonFromTau
    gen_matched_muon_tau = events.GenTauMuon
    
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

    def calculate_observables(
        events: ak.Array,
        matched_lepton: ak.Array,
        matched_lepton_tau: ak.Array,
        positive_mask: ak.Array,
        negative_mask: ak.Array,
        flavor: str="e",
        is_gen: bool=True,
        prefix: str = "",
    ):
        z = matched_lepton.energy / matched_lepton_tau.energy
        beta_lepton = matched_lepton.pvec.absolute() / matched_lepton.energy
        matched_lepton_p = np.sqrt(matched_lepton_tau.px**2 + matched_lepton_tau.py**2 + matched_lepton_tau.pz**2)
        beta_lepton_tau = matched_lepton_p / matched_lepton_tau.energy
        
        is_real_lepton_tau = matched_lepton_tau.pt != EMPTY_FLOAT
        
        #pos_mask=ak.fill_none(ak.flatten(events.Electron.mass, axis=1), EMPTY_FLOAT)>=-1 & <=1
        #pos_ele_mass=(ak.fill_none(ak.flatten(events.Electron.mass, axis=1), EMPTY_FLOAT))[pos_mask]
        lepton_mass = 0
        tau_mass = 0

        if is_gen:
            # if we study generator level stuff, use pdg mass for leptons
            tau_mass = 1.77686
            if flavor == "e":
                lepton_mass = 0.00051099895
            elif flavor == "m":
                lepton_mass = 0.1056583755
            else:
                raise ValueError(f"Cannot calculate spin observables for flavor {flavor}, only 'e' and 'mu' allowed")
        else:
            lepton_mass = matched_lepton.mass
            tau_mass = matched_lepton_tau.mass
        
        a = lepton_mass/tau_mass
        cos_theta_hat = (2*z - 1 - a**2)/(beta_lepton_tau*(1-a**2))
        cos_omega = (1 - a**2 + (1 + a**2)*cos_theta_hat)/(1 + a**2 + (1 - a**2)*cos_theta_hat)
        
        gamma_lepton_tau =1/np.sqrt(1-beta_lepton_tau**2)
        sin_theta = np.sqrt(1-cos_theta_hat**2)
        cos_omega_new = (1 - a**2 + (1 + a**2)*beta_lepton_tau*cos_theta_hat)/np.sqrt((cos_theta_hat**2 + gamma_lepton_tau**(-2)*sin_theta**2)*(1-a**2)**2+2*(1-a**4)*beta_lepton_tau*cos_theta_hat+beta_lepton_tau**2*(1+a**2)**2)
        
        if is_gen and prefix == "":
            prefix = "gen_"
        
        events = fill_real_finite_values_only(events, f"{prefix}z_{flavor}", is_real_lepton_tau, z)

        events = fill_real_finite_values_only(events, f"{prefix}cos_theta_hat_{flavor}", is_real_lepton_tau, cos_theta_hat)
        
        events = fill_real_finite_values_only(events, f"{prefix}cos_omega_{flavor}", is_real_lepton_tau, cos_omega)
        
        events = fill_real_finite_values_only(events, f"{prefix}cos_omega_{flavor}_new", is_real_lepton_tau, cos_omega_new)

        events = set_ak_column_f32(events, f"{prefix}z_pos_{flavor}", ak.where((is_real_lepton_tau)&positive_mask, z, EMPTY_FLOAT))
        events = set_ak_column_f32(events, f"{prefix}z_neg_{flavor}", ak.where((is_real_lepton_tau)&negative_mask, z, EMPTY_FLOAT))
        return events
    
    events = calculate_observables(
        events,
        gen_matched_ele,
        gen_matched_ele_tau,
        positive_mask=positive_mask,
        negative_mask=negative_mask,
        flavor="e",
        is_gen=True
    )
    events = calculate_observables(
        events,
        gen_matched_muon,
        gen_matched_muon_tau,
        positive_mask=positive_mask_mu,
        negative_mask=negative_mask_mu,
        flavor="m",
        is_gen=True
    )
    
    # calculate detector observables
    events = calculate_observables(
        events,
        events.ElectronFromTau,
        events.RecoTauEle,
        positive_mask=positive_mask,
        negative_mask=negative_mask,
        flavor="e",
        is_gen=False,
        prefix="ideal_reco_",
    )
    events = calculate_observables(
        events,
        events.MuonFromTau,
        events.RecoTauMuon,
        positive_mask=positive_mask_mu,
        negative_mask=negative_mask_mu,
        flavor="m",
        is_gen=False,
        prefix="ideal_reco_",
    )
    return events

@spin_observables.init
def spin_observables_init(self):
    cols = {
        "z{version}".format(version=v)
        for v in ("", "_pos", "_neg")
    }
    
    cols = {x+"_{flavor}" for x in cols}
    
    cols |= {
        "cos_omega_{flavor}",
        "cos_theta_hat_{flavor}",
        "cos_omega_{flavor}_new",
    }
    self.produces |= ({
        "{prefix}{x}".format(prefix = p, x = x.format(flavor=f))
        for x in cols
        for p in ("gen_", "ideal_reco_",)
        for f in ("e", "m")    
    })
    
