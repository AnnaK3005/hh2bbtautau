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

from hbt.production.dau import dau_producer
from hbt.production.tautauNN import tautauNN
from hbt.production.gen_lep_tau import gen_lep_tau

np = maybe_import("numpy")
ak = maybe_import("awkward")

# helpers
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)
set_ak_column_i32 = functools.partial(set_ak_column, value_type=np.int32)

@producer(
    uses=({
        "Electron.*", "Muon.*", "tautauNN_regression_output",
        attach_coffea_behavior, dau_producer,
    }),
    produces=({
        "RecoTau0.*", "RecoTau1.*",
    }
    ),
    exposed=True,
)
def neutrino_energies(
    self: Producer,
    events: ak.Array,
    **kwargs
) -> ak.Array:
    print("entering dau producer")
    events = self[dau_producer](events)
    # attach four momenta logic to dau candidates for convenience
    nu_1_energy = np.sqrt(ak.sum(events.tautauNN_regression_output[:, :3], axis=-1)**2)
    nu_2_energy = np.sqrt(ak.sum(events.tautauNN_regression_output[:, 3:], axis=-1)**2)

    for index, nu_energy in zip((0, 1), (nu_1_energy, nu_2_energy)):
        route = Route(f"dau.energy[:, {index}]")
        route_result = route.apply(events, EMPTY_FLOAT)
        # find non-existing entries by looking for None or EMPTY_FLOAT
        invalid_mask = (ak.is_none(route_result)) | (route_result == EMPTY_FLOAT)
        energy = route_result + nu_energy
                
        events = set_ak_column_f32(
            events,
            f"RecoTau{index}.energy",
            ak.where(~invalid_mask, energy, EMPTY_FLOAT),
        )
        
        for p_idx, p in enumerate(("px", "py", "pz")):
            momentum_route = Route(f"dau.{p}[:, {index}]")
            nu_momentum = events.tautauNN_regression_output[:, index*3+p_idx]
            momentum = momentum_route.apply(events) + nu_momentum
            events = set_ak_column_f32(
                events,
                f"RecoTau{index}.{p}",
                ak.where(~invalid_mask, momentum, EMPTY_FLOAT),
            )
    return events
            

@neutrino_energies.requires
def neutrino_energies_requires(self: Producer, reqs: dict):
    if "tautauNN" in reqs:
        return
    
    from columnflow.tasks.production import ProduceColumns
    reqs["tautauNN"] = ProduceColumns.req(self.task, producer="tautauNN")
    
@neutrino_energies.setup
def neutrino_energies_setup(self: Producer, reqs: dict, inputs: dict, reader_targets) -> None:
    reader_targets["tautauNN"] = inputs["tautauNN"]["columns"]
    
@producer(
    uses=({
        gen_lep_tau.PRODUCES, attach_coffea_behavior,
    }),
    produces=({
        # "IdealNeutrinoEle.*", "IdealNeutrinoMuon.*",
        "RecoTauEle.*", "RecoTauMuon.*", "IdealNeutrinoEle.*", "IdealNeutrinoMuon.*",
    })
)
def ideal_neutrino_energies(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
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
        for f in ("ElectronFromTau")
    })
    
    collections.update({
        f: {
            "type_name": "Muon"
        }
        for f in ("MuonFromTau")
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
    
    gen_nu_ele = events.GenTauEle - events.GenEleFromTau
    gen_nu_mu = events.GenTauMuon - events.GenMuonFromTau
    
    reco_tau_ele = events.ElectronFromTau + gen_nu_ele
    reco_tau_muon = events.MuonFromTau + gen_nu_mu
    
    invalid_mask_el = (ak.is_none(events.GenTauEle.pt)) | (events.GenTauEle.pt == EMPTY_FLOAT)
    invalid_mask_mu = (ak.is_none(events.GenTauMuon.pt)) | (events.GenTauMuon.pt == EMPTY_FLOAT)

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
    def save_output(
        events: ak.Array,
        invalid_mask: ak.Array[bool],
        output_name: str,
        output_array: ak.Array,
    ):
    
        for attribute in ("energy", "px", "py", "pz", "pt", "eta", "phi", "mass"):
            current_col = getattr(output_array, attribute)
            events = fill_real_finite_values_only(
                events,
                f"{output_name}.{attribute}",
                ~invalid_mask,
                current_col,
            )
        return events
    
    events = save_output(events, invalid_mask_el, "RecoTauEle", reco_tau_ele)
    # save the fevents, our momentum sum of the two neutrino in tau->2nu+e
    events = save_output(events, invalid_mask_el, "IdealNeutrinoEle", gen_nu_ele)
    events = save_output(events, invalid_mask_mu, "RecoTauMuon", reco_tau_muon)
    events = save_output(events, invalid_mask_mu, "IdealNeutrinoMuon", gen_nu_mu)
    
    return events
