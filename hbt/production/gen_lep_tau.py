# coding: utf-8

"""
Column production methods related to higher-level features.
"""

import functools

from columnflow.production import Producer, producer
from columnflow.production.util import attach_coffea_behavior

from columnflow.production.categories import category_ids
from columnflow.production.cms.mc_weight import mc_weight
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column


np = maybe_import("numpy")
ak = maybe_import("awkward")

# helpers
set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)
set_ak_column_i32 = functools.partial(set_ak_column, value_type=np.int32)

@producer(
    uses={"Electron.*", "Muon.*", "GenPart.*", attach_coffea_behavior},
    produces=({
        f"{field}.{var}"
        for field in [
            "GenTauEle",
            "GenEleFromTau",
            "ElectronFromTau",
            "GenTauMuon",
            "GenMuonFromTau",
            "MuonFromTau",
        ]
        for var in ["pt", "eta", "phi", "mass"]    
    } | {
        f"{field}.charge"
        for field in [
            "ElectronFromTau",
            "MuonFromTau",
        ]
    }),
    exposed=True,
)
def gen_lep_tau(self, events, **kwargs):
    # from IPython import embed; embed()
    events = self[attach_coffea_behavior](
        events,
        collections={
            f: {
                "type_name": f,
                "check_attr": "metric_table",
                # "skip_fields": "*Idx*G",
            } for f in ["Electron", "Muon", "GenPart"]
        }
    )
    
    # find GenPartons (probably GenElectrons, electrons on generator level before detector simulation)
    # that can be matched to the Electrons on detector level (i.e. after detector simulation)
    # use matching from coffea, which is DeltaR based
    gen_matched_ele = events.Electron.matched_gen.pdgId
    
    from IPython import embed
    embed()
    
    # ask GenPartons for parents, i.e. the particles that produced the GenPartons in their decay
    gen_matched_ele_tau = gen_matched_ele.distinctParent
    
    # create mask to filter out/select GenParents that are Taus (i.e. abs(pdgId) == 15)
    ele_tau_mask = abs(gen_matched_ele_tau.pdgId) == 15
    ele_tau_mask = ak.fill_none(ele_tau_mask, False)
    
    
    # GenTauEle: Generator Taus that decay into Generator Electrons
    # this is the case where ele_tau_mask is True, so create an array with ak.where, where you fill
    #   - gen_matched_ele_tau.pt if the mask is True, i.e. that GenParent is a Tau
    #   - place holder value (EMPTY_FLOAT) where mask is False, i.e. the GenParent is not a Tau
    events = set_ak_column_f32(events, "GenTauEle.pt", ak.where(ele_tau_mask, gen_matched_ele_tau.pt, EMPTY_FLOAT))
    
    # GenEleFromTau: GenPartons (i.e. mostly GenElectrons) that come from a decay of a GenTau
    # this is the case where ele_tau_mask is True, so create an array with ak.where, where you fill
    #   - gen_matched_ele.pt if the mask is True, i.e. that GenParent is a Tau
    #   - place holder value (EMPTY_FLOAT) where mask is False, i.e. the GenParent is not a Tau
    events = set_ak_column_f32(events, "GenEleFromTau.pt", ak.where(ele_tau_mask, gen_matched_ele.pt, EMPTY_FLOAT))
    
    # ElectronFromTau: Detector Electron that come from a decay of a GenTau
    # this is the case where ele_tau_mask is True, so create an array with ak.where, where you fill
    #   - gen_matched_ele.pt if the mask is True, i.e. that GenParent is a Tau
    #   - place holder value (EMPTY_FLOAT) where mask is False, i.e. the GenParent is not a Tau
    events = set_ak_column_f32(events, "ElectronFromTau.pt", ak.where(ele_tau_mask, events.Electron.pt, EMPTY_FLOAT))
    
    events = set_ak_column_f32(events, "GenTauEle.eta", ak.where(ele_tau_mask, gen_matched_ele_tau.eta, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenEleFromTau.eta", ak.where(ele_tau_mask, gen_matched_ele.eta, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "ElectronFromTau.eta", ak.where(ele_tau_mask, events.Electron.eta, EMPTY_FLOAT))
    
    
    events = set_ak_column_f32(events, "GenTauEle.phi", ak.where(ele_tau_mask, gen_matched_ele_tau.phi, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenEleFromTau.phi", ak.where(ele_tau_mask, gen_matched_ele.phi, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "ElectronFromTau.phi", ak.where(ele_tau_mask, events.Electron.phi, EMPTY_FLOAT))
    
    
    events = set_ak_column_f32(events, "GenTauEle.mass", ak.where(ele_tau_mask, gen_matched_ele_tau.mass, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenEleFromTau.mass", ak.where(ele_tau_mask, gen_matched_ele.mass, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "ElectronFromTau.mass", ak.where(ele_tau_mask, events.Electron.mass, EMPTY_FLOAT))
    
    # events = set_ak_column_f32(events, "GenTauEle.charge", ak.where(ele_tau_mask, events.Electron.charge, EMPTY_FLOAT))
    # events = set_ak_column_f32(events, "GenEleFromTau.charge", ak.where(ele_tau_mask, events.Electron.charge, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "ElectronFromTau.charge", ak.where(ele_tau_mask, events.Electron.charge, EMPTY_FLOAT))
    
    
    gen_matched_muon = events.Muon.matched_gen
    
    # ask GenPartons for parents, i.e. the particles that produced the GenPartons in their decay
    gen_matched_muon_tau = gen_matched_muon.distinctParent
    
    # create mask to filter out/select GenParents that are Taus (i.e. abs(pdgId) == 15)
    muon_tau_mask = abs(gen_matched_muon_tau.pdgId) == 15
    muon_tau_mask = ak.fill_none(muon_tau_mask, False)
    
    # GenTauEle: Generator Taus that decay into Generator Electrons
    # this is the case where ele_tau_mask is True, so create an array with ak.where, where you fill
    #   - gen_matched_ele_tau.pt if the mask is True, i.e. that GenParent is a Tau
    #   - place holder value (EMPTY_FLOAT) where mask is False, i.e. the GenParent is not a Tau
    events = set_ak_column_f32(events, "GenTauMuon.pt", ak.where(muon_tau_mask, gen_matched_muon_tau.pt, EMPTY_FLOAT))
    
    # GenEleFromTau: GenPartons (i.e. mostly GenElectrons) that come from a decay of a GenTau
    # this is the case where ele_tau_mask is True, so create an array with ak.where, where you fill
    #   - gen_matched_ele.pt if the mask is True, i.e. that GenParent is a Tau
    #   - place holder value (EMPTY_FLOAT) where mask is False, i.e. the GenParent is not a Tau
    events = set_ak_column_f32(events, "GenMuonFromTau.pt", ak.where(muon_tau_mask, gen_matched_muon.pt, EMPTY_FLOAT))
    
    # ElectronFromTau: Detector Electron that come from a decay of a GenTau
    # this is the case where ele_tau_mask is True, so create an array with ak.where, where you fill
    #   - gen_matched_ele.pt if the mask is True, i.e. that GenParent is a Tau
    #   - place holder value (EMPTY_FLOAT) where mask is False, i.e. the GenParent is not a Tau
    events = set_ak_column_f32(events, "MuonFromTau.pt", ak.where(muon_tau_mask, events.Muon.pt, EMPTY_FLOAT))
    
    events = set_ak_column_f32(events, "GenTauMuon.eta", ak.where(muon_tau_mask, gen_matched_muon_tau.eta, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenMuonFromTau.eta", ak.where(muon_tau_mask, gen_matched_muon.eta, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "MuonFromTau.eta", ak.where(muon_tau_mask, events.Muon.eta, EMPTY_FLOAT))
    
    
    events = set_ak_column_f32(events, "GenTauMuon.phi", ak.where(muon_tau_mask, gen_matched_muon_tau.phi, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenMuonFromTau.phi", ak.where(muon_tau_mask, gen_matched_muon.phi, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "MuonFromTau.phi", ak.where(muon_tau_mask, events.Muon.phi, EMPTY_FLOAT))
    
    
    events = set_ak_column_f32(events, "GenTauMuon.mass", ak.where(muon_tau_mask, gen_matched_muon_tau.mass, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenMuonFromTau.mass", ak.where(muon_tau_mask, gen_matched_muon.mass, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "MuonFromTau.mass", ak.where(muon_tau_mask, events.Muon.mass, EMPTY_FLOAT))

    # events = set_ak_column_f32(events, "GenTauMuon.charge", ak.where(muon_tau_mask, events.Muon.charge, EMPTY_FLOAT))
    # events = set_ak_column_f32(events, "GenMuonFromTau.charge", ak.where(muon_tau_mask, events.Muon.charge, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "MuonFromTau.charge", ak.where(muon_tau_mask, events.Muon.charge, EMPTY_FLOAT))

    return events