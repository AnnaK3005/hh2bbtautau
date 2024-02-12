from __future__ import annotations
import functools
from columnflow.production import Producer, producer
from columnflow.production.util import attach_coffea_behavior
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

ak = maybe_import("awkward")
np = maybe_import("numpy")


set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)

@producer(
    uses=({
        "Tau.*", "Electron.*", "Muon.*", attach_coffea_behavior,
    }),
    produces=({
        "dau.*",
    }),
)
def dau_producer(self: Producer, events: ak.Array, **kwargs):
    """
    Helper function to select DAUs.
    DAUs are daughter lepton coming from Higgs.
    The selection of electron, mkuon and taus always result in single or dual entries.
    Therefore, concatenate of single leptons with another single lepton always will result in
    a dual-lepton entry.
    There are three different dau-types: ElectronTau, MuonTau and TauTau.
    """

    # combine single-lepton arrays to create dual-lepton entries
    # filter dual at least 2 taus out
    tau_tau_mask = ak.num(events.Tau) > 1
    tau_tau = ak.mask(events.Tau, tau_tau_mask)

    # filter single taus and single electron and muon out
    single_tau_mask = ak.num(events.Tau) == 1
    tau = ak.mask(events.Tau, single_tau_mask)

    electron_tau = ak.concatenate([events.Electron, tau], axis=1)
    electron_tau_mask = ak.num(electron_tau) == 2
    electron_tau = ak.mask(electron_tau, electron_tau_mask)

    muon_tau = ak.concatenate([events.Muon, tau], axis=1)
    muon_tau_mask = ak.num(muon_tau) == 2
    muon_tau = ak.mask(muon_tau, muon_tau_mask)

    # combine different dual-lepton arrays together
    # order is preserved, since previous masking left Nones, where otherwise an entry would be
    # thus no dual-lepton entry is stacked on top of another dual-lepton
    dau = ak.drop_none(ak.concatenate((electron_tau, muon_tau, tau_tau), axis=1))
    events = set_ak_column_f32(events, "dau", dau)
    # explicitely set columns from behavior for later convenience
    print("in dau: attaching behavior")
    collections = dict()
    collections["dau"] = {"type_name": "PFCand", "check_attr": "metric_table"}
    events = self[attach_coffea_behavior](events, collections=collections)
    
    for quantity in ("energy", "px", "py", "pz"):
        print(f"... attaching {quantity}")
        events = set_ak_column_f32(events, f"dau.{quantity}", getattr(events.dau, quantity))
    print("leaving dau producer")
    return events