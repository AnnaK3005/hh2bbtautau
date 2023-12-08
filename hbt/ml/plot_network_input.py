
import awkward as ak
import numpy as np
import os
import create_dnn_plots as dnnplots
import sklearn
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

thisdir = os.path.realpath(os.path.dirname(__file__))
EMPTY_FLOAT = np.array(-9999.0)



data_path = {
"hh_ggf_bbtautau":["/nfs/dust/cms/user/kindsvat/cf_cache/hbt_store/analysis_hbt/cf.UniteColumns/run2_2017_nano_uhh_v11_limited/hh_ggf_bbtautau_madgraph/nominal/calib__default/sel__onlyMET/prod__onlyMET/123/data_0.parquet"],
"tt_dl":["/nfs/dust/cms/user/kindsvat/cf_cache/hbt_store/analysis_hbt/cf.UniteColumns/run2_2017_nano_uhh_v11_limited/tt_dl_powheg/nominal/calib__default/sel__onlyMET/prod__onlyMET/123/data_0.parquet",
        "/nfs/dust/cms/user/kindsvat/cf_cache/hbt_store/analysis_hbt/cf.UniteColumns/run2_2017_nano_uhh_v11_limited/tt_dl_powheg/nominal/calib__default/sel__onlyMET/prod__onlyMET/123/data_1.parquet",]
}




input_names = [
                'gen_z_neg_e',
                #'gen_z_neg_m',
            ]


def open_parquet_files(files):
    return ak.concatenate([ak.from_parquet(file) for file in files],axis=0)


data_hh_ggf = open_parquet_files(data_path["hh_ggf_bbtautau"])
data_tt_dl = open_parquet_files(data_path["tt_dl"])


def cut_empty_values(dataset):
        ak_array=ak.concatenate([dataset[name] for name in input_names],axis=0)
        flattened_array = ak.fill_none(ak.flatten(ak_array, axis=1), EMPTY_FLOAT)
        structured_numpy_array = flattened_array.to_numpy()
        no_empty_mask=structured_numpy_array>0
        variable_no_empty=structured_numpy_array[no_empty_mask]
        return variable_no_empty



def load_filtered_data(data, column, threshold = 0):
    
    # first, filter out invalid numbers. In case of energy fraction z,
    # value must be in range [0, 1]
    array = data[column]
    array = array[array >= threshold]
    # next, only select events where you actually have events and not
    # empty arrays
    event_mask = ak.num(array, axis=-1) > 0
    
    return event_mask, array[event_mask]
    

count_hh_mask, gen_z_neg_e_hh_ggf = load_filtered_data(data_hh_ggf, "gen_z_neg_e")
count_tt_dl_mask, gen_z_neg_e_tt_dl = load_filtered_data(data_tt_dl, "gen_z_neg_e")

gen_z_neg_e_hh_ggf_flattened = ak.flatten(gen_z_neg_e_hh_ggf, axis=1)
gen_z_neg_e_tt_dl_flattened = ak.flatten(gen_z_neg_e_tt_dl, axis=1)


numpy_h = ak.to_numpy(gen_z_neg_e_hh_ggf_flattened)
numpy_tt = ak.to_numpy(gen_z_neg_e_tt_dl_flattened)



n_bins_h = 100
n_bins_tt = 100


h_count, edges, _ = plt.hist(numpy_h, n_bins_h, density=True, color='b', range=[0,1])
t_count, edges, _ = plt.hist(numpy_tt, n_bins_tt, density=True, color='g', range=[0,1])

#plt.savefig('/afs/desy.de/user/k/kindsvat/Documents/hh2bbtautau/hbt/ml/dnn_models/plots/proof_of_concept_model/network_input_gen_lvl')


def plot_z_input_ratios():
    h_count, edges, _ = plt.hist(numpy_h, n_bins_h, density=True, color='b', range=[0,1])
    t_count, edges, _ = plt.hist(numpy_tt, n_bins_tt, density=True, color='g', range=[0,1])
    tt_mask = t_count > 0
    t_count = t_count[tt_mask]
    h_count = h_count[tt_mask]
    z_ratios = h_count / t_count
    xcenters = (edges[:-1] + edges[1:]) / 2
    xcenters = xcenters[tt_mask]
    plt.clf()
    plt.plot(xcenters, z_ratios)
    plt.savefig('/afs/desy.de/user/k/kindsvat/Documents/hh2bbtautau/hbt/ml/dnn_models/plots/gen_model/output_hist/z_input_ratios')


plot_z_input_ratios()
