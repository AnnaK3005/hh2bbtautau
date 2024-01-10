import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from scipy.stats import truncnorm
import math


#tt=1
#H=0


# Histogramm erstellen
def plot_output_seperatly(data_name,target):
    loaded_output = np.load('/afs/desy.de/user/k/kindsvat/Documents/hh2bbtautau/hbt/ml/dnn_models/output_gen_model/model_output_'+ data_name +'.npy')
    plt.hist(loaded_output, bins=50, edgecolor='black')
    plt.xlabel('Modell-Output')
    plt.ylabel('Anzahl der Beobachtungen')
    plt.title('Histogramm des Modell-Outputs,'+ target)
    plt.savefig('/afs/desy.de/user/k/kindsvat/Documents/hh2bbtautau/hbt/ml/dnn_models/plots/gen_model/output_hist/model_output_' + data_name +'.png')


def plot_output_combined(model_name):
    loaded_output_test_1 = np.load('/afs/desy.de/user/k/kindsvat/Documents/hh2bbtautau/hbt/ml/dnn_models/output_' + model_name + '/model_output_test_mask1.0.npy')
    loaded_output_test_0 = np.load('/afs/desy.de/user/k/kindsvat/Documents/hh2bbtautau/hbt/ml/dnn_models/output_' + model_name + '/model_output_test_mask0.0.npy')
    loaded_output_train_1 = np.load('/afs/desy.de/user/k/kindsvat/Documents/hh2bbtautau/hbt/ml/dnn_models/output_' + model_name + '/model_output_train_mask1.0.npy')
    loaded_output_train_0 = np.load('/afs/desy.de/user/k/kindsvat/Documents/hh2bbtautau/hbt/ml/dnn_models/output_' + model_name + '/model_output_train_mask0.0.npy')
    plt.hist(loaded_output_test_1, bins=50, edgecolor='black', label='Top', color='yellow', range=(0,1), alpha=0.5)
    plt.hist(loaded_output_test_0, bins=50, edgecolor='black', label='Higgs', color='blue', range=(0,1), alpha=0.5)
    #plt.hist(loaded_output_train_1, bins=50, edgecolor='black', label='Top', color='yellow', range=(0,1), alpha=0.5)
    #plt.hist(loaded_output_train_0, bins=50, edgecolor='black', label='Higgs', color='blue', range=(0,1), alpha=0.5)
    plt.xlabel('Modell-Output')
    plt.ylabel('Anzahl der Beobachtungen')
    plt.title('Histogramm des Modell-Outputs,')
    plt.legend()
    plt.savefig('/afs/desy.de/user/k/kindsvat/Documents/hh2bbtautau/hbt/ml/dnn_models/plots/'+ model_name + '/output/model_output_combined_' + model_name +'_with_z_pos.png')


def plot_calib_curve(model_name):
    loaded_output_test_1 = np.load('/afs/desy.de/user/k/kindsvat/Documents/hh2bbtautau/hbt/ml/dnn_models/output_'+ model_name + '/model_output_test_mask1.0.npy')
    loaded_output_test_0 = np.load('/afs/desy.de/user/k/kindsvat/Documents/hh2bbtautau/hbt/ml/dnn_models/output_'+ model_name + '/model_output_test_mask0.0.npy')
    hh_counts, edges, _ = plt.hist(loaded_output_test_0, bins=50, edgecolor='black')
    tt_counts, edges, _ = plt.hist(loaded_output_test_1, bins=50, edgecolor='black')
    calib_curve = tt_counts / (tt_counts + hh_counts)
    mask=calib_curve>0
    calib_curve=calib_curve[mask]
    xcenters = (edges[:-1] + edges[1:]) / 2
    xcenters=xcenters[mask]
    plt.clf()
    plt.plot(xcenters, calib_curve, label='calibration curve')
    plt.plot([0, 1], [0, 1], linestyle="dashed", color="black", linewidth=1)
    plt.savefig('/afs/desy.de/user/k/kindsvat/Documents/hh2bbtautau/hbt/ml/dnn_models/plots/'+ model_name + '/output/calib_curve_' + model_name + '_with_z_pos.png')
    
    
def plot_z_output_ratios():
    loaded_output_test_1 = np.load('/afs/desy.de/user/k/kindsvat/Documents/hh2bbtautau/hbt/ml/dnn_models/output_gen_model/model_output_test_mask1.0.npy')
    loaded_output_test_0 = np.load('/afs/desy.de/user/k/kindsvat/Documents/hh2bbtautau/hbt/ml/dnn_models/output_gen_model/model_output_test_mask0.0.npy')
    hh_counts, edges, _ = plt.hist(loaded_output_test_0, bins=100, edgecolor='black')
    tt_counts, edges, _ = plt.hist(loaded_output_test_1, bins=100, edgecolor='black')
    tt_mask = tt_counts > 0
    tt_counts = tt_counts[tt_mask]
    hh_counts = hh_counts[tt_mask]
    z_ratios = hh_counts / tt_counts
    xcenters = (edges[:-1] + edges[1:]) / 2
    xcenters = xcenters[tt_mask]
    plt.clf()
    plt.plot(xcenters, z_ratios)
    plt.savefig('/afs/desy.de/user/k/kindsvat/Documents/hh2bbtautau/hbt/ml/dnn_models/plots/gen_model/output/z_output_ratios')
    
    
def plot_gaussians():
    lower=0
    upper=1

    loc_h_l=0.48
    loc_h_r=0.46

    loc_h=0.45
    loc_tt=0.55
    scale_h=0.2
    scale_tt=0.25
    size_h=26000
    size_tt=34000
    
    scale_h_l=0.2
    scale_h_r=0.25
    
    
    #loc_h_new= loc_h_l + loc_h_r
    #scale_h_new= math.sqrt(scale_h_l**2+scale_h_r**2)

    #higgs_l=truncnorm((lower - loc_h_l) / scale_h_l, (upper - loc_h_l) / scale_h_l, loc=loc_h_l, scale=scale_h_l)
    #higgs_r=truncnorm((lower - loc_h_r) / scale_h_r, (upper - loc_h_r) / scale_h_r, loc=loc_h_r, scale=scale_h_r)

    #data_hh_ggf = higgs_l + higgs_r


    
    #data_hh_ggf= truncnorm((lower - loc_h_new) / scale_h_new, (upper - loc_h_new) / scale_h_new, loc=loc_h_new, scale=scale_h_new)
    #data_hh_ggf= truncnorm((lower - loc_h) / scale_h, (upper - loc_h) / scale_h, loc=loc_h, scale=scale_h)
    data_tt_dl=truncnorm((lower - loc_tt) / scale_tt, (upper - loc_tt) / scale_tt, loc=loc_tt, scale=scale_tt)

    data_hh_ggf=data_hh_ggf.rvs(size_h)
    data_tt_dl=data_tt_dl.rvs(size_tt)

    count, bins, ignored = plt.hist(data_hh_ggf, n_bins_h, density=True)
    plt.plot(bins, 1/(scale_h * np.sqrt(2 * np.pi)) *

               np.exp( - (bins - loc_h)**2 / (2 * scale_h**2) ),

         linewidth=2, color='r')

    count, bins, ignored = plt.hist(data_tt_dl, n_bins_tt, density=True)
    plt.plot(bins, 1/(scale_tt * np.sqrt(2 * np.pi)) *

               np.exp( - (bins - loc_tt)**2 / (2 * scale_tt**2) ),

         linewidth=2, color='g')

    plt.savefig('/afs/desy.de/user/k/kindsvat/Documents/hh2bbtautau/hbt/ml/dnn_models/plots/proof_of_concept_model/two_gaussians.png')


n_bins_h= 50
n_bins_tt=50

plot_output_combined('gen_model_z_pos_and_neg_separate_inputs_only_mu')
 
plt.clf()

plot_calib_curve('gen_model_z_pos_and_neg_separate_inputs_only_mu')


