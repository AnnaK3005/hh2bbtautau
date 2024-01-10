import tensorflow as tf
import keras
from keras import layers
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
                'gen_z_pos_e',
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
    



count_hh_mask_neg, gen_z_neg_e_hh_ggf = load_filtered_data(data_hh_ggf, "gen_z_neg_e")
count_tt_dl_mask_neg, gen_z_neg_e_tt_dl = load_filtered_data(data_tt_dl, "gen_z_neg_e")

count_hh_mask_pos, gen_z_pos_e_hh_ggf = load_filtered_data(data_hh_ggf, "gen_z_pos_e")
count_tt_dl_mask_pos, gen_z_pos_e_tt_dl = load_filtered_data(data_tt_dl, "gen_z_pos_e")

#e_neg_pt_hh = data_hh_ggf.Electron["pt"]
#e_neg_pt_hh = e_neg_pt_hh[count_hh_mask_neg]

#e_neg_pt_tt = data_tt_dl.Electron["pt"]
#e_neg_pt_tt = e_neg_pt_tt[count_tt_dl_mask_neg]

def calculate_event_weights(data, mask, broad_cast_target):
    # first, get weights you want to consider, e.g. MC weights
    weights = data.mc_weight
    # apply event-level mask to select interesting weights
    weights = weights[mask]
    # now, perform broadcasting to ensure same dimensions
    # note: broadcasted weight array is first entry in list
    # returned by ak.broadcast_arrays
    broadcasted_weights = ak.broadcast_arrays(weights, broad_cast_target)[0]
    return broadcasted_weights/np.sum(broadcasted_weights)

hh_weights_neg = calculate_event_weights(data_hh_ggf, count_hh_mask_neg, gen_z_neg_e_hh_ggf)
tt_dl_weights_neg = calculate_event_weights(data_tt_dl, count_tt_dl_mask_neg, gen_z_neg_e_tt_dl)

hh_weights_pos = calculate_event_weights(data_hh_ggf, count_hh_mask_pos, gen_z_pos_e_hh_ggf)
tt_dl_weights_pos = calculate_event_weights(data_tt_dl, count_tt_dl_mask_pos, gen_z_pos_e_tt_dl)



def add_charge_information(data,variable):
    positive_mask = data.ElectronFromTau.charge > 0
    charge_array = ak.full_like(data[variable], 0)
    charge_array = charge_array[positive_mask] = 1
    flat_charge_array = ak.flatten(charge_array, axis=-1)
    return flat_charge_array

def add_target_to_array(variable, *args, charge_value, target_value=0, dtype=np.int8):
    # first, flatten the input arrays
    flat_variables = ak.flatten(variable, axis=-1)
    flat_args = [ak.flatten(x, axis=-1) for x in args]
    
    # now create new array with same strucutre as main input,
    # which is filled with target value and has data type dtype
    target_array = ak.full_like(flat_variables, target_value, dtype=dtype)
    charge_array = ak.full_like(flat_variables, charge_value, dtype=dtype)
    
    # finally stack everything together
    variable_plus_zero=np.column_stack((flat_variables, charge_array, target_array, *flat_args))
    return variable_plus_zero


gen_z_neg_e_hh_ggf_plus_one = add_target_to_array(gen_z_neg_e_hh_ggf, hh_weights_neg, charge_value=1, target_value=0)
gen_z_neg_e_tt_dl_plus_zero = add_target_to_array(gen_z_neg_e_tt_dl, tt_dl_weights_neg, charge_value=1, target_value=1)

gen_z_pos_e_hh_ggf_plus_one = add_target_to_array(gen_z_pos_e_hh_ggf, hh_weights_pos, charge_value=0, target_value=0)
gen_z_pos_e_tt_dl_plus_zero = add_target_to_array(gen_z_pos_e_tt_dl, tt_dl_weights_pos, charge_value=0, target_value=1)

combined_array = np.concatenate((gen_z_neg_e_hh_ggf_plus_one, gen_z_neg_e_tt_dl_plus_zero, gen_z_pos_e_hh_ggf_plus_one, gen_z_pos_e_tt_dl_plus_zero))

shuffled_array= np.random.permutation(combined_array)


z_array, charge_array, output_array, weights = np.split(shuffled_array, 4, axis=1)

input_array =np.column_stack((z_array, charge_array))


input_tensor = tf.constant(input_array)
output_tensor = tf.constant(output_array)
weights_tensor = tf.constant(weights)

dataset_x = tf.data.Dataset.from_tensor_slices((input_tensor))
dataset_y = tf.data.Dataset.from_tensor_slices((output_tensor))
dataset_weights = tf.data.Dataset.from_tensor_slices((weights_tensor))



def split_dataset(dataset, split_ratio=0.2, batch_size=256):
    num_samples = dataset.cardinality().numpy()

    num_test_samples = int((1-split_ratio) * num_samples)
    train= dataset.take(num_test_samples)
    test = dataset.skip(num_test_samples).batch(batch_size)
    train = train.shuffle(buffer_size=num_test_samples,reshuffle_each_iteration=True).batch(batch_size)
    
    return train, test

def split_tf_dataset_into_components(input):
    input_features = list()
    labels = list()
    weights = list()
    for x, y, w in input.unbatch().as_numpy_iterator():
        input_features.append(x)
        labels.append(y)
        weights.append(w)
    return np.array(input_features), np.array(labels), np.array(weights)
        

dataset_combined=tf.data.Dataset.zip((dataset_x, dataset_y, dataset_weights))
train, test = split_dataset(dataset_combined)
        
x_train, y_train, weights_train = split_tf_dataset_into_components(train)
x_test, y_test, weights_test = split_tf_dataset_into_components(test)



epochs=100
model_name = f"gen_model_5_layers_10_nodes_{epochs}_epochs_with_z_pos_and_charge_information"
model = keras.Sequential(
        [
            layers.Dense(1, activation=None, name="layer1"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("PReLU"),
            layers.Dense(10, activation=None, name="layer2"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("PReLU"),
            layers.Dense(10, activation=None, name="layer3"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("PReLU"),
            layers.Dense(10, activation=None, name="layer4"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("PReLU"),
            layers.Dense(1, activation="sigmoid",name="output"),

        ]
    )

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=[
                    'binary_accuracy',
                    'binary_crossentropy',
                ])


lr_scheduler_callback  = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.8,
    patience=10,
    verbose=0,
    mode='auto',
    min_delta=1e-7,
    cooldown=0,
    min_lr=0,
)

history = model.fit(train, validation_data=test, epochs=epochs, callbacks=[lr_scheduler_callback])

    
dnn_output_path = os.path.join(thisdir, "dnn_models")
if not os.path.exists(dnn_output_path):
        os.makedirs(dnn_output_path)
final_path = os.path.join(dnn_output_path, f"{model_name}")
model.save(final_path)
    
    # save training history
hist_array = ak.Array(history.history)
ak.to_parquet(hist_array, os.path.join(final_path, "history.parquet"))



def draw_roc(y_test, y_pred, output_path, label, weights=None, style="solid"):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred, sample_weight=weights)
    idx = np.argsort(fpr)
    AUC = np.trapz(tpr[idx], fpr[idx])
    
    plt.figure(1)
    plt.plot([0, 1], [0, 1], linestyle="dashed", color="black", linewidth=1)
    plt.plot(fpr, tpr, label=f'{label} (area = {AUC:.3f})', linestyle=style, linewidth=2)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(output_path)


mask_class0 = y_test == 0
mask_class1 = y_test == 1

y_pred = model.predict(x_test).ravel()
output_path = os.path.join(
    thisdir,
    'dnn_models', 'plots', 'gen_model_with_z_pos', 'ROC_plots',
    model_name
)
draw_roc(
    y_test = y_test,
    y_pred=y_pred,
    weights=weights_test,
    output_path = output_path,
    label="DNN gen model",
    style="solid"
)

output_path = os.path.join(
    thisdir,
    'dnn_models', 'plots', 'gen_model_with_z_pos', 'ROC_plots', model_name +
    "_energy_fractions"
)

draw_roc(
    y_test=output_array,
    y_pred=z_array,
    output_path=output_path,
    weights=weights,
    label="Energy fraction",
    style="-."
)

plt.clf()

def plot_loss():
    y= np.array(hist_array["loss"])
    x= np.arange(0,len(y))
    plt.plot(x,y, label="loss")
    a= np.array(hist_array["val_loss"])
    plt.plot(a, label="validation loss")
    plt.legend(loc='upper left')
    plt.savefig('/afs/desy.de/user/k/kindsvat/Documents/hh2bbtautau/hbt/ml/dnn_models/plots/gen_model_with_z_pos/loss_and_accuracy/gen_model_'+ model_name +"_loss_and_val_loss")
    
def plot_accuracy():
    y= np.array(hist_array["binary_accuracy"])
    x= np.arange(0,len(y))
    plt.plot(x,y, label="binary accuracy")
    a= np.array(hist_array["val_binary_accuracy"])
    plt.plot(a, label="validation binary accuracy")
    plt.legend(loc='upper left')
    plt.savefig('/afs/desy.de/user/k/kindsvat/Documents/hh2bbtautau/hbt/ml/dnn_models/plots/gen_model_with_z_pos/loss_and_accuracy/gen_model_'+ model_name +"_binary_accuracy_and_val_binary_accuracy")

plot_loss()

plt.clf()

plot_accuracy()


for label, data, truth_labels in zip(
    ["train", "test"],
    [x_train, x_test],
    [y_train, y_test]
):
    for mask_value in np.unique(truth_labels):
        mask = truth_labels == mask_value
        mask = mask.flatten()
        sub_input = data[mask]
        output = model.predict(sub_input)
        output_folder = os.path.join(thisdir, 'dnn_models', 'output_gen_model_with_z_pos_charge_information')
        output_file_np = os.path.join(output_folder, f'model_output_{label}_mask{mask_value}.npy')
        np.save(output_file_np, output)

