import tensorflow as tf
import keras
from keras import layers
import awkward as ak
import numpy as np
import os
import create_dnn_plots as dnnplots

thisdir = os.path.realpath(os.path.dirname(__file__))

EMPTY_FLOAT = np.array(-9999.0)


data_path = {
"hh_ggf_bbtautau":["/nfs/dust/cms/user/kindsvat/cf_cache/hbt_store/analysis_hbt/cf.UniteColumns/run2_2017_nano_uhh_v11/hh_ggf_bbtautau_madgraph/nominal/calib__default/sel__default/prod__default/v1/data_0.parquet"],
"tt_dl":["/nfs/dust/cms/user/kindsvat/cf_cache/hbt_store/analysis_hbt/cf.UniteColumns/run2_2017_nano_uhh_v11/tt_dl_powheg/nominal/calib__default/sel__default/prod__default/v1/data_0.parquet",
"/nfs/dust/cms/user/kindsvat/cf_cache/hbt_store/analysis_hbt/cf.UniteColumns/run2_2017_nano_uhh_v11/tt_dl_powheg/nominal/calib__default/sel__default/prod__default/v1/data_1.parquet",
"/nfs/dust/cms/user/kindsvat/cf_cache/hbt_store/analysis_hbt/cf.UniteColumns/run2_2017_nano_uhh_v11/tt_dl_powheg/nominal/calib__default/sel__default/prod__default/v1/data_2.parquet",
"/nfs/dust/cms/user/kindsvat/cf_cache/hbt_store/analysis_hbt/cf.UniteColumns/run2_2017_nano_uhh_v11/tt_dl_powheg/nominal/calib__default/sel__default/prod__default/v1/data_3.parquet",]
}





particle_names = ["Tau_neg",
                "Tau_pos"]

input_names = [
                'decayMode',
                'dxy',
                'dz',
                'eta',
                'mass',
                'phi',
                'pt',
                'rawDeepTau2017v2p1VSe',
                'rawDeepTau2017v2p1VSjet',
                'rawDeepTau2017v2p1VSmu',
                'rawDeepTau2018v2p5VSe',
                'rawDeepTau2018v2p5VSjet',
                'rawDeepTau2018v2p5VSmu',
            ]

target_names = ['charge',]



def open_parquet_files(files):
    return ak.concatenate([ak.from_parquet(file) for file in files],axis=0)

def combine_particle_columns(ak_array, input_names, feature):
    return ak.concatenate([ak_array[name][feature] for name in input_names],axis=0)

def standardize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data-mean)/(std)

def prepare_input_data(
    data_path=data_path,
    particle_names=particle_names,
    input_names=input_names,
    target_names=target_names,
    dataset_name="tt_dl"
):
    data = open_parquet_files(data_path[dataset_name])


    target_array = combine_particle_columns(data, particle_names, target_names)
    input_array = combine_particle_columns(data, particle_names, input_names)


    all_taus_flattened_array = ak.fill_none(ak.flatten(input_array, axis=1), EMPTY_FLOAT)
    dtype = all_taus_flattened_array["dxy"].type.content
    x = ak_2_regular_array(input_array, input_names, dtype=dtype)
    y = target_array_to_output_tensor(target_array, dtype=dtype)


    

    dataset_x = tf.data.Dataset.from_tensor_slices((x))
    dataset_y = tf.data.Dataset.from_tensor_slices((y))

    return tf.data.Dataset.zip((dataset_x, dataset_y))
    


def ak_2_regular_array(input_array, input_names, dtype):
    # flatten events, such that only a 2D array with the taus and their features remain
    # use fill_none to get correct type of array
    all_taus_flattened_array = ak.fill_none(ak.flatten(input_array, axis=1), EMPTY_FLOAT)
    
    # change type of decayMode column to float -> BEWARE : this changes the ordering of the fields, now decayMode is the last one
    #all_taus_flattened_array["decayMode"] = ak.enforce_type(all_taus_flattened_array["decayMode"], all_taus_flattened_array["dxy"].type.content)
    all_taus_flattened_array["decayMode"] = ak.enforce_type(all_taus_flattened_array["decayMode"], dtype)
    
    # change ak array to structured numpy array
    structured_numpy_array = all_taus_flattened_array.to_numpy()
    
    # do preprocessing
    for feature in input_names:
        standardized = standardize(structured_numpy_array[feature])
        structured_numpy_array[feature] = standardized
    
    # change structured numpy array to numpy array and then to a tensor
    numpy_array = structured_numpy_array.view(np.float32).reshape(structured_numpy_array.shape + (-1,))
    input_tensor = tf.constant(numpy_array)
    
    # max_shape = ak.max(ak.count(input_array[input_name],axis=1))
    # padded_array = ak.fill_none(ak.to_regular(ak.pad_none(ak_array,max_shape)),     EMPTY_FLOAT)
    #from IPython import embed;embed()
  
    # tf.Tensor(ak.to_numpy(padded_array))
    
    return input_tensor


def target_array_to_output_tensor(target_array, dtype): 
    all_taus_flattened_array_out = ak.fill_none(ak.flatten(target_array, axis=1), EMPTY_FLOAT)
    # all_taus_flattened_array_out["charge"] = ak.enforce_type(all_taus_flattened_array_out["charge"], all_taus_flattened_array["dxy"].type.content)
    all_taus_flattened_array_out["charge"] = ak.enforce_type(all_taus_flattened_array_out["charge"], dtype)

    structured_numpy_array_out = all_taus_flattened_array_out.to_numpy()
    numpy_array_out = structured_numpy_array_out.view(np.float32).reshape(structured_numpy_array_out.shape + (-1,))
    numpy_array_out[numpy_array_out<0] = 0
    target_tensor = tf.constant(numpy_array_out)
    return target_tensor

def split_dataset(dataset, split_ratio=0.2, batch_size=256):
    num_samples = dataset.cardinality().numpy()

    num_test_samples = int((1-split_ratio) * num_samples)
    train= dataset.take(num_test_samples)
    test = dataset.skip(num_test_samples).batch(batch_size)
    train = train.shuffle(buffer_size=num_test_samples,reshuffle_each_iteration=True).batch(batch_size)
    
    return train, test
    
if __name__ == '__main__':
    
    dataset_combined = prepare_input_data()
    train, test = split_dataset(dataset_combined)
 

    
    
    model_name = "very_simple_model_100_epochs_6_layers_128_nodes"
    model = keras.Sequential(
        [
            layers.Dense(128, activation="relu", name="layer1"),
            layers.Dense(128, activation="relu", name="layer2"),
            layers.Dense(128, activation="relu", name="layer3"),
            layers.Dense(128, activation="relu", name="layer4"),
            layers.Dense(128, activation="relu", name="layer5"),
            layers.Dense(1, name="output"),

        ]
    )

    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=[
                    'binary_accuracy',
                    'binary_crossentropy',
                ])
    # from IPython import embed
    # embed()
    history = model.fit(train, validation_data=test, epochs=100)

    
    dnn_output_path = os.path.join(thisdir, "dnn_models")
    if not os.path.exists(dnn_output_path):
        os.makedirs(dnn_output_path)
    final_path = os.path.join(dnn_output_path, f"{model_name}")
    model.save(final_path)
    
    # save training history
    hist_array = ak.Array(history.history)
    ak.to_parquet(hist_array, os.path.join(final_path, "history.parquet"))
