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
from scipy import stats
from scipy.stats import truncnorm

thisdir = os.path.realpath(os.path.dirname(__file__))
EMPTY_FLOAT = np.array(-9999.0)



loc_h=0.47
loc_tt=0.51
scale_h=0.2
scale_tt=0.25
size_h=26000
size_tt=34000

lower=0
upper=1

loc_h_l=0.48
loc_h_r=0.46

scale_h_l=0.2
scale_h_r=0.25



higgs_l=truncnorm((lower - loc_h_l) / scale_h_l, (upper - loc_h_l) / scale_h_l, loc=loc_h_l, scale=scale_h_l)
higgs_r=truncnorm((lower - loc_h_r) / scale_h_r, (upper - loc_h_r) / scale_h_r, loc=loc_h_r, scale=scale_h_r)
data_hh_ggf= truncnorm(higgs_l + higgs_r)
data_tt_dl=truncnorm((lower - loc_tt) / scale_tt, (upper - loc_tt) / scale_tt, loc=loc_tt, scale=scale_tt)

data_hh_ggf=data_hh_ggf.rvs(size_h)
data_tt_dl=data_tt_dl.rvs(size_tt)


def add_target_to_array(variable, *args, target_value=0, dtype=np.int8):
    # first, flatten the input arrays

    # now create new array with same strucutre as main input,
    # which is filled with target value and has data type dtype
    target_array = np.full_like(variable, target_value, dtype=dtype)
    
    # finally stack everything together
    variable_plus_target=np.column_stack((variable, target_array))
    return variable_plus_target


gen_z_neg_e_hh_ggf_plus_zero=add_target_to_array(data_hh_ggf, target_value=0)
gen_z_neg_e_tt_dl_plus_one=add_target_to_array(data_tt_dl, target_value=1)

combined_array = np.concatenate((gen_z_neg_e_hh_ggf_plus_zero, gen_z_neg_e_tt_dl_plus_one))

shuffled_array= np.random.permutation(combined_array)

input_array, output_array = np.split(shuffled_array, 2, axis=1)

input_tensor = tf.constant(input_array)
output_tensor = tf.constant(output_array)
#weights_tensor = tf.constant(weights)

dataset_x = tf.data.Dataset.from_tensor_slices((input_tensor))
dataset_y = tf.data.Dataset.from_tensor_slices((output_tensor))
#dataset_weights = tf.data.Dataset.from_tensor_slices((weights_tensor))



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
    #weights = list()
    for x, y in input.unbatch().as_numpy_iterator():
        input_features.append(x)
        labels.append(y)
        #weights.append(w)
    return np.array(input_features), np.array(labels)#, np.array(weights)
        

dataset_combined=tf.data.Dataset.zip((dataset_x, dataset_y))#, dataset_weights))
train, test = split_dataset(dataset_combined)
        
#x_train, y_train, weights_train = split_tf_dataset_into_components(train)
#x_test, y_test, weights_test = split_tf_dataset_into_components(test)

x_train, y_train = split_tf_dataset_into_components(train)
x_test, y_test = split_tf_dataset_into_components(test)

loc_higgs=47
loc_ttbar=51

epochs=100
model_name = f"proof_of_concept_model_5_layers_10_nodes_{epochs}_epochs_2_gaussians"
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

    
dnn_output_path = os.path.join(thisdir, "dnn_models","hist_proof_of_concept_model")
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
    'dnn_models', 'plots', 'proof_of_concept_model', 'ROC_plots',
    model_name
)
draw_roc(
    y_test = y_test,
    y_pred=y_pred,
    #weights=weights_test,
    output_path = output_path,
    label="DNN gen model",
    style="solid"
)

output_path = os.path.join(
    thisdir,
    'dnn_models', 'plots', 'proof_of_concept_model', 'ROC_plots',
    "energy_fractions"
)
def roc_curve_manually(y_test, y_pred, output_path):
    tpr = np.cumsum(y_test) / np.sum(y_test)
    from IPython import embed; embed()
    fpr = np.cumsum(y_pred) / np.sum(y_pred)
    AUC = np.trapz(tpr, fpr)
    
    plt.figure(1)
    plt.plot([0, 1], [0, 1])
    plt.plot(fpr, tpr, label='gen model (area = {:.3f})'.format(AUC))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(output_path)

draw_roc(
    y_test=y_test,
    y_pred=x_test,
    output_path=output_path,
    #weights=weights_test,
    label="Energy fraction",
    style="-."
)


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
        output_folder = os.path.join(thisdir, 'dnn_models', 'output_proof_of_concept_model')
        output_file_np = os.path.join(output_folder, f'model_output_{label}_mask{mask_value}.npy')
        np.save(output_file_np, output)

