import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

model_name = "5_layers_10_nodes_100_epochs_sigmoid_tt_1_hh_0"
data_path = "/afs/desy.de/user/k/kindsvat/Documents/hh2bbtautau/hbt/ml/dnn_models/gen_model_"+ model_name + "/history.parquet"
history_array = ak.from_parquet(data_path)



y= np.array(history_array["loss"])
x= np.arange(0,len(y))
plt.plot(x,y, label="loss")
a= np.array(history_array["val_loss"])
plt.plot(a, label="validation loss")
plt.legend(loc='upper left')
plt.savefig('/afs/desy.de/user/k/kindsvat/Documents/hh2bbtautau/hbt/ml/dnn_models/plots/gen_model/loss_and_accuracy/gen_model_'+ model_name +"_loss_and_val_loss")
#def plot_variable(variable, variable2):
    #y= np.array(history_array[variable])
    #x= np.arange(1,len(y)+1)
    #plt.xlabel("number of epochs")
    #plt.ylabel("loss functions")
    #plt.plot(x,y, label="loss")
    #a= np.array(history_array[variable2])
   # b= np.arange(1,len(a)+1)
  #  plt.plot(a,b, label="validation loss")
 #   plt.show() 
#    plt.savefig('/afs/desy.de/user/k/kindsvat/Documents/hh2bbtautau/hbt/ml/dnn_models/plots/'+ model_name +"_"+variable)




#loss
#val_loss
#binary_accuracy
#val_binary_accuracy