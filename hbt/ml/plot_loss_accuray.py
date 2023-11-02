import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

model_name = "100_epochs_6_layers_128_nodes"
data_path = "/afs/desy.de/user/k/kindsvat/Documents/hh2bbtautau/hbt/ml/dnn_models/very_simple_model_"+ model_name + "/history.parquet"
history_array = ak.from_parquet(data_path)



y= np.array(history_array["binary_accuracy"])
x= np.arange(1,len(y)+1)
plt.plot(x,y, label="binary accuracy")
a= np.array(history_array["val_binary_accuracy"])
plt.plot(a, label="validation binary accuracy")
plt.legend(loc='upper left')
plt.savefig('/afs/desy.de/user/k/kindsvat/Documents/hh2bbtautau/hbt/ml/dnn_models/plots/'+ model_name +"_binary_accuracy_and_val_binary_accuracy")
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

    
#plot_variable("loss", "val_loss")


#loss
#val_loss
#binary_accuracy
#val_binary_accuracy