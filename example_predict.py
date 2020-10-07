import sys
from netCDF4 import Dataset as nc
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add, Dropout, concatenate, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Conv3D, MaxPool3D, UpSampling3D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from utils import *
from data_generator import DataGenerator
from models import *


#Reading data
try:
	print("Reading data...")
	data = []
	data.append(nc('Data/CUR_uo.nc'))
	data.append(nc('Data/CUR_vo.nc'))
	data.append(nc('Data/SAL.nc'))
	data.append(nc('Data/SSH.nc'))
except:
	print("\nData not found. Please decompress the data inside '/Data/'")
	sys.exit()


#Loading best model
filepath="saved_models/best_model.hdf5"
model = load_model(filepath)


# Extracting max and min values of each variable from training data
ts_train = 8760
maxValues, minValues = get_min_max_values_data(data, ts_train)


#Uncomment to use samples in summer
t_steps_ini = 12215
t_steps_fin = 12695 

#Uncomment to use samples in winter
# t_steps_ini = 16632
# t_steps_fin = 17136 

#Extracting variables
tempVar1 = data[0].variables['uo'][t_steps_ini:t_steps_fin,0,:,:] 
tempVar2 = data[1].variables['vo'][t_steps_ini:t_steps_fin,0,:,:] 
tempVar3 = data[2].variables['so'][t_steps_ini:t_steps_fin,0,:,:] 
tempVar4 = data[3].variables['zos'][t_steps_ini:t_steps_fin,:,:]


#Normalization & replace masked values
wind = normalize(tempVar1)
wind2 = normalize(tempVar2)
salinity = normalize(tempVar3)
seaLevel = normalize(tempVar4)

#Generating targets and labels
lags=10
ts_ahead=48
vars_target = [wind, wind2, salinity, seaLevel]
vars_labels = [wind, wind2, salinity, seaLevel]
x, y = generate_targets_labels_multivar(vars_target, vars_labels, lags, ts_ahead)

#Predict
pred = model.predict(x)

#Showing and saving predictions
sample=27 #Timestep to show
n=1
titles = ["Eastward seawater velocity (m/s)","Nortward seawater velocity (m/s)",
          'Sea water salinity (psu)', 'Sea surface height (m)']
for feat,t in enumerate(titles):
    #Denormalizing data
    temp = denormalize(pred[:,:,:,:,feat], maxValues[feat], minValues[feat])
    #Plotting and saving
    fig, ax = plt.subplots(1,2, figsize=(8,5), gridspec_kw={'width_ratios': [3.21, 4]})
    fig.suptitle(t, fontsize=18)
    ax[0].imshow(y[sample,0,:,:, feat], origin='lower')
    ax[0].set_title("Real", fontsize=16)
    ax[0].axis('off')
    im=ax[1].imshow(temp[sample,0,:,:], origin='lower') 
    ax[1].set_title("Prediction", fontsize=16)
    ax[1].axis('off')
    fig.tight_layout()
    fig.colorbar(im, shrink=0.71)
    fig.savefig(str(feat)+str('_pred_summer_.pdf'), bbox_inches='tight', dpi = 300)
    plt.show()