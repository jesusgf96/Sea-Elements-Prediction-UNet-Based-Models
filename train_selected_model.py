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
    print("\nData not found. Please decompress the data inside '/Data/' in the same directory where the scripts are")
    sys.exit()


########### TRAINING ###########

#Uncomment the wanted model
network = UNet_3DDR
# network = UNet_Res3DDR
# network = UNet_InceptionRes3DDR
# network = UNet_AsymmetricInceptionRes3DDR

#Parameters network
lags = 10
lat = 128
long = 128
feats = 4
feats_output = 4
convFilters = 15
dropoutRate = 0.5
optimizer = 'Adam'

#Parameters training
epochs = 1
batch_size = 16
ts_ahead = 72
batch_size_added = batch_size + lags + ts_ahead - 1

#Timesteps
ts_train = 8760
ts_val = [[9503,10007],[11687,12191],[13895,14399],[16104,16608]]

#Extracting the maximum and minimum values from the training data
maxValues, minValues = get_min_max_values_data(data, ts_train)

#Custom metrics (denormalized mse)
mse_denorm = MSE_denormalized(maxValues, minValues)

#Instantiation model
model = network(lags, lat, long, feats, feats_output, convFilters, dropoutRate)
model.compile(loss=mse_denorm.mse_all, optimizer=optimizer, metrics=[mse_denorm.mse_all])
model.summary()

#Checkpoint to save best model
filepath="saved_models/best_model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')
callbacks_list = [checkpoint]

#Instantiating generators
training_generator = DataGenerator(data, batch_size_added, lags, ts_ahead, 'train', ts_train=ts_train)
validation_generator = DataGenerator(data, batch_size_added, lags, ts_ahead, 'val', ts_val=ts_val)

#Training
print("Starting training...")
history = model.fit_generator(generator=training_generator, epochs=epochs, validation_data=validation_generator, 
							  callbacks=callbacks_list, use_multiprocessing=False)

#Showing training history
fig = plt.figure(figsize=(10,7))
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.title("Training history")
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.ylim(top=0.005,bottom=0) #Limit
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
fig.subplots_adjust(right=0.80, top=0.88)
plt.grid(b=None)
plt.savefig('training_results_loss.png', dpi = 300)
plt.show()


########### EVALUATION ###########

print("\nEvaluating model...")

#Loading best model
model = load_model(filepath, compile=False)

#Compiling with denormalizing metrics
mse_denorm = MSE_denormalized(maxValues, minValues)
model.compile(loss=[mse_denorm.mse1, mse_denorm.mse2, mse_denorm.mse3, mse_denorm.mse4], optimizer=optimizer, 
	metrics=[mse_denorm.mse1, mse_denorm.mse2, mse_denorm.mse3, mse_denorm.mse4])

#Evaluate with the generator - All seasons
ts_test = [[10031,10511],[12215,12695],[14423,14928],[16632,17136]]
test_generator = DataGenerator(data, batch_size_added, lags, ts_ahead, 'test', ts_test=ts_test)
result = model.evaluate_generator(test_generator)

print("\n>> All seasons:")
print("Denormalized MSE:", np.mean(result[1:]))
print("Denormalized MSE separated variables [Eastward wind, Northward wind, Sea water salinity, Sea surface height]:")
print(result[1:])

#Evaluate with the generator - Seaparated seasons
seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
ts_seasons = [[[10031,10511]],[[12215,12695]],[[14423,14928]],[[16632,17136]]]
for season, ts_season in zip(seasons, ts_seasons):
  test_generator = DataGenerator(data, batch_size_added, lags, ts_ahead, 'test', ts_test=ts_season, sep_seasons=True)
  result = model.evaluate_generator(test_generator)
  print("\n>>", season)
  print("Denormalized MSE:", np.mean(result[1:]))

