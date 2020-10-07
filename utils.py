import numpy as np
import tensorflow as tf

#Receives a list of target features and label features
# and returns the X & Y.
#It also crops the image to 128,128
def generate_targets_labels_multivar(data_target, data_label, lags, ts_ahead=1):
    x=[]
    y=[]
    ts_ahead-=1
    for i in range(lags, len(data_target[0])-ts_ahead):
        features = []
        for feature in data_target:
            features.append(np.expand_dims(feature[i-lags:i, -128:,-128:], axis=-1))
        temp = np.concatenate(features, axis=-1)
        x.append(temp)
        features = []
        for feature in data_label:
            features.append([np.expand_dims(feature[i+ts_ahead, -128:,-128:], axis=-1)])
        temp = np.concatenate(features, axis=-1)        
        y.append(temp)    
    return np.array(x), np.array(y)


# Normalize in the range given or [0.1, 1] if none given
# and assigns 0 to the ground (masked values)
def normalize(data, maxVal=None, minVal=None):
    # Normalize with min/max values of the data
    if maxVal is None or minVal is None: 
        data = 0.1+((data-np.min(data))*(1-0.1))/(np.max(data)-np.min(data))
    # Normalize with max and min given
    else:
        data = 0.1+((data-minVal)*(1-0.1))/(maxVal-minVal)
    data = data.filled(fill_value=0)
    return data


# Denormalize the data with man and mix given
def denormalize(dataNorm, maxVal, minVal):
    denormalized = minVal+((dataNorm-0.1)*(maxVal-minVal))/(1-0.1)
    return denormalized


# Returns the max and min value of each variable for the data given
def get_min_max_values_data(data, n):
	# Extracting the maximum values from the data
	maxValues = []
	maxValues.append(np.max(data[0].variables['uo'][:n,0,:,:]))
	maxValues.append(np.max(data[1].variables['vo'][:n,0,:,:]))
	maxValues.append(np.max(data[2].variables['so'][:n,0,:,:]))
	maxValues.append(np.max(data[3].variables['zos'][:n,:,:]))

	# Extracting the maximum values from the data
	minValues = []
	minValues.append(np.min(data[0].variables['uo'][:n,0,:,:]))
	minValues.append(np.min(data[1].variables['vo'][:n,0,:,:]))
	minValues.append(np.min(data[2].variables['so'][:n,0,:,:]))
	minValues.append(np.min(data[3].variables['zos'][:n,:,:]))

	return maxValues, minValues


# Custom metric to calculate the denormalized mse per variable
class MSE_denormalized:
	def __init__(self, maxValues, minValues):
		self.maxValues = maxValues
		self.minValues = minValues
		self.mse = tf.keras.losses.MeanSquaredError()

	#  Eastward wind
	def mse1(self, y_true, y_pred):
		y_pred = denormalize(y_pred[:,:,:,:,0], self.maxValues[0], self.minValues[0])
		y_true = denormalize(y_true[:,:,:,:,0], self.maxValues[0], self.minValues[0])
		return self.mse(y_true, y_pred)

	# Northward wind
	def mse2(self, y_true, y_pred):
		y_pred = denormalize(y_pred[:,:,:,:,1], self.maxValues[1], self.minValues[1])
		y_true = denormalize(y_true[:,:,:,:,1], self.maxValues[1], self.minValues[1])
		return self.mse(y_true, y_pred)

	# Sea water salinity
	def mse3(self, y_true, y_pred):
		y_pred = denormalize(y_pred[:,:,:,:,2], self.maxValues[2], self.minValues[2])
		y_true = denormalize(y_true[:,:,:,:,2], self.maxValues[2], self.minValues[2])
		return self.mse(y_true, y_pred)

	# Sea surface height
	def mse4(self, y_true, y_pred):
		y_pred = denormalize(y_pred[:,:,:,:,3], self.maxValues[3], self.minValues[3])
		y_true = denormalize(y_true[:,:,:,:,3], self.maxValues[3], self.minValues[3])
		return self.mse(y_true, y_pred)
