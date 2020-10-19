import numpy as np
import tensorflow as tf
from utils import *


#USAGE OF THE GENERATOR:
# -Fill the parameter purpose with 'train', 'val' or 'test'
# -Fill the parameter ts_train=0, ts_val=[] or ts_test=[] depending on the purpose
# -ts_train is a int to pick from 0 to ts_train
# -ts_val=[] and ts_test=[] are list with the shape: (season, range_timeSteps) -> [[-,-],[-,-],[-,-],[-,-]] 
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size, lags, ts_ahead, purpose, ts_train=0, ts_val=[], ts_test=[], debug=False, sep_seasons=False, norm_values=None):
        self.data = data
        self.b_size = batch_size
        self.lags = lags
        self.ts_ahead = ts_ahead
        self.purpose = purpose #'train', 'val' or 'test'
        self.debug = debug #Show msgs with indexes used in each moment (just to check correct functioning)
        self.sep_seasons = sep_seasons
        if norm_values is not None:
          self.norm_max = norm_values[0]
          self.norm_min = norm_values[1]
        else:
          self.norm_max = self.norm_min = [None, None, None, None]
        #Training data
        if self.purpose=='train': 
            self.tempVar1 = data[0].variables['uo'][:ts_train,0,:,:] 
            self.tempVar2 = data[1].variables['vo'][:ts_train,0,:,:]
            self.tempVar3 = data[2].variables['so'][:ts_train,0,:,:]
            self.tempVar4 = data[3].variables['zos'][:ts_train,:,:]
            self.time_steps = ts_train
        #Validation or test data
        else:
            self.tempVar1 = []
            self.tempVar2 = []
            self.tempVar3 = []
            self.tempVar4 = []
            #Picking validation or test timesteps
            if self.purpose=='val': ts_ranges = ts_val
            elif self.purpose=='test': ts_ranges = ts_test
            #Extracting timesteps by seasons
            for season in ts_ranges:
                self.tempVar1.append(data[0].variables['uo'][season[0]:season[1],0,:,:])
                self.tempVar2.append(data[1].variables['vo'][season[0]:season[1],0,:,:])
                self.tempVar3.append(data[2].variables['so'][season[0]:season[1],0,:,:])
                self.tempVar4.append(data[3].variables['zos'][season[0]:season[1],:,:])
            self.time_steps = int(ts_ranges[0][1]-ts_ranges[0][0])
              

    
    #Calculates the number of batches: samples/batch_size
    def __len__(self):
        #Calculating the number of batches 
        return int(self.time_steps/self.b_size)

    
    #Obtains one batch of data (Reading>Preprocessing>GeneratingXY)
    def __getitem__(self, idx):       
        #--Training--#
        if self.purpose=='train':
            #Reading each variable independently
            tempVar1 = self.tempVar1[idx*self.b_size:(idx+1)*self.b_size,:,:] 
            tempVar2 = self.tempVar2[idx*self.b_size:(idx+1)*self.b_size,:,:] 
            tempVar3 = self.tempVar3[idx*self.b_size:(idx+1)*self.b_size,:,:] 
            tempVar4 = self.tempVar4[idx*self.b_size:(idx+1)*self.b_size,:,:]
            #Normalizing variables
            tempVar1 = normalize(tempVar1, self.norm_max[0], self.norm_min[0])
            tempVar2 = normalize(tempVar2, self.norm_max[1], self.norm_min[1])
            tempVar3 = normalize(tempVar3, self.norm_max[2], self.norm_min[2])
            tempVar4 = normalize(tempVar4, self.norm_max[3], self.norm_min[3])
            #Generating targets and labels
            vars_target = [tempVar1, tempVar2, tempVar3, tempVar4]
            vars_labels = [tempVar1, tempVar2, tempVar3, tempVar4]
            x, y = generate_targets_labels_multivar(vars_target, vars_labels, self.lags, self.ts_ahead)
        #--Validation/Test--#
        else:
            x = []
            y = []
            #Individual seasons or all of them together
            if self.sep_seasons: n=1
            else: n=4
            #Iterating different seasons
            for season in range(n):
                tempVar1 = self.tempVar1[season]
                tempVar2 = self.tempVar2[season]
                tempVar3 = self.tempVar3[season]
                tempVar4 = self.tempVar4[season]
                #Normalizing variables
                tempVar1 = normalize(tempVar1[idx*self.b_size:(idx+1)*self.b_size,:,:], self.norm_max[0], self.norm_min[0])
                tempVar2 = normalize(tempVar2[idx*self.b_size:(idx+1)*self.b_size,:,:], self.norm_max[1], self.norm_min[1])
                tempVar3 = normalize(tempVar3[idx*self.b_size:(idx+1)*self.b_size,:,:], self.norm_max[2], self.norm_min[2])
                tempVar4 = normalize(tempVar4[idx*self.b_size:(idx+1)*self.b_size,:,:], self.norm_max[3], self.norm_min[3])
                #Generating targets and labels
                vars_target = [tempVar1, tempVar2, tempVar3, tempVar4]
                vars_labels = [tempVar1, tempVar2, tempVar3, tempVar4]
                temp_x, temp_y = generate_targets_labels_multivar(vars_target, vars_labels, self.lags, self.ts_ahead)
                x.append(temp_x)
                y.append(temp_y)                
                del temp_x
                del temp_y
            #Concatenating all the seasons
            x = np.concatenate(np.array(x))
            y = np.concatenate(np.array(y))
        #Freeing memory        
        del tempVar1
        del tempVar2
        del tempVar3
        del tempVar4
        del vars_target
        del vars_labels
        #Returning targets and labels
        return x, y