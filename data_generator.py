import numpy as np
import tensorflow as tf


#USAGE OF THE GENERATOR:
# -Fill the parameter purpose with 'train', 'val' or 'test'
# -Fill the parameter ts_train=maxTimeStep, ts_val=[] or ts_test=[] depending on the purpose
# -ts_train is a int to pick from 0 to maxTimeStep
# -ts_val=[] and ts_test=[] are list with the shape: (season, range_timeSteps) -> [[-,-],[-,-],[-,-],[-,-]] 
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size, lags, ts_ahead, purpose, ts_train=0, ts_val=[], ts_test=[], debug=False):
        self.data = data
        self.b_size = batch_size
        self.lags = lags
        self.ts_ahead = ts_ahead
        self.purpose = purpose #'train', 'val' or 'test'
        self.debug=debug #Show msgs with indexes used in each moment (just to check correct functioning)
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
            tempVar1 = self.normalize(tempVar1)
            tempVar2 = self.normalize(tempVar2)
            tempVar3 = self.normalize(tempVar3)
            tempVar4 = self.normalize(tempVar4)
            #Generating targets and labels
            vars_target = [tempVar1, tempVar2, tempVar3, tempVar4]
            vars_labels = [tempVar1, tempVar2, tempVar3, tempVar4]
            x, y = self.generate_targets_labels_multivar(vars_target, vars_labels, self.lags, self.ts_ahead)
        #--Validation/Test--#
        else:
            x = []
            y = []
            #Iterating different seasons
            for season in range(4):
                tempVar1 = self.tempVar1[season]
                tempVar2 = self.tempVar2[season]
                tempVar3 = self.tempVar3[season]
                tempVar4 = self.tempVar4[season]
                #Normalizing variables
                tempVar1 = self.normalize(tempVar1[idx*self.b_size:(idx+1)*self.b_size,:,:])
                tempVar2 = self.normalize(tempVar2[idx*self.b_size:(idx+1)*self.b_size,:,:])
                tempVar3 = self.normalize(tempVar3[idx*self.b_size:(idx+1)*self.b_size,:,:])
                tempVar4 = self.normalize(tempVar4[idx*self.b_size:(idx+1)*self.b_size,:,:])
                #Generating targets and labels
                vars_target = [tempVar1, tempVar2, tempVar3, tempVar4]
                vars_labels = [tempVar1, tempVar2, tempVar3, tempVar4]
                temp_x, temp_y = self.generate_targets_labels_multivar(vars_target, vars_labels, self.lags, self.ts_ahead)
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
    

    #Auxiliar function to generate X and Y
    def generate_targets_labels_multivar(self, data_target, data_label, lags, ts_ahead=1):
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


    #Auxiliar function to normalize data and replace masked values with 0
    def normalize(self, data):
        data = 0.1+((data-np.min(data))*(1-0.1))/(np.max(data)-np.min(data))
        data = data.filled(fill_value=0)
        return data
    
    