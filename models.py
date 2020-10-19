
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add, Dropout, concatenate, BatchNormalization, Activation
from tensorflow.keras.layers import Conv3D, MaxPool3D, UpSampling3D


# The data has 3-dimensional shape in the encoder and 2-dimensional shape in the decoder
def UNet_3DDR(lags, latitude, longitude, features, features_output, filters, dropout, kernel_init=tf.keras.initializers.GlorotUniform(seed=50)):
    
    #--- Contracting part / encoder ---#
    inputs = Input(shape = (lags, latitude, longitude, features)) 
    conv1 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(inputs)
    conv1 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv1)
    pool1 = MaxPool3D(pool_size=(1, 2, 2))(conv1)
    
    conv2 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool1)
    conv2 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv2)
    pool2 = MaxPool3D(pool_size=(1, 2, 2))(conv2)
    
    conv3 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool2)
    conv3 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv3)
    pool3 = MaxPool3D(pool_size=(1, 2, 2))(conv3)
    
    conv4 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool3)
    conv4 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv4)
    drop4 = Dropout(dropout)(conv4)
    
    #--- Bottleneck part ---#
    pool4 = MaxPool3D(pool_size=(1, 2, 2))(drop4)
    conv5 = Conv3D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool4)
    compressLags = Conv3D(16*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv5)
    conv5 = Conv3D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(compressLags)
    drop5 = Dropout(dropout)(conv5)
    
    #--- Expanding part / decoder ---#
    up6 = Conv3D(8*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(drop5))
    compressLags = Conv3D(8*filters, (lags,1,1),activation = 'relu', padding = 'valid')(drop4)
    merge6 = concatenate([compressLags,up6], axis = -1)
    conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge6)
    conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv6)

    up7 = Conv3D(4*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv6))
    compressLags = Conv3D(4*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv3)
    merge7 = concatenate([compressLags,up7], axis = -1)
    conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge7)
    conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv7)

    up8 = Conv3D(2*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv7))
    compressLags = Conv3D(2*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv2)
    merge8 = concatenate([compressLags,up8], axis = -1)
    conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge8)
    conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv8)

    up9 = Conv3D(filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv8))
    compressLags = Conv3D(filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv1)
    merge9 = concatenate([compressLags,up9], axis = -1)
    conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge9)
    conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    conv9 = Conv3D(2*features_output, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    
    conv10 = Conv3D(features_output, 1, activation = 'relu')(conv9) #Reduce last dimension    

    return Model(inputs = inputs, outputs = conv10)
    
    

# Includes residual connections and more consecutive convolutional operations
def UNet_Res3DDR(lags, latitude, longitude, features, features_output, filters, dropout, kernel_init=tf.keras.initializers.GlorotUniform(seed=50)):
    
    def residual_block(x, f, k):
        shortcut=x
        #First component
        x = Conv3D(f, k, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        #Second component
        x = Conv3D(f, k, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x = BatchNormalization()(x)
        #Addition
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x     
        
    #--- Contracting part / encoder ---#
    inputs = Input(shape = (lags, latitude, longitude, features)) 
    conv1 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(inputs)
    conv1 = residual_block(conv1, filters, 3)
    pool1 = MaxPool3D(pool_size=(1, 2, 2))(conv1)
    
    conv2 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool1)
    conv2 = residual_block(conv2, 2*filters, 3)
    pool2 = MaxPool3D(pool_size=(1, 2, 2))(conv2)
    
    conv3 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool2)
    conv3 = residual_block(conv3, 4*filters, 3)
    pool3 = MaxPool3D(pool_size=(1, 2, 2))(conv3)
    
    conv4 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool3)
    conv4 = residual_block(conv4, 8*filters, 3)
    drop4 = Dropout(dropout)(conv4)
    
    #--- Bottleneck part ---#
    pool4 = MaxPool3D(pool_size=(1, 2, 2))(drop4)
    conv5 = Conv3D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool4)
    conv5 = residual_block(conv5, 16*filters, 3)
    compressLags = Conv3D(16*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv5)
    drop5 = Dropout(dropout)(compressLags)
    
    #--- Expanding part / decoder ---#
    up6 = Conv3D(8*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(drop5))
    compressLags = Conv3D(8*filters, (lags,1,1),activation = 'relu', padding = 'valid')(drop4)
    merge6 = concatenate([compressLags,up6], axis = -1)
    conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge6)
    conv6 = residual_block(conv6, 8*filters, 3)

    up7 = Conv3D(4*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv6))
    compressLags = Conv3D(4*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv3)
    merge7 = concatenate([compressLags,up7], axis = -1)
    conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge7)
    conv7 = residual_block(conv7, 4*filters, 3)

    up8 = Conv3D(2*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv7))
    compressLags = Conv3D(2*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv2)
    merge8 = concatenate([compressLags,up8], axis = -1)
    conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge8)
    conv8 = residual_block(conv8, 2*filters, 3)

    up9 = Conv3D(filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv8))
    compressLags = Conv3D(filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv1)
    merge9 = concatenate([compressLags,up9], axis = -1)
    conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge9)
    conv9 = residual_block(conv9, filters, 3)
    conv9 = Conv3D(2*features_output, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    
    conv10 = Conv3D(features_output, 1, activation = 'relu')(conv9) #Reduce last dimension    

    return Model(inputs = inputs, outputs = conv10)



# Includes parallel convolutions and residual connections
def UNet_InceptionRes3DDR(lags, latitude, longitude, features, features_output, filters, dropout, kernel_init=tf.keras.initializers.GlorotUniform(seed=50)):
       
    def res_inception_block(x, f, k):
        shortcut=x
        x1 = Conv3D(f, 1, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x2 = Conv3D(f, 1, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x2 = Conv3D(f, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x2)
        x3 = Conv3D(f, 1, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x3 = Conv3D(f, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x3)
        x3 = Conv3D(f, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x3)
        x = concatenate([x1, x2, x3], axis = -1)
        x = Conv3D(f, 1, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

    #--- Contracting part / encoder ---#
    inputs = Input(shape = (lags, latitude, longitude, features)) 
    conv1 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(inputs)
    conv1 = res_inception_block(conv1, filters, 3)
    pool1 = MaxPool3D(pool_size=(1, 2, 2))(conv1)
    
    conv2 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool1)
    conv2 = res_inception_block(conv2, 2*filters, 3)
    pool2 = MaxPool3D(pool_size=(1, 2, 2))(conv2)
    
    conv3 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool2)
    conv3 = res_inception_block(conv3, 4*filters, 3)
    pool3 = MaxPool3D(pool_size=(1, 2, 2))(conv3)
    
    conv4 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool3)
    conv4 = res_inception_block(conv4, 8*filters, 3)
    drop4 = Dropout(dropout)(conv4)
    
    #--- Bottleneck part ---#
    pool4 = MaxPool3D(pool_size=(1, 2, 2))(drop4)
    conv5 = Conv3D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool4)
    conv5 = res_inception_block(conv5, 16*filters, 3)
    compressLags = Conv3D(16*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv5)
    drop5 = Dropout(dropout)(compressLags)
    
    #--- Expanding part / decoder ---#
    up6 = Conv3D(8*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(drop5))
    compressLags = Conv3D(8*filters, (lags,1,1),activation = 'relu', padding = 'valid')(drop4)
    merge6 = concatenate([compressLags,up6], axis = -1)
    conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge6)
    conv6 = res_inception_block(conv6, 8*filters, 3)

    up7 = Conv3D(4*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv6))
    compressLags = Conv3D(4*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv3)
    merge7 = concatenate([compressLags,up7], axis = -1)
    conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge7)
    conv7 = res_inception_block(conv7, 4*filters, 3)

    up8 = Conv3D(2*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv7))
    compressLags = Conv3D(2*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv2)
    merge8 = concatenate([compressLags,up8], axis = -1)
    conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge8)
    conv8 = res_inception_block(conv8, 2*filters, 3)

    up9 = Conv3D(filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv8))
    compressLags = Conv3D(filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv1)
    merge9 = concatenate([compressLags,up9], axis = -1)
    conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge9)
    conv9 = res_inception_block(conv9, filters, 3)
    conv9 = Conv3D(2*features_output, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    
    conv10 = Conv3D(features_output, 1, activation = 'relu')(conv9) #Reduce last dimension    

    return Model(inputs = inputs, outputs = conv10)



# Includes parallel convolutions, asymmetric convolutions and residual connections
def UNet_AsymmetricInceptionRes3DDR(lags, latitude, longitude, features, features_output, filters, dropout, kernel_init=tf.keras.initializers.GlorotUniform(seed=50)):    

    def res_inception_block(x, f, k):
        shortcut=x
        x1 = Conv3D(f, 1, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x2 = Conv3D(f, (1,1,3), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x2 = Conv3D(f, (1,3,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x2)
        x2 = Conv3D(f, (3,1,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x2)
        x3 = Conv3D(f, (1,1,5), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x3 = Conv3D(f, (1,5,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x3)
        x3 = Conv3D(f, (5,1,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x3)
        x4 = Conv3D(f, (1,1,3), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x4 = Conv3D(f, (1,3,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x4)
        x4 = Conv3D(f, (3,1,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x4)
        x5 = Conv3D(f, (1,1,3), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x5 = Conv3D(f, (1,3,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x5)
        x5 = Conv3D(f, (3,1,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x5)
        x = concatenate([x1, x2, x3, x4, x5], axis = -1)
        x = Conv3D(f, 1, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x
    
    #--- Contracting part / encoder ---#
    inputs = Input(shape = (lags, latitude, longitude, features)) 
    conv1 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(inputs)
    conv1 = res_inception_block(conv1, filters, 3)
    pool1 = MaxPool3D(pool_size=(1, 2, 2))(conv1)
    
    conv2 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool1)
    conv2 = res_inception_block(conv2, 2*filters, 3)
    pool2 = MaxPool3D(pool_size=(1, 2, 2))(conv2)
    
    conv3 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool2)
    conv3 = res_inception_block(conv3, 4*filters, 3)
    pool3 = MaxPool3D(pool_size=(1, 2, 2))(conv3)
    
    conv4 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool3)
    conv4 = res_inception_block(conv4, 8*filters, 3)
    drop4 = Dropout(dropout)(conv4)
    
    #--- Bottleneck part ---#
    pool4 = MaxPool3D(pool_size=(1, 2, 2))(drop4)
    conv5 = Conv3D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool4)
    conv5 = res_inception_block(conv5, 16*filters, 3)
    compressLags = Conv3D(16*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv5)
    drop5 = Dropout(dropout)(compressLags)
    
    #--- Expanding part / decoder ---#
    up6 = Conv3D(8*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(drop5))
    compressLags = Conv3D(8*filters, (lags,1,1),activation = 'relu', padding = 'valid')(drop4)
    merge6 = concatenate([compressLags,up6], axis = -1)
    conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge6)
    conv6 = res_inception_block(conv6, 8*filters, 3)

    up7 = Conv3D(4*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv6))
    compressLags = Conv3D(4*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv3)
    merge7 = concatenate([compressLags,up7], axis = -1)
    conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge7)
    conv7 = res_inception_block(conv7, 4*filters, 3)

    up8 = Conv3D(2*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv7))
    compressLags = Conv3D(2*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv2)
    merge8 = concatenate([compressLags,up8], axis = -1)
    conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge8)
    conv8 = res_inception_block(conv8, 2*filters, 3)

    up9 = Conv3D(filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv8))
    compressLags = Conv3D(filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv1)
    merge9 = concatenate([compressLags,up9], axis = -1)
    conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge9)
    conv9 = res_inception_block(conv9, filters, 3)
    conv9 = Conv3D(2*features_output, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    
    conv10 = Conv3D(features_output, 1, activation = 'relu')(conv9) #Reduce last dimension    

    return Model(inputs = inputs, outputs = conv10)