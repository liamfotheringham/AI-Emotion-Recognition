

#Import Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import plot_model
import pickle
import time

# Load the Drive helper and mount
from google.colab import drive
drive.mount('/content/drive')

#Establish dense layers to loop through
dense_layers = [0,1,2]

#Establish number of nodes per layer
layer_sizes = [32,64,128,256,512]

#Establish convolutional layers to loop through
conv_layers = [1,2,3,4,5]

#Import data from Piclkle files
X = pickle.load(open("/content/drive/My Drive/Emotions/X.pickle","rb"))#Pixel Data
y = pickle.load(open("/content/drive/My Drive/Emotions/y.pickle","rb"))#Label Data

#Normalise data(Number of pixel colours)
X = X/255.0

#Loop through the multiple parameters
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            
            #Create name of network for saving
            NAME = "{}-conv-{}-nodes-{}-dense".format(conv_layer, layer_size, dense_layer)
            
            #Initialise Keras call backs
            tensorboard = TensorBoard(log_dir=f"/content/drive/My Drive/Emotions/logs/{NAME}")#Tensorboard callback
            stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0, mode='auto', baseline=None, restore_best_weights=False)#Early Stopping Callbacks
            
            #Define the model as a sequential model
            model = Sequential()
            
            #Layer of model
            model.add(Conv2D(layer_size, (3,3), input_shape = X.shape[1:]))#Add convolutional layer (Number of nodes, size of convolutional kernel, shape of image)
            model.add(Activation("relu"))#ReLU activation
            model.add(MaxPooling2D(pool_size=(2,2)))#Pool convolutional result
            
            #Layer of model
            for l in range(conv_layer - 1):#Loop through conv layers
                model.add(Conv2D(layer_size, (3,3)))#Add Convolutional Layer
                model.add(Activation("relu"))#ReLU activation
                model.add(MaxPooling2D(pool_size=(2,2)))#Pool convolutional result
            
            #Flatten Layers
            model.add(Flatten())
            
            #Layer of Model
            for l in range(dense_layer):#Loop through dense layers
                model.add(Dense(layer_size))#Add Dense layer
                model.add(Activation("relu"))#ReLU activation
            
            
            model.add(Dropout(0.3))#Dropout rate 30%
            model.add(Dense(6))#Add Dense layer of 6
            model.add(Activation("softmax"))#Softmax activation
            
            #Compile model
            model.compile(loss = "sparse_categorical_crossentropy", 
                          optimizer="adam",
                          metrics=["accuracy"])

            #Fit data for model
            model.fit(X, y
                      , batch_size=32, epochs = 15, validation_split=0.2, callbacks=[tensorboard, stopping])#(Image data, labels, number of images fed, number of epochs, validation split, callback functions)

            #Save model
            model.save(f'/content/drive/My Drive/Emotions/Models/{NAME}.model')