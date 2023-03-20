#This is the core codes of MIML-DCNN in `` Automatic Waveform Recognition of Overlapping LPI Radar Signals Based on Multi-Instance Multi-Label Learning''
#

from keras.applications.vgg16 import VGG16
from keras.callbacks import TensorBoard
from keras.models import Sequential, load_model
from keras.applications.vgg16 import preprocess_input
from keras.layers import Activation, Dropout, Flatten, Reshape, Dense, concatenate,MaxPooling2D,AveragePooling2D,GlobalAveragePooling2D,GlobalMaxPooling2D,Permute
from keras.layers import BatchNormalization, Input, Conv2D, Lambda, Average, Multiply,add
import numpy as np
import tensorflow as tf
from keras.utils import plot_model
from keras.models import Model
from tensorflow.python.keras.optimizers import SGD


# load Data
#X_train=np.load('E:/MIML/X_train.npy')
#Y_label_train=np.load('E:/MIML/Y_train.npy')

#Train Network
base_model = VGG16(weights='imagenet', include_top=False,input_shape=(224,224,3))
base_model2=Sequential(layers=base_model.layers[:-4])
x1=base_model2.output
x2=Conv2D(1024,3,strides=1,padding='same',activation='relu')(x1)
x3=BatchNormalization(momentum=0.8)(x2)
x4=Conv2D(2048,3,strides=1,padding='same',activation='relu')(x3)
x5=BatchNormalization(momentum=0.8)(x4)
x6=Conv2D(200,3,strides=1,padding='same',activation='relu')(x5)
x7=BatchNormalization(momentum=0.8)(x6)
x9=Reshape((196,200))(xx)
x10=Dense(128,activation='relu')(x9)
x11=Dense(128,activation='relu')(x10)
x12=Dense(4,activation='sigmoid')(x11)
x13=Reshape((196,4,1))(x12)
x14=MaxPooling2D((196,1))(x13)
x15=Reshape((4,))(x14)

model_miml1=Model(inputs=base_model.input,outputs=x15)
for layer in model_miml1.layers[:14]:
    layer.trainable=False
print(model_miml1.summary())
print("Compiling Deep MIML Model...")

sgd =SGD(lr=0.01, decay=1e-6, momentum=0.8, nesterov=True)
model_miml.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])
EPOCHS=50
BS=32
H=model_miml.fit(X_train,Y_label_train,batch_size=BS,epochs=EPOCHS,
                 validation_split=0.2)
#Save Model
model_miml.save('E:/MIML/MIML_model_epoch.h5')
