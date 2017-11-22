
from keras.models import Sequential
from keras.layers import Conv3D
from keras.layers.convolutional import MaxPooling3D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, RMSprop
import numpy as np
from sklearn.model_selection import train_test_split
from keras import backend as K

tag_name = np.load('./CS446-project/tag_name.npy')
train_Y  = np.load('./CS446-project/train_binary_Y.npy')
train_X  = np.load('./CS446-project/train_X.npy')
test_X   = np.load('./CS446-project/valid_test_X.npy')
print(tag_name)
print("tag_name shape =", tag_name.shape)
print("train_Y shape =", train_Y.shape)
print("train_X shape =", train_X.shape)
print("test_X shape =", test_X.shape)

x_train, x_test, y_train, y_test = train_test_split(train_X, train_Y, test_size = .25)

#https://stackoverflow.com/questions/42699956/3d-convolutional-neural-network-input-shape
img_rows = 26
img_cols = 31
img_depth = 23
input_shape = None
if K.image_dim_ordering() == 'th':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols, img_depth)
    input_shape = (1, img_rows, img_cols, img_depth)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_depth, 1)
    input_shape = (img_rows, img_cols, img_depth, 1)

#adapted from: https://github.com/MinhazPalasara/keras/blob/master/examples/shapes_3d_cnn.py
filters = [16,32]
kernel = [(7,7,7),(3,3,3)]
pool = [3,3]

model = Sequential()
model.add(Conv3D(filters[0],kernel[0], input_shape=input_shape))
model.add(MaxPooling3D(pool_size=(pool[0], pool[0], pool[0])))
model.add(Dropout(0.5))
model.add(Conv3D(filters[1],kernel[1]))
model.add(MaxPooling3D(pool_size=(pool[1], pool[1], pool[1])))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(16, kernel_initializer='normal', activation='relu'))
model.add(Dense(19, kernel_initializer='normal'))
model.add(Activation('softmax'))
 
# Optimize with SGD
model.compile(loss='categorical_crossentropy', 
              optimizer='sgd', metrics=['accuracy'])
 
# Fit model in batches
print("training model")
model.fit(x_train, y_train, nb_epoch=5, batch_size=500)
print("training completed")
# Evaluate model
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=500)