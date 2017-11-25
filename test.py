
from keras.models import Sequential
from keras.layers import Conv3D
from keras.layers.convolutional import MaxPooling3D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import model_from_json
from keras.optimizers import SGD, RMSprop
import numpy as np
from sklearn.preprocessing import StandardScaler
import cPickle
from sklearn.model_selection import train_test_split
from keras import backend as K
import sklearn
import random
from keras.preprocessing.image import ImageDataGenerator

def load():
	tag_name = np.load('./CS446-project/tag_name.npy')
	train_Y  = np.load('./CS446-project/train_binary_Y.npy')
	train_X  = np.load('./CS446-project/train_X.npy')
	test_X   = np.load('./CS446-project/valid_test_X.npy')
	print(tag_name)
	print("tag_name shape =", tag_name.shape)
	print("train_Y shape =", train_Y.shape)
	print("train_X shape =", train_X.shape)
	print("test_X shape =", test_X.shape)
	return tag_name, train_Y, train_X, test_X

def normalizeData(train_X, train_Y):
	### Reshape to 2D array ###
	img_rows = 26
	img_cols = 31
	img_depth = 23
	train_X_2d = train_X.reshape((len(train_X), \
	           len(train_X[0])*len(train_X[0][0])*len(train_X[0][0][0])))

	### Scale Data ###
	scaler = StandardScaler()
	scaler.fit(train_X_2d)
	train_X_2d = scaler.transform(train_X_2d)

	train_X = train_X_2d.reshape((len(train_Y), img_rows, img_cols, img_depth))
	return train_X

def main():
	tag_name, train_Y, train_X, test_X = load()

	x_train, x_test, y_train, y_test = train_test_split(train_X, train_Y, test_size = .2)
	#https://stackoverflow.com/questions/42699956/3d-convolutional-neural-network-input-shape
	img_rows = 26
	img_cols = 31
	img_depth = 23
	input_shape = None
	if K.image_dim_ordering() == 'th':
	    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols, img_depth)
	    x_test = x_test.reshape(x_test.shape[0], 1, img_rows,img_cols,img_depth)
	    test_X = test_X.reshape(test_X.shape[0],1,img_rows,img_cols,img_depth)
	    input_shape = (1, img_rows, img_cols, img_depth)
	else:
	    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_depth, 1)
	    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_depth, 1)
	    test_X = test_X.reshape(test_X.shape[0], img_rows, img_cols, img_depth, 1)
	    input_shape = (img_rows, img_cols, img_depth, 1)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	#adapted from: https://github.com/MinhazPalasara/keras/blob/master/examples/shapes_3d_cnn.py
	filters = [32,32]
	kernel = [(3,3,3),(3,3,3)]
	pool = [2,2]

	model = Sequential()
	model.add(Conv3D(filters[0],kernel[0], input_shape=input_shape, activation="relu"))
	model.add(Conv3D(filters[1],kernel[1], activation="relu"))
	model.add(MaxPooling3D(pool_size=(pool[0], pool[0], pool[0])))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation="relu"))
	model.add(Dropout(0.25))
	model.add(Dense(19, kernel_initializer='normal', activation="sigmoid"))

	# Optimize with SGD
	rms = RMSprop(lr=0.1, rho=0.9, epsilon=1e-08, decay=0)
	model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])

	# Fit model in batches
	print("training model")
	#create class weights
	counts = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	for y in y_train:
		for i in range(len(y)):
			if y[i] == 1:
				counts[i] += 1
	classWeights = {}
	total = len(y_train)
	for i in range(len(counts)):
		avg = total/19.0
		classWeights[i] = avg/counts[i]
	model.fit(x_train, y_train, epochs=2, batch_size=500, validation_data=(x_test, y_test), class_weight=classWeights)
	print("training completed")

	print("saving model")
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
	    json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("model.h5")
	print("saved to disk")

	score = model.evaluate(x_test, y_test, batch_size=64)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])

	# predictions = model.predict(x_test, batch_size=64, verbose=0)
	# print("Test predictions")
	# print(sklearn.metrics.accuracy_score(y_test, predictions))
	# predictions = model.predict(train_X, batch_size=64, verbose=0)
	# print("Train predictions")
	# print(sklearn.metrics.accuracy_score(train_Y, predictions))
	print("saving other variables")
	with open('x_train.pkl', 'wb') as fid:
	    cPickle.dump(x_train, fid)

	with open('x_test.pkl', 'wb') as fid:
	    cPickle.dump(x_test, fid)

	with open('y_train.pkl', 'wb') as fid:
	    cPickle.dump(y_train, fid)

	with open('y_test.pkl', 'wb') as fid:
	    cPickle.dump(y_test, fid)
	print("other variables saved")

	predictions = model.predict(test_X)
	with open('myPredictions.pkl', 'wb') as fid:
		cPickle.save(predictions, fid)


def run():
	# load json and create model
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("model.h5")
	print("Loaded model from disk")

	tag_name = np.load('./CS446-project/tag_name.npy')
	train_Y  = np.load('./CS446-project/train_binary_Y.npy')
	train_X  = np.load('./CS446-project/train_X.npy')
	test_X   = np.load('./CS446-project/valid_test_X.npy')
	img_rows = 26
	img_cols = 31
	img_depth = 23
	x_train = None
	x_test = None
	y_train = None
	y_test = None
	print("load data")
	# with open('x_train.pkl', 'rb') as fid:
	#     x_train = cPickle.load(fid)
	# with open('x_test.pkl', 'rb') as fid:
	#     x_test = cPickle.load(fid)
	# with open('y_train.pkl', 'rb') as fid:
	#     y_train = cPickle.load(fid)
	# with open('y_test.pkl', 'rb') as fid:
	#     y_test = cPickle.load(fid)
	print("loading data completed")

	if K.image_dim_ordering() == 'th':
	    train_X = train_X.reshape(train_X.shape[0], 1, img_rows, img_cols, img_depth)
	    test_X = test_X.reshape(test_X.shape[0], 1, img_rows,img_cols,img_depth)
	    # x_test = x_test.reshape(x_test.shape[0], 1, img_rows,img_cols,img_depth)
	else:
	    train_X = train_X.reshape(train_X.shape[0], img_rows, img_cols, img_depth, 1)
	    test_X = test_X.reshape(test_X.shape[0], img_rows, img_cols, img_depth, 1)
	    # x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_depth, 1)
	pred = loaded_model.predict(test_X)
	for i in range(len(pred)):
		for j in range(len(pred[i])):
			if pred[i][j] >=.5:
				pred[i][j]=1
			else:
				pred[i][j]=0

	myId = 0
	for row in pred:
		myStr = ""+str(myId)+","
		for x in row:
			myStr += str(x)+","
		print myStr[0:len(myStr)-1]
		myId += 1

if __name__ == "__main__":
    # pdb.set_trace()
    main()
    # print("--- %s seconds ---" % (time.time() - start_time))
    #run()