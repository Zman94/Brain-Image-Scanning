
from keras.models import Sequential
from keras.layers import Conv3D
from keras.layers.convolutional import MaxPooling3D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import model_from_json
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras import backend as K
import sklearn
import random

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

def normalizeData(train_X):
	### Reshape to 2D array ###
	num_dims = train_X.shape[0]
	img_rows = train_X.shape[1]
	img_cols = train_X.shape[2]
	img_depth = train_X.shape[3]
	train_X_2d = train_X.reshape((len(train_X), \
	           len(train_X[0])*len(train_X[0][0])*len(train_X[0][0][0])))

	### Scale Data ###
	scaler = StandardScaler()
	scaler.fit(train_X_2d)
	train_X_2d = scaler.transform(train_X_2d)

	train_X = train_X_2d.reshape((num_dims, img_rows, img_cols, img_depth))
	return train_X

def modifiedSampling(train_X, train_Y):
	myDict = {}
	counts = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	for i in range(len(train_X)):
		key = ""
		for j in range(len(train_Y[i])):
			if train_Y[i][j] == 1:
				key += str(j)+","
				counts[j]+=1
		key = key[:len(key)-1]
		if key not in myDict:
			myDict[key] = []
		myDict[key].append(train_X[i])
	retX = []
	retY = []
	i = 0
	for k in myDict.keys():
		# numSamples = int(50*portion[i])
		numSamples = 250
		newSamples = [random.choice(myDict[k]) for _ in range(numSamples)]
		retX += newSamples
		nums = k.split(',')
		hot_vector = np.zeros(19)
		for n in nums:
			hot_vector[int(n)] = 1
		for _ in range(numSamples):
			retY.append(hot_vector)
		i+=1

	retX = np.array(retX)
	retY = np.array(retY)
	return retX, retY


def main():
	tag_name, train_Y, train_X, test_X = load()

	#normalize training data
	# train_X = normalizeData(train_X)
	# test_X = normalizeData(test_X)

	# #sample 200 from each label pairing	
	# modified_X, modified_Y = modifiedSampling(train_X, train_Y)

	x_train, x_test, y_train, y_test = train_test_split(train_X, train_Y, test_size = .2)
	
	x_train, y_train = modifiedSampling(x_train, y_train)

	#https://stackoverflow.com/questions/42699956/3d-convolutional-neural-network-input-shape
	#reshape for input for Conv3D layer
	img_rows = x_train.shape[1]
	img_cols = x_train.shape[2]
	img_depth = x_train.shape[3]
	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_depth, 1)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_depth, 1)
	test_X = test_X.reshape(test_X.shape[0], img_rows, img_cols, img_depth, 1)
	input_shape = (img_rows, img_cols, img_depth, 1)

	#adapted from: https://github.com/MinhazPalasara/keras/blob/master/examples/shapes_3d_cnn.py
	filters = [32,64]
	kernel = [(7,7,7), (3,3,3)]
	pool = [3,3]

	model = Sequential()
	model.add(Conv3D(filters[0],kernel[0], input_shape=input_shape, activation="relu"))
	model.add(Conv3D(filters[1],kernel[1], activation="relu"))
	model.add(MaxPooling3D(pool_size=(pool[0], pool[0], pool[0])))
	# model.add(MaxPooling3D(pool_size=(pool[1], pool[1], pool[1])))
	model.add(Flatten())
	model.add(Dropout(.25))
	# model.add(Dense(512, activation="relu"))
	model.add(Dense(128, activation="relu"))
	model.add(Dense(19, kernel_initializer='normal', activation="sigmoid"))

	# Optimize with SGD
	# rms = RMSprop(lr=0.1, rho=0.9, epsilon=1e-08, decay=0)
	optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

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
	# print classWeights
	model.fit(x_train, y_train, epochs=3, batch_size=32, validation_data=(x_test, y_test), class_weight=classWeights)
	print("training completed")

	print("saving model")
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
	    json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("model.h5")
	print("saved to disk")

	score = model.evaluate(x_test, y_test, batch_size=32)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])

	score = model.evaluate(train_X, train_Y, batch_size=32)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])

	pred = model.predict(test_X)
	print pred
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
			myStr += str(int(x))+","
		print myStr[0:len(myStr)-1]
		myId += 1


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

	# test_X = normalizeData(test_X)

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
			myStr += str(int(x))+","
		print myStr[0:len(myStr)-1]
		myId += 1

if __name__ == "__main__":
    # pdb.set_trace()
    main()
    # print("--- %s seconds ---" % (time.time() - start_time))
    #run()