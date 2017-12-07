import time
# import pdb
import sys
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import random
from keras.models import Sequential
from keras.layers import Conv3D
from keras.layers.convolutional import MaxPooling3D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import model_from_json
from keras.optimizers import SGD, RMSprop, Adam

def load_data():
    tag_name = np.load('./CS446-project/tag_name.npy')
    train_Y  = np.load('./CS446-project/train_binary_Y.npy')
    train_X  = np.load('./CS446-project/train_X.npy')
    test_X   = np.load('./CS446-project/valid_test_X.npy')
    return tag_name, train_Y, train_X, test_X

def manual_reshape(train_X):
    train_X_2d = []
    for i in range(len(train_X)):
        train_X_2d.append([])
        for j in range(len(train_X[i])):
            for k in range(len(train_X[i][j])):
                train_X_2d[i] += list(train_X[i][j][k])
    for i in range(len(train_X_2d)):
        train_X_2d[i] = np.array(train_X_2d[i])
    train_X_2d = np.array(train_X_2d)

### 46% ###
def vanilla_nn(train_X, train_Y, test_X):
    mlp = MLPClassifier()
    mlp.fit(train_X, train_Y)
    predictions = mlp.predict(test_X)

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

    # train_X = train_X_2d.reshape((num_dims, img_rows, img_cols, img_depth))
    return train_X_2d

def main():
    ### Loading Data ###
    tag_name, train_Y, train_X, test_X = load_data()
    print(tag_name)
    print("tag_name shape =", tag_name.shape)
    print("train_Y shape =", train_Y.shape)
    print("train_X shape =", train_X.shape)
    print("test_X shape =", test_X.shape)
    train_X = normalizeData(train_X)
    img_rows = 26
    img_cols = 31
    img_depth = 23
    input_shape = (img_rows, img_cols, img_depth, 1)
    ### Reshape to 2D array ###
    # train_X_2d = train_X.reshape((len(train_X), \
    #            len(train_X[0])*len(train_X[0][0])*len(train_X[0][0][0])))

    ### Scale Data ###
    # scaler = StandardScaler()
    # scaler.fit(train_X_2d)
    # train_X_2d = scaler.transform(train_X_2d)

    ### Train/Test Split ###
    # x_train, x_test, y_train, y_test = train_test_split(train_X_2d, train_Y, test_size = .2)

    myDict = {}
    for i in range(len(train_X)):
        for j in range(len(train_Y[i])):
            if j not in myDict.keys():
                myDict[j] = [[],[]]
            if train_Y[i][j] == 1:
                myDict[j][0].append(train_X[i])
            else:
                myDict[j][1].append(train_X[i])

    ### Train Classifier ###
    # vanilla_nn(train_X_2d, train_Y, test_X)

    ### Train Classifier (train/test) ###
    predictions = []
    classifiers = []
    for i in range(len(tag_name)):
        if i != 9 and i != 14:
            continue
        # mlp = MLPClassifier(solver='lbfgs', alpha=.3, activation='relu', learning_rate='adaptive', max_iter=1000, validation_fraction=.2, early_stopping=True)
        myX = []
        myY = []
        samples = myDict[i]
        for _ in range(5000):
            myX.append(random.choice(myDict[i][0]))
            myY.append(1)
        for _ in range(5000):
            myX.append(random.choice(myDict[i][1]))
            myY.append(0)
        myX = np.array(myX)
        myY = np.array(myY)
        x_train, x_test, y_train, y_test = train_test_split(myX, myY, test_size = .2)
        '''
            cnn for each label
        
        #adapted from: https://github.com/MinhazPalasara/keras/blob/master/examples/shapes_3d_cnn.py
        filters = [32,64]
        kernel = [(7,7,7), (3,3,3)]
        pool = [3,3]

        model = Sequential()
        model.add(Conv3D(filters[0],kernel[0], input_shape=input_shape, activation="relu"))
        # model.add(Conv3D(filters[1],kernel[1], activation="relu"))
        model.add(MaxPooling3D(pool_size=(pool[0], pool[0], pool[0])))
        # model.add(MaxPooling3D(pool_size=(pool[1], pool[1], pool[1])))
        model.add(Flatten())
        # model.add(Dropout(.25))
        # model.add(Dense(512, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(1, kernel_initializer='normal', activation="sigmoid"))

        # Optimize with SGD
        # rms = RMSprop(lr=0.1, rho=0.9, epsilon=1e-08, decay=0)
        # optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='binary_crossentropy', optimizer="sgd", metrics=['accuracy'])
            cnn ended
        '''
        classifier = svm.SVC()
        print "Training "+str(i)
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)
        print "score: ", accuracy_score(y_test, predictions)
        

        

    # np_predict = np.column_stack((np.array(i) for i in predictions))
    # print(accuracy_score(y_test, np_predict))

    ### Reshape to 2D array ###
    # test_X_2d = test_X.reshape((len(test_X), \
    #            len(test_X[0])*len(test_X[0][0])*len(test_X[0][0][0])))

    # ### Scale Data ###
    # scaler = StandardScaler()
    # scaler.fit(test_X_2d)
    # test_X = scaler.transform(test_X_2d)

    # myId = 0
    # for x in test_X:
    #     retVal =""+str(myId)+","
    #     for i in range(len(tag_name)):
    #         pred = classifiers[i].predict([x])[0]
    #         if int(pred) == 1:
    #             retVal+=tag_name[i]+" "
    #     retVal = retVal[:len(retVal)-1]
    #     print retVal
    #     myId+=1

    # print(classification_report(y_test, predictions))
    # print(predictions)

if __name__ == "__main__":
    # pdb.set_trace()
    # start_time = time.time()
    main()
    # print("--- {0} minutes {1} seconds ---".format( \
    #    int((time.time() - start_time)/60), int((time.time() - start_time)%60)))
