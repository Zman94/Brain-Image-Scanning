import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
import cPickle

'''
    Load data from npy files
'''
def load_data():
    tag_name = np.load('./CS446-project/tag_name.npy')
    train_Y  = np.load('./CS446-project/train_binary_Y.npy')
    train_X  = np.load('./CS446-project/train_X.npy')
    test_X   = np.load('./CS446-project/valid_test_X.npy')
    return tag_name, train_Y, train_X, test_X

'''
    Scale data and flatten to one vector for each sample
'''
def normalizeData(train_X):
    train_X_2d = train_X.reshape((len(train_X), \
               len(train_X[0])*len(train_X[0][0])*len(train_X[0][0][0])))
    scaler = StandardScaler()
    scaler.fit(train_X_2d)
    train_X_2d = scaler.transform(train_X_2d)

    return train_X_2d

'''
    prints outs csv for test_X and list of trained classifiers for each label
'''
def predictCSV(test_X, classifiers):
    for i in range(len(test_X)):
        x = test_X
        csvLine = ""+str(i)+","
        for c in classifiers:
            pred = c.predict([x])[0]
            csvLine += str(pred)+","
        print csvLine[0:len(csvLine)-1]

'''
    Load trained classifiers and predict on training and predict on test
    (only call if you have models saved through cPickle)
'''
def loadAndPredict():
    tag_name, train_Y, train_X, test_X = load_data()
    print(tag_name)
    print("tag_name shape =", tag_name.shape)
    print("train_Y shape =", train_Y.shape)
    print("train_X shape =", train_X.shape)
    print("test_X shape =", test_X.shape)

    classifiers = []
    for i in range(19):
        print "loading "+str(i)
        with open('./Classifiers/c'+str(i)+'.pkl', 'rb') as fid:
            classifiers.append(cPickle.load(fid))
        print "done loading "+str(i)

    train_X = normalizeData(train_X)
    test_X = normalizeData(test_X)

    predictions = []
    for i in range(len(train_X)):
        x = train_X[i]
        predictions.append(np.zeros(19))
        for j in len(classifiers):
            predictions[i][j] = classifiers[j].predict([x])[0]
    predictions = np.array(predictions)
    print "training score: ", accuracy_score(train_Y, predictions)

    predictCSV(test_X, classifiers)


def train():
    ### Loading Data ###
    tag_name, train_Y, train_X, test_X = load_data()
    print(tag_name)
    print("tag_name shape =", tag_name.shape)
    print("train_Y shape =", train_Y.shape)
    print("train_X shape =", train_X.shape)
    print("test_X shape =", test_X.shape)

    ### Normalize data ###
    train_X = normalizeData(train_X)

    ### Create a dictionary for positive/negative examples for each label ###
    #(key, value) -> (label, [[list of positive examples],[list of negative examples]])
    myDict = {}
    for i in range(len(train_X)):
        for j in range(len(train_Y[i])):
            if j not in myDict.keys():
                myDict[j] = [[],[]]
            if train_Y[i][j] == 1:
                myDict[j][0].append(train_X[i])
            else:
                myDict[j][1].append(train_X[i])

    #train a classifier for each label
    classifiers = []
    for i in range(len(tag_name)):
        myX = []
        myY = []
        # sample 5000 positive examples
        for _ in range(5000):
            myX.append(random.choice(myDict[i][0]))
            myY.append(1)
        # sample 5000 negative examples
        for _ in range(5000):
            myX.append(random.choice(myDict[i][1]))
            myY.append(0)
        myX = np.array(myX)
        myY = np.array(myY)

        #train/test split on sampled data
        x_train, x_test, y_train, y_test = train_test_split(myX, myY, test_size = .2)
        
        classifier = svm.SVC()
        
        print "Training "+str(i)
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)
        print "score: ", accuracy_score(y_test, predictions)
        print "saving classifier "+str(i)
        with open('./Classifiers/c'+str(i)+'.pkl', 'wb') as fid:
            cPickle.dump(classifier, fid)
        print "classifier "+str(i)+" saved"
        # x = x_test[0]
        # print classifier.predict([x])[0]
        # print y_test[0]
        classifiers.append(classifier)

    #predict
    test_X = normalizeData(test_X)
    predictCSV(test_X, classifiers)

if __name__ == "__main__":
    train()
    #loadAndPredict()
