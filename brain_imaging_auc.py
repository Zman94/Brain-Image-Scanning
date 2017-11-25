import pdb
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sklearn
import cPickle

def main():
    train()
    test()


def train():
    ### Loading Data ###
    tag_name = np.load('./CS446-project/tag_name.npy')
    train_Y  = np.load('./CS446-project/train_binary_Y.npy')
    train_X  = np.load('./CS446-project/train_X.npy')
    test_X   = np.load('./CS446-project/valid_test_X.npy')
    print(tag_name)
    print("tag_name shape =", tag_name.shape)
    print("train_Y shape =", train_Y.shape)
    print("train_X shape =", train_X.shape)
    print("test_X shape =", test_X.shape)
    counts = np.zeros((19,))
    for y in train_Y:
        for i in range(len(y)):
            if y[i] == 1:
                counts[i] += 1
    print counts
    print len(train_Y)
    input_train = []
    for image in train_X:
        flatten_image = image.flatten()
        input_train.append(flatten_image)

    x_train, x_test, y_train, y_test = train_test_split(input_train, train_Y, test_size = .25)
    classifier = MLPClassifier(solver='adam', alpha=.3, activation='logistic', learning_rate='adaptive', max_iter=500, validation_fraction=.3, early_stopping=True)
    #classifier = MLPClassifier()
    print("training classifier")
    classifier.fit(x_train, y_train)
    print("training completed")

    print("saving data")
    
    #https://stackoverflow.com/questions/10592605/save-classifier-to-disk-in-scikit-learn
    with open('my_dumped_classifier.pkl', 'wb') as fid:
        cPickle.dump(classifier, fid)

    with open('x_train.pkl', 'wb') as fid:
        cPickle.dump(x_train, fid)

    with open('x_test.pkl', 'wb') as fid:
        cPickle.dump(x_test, fid)

    with open('y_train.pkl', 'wb') as fid:
        cPickle.dump(y_train, fid)

    with open('y_test.pkl', 'wb') as fid:
        cPickle.dump(y_test, fid)

    print("saving completed")

def test():
    # load it again
    print("load classifier")
    classifier = None
    with open('my_dumped_classifier.pkl', 'rb') as fid:
        classifier = cPickle.load(fid)
    print("load classifier completed")

    x_train = None
    x_test = None
    y_train = None
    y_test = None
    print("load data")
    with open('x_train.pkl', 'rb') as fid:
        x_train = cPickle.load(fid)
    with open('x_test.pkl', 'rb') as fid:
        x_test = cPickle.load(fid)
    with open('y_train.pkl', 'rb') as fid:
        y_train = cPickle.load(fid)
    with open('y_test.pkl', 'rb') as fid:
        y_test = cPickle.load(fid)
    print("loading data completed")

    train_predictions = classifier.predict(x_train)
    print(sklearn.metrics.accuracy_score(y_train, train_predictions))
    test_predictions = classifier.predict(x_test)
    print(sklearn.metrics.accuracy_score(y_test, test_predictions))

if __name__ == "__main__":
    # pdb.set_trace()
    main()
    # print("--- %s seconds ---" % (time.time() - start_time))
