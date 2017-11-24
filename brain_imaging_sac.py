import time
# import pdb
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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

def main():
    ### Loading Data ###
    tag_name, train_Y, train_X, test_X = load_data()
    print(tag_name)
    print("tag_name shape =", tag_name.shape)
    print("train_Y shape =", train_Y.shape)
    print("train_X shape =", train_X.shape)
    print("test_X shape =", test_X.shape)

    ### Reshape to 2D array ###
    train_X_2d = train_X.reshape((len(train_X), \
               len(train_X[0])*len(train_X[0][0])*len(train_X[0][0][0])))

    ### Scale Data ###
    scaler = StandardScaler()
    scaler.fit(train_X_2d)
    train_X_2d = scaler.transform(train_X_2d)

    ### Train/Test Split ###
    x_train, x_test, y_train, y_test = train_test_split(train_X_2d, train_Y, test_size = .25)

    ### Train Classifier ###
    # vanilla_nn(train_X_2d, train_Y, test_X)

    ### Train Classifier (train/test) ###
    predictions = []
    for i in range(len(y_train[0])):
        mlp = MLPClassifier()
        mlp.fit(x_train, y_train[:, i])
        predictions.append(mlp.predict(x_test))
        print("----", tag_name[i] + ",", i, "----")
        print(accuracy_score(y_test[:, i], predictions[i]))

    np_predict = np.column_stack((np.array(i) for i in predictions))
    print(accuracy_score(y_test, np_predict))

    # print(classification_report(y_test, predictions))
    # print(predictions)

if __name__ == "__main__":
    # pdb.set_trace()
    start_time = time.time()
    main()
    print("--- {0} minutes {1} seconds ---".format( \
       int((time.time() - start_time)/60), int((time.time() - start_time)%60)))
