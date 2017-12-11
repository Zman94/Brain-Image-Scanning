'''
'
'''
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def load_data():
    '''
    '
    '''
    tag_name = np.load('./CS446-project/tag_name.npy')
    trainY = np.load('./CS446-project/train_binary_Y.npy')
    trainX = np.load('./CS446-project/train_X.npy')
    testX = np.load('./CS446-project/valid_test_X.npy')
    return tag_name, trainY, trainX, testX

def manual_reshape(trainX):
    '''
    '
    '''
    train_X_2d = []
    for i in enumerate(trainX):
        train_X_2d.append([])
        for j in enumerate(trainX[i]):
            for k in enumerate(trainX[i][j]):
                train_X_2d[i] += list(trainX[i][j][k])
    for i in enumerate(train_X_2d):
        train_X_2d[i] = np.array(train_X_2d[i])
    train_X_2d = np.array(train_X_2d)

def vanilla_nn(trainX, trainY, testX):
    '''
    ' 46% run
    '''
    mlp = MLPClassifier()
    mlp.fit(trainX, trainY)
    predictions = mlp.predict(testX)

def main():
    '''
    '
    '''
    ### Loading Data ###
    tag_name, trainY, trainX, testX = load_data()
    print(tag_name)
    print("tag_name shape =", tag_name.shape)
    print("trainY shape =", trainY.shape)
    print("trainX shape =", trainX.shape)
    print("testX shape =", testX.shape)

    ### Reshape to 2D array ###
    train_X_2d = trainX.reshape((len(trainX), \
               len(trainX[0])*len(trainX[0][0])*len(trainX[0][0][0])))
    test_X_2d = testX.reshape((len(testX), \
               len(testX[0])*len(testX[0][0])*len(testX[0][0][0])))

    ### Scale Data ###
    scaler = StandardScaler()
    scaler.fit(train_X_2d)
    train_X_2d = scaler.transform(train_X_2d)

    ### Generate Labels ###
    trainY_temp = np.empty((len(trainY)), dtype=object)
    for i, label in enumerate(trainY):
        temp_str = ""
        for j, val in enumerate(label):
            if val == 1:
                temp_str += (str(j)+",")
        temp_str = temp_str[:-1]
        trainY_temp[i] = temp_str

    ### Train/Test Split ###
    x_train, x_test, y_train, y_test = train_test_split(train_X_2d[:250], trainY[:250], test_size = .25)
    # x_train = train_X_2d
    # y_train = trainY
    # x_test = test_X_2d


    ### Train Classifier ###
    # vanilla_nn(train_X_2d, trainY, testX)

    ### Train Classifier (train/test) ###
    predictions = []
    mlp = MLPClassifier(hidden_layer_sizes=(500, 50))
    mlp.fit(x_train, y_train)
    predictions = mlp.predict(x_test)
    predictions_proba = mlp.predict_log_proba(x_test)
    # for i in range(len(y_train[0])):
        # mlp = MLPClassifier(hidden_layer_sizes=(1500, 50))
        # mlp.fit(x_train, y_train[:, i])
        # predictions.append(mlp.predict(x_test))
        # print("----", tag_name[i] + ",", i, "----")
        # print(accuracy_score(y_test[:, i], predictions[i]))

    # np_predict = np.column_stack((np.array(i) for i in predictions))
    # print()

    # retVal = ''
    # for count, p in enumerate(np_predict):
        # retVal += ','.join(str(x) for x in p)
        # retVal = str(count)+','+retVal
        # print(retVal)
        # retVal = ''

    print(accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(predictions)
    print(predictions_proba)

if __name__ == "__main__":
    # pdb.set_trace()
    STARTTIME = time.time()
    main()
    print("--- {0} minutes {1} seconds ---".format( \
       int((time.time() - STARTTIME)/60), int((time.time() - STARTTIME)%60)))
