import pdb
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

def main():
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

    ### Scale Data ###

    ### Train Classifier ###


if __name__ == "__main__":
    # pdb.set_trace()
    main()
    # print("--- %s seconds ---" % (time.time() - start_time))
