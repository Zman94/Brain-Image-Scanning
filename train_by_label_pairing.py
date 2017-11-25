import random
from sklearn.preprocessing import StandardScaler
import numpy as np
import cPickle
from sklearn.model_selection import train_test_split
import random
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn
img_rows = 26
img_cols = 31
img_depth = 23
tag_name = np.load('./CS446-project/tag_name.npy')
train_Y  = np.load('./CS446-project/train_binary_Y.npy')
train_X  = np.load('./CS446-project/train_X.npy')
test_X   = np.load('./CS446-project/valid_test_X.npy')
### Reshape to 2D array ###
train_X_2d = train_X.reshape((len(train_X), \
           len(train_X[0])*len(train_X[0][0])*len(train_X[0][0][0])))

### Scale Data ###
scaler = StandardScaler()
scaler.fit(train_X_2d)
train_X = scaler.transform(train_X_2d)


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
print myDict.keys()
print len(myDict.keys())

myClassifiers = []
for k in myDict.keys():
	myListP = myDict[k]
	noOverlap = []
	someOverlap = []
	classes = k.split(',')
	
	for otherKeys in myDict.keys():
		if otherKeys == k:
			continue
		noOverlapFlag = True
		for c in classes:
			if c in otherKeys:
				noOverlapFlag = False
				break
		if noOverlapFlag:
			noOverlap += myDict[otherKeys]
		else:
			someOverlap += myDict[otherKeys]
	positiveSamples = [random.choice(myListP) for _ in range(3000)]
	# negativeSamples = [random.choice(noOverlap) for _ in range(2000)]
	negativeSamples = []
	for i in range(3000):
		if i < 2000:
			negativeSamples.append(random.choice(noOverlap))
		else:
			negativeSamples.append(random.choice(someOverlap))
	myX = []
	myY = []
	for p in positiveSamples:
		myX.append(p)
		myY.append(1)
	for n in negativeSamples:
		myX.append(n)
		myY.append(0)

	x_train, x_test, y_train, y_test = train_test_split(myX, myY, test_size = .25)
	classifier = MLPClassifier(solver='lbfgs', alpha=.3, activation='relu', learning_rate='adaptive', max_iter=1000, validation_fraction=.2, early_stopping=True)
	print "training for: "+k
	classifier.fit(x_train, y_train)
	predictions = classifier.predict(x_test)
	print "score: ", accuracy_score(y_test, predictions)

	finalVector = np.zeros(19)
	for c in classes:
		finalVector[int(c)] = 1
	#print finalVector
	myClassifiers.append((finalVector,classifier))

### Reshape to 2D array ###
test_X_2d = test_X.reshape((len(test_X), \
           len(test_X[0])*len(test_X[0][0])*len(test_X[0][0][0])))

### Scale Data ###
scaler = StandardScaler()
scaler.fit(test_X_2d)
test_X = scaler.transform(test_X_2d)

myId = 0
for x in test_X:
	retVal =""+str(myId)+","
	pred = [c[1].predict_proba([x])[0][1] for c in myClassifiers]
	pred = np.array(pred)
	best = np.argmax(pred)
	finalPrediction = myClassifiers[best][0]
	for p in finalPrediction:
		retVal+=str(int(p))+","
	retVal = retVal[:len(retVal)-1]
	print retVal
	myId+=1
