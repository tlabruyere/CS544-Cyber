from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils
import numpy as np
from scipy.ndimage import imread
import os
from PIL import Image


os.chdir('trainimg/')
numFiles = num_files = sum(os.path.isfile(os.path.join(os.getcwd(), f)) for f in os.listdir(os.getcwd()))
fileNames = []
for (dirp, dirn, f) in os.walk(os.getcwd()):
    fileNames.extend(f)

#Selecting a random 100 images to traint on
indices = np.random.permutation(numFiles)[:1000]

trainFileNames = []
for a in indices:
    trainFileNames.append(fileNames[a])

#Getting labels of the 100 files and converting files to numpy array
os.chdir('..')
allLabels = {}

with open('trainLabels.csv','r') as labels:
    for line in labels:
        line = line.replace('"','')
        line = line.strip('\n')
        line = line.split(',')
        allLabels[line[0]] = int(line[1])-1

trainLabels = []
testLabels = []
xtrain = []
xtest = []
os.chdir('trainimg/')
for a in trainFileNames[:-100]:
    trainLabels.append(allLabels[a[:-10]])
    img = Image.open(a)
    img = img.resize((1024,1024), Image.BILINEAR)
    xtrain.append(list(img.getdata()))

for a in trainFileNames[-100:]:
    testLabels.append(allLabels[a[:-10]])
    img = Image.open(a)
    img = img.resize((1024,1024), Image.BILINEAR)
    xtest.append(list(img.getdata()))

trainCategorical = np_utils.to_categorical(trainLabels)
testCategorical = np_utils.to_categorical(testLabels)
xtrain = np.array(xtrain)
xtest = np.array(xtest)

print(trainCategorical.shape)
print(testCategorical.shape)
print(xtest.shape)
print(xtrain.shape)
print(xtest[1].shape)
print(xtrain[1].shape)
print(xtrain[1][1])

#Build Keras model
model = Sequential()
model.add(Dense(16, input_dim=xtrain.shape[1]))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(trainCategorical.shape[1]))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta')
model.fit(xtrain, trainCategorical, nb_epoch=10)

#Running test data on keras evaluator
score,acc = model.evaluate(xtest, testCategorical, show_accuracy=True)


#adding accuracy to array for final cross validation mean calculation
print ('Accuracy: ' + str(acc))
