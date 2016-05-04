import numpy as np
from scipy.ndimage import imread
import os
from PIL import Image
import leargist
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import time
import matplotlib.pyplot as plt
from matplotlib import pylab
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier 
from sklearn import svm


def validate(X_test, y_test, pipe, title, fileName):
    
    print('Test Accuracy: %.3f' % pipe.score(X_test, y_test))

    y_predict = pipe.predict(X_test)

    confusion_matrix = np.zeros((9,9))

    for p,r in zip(y_predict, y_test):
        confusion_matrix[p-1,r-1] = confusion_matrix[p-1,r-1] + 1

    print (confusion_matrix) 

    confusion_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    print (confusion_normalized)

    pylab.clf()
    pylab.matshow(confusion_normalized, fignum=False, cmap='Blues', vmin=0.0, vmax=1.0)
    ax = pylab.axes()
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families,  fontsize=4)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families, fontsize=4)
    pylab.title(title)
    pylab.colorbar()
    pylab.grid(False)
    pylab.grid(False)
    pylab.savefig(fileName, dpi=900)


#os.chdir('trainimg/')
#numFiles = num_files = sum(os.path.isfile(os.path.join(os.getcwd(), f)) for f in os.listdir(os.getcwd()))
#fileNames = []
#for (dirp, dirn, f) in os.walk(os.getcwd()):
#    fileNames.extend(f)

X = []
Y = []
pp = True
with open('gist2.data','r') as d:
    for line in d:
        line = line[1:-2]
        data = line.split(',')
        X.append(data[0:320])
        Y.append(data[320]) 


numFiles = len(Y)
indices = np.random.permutation(numFiles).tolist()
print(numFiles)
trainAmt = int(numFiles/20)
xtrain = []
ytrain = []
for a in indices[:trainAmt]:
    xtrain.append(X[a])
    ytrain.append(Y[a])

xtest = []
ytest = []
for a in indices[trainAmt:]:
    xtest.append(X[a])
    ytest.append(Y[a])

clf = []
clf = KNeighborsClassifier(1,weights='distance') 
tic = time.time()
clf.fit(xtrain,ytrain) 
toc = time.time()
print 'model fit in: ' + str(toc-tic) + ' Seconds.'

 # Testing
predict = []
#tic = time.time()
predict = clf.predict(xtest) # output is labels and not indices
#toc = time.time()
print('Accuracy: ' + str(clf.score(xtest,ytest)))
conf_m = confusion_matrix(ytest,predict)
print(conf_m)

forest = RandomForestClassifier(n_estimators = 1000)
forest = forest.fit(xtrain,ytrain)
print 'Forest Accuracy: ' + str(forest.score(xtest,ytest)) 


sv = svm.SVC(kernel='rbf',C=100.0, random_state=1)
sv.fit(xtrain,ytrain)
print 'SVM Accuracy: ' + str(sv.score(xtest,ytest))
