
import matplotlib

matplotlib.use('Agg')

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt
from matplotlib import pylab
from sklearn.externals import joblib



families = ['Ramnit','Lollipop','Kelihos_ver3','Vundo','Simda','Tracur','Kelihos_ver1','Obfuscator.ACY','Gatak']

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





X_test = joblib.load('models/X_test.pkl')
y_test = joblib.load('models/y_test.pkl')

pipe_dt = joblib.load('models/DecisionTreePipeline.pkl') 
validate(X_test, y_test, pipe_dt, "Confusion Matrix for Decision Tree", "DecisionTreeConfusionMatrix.png")


pipe_nb = joblib.load('models/NaiveBayesPipeline.pkl')
validate(X_test, y_test, pipe_nb, "Confusion Matrix for Naive Bayes", "NaiveBayesConfusionMatrix.png")



pipe_rf = joblib.load('models/RandomForestPipeline.pkl')
validate(X_test, y_test, pipe_rf, "Confusion Matrix for Random Forest", "RandomForestConfusionMatrix.png")


pipe_svm = joblib.load('models/StochasticSVMPipeline.pkl')
validate(X_test, y_test, pipe_svm, "Confusion Matrix for SVM", "SVMConfusionMatrix.png")


pipe_log = joblib.load('models/StochasticLogisticRegressionPipeline.pkl')
validate(X_test, y_test, pipe_log, "Confusion Matrix for Logistic Regression", "LogisticRegressionMatrix.png")




