import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle
from sklearn.externals import joblib


X = np.zeros((10868,65536))
y = np.zeros((10868,))
row = 0
column = 0
with open('train.csv', 'r') as data:
    for line in data:
        line = line.rstrip()
        tokens = line.split(',')
        y[row] = float(tokens[0])
        column = 0
        for token in tokens[1:]:
            X[row,column] = float(token)
            column = column + 1
        row = row + 1
        if row % 1000 == 0:
            print(row)



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=1)


#joblib.dump(X_train, 'models/X_train.pkl')

#joblib.dump(y_train, 'models/y_train.pkl')


print("split data into train and test")

#pipe_lr = Pipeline([('scl', StandardScaler()) , ('clf', LogisticRegression(random_state=1))])
#pipe_lr = Pipeline([ ('clf', LogisticRegression(random_state=1))])
#pipe_lr = Pipeline([('scl', StandardScaler()) ,('clf',SVC(kernel='linear', C=10.0, random_state=1))])
#pipe_lr = Pipeline([('clf',SVC(kernel='linear', C=10.0, random_state=1))])
pipe_dt = Pipeline([('clf', DecisionTreeClassifier(criterion='entropy',max_depth=40, random_state=1))])
pipe_rf = Pipeline([('clf', RandomForestClassifier(criterion='entropy',n_estimators=1000,n_jobs=-1, random_state=1))])
pipe_kn = Pipeline([('scl', StandardScaler()),('clf', KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski'))])
#pipe_lr = Pipeline([('scl', StandardScaler()),('clf', GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)  )])
#pipe_lr = Pipeline([('scl', StandardScaler()),('clf', GaussianNB()  )])
pipe_nb = Pipeline([('clf', MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)  )])



#stochastic svm
pipe_stochastic_svm = Pipeline([ ('scl', StandardScaler()), ('clf', SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1,
       penalty='l2', power_t=0.5, random_state=1, shuffle=True,
       verbose=0, warm_start=False))])




#stochastic logistic regression
pipe_lr = Pipeline([ ('scl', StandardScaler()),('clf', SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='log', n_iter=5, n_jobs=1,
       penalty='l2', power_t=0.5, random_state=None, shuffle=True,
       verbose=0, warm_start=False))])




print("***** Random Forest ******")

scores = cross_val_score(estimator=pipe_rf, X = X_train, y=y_train, cv=10, n_jobs=-1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))

pipe_rf.fit(X_train, y_train)
print('Test Accuracy: %.3f' % pipe_rf.score(X_test, y_test))

joblib.dump(pipe_rf, 'models/RandomForestPipeline.pkl') 



print("***** Stochastic SVM ******")
scores = cross_val_score(estimator=pipe_stochastic_svm, X = X_train, y=y_train, cv=10, n_jobs=-1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
pipe_stochastic_svm.fit(X_train, y_train)
print('Test Accuracy: %.3f' % pipe_stochastic_svm.score(X_test, y_test))
joblib.dump(pipe_stochastic_svm, 'models/StochasticSVMPipeline.pkl') 


print("***** Decision Tree ******")
scores = cross_val_score(estimator=pipe_dt, X = X_train, y=y_train, cv=10, n_jobs=-1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
pipe_dt.fit(X_train, y_train)
print('Test Accuracy: %.3f' % pipe_dt.score(X_test, y_test))
joblib.dump(pipe_dt, 'models/DecisionTreePipeline.pkl')


print("***** Logistic Regression ******")
scores = cross_val_score(estimator=pipe_lr, X = X_train, y=y_train, cv=10, n_jobs=-1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
pipe_lr.fit(X_train, y_train)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))
joblib.dump(pipe_lr, 'models/StochasticLogisticRegressionPipeline.pkl')



print("***** Naive Bayes ******")
scores = cross_val_score(estimator=pipe_nb, X = X_train, y=y_train, cv=10, n_jobs=-1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
pipe_nb.fit(X_train, y_train)
print('Test Accuracy: %.3f' % pipe_nb.score(X_test, y_test))
joblib.dump(pipe_nb, 'models/NaiveBayesPipeline.pkl')


'''
print("***** K Nearest Neighbors ******")
scores = cross_val_score(estimator=pipe_kn, X = X_train, y=y_train, cv=10, n_jobs=-1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
pipe_kn.fit(X_train, y_train)
print('Test Accuracy: %.3f' % pipe_kn.score(X_test, y_test))
joblib.dump(pipe_kn, 'models/KNearestNeighborsPipeline.pkl')
'''
