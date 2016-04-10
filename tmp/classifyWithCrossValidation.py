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


#X, y = shuffle(X, y, random_state=0)
#print(y)
'''    

df = pd.read_csv('train.csv', header=None)
print("read data into pandas DF")

imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df)
df_imputed = pd.DataFrame(imr.transform(df.values))

print("removed missing data")

X = df_imputed.loc[:, 1:].values
y = df_imputed.loc[:, 0].values
'''



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=1)

#print(y_test)

print("split data into train and test")

#pipe_lr = Pipeline([('scl', StandardScaler()), ('pca',PCA()) , ('clf', LogisticRegression(random_state=1))])
#pipe_lr = Pipeline([ ('clf', LogisticRegression(random_state=1))])
#pipe_lr = Pipeline([('scl', StandardScaler()) ,('clf',SVC(kernel='linear', C=10.0, random_state=1))])
#pipe_lr = Pipeline([('clf',SVC(kernel='linear', C=10.0, random_state=1))])
pipe_lr = Pipeline([('clf', DecisionTreeClassifier(criterion='entropy',max_depth=40, random_state=0))])
#pipe_lr = Pipeline([('scl', StandardScaler()),('clf', RandomForestClassifier(criterion='entropy',n_estimators=1000,n_jobs=4, random_state=1))])
#pipe_lr = Pipeline([('scl', StandardScaler()),('clf', KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski'))])
#pipe_lr = Pipeline([('scl', StandardScaler()),('clf', GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)  )])
#pipe_lr = Pipeline([('scl', StandardScaler()),('clf', GaussianNB()  )])
#pipe_lr = Pipeline([('clf', MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)  )])



'''
pipe_lr = Pipeline([ ('clf', SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='log', n_iter=5, n_jobs=1,
       penalty='l2', power_t=0.5, random_state=None, shuffle=True,
       verbose=2, warm_start=False))])
'''

print("starting to build classifier")

#pipe_lr.fit(X_train, y_train)
#print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))


scores = cross_val_score(estimator=pipe_lr, X = X_train, y=y_train, cv=10, n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
