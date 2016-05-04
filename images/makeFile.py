import numpy as np
from scipy.ndimage import imread
import os
from PIL import Image
import leargist
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils import shuffle
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier
import time


os.chdir('trainimg/')
numFiles = num_files = sum(os.path.isfile(os.path.join(os.getcwd(), f)) for f in os.listdir(os.getcwd()))
fileNames = []
for (dirp, dirn, f) in os.walk(os.getcwd()):
    fileNames.extend(f)

os.chdir('..')
allLabels = {}
with open('trainLabels.csv','r') as labels:
    for line in labels:
        line = line.replace('"','')
        line = line.strip('\n')
        line = line.split(',')
        allLabels[line[0]] = int(line[1])-1
#os.chdir('trainimg/')

cnt=0
wr = open('gist2.data','w')
for a in fileNames:
    label = allLabels[a[:-10]]
    img = Image.open('trainimg/'+a)
    img = img.resize((64,64), Image.ANTIALIAS)
    des = leargist.color_gist(img)
    t = des[0:320]
    t2 = str(np.append(t,int(label)).tolist()) + '\n'
    wr.write(t2)
    cnt +=1
    print(cnt)

