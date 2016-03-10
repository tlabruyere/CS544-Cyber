#/usr/bin/python

import numpy as np
#import sklearn.feature_extraction.text as CountVectorizer 
import sklearn

class BuildTrainData:
    byteExt = '.bytes'
    asmExt = '.asm'

    def build(self, csvLabels, trainDir):
       data = np.loadtxt(open(csvLabels, 'rb'), delimiter=',')
       return data

if __name__=='__main__':
    loader = BuildTrainData()
    dat = loader.build('/home/tom/idLabels.csv', '/var/data/train')
    print dat.shape
    print dat[1,:]
