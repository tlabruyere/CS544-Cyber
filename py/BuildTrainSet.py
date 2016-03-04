#/usr/bin/python

import numpy as np

class BuildTrainData:
    def build(self, csvLabels, trainDir):
       data = np.loadtxt(open(csvLabels, 'rb'), delimiter=',')

if __name__=='__main__':
    loader = BuildTrainData()
    dat = loader.build('/var/data/trainLabels.csv', '/var/data/train')
    print dat[1,:]
