#/usr/bin/python

import numpy as np
import sklearn.feature_extraction.text as CountVectorizer 
import sklearn
from os import listdir
from os.path import isfile, join
import time

class BuildTrainData:
    byteExt = '.bytes'
    asmExt = '.asm'
    id2FileName_file = ''
    id2Label_file = ''
    trainingDir = ''
    id2FileMap = {}
    trainDataMatrix = np.empty([1,1])

    def __init__(self, id2FileNameFile, csvLabels, trainDir):
        # setup local variables
        self.id2FileName_file = id2FileNameFile
        self.id2Label_file = csvLabels
        self.trainingDir = trainDir
        # setup lookup data structures 
        self.id2FileMap = self.loadMappingFile(id2FileNameFile)
        self.loadIdAndLabel(csvLabels)

    def genNgramsforFile(self):
        onlyfiles = [join(self.trainingDir, f) for f in listdir(self.trainingDir)]# if isfile(join(self.trainingDir, f))]
        oldtime = time.time()
        vectorizor = CountVectorizer.CountVectorizer(input='filename',ngram_range=(4,4),decode_error='ignore')
        data = vectorizor.fit_transform(onlyfiles)
        print 'vectorizing takes ',time.time()-oldtime
        print 'data size = ',data.shape

        
    def loadASMFileById(self, ID):
        # ID must be an int
        fileName = self.id2FileMap[ID] + self.asmExt
        oldTime= time.time()
        fileObj = open(os.path.join(self.trainingDir, fileName))
        oldTime = time.time()
        fileContents = fileObj.read()
        return fileContents

    def loadMappingFile(self, fileId):
        fid = open(fileId)
        retMap = {}
        with  open(fileId) as infile:
            retMap = dict((int(rows.split(',')[0].strip()),rows.split(',')[1].strip()) for rows in infile)
        return retMap

    def loadIdAndLabel(self, csvLabels):
       self.trainDataMatrix = np.loadtxt(open(csvLabels, 'rb'), delimiter=',')


if __name__=='__main__':
    loader = BuildTrainData('data/mappingFile.csv', 'data/idLabels.csv', 'data/train/asm')
    loader.genNgramsforFile()
#    dat = loader.build('data/mappingFile.csv', 'data/idLabels.csv', 'data/train')
#    print dat.shape
#    print dat[1,:]
