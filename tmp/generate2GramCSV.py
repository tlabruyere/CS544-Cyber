import numpy as np

numberOfWordsInVocabulary = 65536
numberOfSamples = 10868

sampleMatrix = np.zeros((numberOfSamples,numberOfWordsInVocabulary))

fileIdToLabel = {}
fileIndexToLabel = []

labels = np.zeros((numberOfSamples,))
with open('trainLabels.csv', 'r') as trainLabels:
    next(trainLabels) #skip first line
    index = 0
    for line in trainLabels:
        line = line.rstrip()
        line = line.replace("\"","")
        tokens = line.split(",")
        label = int(tokens[1])
        fileId = tokens[0]
        fileIdToLabel[fileId] = label
        #fileIndexToLabel.append(label)
        index = index + 1



rowIndex = 0
lastFileId = ''
count = 0
with open('train-full.data','r') as trainData:
    for line in trainData:
        values = line.rstrip('\n').split(" ")
        fileId = values[0]
        #check for first iteration when lastFileId is uninitialized
        if lastFileId == '':
            lastFileId = fileId
        if lastFileId != fileId:
            rowIndex = rowIndex + 1
            lastFileId = fileId
        
        label = fileIdToLabel[values[0]]
        labels[rowIndex] = int(label)
        wordId = int(values[1])     
        wordCount = int(values[2])
        sampleMatrix[rowIndex,wordId] = wordCount
        count = count + 1
        if count % 10000000 == 0:
	    print("x " + str(count))
        
        
with open('train.csv','a') as trainCSV:
    for row in range(0,numberOfSamples):
        rowString = str(labels[row])
        for column in range(0,numberOfWordsInVocabulary):
            rowString = rowString + ',' + str(sampleMatrix[row,column])
        rowString = rowString + '\n'
        trainCSV.write(rowString)
        if row % 100 == 0:
            print("y " + str(row))
