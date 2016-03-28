import numpy as np
import math
import os

trainDir = "/var/data/train/"

def divideData():
    
    labelToFileId = {}
    validationList = []
    trainList = []
    with open('trainLabels.csv', 'r') as trainLabels:
        next(trainLabels) #skip first line
        for line in trainLabels:
            line = line.rstrip()
            line = line.replace("\"","")
            tokens = line.split(",")
            label = int(tokens[1])
            fileId = tokens[0]
            if label in labelToFileId.keys():
                labelToFileId[label].append(fileId)
            else:
                idList = []
                idList.append(fileId)
                labelToFileId[label] = idList

            
    for k,v in labelToFileId.iteritems():
        idList = v
        sizeOfCategory = len(v)
        numberOfValidation = sizeOfCategory / 10
        print(k,sizeOfCategory,numberOfValidation)
        for i in range(0,sizeOfCategory):
            if i < numberOfValidation:
                validationList.append(idList[i] + "," + str(k))
            else:
                trainList.append(idList[i] + "," + str(k))

    print(len(validationList))
    print(len(trainList)) 
    

    with open("train.label","w") as f:
        for l in trainList:
            f.write(l + "\n")

    with open("validation.label","w") as f:   
        for l in validationList:
            f.write(l + "\n")



divideData()

