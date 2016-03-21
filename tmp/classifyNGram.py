import numpy as np
import math
import os

trainDir = "/var/data/train/"

def classify(beta):
    numberOfWordsInVocabulary = 65536
    numberOfClasses = 9
    numberOfTestingExamples = 10873 
    

    classProbabilityMatrix = np.zeros((numberOfClasses,))
    wordProbabilityMatrix = np.zeros((numberOfWordsInVocabulary,numberOfClasses))
    wordCountMatrix = np.zeros((numberOfWordsInVocabulary,numberOfClasses))

    fileIdToLabel = {}
    
    with open('trainLabels.csv', 'r') as trainLabels:
        next(trainLabels) #skip first line
        for line in trainLabels:
            line = line.rstrip()
            line = line.replace("\"","")
            tokens = line.split(",")
            label = int(tokens[1])
            fileId = tokens[0]
            fileIdToLabel[fileId] = label
            classProbabilityMatrix[label-1] = classProbabilityMatrix[label-1] + 1

    print(classProbabilityMatrix)
    totalTrainingExamples = classProbabilityMatrix.sum()
    classProbabilityMatrix = np.log2(classProbabilityMatrix/totalTrainingExamples)


    count = 0
    with open('train.data','r') as trainData:
        for line in trainData:
            values = line.rstrip('\n').split(" ")
            label = fileIdToLabel[values[0]]
            wordId = int(values[1])
            wordCount = int(values[2])
            matrixValue = wordCountMatrix[wordId-1,label-1]
            newValue = matrixValue + wordCount
            wordCountMatrix[wordId-1,label-1] = newValue
            if count % 5000000 == 0:
                print(count)
            count = count + 1
    #beta = 1.0/numberOfWordsInVocabulary
    #beta = b
    #print(wordCountMatrix)

    vocabSize = numberOfWordsInVocabulary
    for v in range(0,numberOfClasses):
        totalWordsInClassV = wordCountMatrix[:,v].sum()
    
        for w in range(0,numberOfWordsInVocabulary):
            wordId = int(w)
            label = int(v)
            countOfWInClassV = wordCountMatrix[w,v]
            probabilityOfWGivenV = ((countOfWInClassV) + beta)/ (totalWordsInClassV + beta*vocabSize)
            wordProbabilityMatrix[wordId,label] = probabilityOfWGivenV

    wordProbabilityMatrix = np.log2(wordProbabilityMatrix)
    #print(wordProbabilityMatrix)


    testMatrix = np.zeros((numberOfTestingExamples,numberOfWordsInVocabulary))


    listOfTestFileNames = []
    lastFileName = ''
    with open('test.data','r') as testData:
        for line in testData:
            values = line.rstrip('\n').split(" ")
           
            docId = int(values[0])
            if docId != lastFileName:
                listOfTestFileNames.append(docId)
                lastFileName = docId
            wordId = int(values[1])
            wordCount = int(values[2])
            testMatrix[len(listOfTestFileNames)-1,wordId] = wordCount

    classifySumMatrix = np.dot(testMatrix,wordProbabilityMatrix)

    classifyProbabilityMatrix = classifySumMatrix + classProbabilityMatrix

    classifyHeader = "\"Id\",\"Prediction1\",\"Prediction2\",\"Prediction3\",\"Prediction4\",\"Prediction5\",\"Prediction6\",\"Prediction7\",\"Prediction8\",\"Prediction9\""
    classifyStrings = []
    classifyStrings.append('1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0')
    classifyStrings.append('0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0')
    classifyStrings.append('0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0')
    classifyStrings.append('0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0')
    classifyStrings.append('0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0')
    classifyStrings.append('0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0')
    classifyStrings.append('0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0')
    classifyStrings.append('0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0')
    classifyStrings.append('0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0')
    
    with open("submissionNGramsBytes-001.csv","w") as f:
        f.write(classifyHeader + "\n")
        for e in range(0,numberOfTestingExamples):
            prediction = np.argmax(classifyProbabilityMatrix[e,:]) 
            fileId = listOfTestFileNames[e]
            line = "\"" + fileId + "\"" + "," + classifyStrings[prediction] + "\n"
            f.write(line)

    print("DONE")
           
classify(.001)
