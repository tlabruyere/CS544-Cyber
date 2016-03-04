#!/usr/bin/python

def convertStrtoId(csvLabels, idCsvLabels, mappingFile):
    inFile = open(csvLabels, 'r')
    mappingFileIO = open(mappingFile, 'w')
    outFile = open(idCsvLabels, 'w')
    lines = inFile.readlines()
    idx = 0
    for line in lines[1:]:
       lineSpl = line.split(',') 
       outFile.write(','.join((str(idx), lineSpl[1])))
       mappingFileIO.write(','.join((str(idx), lineSpl[0][1:-1])))
       mappingFileIO.write('\n')
       idx+=1
    outFile.close()
    mappingFileIO.close()

if __name__=='__main__':
    convertStrtoId(
        '/var/data/trainLabels.csv',
        '/home/tom/idLabels.csv',
        '/home/tom/mappingFile.csv')
