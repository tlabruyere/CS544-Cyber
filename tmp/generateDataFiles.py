import os

trainDir = "/var/data/train/"
testDir = "/var/data/test/"

#ngrams = {}

trainFileName = "train.data"
testFileName = "test.data"

def appendDataFile(dataFileName, ngrams, fileId):
    with open(dataFileName,"a") as f:
        for w in ngrams:
            f.write(fileId + " " + str(w) + " " + str(ngrams[w]) +  "\n") 


def parseNGrams(s,ngrams):
    tokens = s.split(' ')
    for i in range(len(tokens)-2):
        ngram_s = tokens[i] + tokens[i+1]
        ngram = 0
        if '?' not in ngram_s:
            ngram = int(ngram_s,16)
        if ngram in ngrams:
            ngrams[ngram] = ngrams[ngram] + 1
        else:
            ngrams[ngram] = 1

count = 1
for fileName in os.listdir(testDir):
    
    if fileName.endswith(".bytes"):
        #if count > 1000:
            #break
        if count % 100 == 0:
            print(count)
        count = count + 1
        ngrams = {}
        with open(testDir + fileName) as f:
            for line in f:
                parseNGrams(line[9:].rstrip(),ngrams)
        fileId = fileName.split(".")[0]
        appendDataFile(testFileName,ngrams,fileId)