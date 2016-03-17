

import os

trainDir = "/var/data/train/"

ngrams = {}
count = 1

def parseNGrams(s):
    tokens = s.split(' ')
    for i in range(len(tokens)-3):
        ngram = tokens[i] + tokens[i+1] + tokens[i+2] + tokens[i+3]
        if ngram in ngrams:
            ngrams[ngram] = ngrams[ngram] + 1
        else:
            ngrams[ngram] = 1


for fileName in os.listdir(trainDir):
    s = ''
    if fileName.endswith(".bytes"):
        if( count == 1 or count == 2 or count == 3 or count == 4 or count == 5 or count == 6 or count == 7 or count == 8 or count == 100 or count == 500 or count == 1000 or count == 2000 or count == 4000 or count == 5000):
            print(count)
        count = count + 1
        if count > 20:
            break
        with open(trainDir + fileName) as f:
            for line in f:
                s = line[9:].rstrip('\n')
                parseNGrams(s)


#todo save most common ngrams to file: ngram, ngramId, count
for w in sorted(ngrams, key=ngrams.get, reverse=True):
    if ngrams[w] > 500:
        #write ngrams that apper more than 500 times
        #print(w, ngrams[w])