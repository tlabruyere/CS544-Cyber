

import os

trainDir = "/var/data/train/"
count = 1
ngrams = {}

def parseNGrams(s):
    tokens = s.split(' ')
    for i in range(len(tokens)-3):
        ngram = tokens[i] + tokens[i+1] + tokens[i+2] + tokens[i+3]
        if ngram in ngrams:
            ngrams[ngram] = ngrams[ngram] + 1
        else:
            ngrams[ngram] = 1


for fileName in os.listdir(trainDir):
    
    if fileName.endswith(".bytes"):
        count = count + 1
        if count % 100 == 0:
            print(count)
        with open(trainDir + fileName) as f:
            for line in f:
                parseNGrams(line[9:].rstrip())

print("writing to file")
#todo save most common ngrams to file: ngram, ngramId, count
for w in sorted(ngrams, key=ngrams.get, reverse=True):
    if ngrams[w] > 100:
        #write ngrams that apper more than 100 times
        with open("topNGrams.csv","w") as f:
            f.write(w + " " + ngrams[w] + "\n") 
            #print(w, ngrams[w])
