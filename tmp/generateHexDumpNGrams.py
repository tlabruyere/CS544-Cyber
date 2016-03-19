

import os

trainDir = "/var/data/train/"
count = 1
ngrams = {}

def parseNGrams(s):
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


for fileName in os.listdir(trainDir):
    
    if fileName.endswith(".bytes"):
        count = count + 1
        if count % 1000 == 0:
            print(count)
        #if count > 5000:
            #break
        with open(trainDir + fileName) as f:
            for line in f:
                parseNGrams(line[9:].rstrip())

print("writing to file")
c = 1
#todo save most common ngrams to file: ngram, ngramId, count
with open("topNGrams.csv","w") as f:
    for w in sorted(ngrams, key=ngrams.get, reverse=True):
    #for w in ngrams: 
        if ngrams[w] > 1:
            #if c % 10000 == 0:
                #print(c)
            #c = c + 1
            #write ngrams that apper more than 100 times
            f.write(str(w) + " " + str(ngrams[w]) + "\n") 
            #print(w, ngrams[w])
            c = c + 1
            if c % 100 == 0:
                print(c)
            #if c > 500:
                #break
