validationDict = {}
trainDict = {}

with open("validation.label","r") as validation:
    for line in validation:
        line = line.rstrip();
        tokens = line.split(",")
        fileId = tokens[0]
        validationDict[fileId] = 1



with open("train.label","r") as train:
     for line in train:
        line = line.rstrip();
        tokens = line.split(",")
        fileId = tokens[0]
        trainDict[fileId] = 1

with open("train-full.data","r") as full:
    
    for line in full:
        
        tokens = line.split(" ")
        fileId = tokens[0]
        if fileId in trainDict.keys():
            with open("train.data","a") as train:
                train.write(line)
        else:
            with open("validation.data","a") as validation:   
                validation.write(line)



