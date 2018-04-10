from __future__ import print_function

import os
import sys
import re
import numpy
import pickle
import glob

from src.FeatureExtractor import W2VExtractor, HandcraftExtractor
from src.CNNModel import CNNModel

BATCH_FORMAT = ".batch"

DATA_FORMAT = {
    "word": 0,
    "pos": 1,
    "sct": 2,
    "net": 3
}

NE_LABEL = {
    "O": 0,
    "ORG": 1,
    "PER": 2,
    "LOC": 3,
    "MISC": 4
}

OMITED_DATA = ["\n", "-DOCSTART- -X- O O\n"]

def convert_data(rawDataFile, outputFolder, w2vEx, hcEx, batchSize=None):
    inputFile = open(rawDataFile, "r")
    lines = inputFile.readlines()
    if batchSize is None:
        batchSize = len(lines)
    w2vFeatures = []
    hcFeatures = []
    labels = []
    batch = 0
    for i in range(len(lines)):
        if (len(labels) and len(labels)%batchSize == 0) or i == len(lines)-1:
            batchData = {
                "w2v": numpy.array(w2vFeatures),
                "hc": numpy.array(hcFeatures),
                "labels": labels
            }
            with open(os.path.join(outputFolder, str(batch))+BATCH_FORMAT, "wb") as output:
                pickle.dump(batchData, output, pickle.HIGHEST_PROTOCOL)
            print("\rSaved batch "+str(batch), end="")
            w2vFeatures = []
            hcFeatures = []
            labels = []
            batch += 1

        if lines[i] in OMITED_DATA:
            continue
        wordData = lines[i][:-1].split(" ") # cut out '\n'
        word = wordData[DATA_FORMAT["word"]]
        w2vFeatures.append(w2vEx.extract(word))

        preWordData = lines[i-1].split(" ")
        posWordData = lines[i+1].split(" ")
        hcFeatures.append(hcEx.extract(word, posWordData[0]=="", preWordData[0]==""))

        labels.append(NE_LABEL[re.sub(r".*-", "", wordData[DATA_FORMAT["net"]])])
        
    print("\n")

def load_batch(dataFile):
    inFile = open(dataFile, "rb")
    batchData = pickle.load(inFile)
    inFile.close()
    return [batchData["w2v"], batchData["hc"]], batchData["labels"]

def train(model, folderTrain, modelName, weightName):
    batch_data_files = glob.glob(folderTrain+"/*"+BATCH_FORMAT)
    for batch_file in batch_data_files:
        X, Y = load_batch(batch_file)
        Y = CNNModel.convert_labels(Y, numClass=5)
        model.train(X, Y)
    model.save(modelName, weightName)

def test(model, folderTest, modelName, weightName):
    model.load("model.json", "w01.h5")
    batch_data_files = glob.glob(folderTest+"/*"+BATCH_FORMAT)
    for batch_file in batch_data_files:
        X, Y = load_batch(batch_file)
        Y = CNNModel.convert_labels(Y, numClass=5)
        print(model.test(X, Y))


if __name__ == '__main__':
    option = "test"
    batch_size = 100

    #rawDataFile = "dat/eng.train"
    #outputFolder = "dat/batch_train"

    rawDataFile = "dat/eng.testa"
    outputFolder = "dat/batch_testa"

    w2vEx = W2VExtractor()
    w2vEx.load("dat/vectors.bin")
    hcEx = HandcraftExtractor()

    model = CNNModel()

    if option == "train":
        convert_data(rawDataFile, outputFolder, w2vEx, hcEx, batchSize)
        train(model, outputFolder, "model.json", "w01.h5")
    elif option == "test":
        convert_data(rawDataFile, outputFolder, w2vEx, hcEx)
        test(model, outputFolder, "model.json", "w01.h5")
        
