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

def convert_data(rawDataFile, outputFolder, batchSize, w2vEx, hcEx):
    inputFile = open(rawDataFile, "r")
    lines = inputFile.readlines()
    w2vFeatures = []
    hcFeatures = []
    labels = []
    batch = 0
    for i in range(len(lines)):
        if lines[i] in OMITED_DATA:
            continue
        wordData = lines[i][:-1].split(" ") # cut out '\n'
        word = wordData[DATA_FORMAT["word"]]
        w2vFeatures.append(w2vEx.extract(word))

        preWordData = lines[i-1].split(" ")
        posWordData = lines[i+1].split(" ")
        hcFeatures.append(hcEx.extract(word, posWordData[0]=="", preWordData[0]==""))

        labels.append(NE_LABEL[re.sub(r".*-", "", wordData[DATA_FORMAT["net"]])])

        if len(labels)%batchSize == 0 or i == len(lines)-1:
            batchData = {
                "w2v": numpy.concatenate(w2vFeatures).reshape((batchSize, len(w2vFeatures[0]))),
                "hc": numpy.concatenate(hcFeatures).reshape((batchSize, len(hcFeatures[0]))),
                "labels": labels
            }
            with open(os.path.join(outputFolder, str(batch))+BATCH_FORMAT, "wb") as output:
                pickle.dump(batchData, output, pickle.HIGHEST_PROTOCOL)
            print("\rSaved batch "+str(batch), end="")
            w2vFeatures = []
            hcFeatures = []
            labels = []
            batch += 1

def load_batch(dataFile):
    inFile = open(dataFile, "rb")
    batchData = pickle.load(inFile)
    inFile.close()
    return [batchData["w2v"], batchData["hc"]], batchData["labels"]

if __name__ == '__main__':
    rawDataFile = "dat/eng.train"
    outputFolder = "dat/batch_data"

    w2vEx = W2VExtractor()
    w2vEx.load("dat/vectors.bin")
    hcEx = HandcraftExtractor()

    model = CNNModel()

    #convert_data(rawDataFile, outputFolder, 100, w2vEx, hcEx)
    X, Y = load_batch("dat/batch_data/0.batch")
    print(X[0].shape, X[1].shape, len(Y))