from __future__ import print_function

import os
import sys
import re
import numpy
import pickle
import glob
import argparse
import shutil
import collections

from src.FeatureExtractor import W2VExtractor, HandcraftExtractor
from src.CNNModel import CNNModel

BATCH_FORMAT = ".batch"
MODEL_FORMAT = ".json"
WEIGHT_FORMAT = ".h5"

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
    model.load(modelName, weightName)
    batch_data_files = glob.glob(folderTest+"/*"+BATCH_FORMAT)
    for batch_file in batch_data_files:
        X, Y = load_batch(batch_file)
        print(collections.Counter(Y))
        print(model.test(X, Y))
        Y = CNNModel.convert_labels(Y, numClass=5)
        print(model.test_auto(X, Y))

if __name__ == '__main__':
    # python main.py -m train -e 1 -d dat/eng.train -o dat/ -w dat/vectors.bin 
    # python main.py -m test -d dat/eng.testa -o dat/ -w dat/vectors.bin -c dat/w_0.h5 -a dat/model.json 

    parser = argparse.ArgumentParser(description="main program to train or test cnn model")
    parser.add_argument("-m", "--mode", 
        help="[train, test] train data file/test data file", required=True)
    parser.add_argument("-d", "--data", help="data file", required=True)
    parser.add_argument("-o", "--output", help="output prefix", required=False, default="")
    parser.add_argument("-s", "--batch_size", 
        help="use for mode 'train', size of each batch file (default: 100)", 
        required=False)
    parser.add_argument("-e", "--epoch", 
        help="use for mode 'train', number of epoch (default: 5)", 
        required=False)
    parser.add_argument("-w", "--w2v_file", 
        help="word2vec pretrained model file", 
        required=True)
    parser.add_argument("-c", "--weight_file", 
        help="use for mode 'test', cnn pretrained weight file", 
        required=False)
    parser.add_argument("-a", "--architech_file", 
        help="use for mode 'test', cnn architech file", 
        required=False)


    args = parser.parse_args()

    option = args.mode.lower()
    
    batchSize = 100
    if args.batch_size:
        batchSize = int(args.batch_size)

    epoch = 5
    if args.epoch:
        epoch = int(args.epoch)

    rawDataFile = args.data
    outputFolder = args.output

    w2vEx = W2VExtractor()
    w2vEx.load(args.w2v_file)
    hcEx = HandcraftExtractor()

    model = CNNModel()

    if os.path.isdir(os.path.join(outputFolder, "batches")):
        shutil.rmtree(os.path.join(outputFolder, "batches"))

    os.mkdir(os.path.join(outputFolder, "batches"))

    if option == "train":
        convert_data(
            rawDataFile, os.path.join(outputFolder, "batches"), 
            w2vEx, hcEx, batchSize)
        for e in range(epoch):
            train(
                model, os.path.join(outputFolder, "batches"), 
                os.path.join(outputFolder, "model"+MODEL_FORMAT), 
                os.path.join(outputFolder, "w_"+str(e)+WEIGHT_FORMAT))
    elif option == "test":
        if not args.weight_file or not args.architech_file:
            print("Error: Missing pretrained model")
            sys.exit()
        convert_data(
            rawDataFile, os.path.join(outputFolder, "batches"), 
            w2vEx, hcEx)
        test(
            model, os.path.join(outputFolder, "batches"),
            args.architech_file,
            args.weight_file)
        
