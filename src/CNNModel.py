from __future__ import print_function

import numpy

from sklearn.metrics import confusion_matrix
from keras import regularizers, layers, models, metrics
from keras.models import model_from_json
from keras import utils

P_DROPOUT = 0.5
LAMBDA = 3

class CNNModel:
    def __init__(self, vectorSizes=(200, 7), numClass=5, kernelSize=5, numFilter=8):
        w2vInput = layers.Input(shape=(None, vectorSizes[0]))
        w2v1 = layers.Conv1D(
            numFilter, kernelSize, 
            activation="relu",
            padding="same",
            kernel_regularizer=regularizers.l2(LAMBDA))(w2vInput)
        w2v2 = layers.Dropout(P_DROPOUT)(w2v1)
        
        hcInput = layers.Input(shape=(None, vectorSizes[1]))
        hc1 = layers.Conv1D(
            numFilter, vectorSizes[1],
            padding="same", 
            activation="relu")(hcInput)

        conc = layers.concatenate([w2v2, hc1])
        dens = layers.Dense(numClass, activation="softmax")(conc)
        self.__model = models.Model(inputs=[w2vInput, hcInput], outputs=dens)
        self.__model.compile(
            loss='categorical_crossentropy',
            optimizer='adadelta',
            metrics=['accuracy', metrics.categorical_accuracy])

    def train(self, X, Y):
        self.__model.train_on_batch(X, Y, sample_weight=None, class_weight=None)

    def test_auto(self, X, Y):
        return dict(zip(self.__model.metrics_names, self.__model.evaluate(X, Y)))

    def test(self, X, Y_raw):
        y = self.predict(X)
        y_raw = numpy.array([numpy.argmax(v) for v in y])
        confusionMat = confusion_matrix(Y_raw, y_raw)
        return confusionMat


    def predict(self, X):
        return self.__model.predict(X)

    def save(self, modelName="model.json", weightName="weight.h5"):
        modelJson = self.__model.to_json()
        with open(modelName, "w") as jsonFile:
            jsonFile.write(modelJson)
        self.__model.save_weights(weightName)

    def load(self, modelName="model.json", weightName="weight.h5"):
        jsonFile = open(modelName, 'r')
        modelJson = jsonFile.read()
        jsonFile.close()
        self.__model = model_from_json(modelJson)
        self.__model.compile(
            loss='mean_squared_error',
            optimizer='adadelta',
            metrics=['accuracy'])
        self.__model.load_weights(weightName)

    @staticmethod
    def convert_labels(labels, numClass):
        return utils.to_categorical(labels, num_classes=numClass).reshape(-1, 1, numClass)