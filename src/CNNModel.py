from __future__ import print_function

from keras import regularizers, layers, models

P_DROPOUT = 0.5
LAMBDA = 3

class CNNModel:
    def __init__(self, vectorSizes=(200, 7), numClass=5, kernelSize=5, numFilter=8):
        w2vInput = layers.Input(shape=(None, vectorSizes[0]))
        w2v1 = layers.Conv1D(
            numFilter, kernelSize, 
            activation='relu', 
            kernel_regularizer=regularizers.l2(LAMBDA))(w2vInput)
        w2v2 = layers.MaxPooling1D(pool_size=3)(w2v1)
        w2v3 = layers.Dropout(P_DROPOUT)(w2v2)
        
        hcInput = layers.Input(shape=(None, vectorSizes[1]))
        hc1 = layers.Conv1D(
            numFilter, vectorSizes[1], 
            activation='relu', 
            kernel_regularizer=regularizers.l2(LAMBDA))(hcInput)

        conc = layers.concatenate([w2v3, hc1])
        dens = layers.Dense(numClass, activation="relu")(conc)
        acti = layers.Activation("softmax")(dens)
        self.__model = models.Model(inputs=[w2vInput, hcInput], outputs=acti)
        self.__model.compile(
            loss='categorical_crossentropy',
            optimizer='adadelta',
            metrics=['accuracy'])

    def train(self, X, Y):
        pass