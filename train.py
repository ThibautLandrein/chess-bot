# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 11:32:24 2022

@author: Thibaut Landrein
"""

import numpy
import tensorflow.keras.callbacks as callbacks
from model import build_model_residual

import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers

def get_dataset():
	container = numpy.load('dataset.npz')
	b, v = container['b'], container['v']
	v = numpy.asarray(v / abs(v).max() / 2 + 0.5, dtype=numpy.float32) # normalization (0 - 1)
	return b, v


if __name__ == "__main__":
    x_train, y_train = get_dataset()
    print(x_train.shape)
    model = build_model_residual(32, 4)

    model.compile(optimizer=optimizers.Adam(5e-4), loss='mean_squared_error')
    model.summary()
    model.fit(x_train, y_train,
              batch_size=2048,
              epochs=1000,
              verbose=1,
              validation_split=0.1,
              callbacks=[callbacks.ReduceLROnPlateau(monitor='loss', patience=10),
                         callbacks.EarlyStopping(monitor='loss', patience=15, min_delta=1e-4)])

    model.save('model.h5')
