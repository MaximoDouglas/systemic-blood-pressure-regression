import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential, layers
import pandas as pd
import numpy as np
from scipy import stats
from tensorflow import keras

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

data_path = "../dataset/"
results_path = "../results/"
model_name = "cnn_3conv_1flt_1dpo_1dense"

thetav = np.loadtxt(data_path + "thetav-1000.csv",delimiter=',')
Vec = np.loadtxt(data_path + 'Vecv-1000.csv',delimiter = ',')

# Data pre-processing ---------------------------------------------
sc = MinMaxScaler()
X_n = sc.fit_transform(Vec)
sc2 = MinMaxScaler()
y_n = sc2.fit_transform(thetav)

X_n = X_n.reshape(-1,8000,1)
select=[1]

train_size = 0.7
lng = len(X_n)
X_treinamento = X_n[:round(lng*0.6)]
y_treinamento = y_n[:round(lng*0.6)]
X_valid = X_n[round(lng*0.6):round(lng*0.85)]
y_valid = y_n[round(lng*0.6):round(lng*0.85)]
X_test = X_n[round(lng*0.85):]
y_test = y_n[round(lng*0.85):]
X_treinamento, X_valid, X_test = X_treinamento.reshape(-1,X_treinamento.shape[1],1), X_valid.reshape(-1,X_treinamento.shape[1],1),  X_test.reshape(-1,X_treinamento.shape[1],1)

X_treinamento.shape,y_treinamento.shape, X_valid.shape, y_valid.shape, X_test.shape, y_test.shape

# Model compilation ------------------------------------------------------------------------------
def get_model():
    input_layer1 = keras.layers.Input(X_treinamento.shape[1:])

    conv1 = keras.layers.Conv1D(filters=32, kernel_size=8,strides=400)(input_layer1)
    conv1 = keras.layers.MaxPooling1D()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3)(conv1)
    conv2 = keras.layers.MaxPooling1D()(conv2)
    conv2 = keras.layers.Activation(activation='relu')(conv2)

    conv3 = keras.layers.Conv1D(filters=96, kernel_size=3)(conv2)
    conv3 = keras.layers.MaxPooling1D()(conv3)
    conv3 = keras.layers.Activation(activation='relu')(conv3)

    flt = keras.layers.Flatten()(conv3)
    dropout = keras.layers.Dropout(0.4)(flt)

    dense2 = keras.layers.Dense(50,activation='relu')(dropout)

    out = keras.layers.Dense(10, activation='linear')(dense2)

    model = keras.models.Model(inputs=input_layer1, outputs=out)


    model.compile(optimizer=tf.keras.optimizers.Adam(),  # Optimizer
                loss='mse')

    return model

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=20,
                                                      min_lr=0.0001)
file_path = '../models/' + model_name + '.hdf5'
model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)
callbacks = [reduce_lr, model_checkpoint]

model = get_model()
model.summary()

print('# Fit model on training data')
history = model.fit(X_treinamento, (y_treinamento),
                    batch_size=10,
                    epochs=300,
                    verbose=0,
                    # We pass some validation for
                    # monitoring validation loss and metrics
                    # at the end of each epoch
                    validation_data=(X_valid, (y_valid)),
                    callbacks=callbacks)
                    

# Results -----------------------------------------------------------------
plt.figure()
plt.plot(history.history['loss'], label='Erro treinamento')
plt.plot(history.history['val_loss'], label='Erro validação')
plt.legend()
plt.show()

CONV1D_MODEL = keras.models.load_model(file_path)
y_pred_CONV1D = CONV1D_MODEL.predict(X_test)

MSE = ((y_test - y_pred_CONV1D)**2).mean(axis=0)
print(MSE)

plt.rcParams.update({'font.size': 16})
plt.figure(figsize = (8,5))
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.xlabel('Épocas')
plt.legend()
plt.savefig(results_path + model_name + '.png',bbox_inches = 'tight', dpi=300)