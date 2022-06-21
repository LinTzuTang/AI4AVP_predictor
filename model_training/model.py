import os
import tensorflow.keras
from tensorflow.keras.layers import Dense, Dropout,Add
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input,Flatten,Masking,BatchNormalization
from tensorflow.keras.layers import LSTM,Conv1D,LeakyReLU
from tensorflow.keras import Model
from tensorflow.keras import optimizers

def train_model(train_data, train_label, validation_data, validation_label, model_name, path = None):
    if not os.path.isdir(path):
        os.mkdir(path)
    input_ = Input(shape=(50,6))
    #mask = Masking(mask_value=0.)(input_)
    cnn1 = Conv1D(64 ,8,activation = 'relu', padding="same")(input_)
    norm1 = BatchNormalization()(cnn1)
    d1 = Dropout(0.5)(norm1)
    cnn2 = Conv1D(32 ,8,activation = 'relu', padding="same")(d1)
    norm2 = BatchNormalization()(cnn2)
    d2 = Dropout(0.5)(norm2)
    cnn3 = Conv1D(16 ,8,activation = 'relu', padding="same")(norm2)
    norm3 = BatchNormalization()(cnn3)
    d3 = Dropout(0.5)(norm3)
    f = Flatten()(d3)
    result = Dense(1, activation = "sigmoid")(f)
    model = Model(inputs=input_,outputs=result)

    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


    best_weights_filepath = path+'./%s_best_weights.h5'%model_name
    saveBestModel = tensorflow.keras.callbacks.ModelCheckpoint(best_weights_filepath, 
                                                        monitor='val_loss', 
                                                        verbose=1, 
                                                        save_best_only=True, 
                                                        mode='auto')
    reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                              patience=50,verbose=1)
    CSVLogger = tensorflow.keras.callbacks.CSVLogger(path+"./%s_csvLogger.csv"%model_name,separator=',', append=False)
    e_s = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss',
                                          min_delta=0,
                                          patience=150,
                                          verbose=0, mode='min')

    history = model.fit(train_data,train_label, validation_data=(validation_data, validation_label),shuffle=True,
                    epochs=2000, batch_size=100,callbacks=[saveBestModel,CSVLogger,reduce_lr,e_s])
    return model,history

def train_ennavia_model(train_data, train_label, validation_data, validation_label, model_name, path = None):
    if not os.path.isdir(path):
        os.mkdir(path)
    input_ = Input(shape=(461))
    #mask = Masking(mask_value=0.)(input_)
    f1 = Dense(1024)(input_)
    norm1 = BatchNormalization()(f1)
    d1 = Dropout(0.3)(norm1)
    f2 = Dense(256)(d1)
    norm2 = BatchNormalization()(f2)
    d2 = Dropout(0.3)(norm2)
    f = Flatten()(d2)
    result = Dense(1, activation = "sigmoid")(f)
    model = Model(inputs=input_,outputs=result)

    model.compile(optimizer=optimizers.Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


    best_weights_filepath = path+'./%s_best_weights.h5'%model_name
    saveBestModel = tensorflow.keras.callbacks.ModelCheckpoint(best_weights_filepath, 
                                                        monitor='val_loss', 
                                                        verbose=1, 
                                                        save_best_only=True, 
                                                        mode='auto')
    reduce_lr = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                              patience=50,verbose=1)
    CSVLogger = tensorflow.keras.callbacks.CSVLogger(path+"./%s_csvLogger.csv"%model_name,separator=',', append=False)
    e_s = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss',
                                          min_delta=0,
                                          patience=150,
                                          verbose=0, mode='min')

    history = model.fit(train_data,train_label, validation_data=(validation_data, validation_label),shuffle=True,
                    epochs=2000, batch_size=100,callbacks=[saveBestModel,CSVLogger,reduce_lr,e_s])
    return model,history