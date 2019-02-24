from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.models import Model
from keras.layers.convolutional import *
import keras.backend as K
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from keras import regularizers
from keras.layers import concatenate
from keras.optimizers import Adam


'''
Wide and deep multi-task neural network. 
'''
def get_wide_deep():
    input = Input(shape=(222,))
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-8))(input)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-8))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-8))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    wide_deep = concatenate([input, x])
    preds = Dense(11, activation='sigmoid', kernel_regularizer=regularizers.l2(1e-8))(wide_deep)
    model = Model(input=input, output=preds)
    opt = Adam(lr=np.exp(-1.0 * 9))
    model.compile(optimizer=opt,
                  loss=masked_multi_weighted_bce,
                  metrics=[masked_weighted_accuracy])
    return model


def get_wide_deep_raw_features():
    input = Input(shape=(166,))
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-8))(input)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-8))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-8))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    wide_deep = concatenate([input, x])
    preds = Dense(11, activation='sigmoid', kernel_regularizer=regularizers.l2(1e-8))(wide_deep)
    model = Model(input=input, output=preds)
    opt = Adam(lr=np.exp(-1.0 * 9))
    model.compile(optimizer=opt,
                  loss=masked_multi_weighted_bce,
                  metrics=[masked_weighted_accuracy])
    return model

def get_deep():
    input = Input(shape=(222,))
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-8))(input)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-8))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-8))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    preds = Dense(11, activation='sigmoid', kernel_regularizer=regularizers.l2(1e-8))(x)
    model = Model(input=input, output=preds)
    opt = Adam(lr=np.exp(-1.0 * 9))
    model.compile(optimizer=opt,
                  loss=masked_multi_weighted_bce,
                  metrics=[masked_weighted_accuracy])
    return model


'''
Wide and deep single task neural network. 
'''
def get_wide_deep_single():
    input = Input(shape=(222,))
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-8))(input)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-8))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-8))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    wide_deep = concatenate([input, x])
    preds = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(1e-8))(wide_deep)
    model = Model(input=input, output=preds)
    opt = Adam(lr=np.exp(-1.0 * 9))
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def get_deep_single():
    input = Input(shape=(222,))
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-8))(input)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-8))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-8))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    preds = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(1e-8))(x)
    model = Model(input=input, output=preds)
    opt = Adam(lr=np.exp(-1.0 * 9))
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def get_wide_deep_preselect(num_snp_indiv):
    input = Input(shape=(num_snp_indiv,))
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-8))(input)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-8))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-8))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    wide_deep = concatenate([input, x])
    preds = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(1e-8))(wide_deep)
    model = Model(input=input, output=preds)
    opt = Adam(lr=np.exp(-1.0 * 9))
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def get_random_forest():
    return RandomForestClassifier(n_estimators=1000, max_features=0.2, min_samples_leaf=0.002)