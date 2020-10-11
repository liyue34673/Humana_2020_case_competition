# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout 
from keras.optimizers import Adam
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

get_ipython().run_cell_magic('time', '', "train_data = pd.read_csv('../data/embedding_train.csv')")

target = ['transportation_issues']
dense_features = list(set(train_data.columns) - set(target))

train, test = train_test_split(train_data, test_size=0.2, random_state=2020)
train_model_input = train[dense_features]
test_model_input = test[dense_features]

model = Sequential([
    Dense(1024, input_dim = len(dense_features)),
    Activation('relu'),
    Dense(256),
    Activation('relu'),
    Dense(64),
    Activation('relu'),
    Dense(1),
    Activation('sigmoid'),
])

opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['binary_crossentropy'])

model.fit(x=train_model_input, y=train[target], validation_split=0.1, batch_size=64, epochs=10, class_weight={0:1, 1:3})

pred_ans = model.predict(test_model_input, batch_size=64)
print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

ans = pd.Series(pred_ans.reshape((-1,)))
ans[ans>=0.5] = 1
ans[ans<0.5] = 0

pd.Series(test[target].transportation_issues).value_counts()

ans.value_counts()

print(classification_report(test[target], ans))

model.summary()
