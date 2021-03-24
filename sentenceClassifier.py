import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd
import numpy as np

import tensorflowjs as tfjs

# import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()


def encodeLabel(labels):
    sentenceTypes = ['irrelevant','introduction','observation','perturbation']
    numLabels = []
    for l in labels:
        print(l)
        try:
            numLabels.append(sentenceTypes.index(l))
        except:
            numLabels.append(0)
    return numLabels


# tokenizer = Tokenizer(num_words = 100, oov_token='<OOV>')
# tokenizer.fit_on_texts(sentences)
# words = tokenizer.word_index
# print(words)

# sequences = tokenizer.texts_to_sequences(sentences)
# paddedSeq = pad_sequences(sequences, padding='pre', maxlen=4, truncating='post')
# print(paddedSeq)

#download dataset

fullDataSets = pd.read_csv('sampleLabelledDataSet.csv',sep=',', header=None)
fullDataSets.columns = ['documentId','sentence','sentenceType']

print(fullDataSets.head())

#trains
trainingRatio = 0.6 
testRatio = 0.2
validateRatio = 0.2

trainingSize = int(trainingRatio*len(fullDataSets.index))
testSize = int(testRatio*len(fullDataSets))
validateSize = int(validateRatio*len(fullDataSets))

trainingDataSet =  fullDataSets.iloc[range(0,trainingSize)]
testDataSet =  fullDataSets.iloc[trainingSize:trainingSize+testSize]
validateDataSet =  fullDataSets.iloc[trainingSize+testSize:]

print(fullDataSets.shape)
print(trainingDataSet.shape)
print(testDataSet.shape)
print(validateDataSet.shape)

print(testDataSet.head())

trainInputs,trainLabels = list(trainingDataSet['sentence']),list(trainingDataSet['sentenceType'])
testInputs,testLabels = list(testDataSet['sentence']),list(testDataSet['sentenceType'])
validateInputs,validateLabels = list(validateDataSet['sentence']),list(validateDataSet['sentenceType'])

volcabularySize = 100
OOV = '<OOV>'
maxLength = 20


tokenizer = Tokenizer(num_words=volcabularySize,oov_token=OOV)
tokenizer.fit_on_texts(trainInputs)

wordList = tokenizer.word_index

print(wordList)

trainSequences = tokenizer.texts_to_sequences(trainInputs)
trainSequences_padded = pad_sequences(trainSequences, maxlen=maxLength, truncating='pre', padding='pre')

validateSequences = tokenizer.texts_to_sequences(validateInputs)
validateSequences_padded = pad_sequences(validateSequences, maxlen=maxLength, truncating='pre', padding='pre')

# validateSequences = tokenizer.texts_to_sequences(validateInputs)

print(trainSequences_padded)
print(validateSequences_padded)

#mdel
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=volcabularySize,output_dim= 16,input_length=maxLength),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(units=24,activation='relu'),
    tf.keras.layers.Dense(units=1,activation='sigmoid'),
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


import numpy as np
trainSequences_padded = np.array(trainSequences_padded)
trainLabels = np.array(encodeLabel(trainLabels))
validateSequences_padded = np.array(validateSequences_padded)
validateLabels = np.array(encodeLabel(validateLabels))

print(trainLabels)
print(validateLabels)

num_epochs = 200
history = model.fit(trainSequences_padded, trainLabels, epochs=num_epochs, validation_data=(validateSequences_padded, validateLabels), verbose=2)
  
# plot_graphs(history, "accuracy")
# plot_graphs(history, "loss")


testSequences = tokenizer.texts_to_sequences(testInputs)
testSequences_padded = pad_sequences(testSequences, maxlen=maxLength, truncating='pre', padding='pre')
print(testSequences_padded)

testSequences_padded = np.array(testSequences_padded)
testLabels = np.array(encodeLabel(testLabels))
print(model.predict(testSequences_padded))


#if model provide reasonable prediction
tfjs.converters.save_keras_model(model, '/tmp/sciDocParser/')

