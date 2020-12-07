import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pandas as pd


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


trainSequences = tokenizer.texts_to_sequences(trainInputs)
trainSequences_padded = pad_sequences(trainSequences, maxlen=maxLength, truncating='pre', padding='pre')


testSequences = tokenizer.texts_to_sequences(testInputs)
testSequences_padded = pad_sequences(testSequences, maxlen=maxLength, truncating='pre', padding='pre')

validateSequences = tokenizer.texts_to_sequences(validateInputs)
validateSequences_padded = pad_sequences(validateSequences, maxlen=maxLength, truncating='pre', padding='pre')

# validateSequences = tokenizer.texts_to_sequences(validateInputs)

print(trainSequences_padded)
print(testSequences_padded)
print(validateSequences_padded)

#mdel
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=volcabularySize,output_dim= 16,input_length=maxLength),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(units=24,activation='relu'),
    tf.keras.layers.Dense(units=2,activation='sigmoid'),
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

num_epochs = 2000
history = model.fit(trainSequences_padded, trainLabels, epochs=num_epochs, validation_data=(validateSequences_padded, validateLabels), verbose=2)

