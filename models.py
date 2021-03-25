from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

import pandas as pd

#data preprocessing
dfFeatureName = pd.read_csv('featureName.txt')
dfMetaData = pd.read_csv('metaData.txt')

npArray = np.zeros((len(dfMetaData[0]),len(dfFeatureName.columns)), dtype = float)

with open('featureWeights.csv','r') as f:
    record = f.readline().split(',')
    npArray[int(record[0]),int(record[1])] = int(record[2])


dfFullDataset = pd.DataFrame(npArray, columns = dfFeatureName.columns)

exit(0)

model = Sequential()
model.add(Dense(64,activation = 'relu', input_dim=vocabularySize))
model.add(Dropout(0.5))
model.add(Dense(64,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(2,activation = 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metric=['precision'])    
model.fit(train_x, train_y, epoch=1000, batch_size=100)
score = model.evaluate(test_x, test_y, batch_size=100,)