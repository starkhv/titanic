import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2

#td = pd.read_csv('train.csv')
td = pd.read_csv('train.csv')

#set of class columns for one hot coding
class_columns = set(['Pclass', 'Sex', 'Cabin', 'Embarked'])

#output columns
output_columns = set(['Survived'])

exclude_columns = set(['PassengerId', 'Name', 'Ticket'])

one_hot_dicts = {}

#one hot sizes to fixed size for each attribute in the input during training and testing
one_hot_sizes = {}
known_indices = {}

#make one hot label dictionaries for class columns
for cls in class_columns:
    one_hot_dicts[cls] = {}
    i = 0
    unique_vals = td[cls].unique()
    temp_known_indices = {}
    for val in unique_vals:
        temp_known_indices[val] = i
        # 1 extra bit for unknown inputs
        one_hot_vals = [0]*(len(unique_vals)+1)
        one_hot_vals[i] = 1
        one_hot_dicts[cls][val] = one_hot_vals
        i += 1
    known_indices[cls] = temp_known_indices
    one_hot_sizes[cls] = len(unique_vals)+1

filename = 'one_hot_sizes.pkl'
pd.to_pickle(one_hot_sizes, filename)
print('saved one_hot_sizes to', filename)
filename = 'known_indices.pkl'
pd.to_pickle(known_indices, filename)
print('saved known indices to', filename)

num_data = td.shape[0]

one_hot_data_list = []
output_list = []

for i in range(num_data):
    this_data_list = []
    row = td.iloc[i]
    for ind in row.index:
        if ind in exclude_columns:
            continue
        val = row[ind]
        if ind in output_columns:
            output_list.append(val)
            continue
        if ind in class_columns:
            this_data_list += one_hot_dicts[ind][val]
        else:
            if pd.isnull(val):
                val = 0
            this_data_list += [val]
    one_hot_data_list.append(this_data_list)

# make dataset
# calculate dimensions of input, output data

input_data = np.array(one_hot_data_list)
pd.to_pickle(input_data, 'input_data.pkl')
#print(np.max(input_data))
output_data = np.array(output_list)
pd.to_pickle(output_data, 'output_data.pkl')
#print(output_data)

checkpoint = ModelCheckpoint('best.model', monitor='loss', verbose=1, save_best_only=True)
model = Sequential()
model.add(Dense(100, input_dim=input_data.shape[1], activation='sigmoid'))
# model.add(Dense(1, input_dim=input_data.shape[1], activation='sigmoid', W_regularizer=l2(0.001)))
# model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam', metrics=['binary_accuracy'])
# model.compile(loss='mse', optimizer='sgd', metrics=['binary_accuracy'])
# model.fit(input_data, output_data, 32, 1000, verbose=2, validation_split=0.2, callbacks=[checkpoint])
model.fit(input_data, output_data, 32, 1000, verbose=2, validation_split=0.0, callbacks=[checkpoint])
