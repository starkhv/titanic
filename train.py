import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

td = pd.read_csv('train.csv')

#set of class columns for one hot coding
class_columns = set(['Pclass', 'Sex', 'Cabin', 'Embarked'])

#output columns
output_columns = set(['Survived'])

exclude_columns = set(['PassengerId', 'Name', 'Ticket'])

one_hot_dicts = {}

#make one hot label dictionaries for class columns
for cls in class_columns:
    one_hot_dicts[cls] = {}
    i = 0
    unique_vals = td[cls].unique()
    for val in unique_vals:
        one_hot_vals = [0]*len(unique_vals)
        one_hot_vals[i] = 1
        one_hot_dicts[cls][val] = one_hot_vals
        i += 1

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
#print(np.max(input_data))
output_data = np.array(output_list)
#print(output_data)

model = Sequential()
model.add(Dense(30, input_dim=input_data.shape[1], activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
model.fit(input_data, output_data, 32, 1000, verbose=2, validation_split=0.2)
