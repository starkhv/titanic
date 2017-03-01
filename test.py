import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2

#td = pd.read_csv('train.csv')
td = pd.read_csv('test.csv')

#set of class columns for one hot coding
class_columns = set(['Pclass', 'Sex', 'Cabin', 'Embarked'])

#output columns
#output_columns = set(['Survived'])

exclude_columns = set(['PassengerId', 'Name', 'Ticket'])

one_hot_dicts = {}

one_hot_sizes = pd.read_pickle('one_hot_sizes.pkl')
known_indices = pd.read_pickle('known_indices.pkl')

#make one hot label dictionaries for class columns
for cls in class_columns:
    one_hot_size = one_hot_sizes[cls]
    one_hot_dicts[cls] = {}
    i = 0
    unique_vals = td[cls].unique()
    for val in unique_vals:
        one_hot_vals = [0]*one_hot_size
        if val in known_indices[cls]:
            one_hot_vals[known_indices[cls][val]] = 1
        else:
            one_hot_vals[one_hot_size-1] = 1
        one_hot_dicts[cls][val] = one_hot_vals
        i += 1

num_data = td.shape[0]

one_hot_data_list = []
#output_list = []

for i in range(num_data):
    this_data_list = []
    row = td.iloc[i]
    for ind in row.index:
        if ind in exclude_columns:
            continue
        val = row[ind]
        # if ind in output_columns:
            # output_list.append(val)
            # continue
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
pd.to_pickle(input_data, 'input_data_test.pkl')
#print(np.max(input_data))
# output_data = np.array(output_list)
# pd.to_pickle(output_data, 'output_data.pkl')
#print(output_data)

checkpoint = ModelCheckpoint('best.model', monitor='binary_accuracy', verbose=1, save_best_only=True)
# model = Sequential()
# model.add(Dense(1, input_dim=input_data.shape[1], activation='sigmoid'))
# # model.add(Dense(1, input_dim=input_data.shape[1], activation='sigmoid', W_regularizer=l2(0.001)))
# #model.add(Dropout(0.5))
# # model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='mse', optimizer='sgd', metrics=['binary_accuracy'])
# # model.compile(loss='mse', optimizer='sgd', metrics=['binary_accuracy'])
# # model.fit(input_data, output_data, 32, 1000, verbose=2, validation_split=0.2, callbacks=[checkpoint])
model = load_model('best.model')
classes = model.predict_classes(input_data)
classes = pd.DataFrame(data = classes[:, :], index=td.PassengerId)
classes.to_csv('predictions.csv')
print(classes)
# model.fit(input_data, output_data, 32, 1000, verbose=2, validation_split=0.0, callbacks=[checkpoint])
