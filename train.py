import pandas as pd
import numpy as np

td = pd.read_csv('train.csv')

#set of class columns for one hot coding
class_columns = set(['Survived', 'Pclass', 'Sex', 'Cabin', 'Embarked'])

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

for i in range(num_data):
    this_data_list = []
    row = td.iloc[i]
    for ind in row.index:
        if ind in exclude_columns:
            continue
        val = row[ind]
        if ind in class_columns:
            this_data_list.append(one_hot_dicts[ind][val])
        else:
            if pd.isnull(val):
                val = 0
            this_data_list.append(val)
    one_hot_data_list.append(this_data_list)


