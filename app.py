import pandas as pd
import numpy as np


def status(feature):

    print ('Processing', feature, ': ok')

def combine_data():

    global combined

    train = pd.read_csv('train.csv')

    test = pd.read_csv('test.csv')

    targets = train.Survived
    train.drop('Survived', 1, inplace=True)

    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index', inplace=True, axis=1)

    # combined.to_csv('combined.csv', index=False)

combined = combine_data()

def add_titles():

    global combined
    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    title_dictionary = {
        "Capt":       "Officer",
        "Col":        "Officer",
        "Major":      "Officer",
        "Jonkheer":   "Royalty",
        "Don":        "Royalty",
        "Sir":       "Royalty",
        "Dr":         "Officer",
        "Rev":        "Officer",
        "the Countess": "Royalty",
        "Dona":       "Royalty",
        "Mme":        "Mrs",
        "Mlle":       "Miss",
        "Ms":         "Mrs",
        "Mr":        "Mr",
        "Mrs":       "Mrs",
        "Miss":      "Miss",
        "Master":    "Master",
        "Lady":      "Royalty"
    }

    combined['Title'] = combined.Title.map(title_dictionary)

    # combined.to_csv('combined.csv', index=False)

def process_age():
    global combined

    def fill_ages(row):
        if row['Sex']=='female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return 30
            elif row['Title'] == 'Mrs':
                return 45
            elif row['Title'] == 'Officer':
                return 49
            elif row['Title'] == 'Royalty':
                return 39

        elif row['Sex']=='female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return 20
            elif row['Title'] == 'Mrs':
                return 30

        elif row['Sex']=='female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return 18
            elif row['Title'] == 'Mrs':
                return 31

        elif row['Sex']=='male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 41.5
            elif row['Title'] == 'Officer':
                return 52
            elif row['Title'] == 'Royalty':
                return 40

        elif row['Sex']=='male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return 2
            elif row['Title'] == 'Mr':
                return 30
            elif row['Title'] == 'Officer':
                return 41.5

        elif row['Sex']=='male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 26

        else:
            return 28

    combined.Age = combined.apply(lambda r : fill_ages(r) if np.isnan(r['Age']) else r['Age'], axis=1)
    # combined.to_csv('combined.csv', index=False)

    status('age')

def process_names():

    global combined

    combined.drop('Name', axis=1, inplace=True)

    titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')
    combined = pd.concat([combined, titles_dummies], axis=1)

    combined.drop('Title', axis=1, inplace=True)

    status('names')

def process_fares():

    global combined

    combined.Fare.fillna(combined.Fare.mean(), inplace=True)

    status('Fare')

def process_embarked():

    global combined

    combined.Embarked.fillna('S', inplace=True)

    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
    combined = pd.concat([combined, embarked_dummies], axis=1)
    combined.drop('Embarked', axis=1, inplace=True)
    status('embarked')

def process_cabin():

    global combined

    combined.Cabin.fillna('U', inplace=True)
    combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])
    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')
    combined = pd.concat([combined, cabin_dummies], axis=1)
    combined.drop('Cabin', axis=1, inplace=True)

    status('cabin')

def process_sex():
    
    global combined

    combined.Sex = combined.Sex.map({'male':1, 'female':0})

    status('sex')

def process_pclass():

    global combined

    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix='Pclass')

    combined = pd.concat([combined, pclass_dummies], axis=1)

    combined.drop('Pclass', axis=1, inplace=True)

    status('pclass')


def process_ticket():

    global combined

    def cleanTicket(ticket):
        ticket = ticket.replace('.', '')
        ticket = ticket.replace('/', '')
        ticket = ticket.split()
        ticket = map(lambda x : x.strip(), ticket)
        ticket = list(filter(lambda t: not t.isdigit(), ticket))
        if len(ticket) > 0:
            return ticket[0]
        else:
            return 'XXX'

    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined.Ticket, prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies], axis=1)
    combined.drop('Ticket', inplace=True, axis=1)

    status('ticket')

def process_family():

    global combined
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1

    combined['Singleton'] = combined['FamilySize'].map(lambda s : 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s : 1 if 5<=s else 0)

    status('family')

def scale_all_features():

    global combined

    features = list(combined.columns)
    features.remove('PassengerId')
    combined[features] = combined[features].apply(lambda x: x/x.max(), axis=0)

    print('Features scaled successfully !')

def recover_train_test_target():
    global combined

    train0 = pd.read_csv('train.csv')

    targets = train0.Survived
    train = combined.ix[0:890, 1:]
    test = combined.ix[891:, 1:]

    return train, test, targets

def process_combined():
    global combined
    combine_data()
    add_titles()
    process_age()
    process_names()
    process_fares()
    process_embarked()
    process_cabin()
    process_sex()
    process_pclass()
    process_ticket()
    process_family()
    scale_all_features()
    train, test, targets = recover_train_test_target()
    return train, test, targets
