from multiprocessing import dummy
import pandas as pd
url = 'https://raw.githubusercontent.com/dataprofessor/streamlit_freecodecamp/main/app_8_classification_penguins/penguins_cleaned.csv'
#Data taken from Data Professor(https://github.com/dataprofessor)

penguins = pd.read_csv(url)
df = penguins.copy()

target = 'species'
encode = ['sex','island']

#encoding the Categorical Variables
for col in encode:
    dummy_var = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy_var], axis=1)
    del df[col]

t_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
def target_encode(val):
    return t_mapper[val]

df['species'] = df['species'].apply(target_encode)

X = df.drop('species', axis=1)

y= df['species']

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, y)

import pickle
pickle.dump(clf, open('pg_clf.pkl', 'wb'))
