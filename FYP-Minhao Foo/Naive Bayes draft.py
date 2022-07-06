import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import datasets

df = pd.read_csv('D:/source/repos/keystroke UI/keytest.csv')

target_names = np.array(['Positive', 'Negative'])

df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.75



train = df[df['is_train']==True]
test = df[df['is_train']==False]

number = LabelEncoder()
df['timeP'] = number.fit_transform(df['timeP'])
df['timeR'] = number.fit_transform(df['timeR'])
df['key'] = number.fit_transform(df['key'])
df['event'] = number.fit_transform(df['event'])
df['duration'] = number.fit_transform(df['duration'])


features = ["timeP","timeR", "key", "event", "duration"]
target = "duration"

features_train, features_test, target_train, target_test = train_test_split(df[features],
df[target], 
test_size = 0.22,
random_state = 54)

model = GaussianNB()
model.fit(features_train, target_train)

pred = model.predict(features_test)
accuracy = accuracy_score(target_test, pred)
print(pred)
print(accuracy)

#find way for algo to compare REGISTERED accuracy and LOGIN accuracy
#EG: If accuracy of typing when logging in < accuracy of typing during registration: Reject login

#try to develop database for it