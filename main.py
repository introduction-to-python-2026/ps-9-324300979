import lab_setup_do_not_edit

import pandas as pd

df = pd.read_csv('parkinsons.csv')
df = df.dropna()

import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(df, hue='status')

selected_features = ['NHR', 'PPE']

X = df[selected_features]
y = df['status']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

import joblib

joblib.dump(model, 'my_model.joblib')
