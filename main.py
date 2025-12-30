!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
!wget https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O /content/lab_setup_do_not_edit.py
import lab_setup_do_not_edit

import pandas as pd

df = pd.read_csv('parkinsons.csv')
df = df.dropna()
df.head()

import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(df, hue='status')

plt.show()

selected_features = ['NHR', 'PPE']

X = df[selected_features]
y = df['status']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled[:5])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
print(f'x_train shape: {x_train.shape}')
print(f'x_test shape: {x_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
model

from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
accuracy

import joblib

joblib.dump(model, 'my_model.joblib')
