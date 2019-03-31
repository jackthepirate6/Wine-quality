import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

wine_data = pd.read_csv('wine_quality.csv')

wq = wine_data.copy()
#copying the data so that no changes are made in the original data

wq['quality_label'] = (wq['quality'] > 6.5)*1

features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
            'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
target_classifier = ['quality_label']

x = wq[features]
y = wq[target_classifier]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=324)

logisticregr = LogisticRegression()

logisticregr.fit(x_train , y_train)
prediction = logisticregr.predict(x_test)
acc = accuracy_score(y_true=y_test, y_pred=prediction)

print("the accuracy is ",acc)

#accuracy is 88.25%






