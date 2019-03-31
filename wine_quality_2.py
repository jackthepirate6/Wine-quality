import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm

wine_data = pd.read_csv('wine_quality.csv')

wq = wine_data.copy()
#copying the data so that no changes are made in the original data

wq['quality_label'] = (wq['quality'] > 6.5)*1

features = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
            'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']
target_classifier = ['quality_label']

x = wq[features]
y = wq[target_classifier]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=324)

parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'],
               'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
#an array of dictonary

classifier = svm.SVC(random_state=0)
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,)
#we search for the most suitable parameters

grid_search.fit(x_train,y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

print("the accuracy is ",best_accuracy)

#accuracy is 89.52%






