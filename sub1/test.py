import numpy as np

import csv
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X = np.zeros((200,3))
Y = np.zeros((200,))


f = open('advertising.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
for (idx, line) in enumerate(rdr):
    if idx == 0:
        continue
    X[idx-1][0] = float(line[1])
    X[idx-1][1] = float(line[2])
    X[idx-1][2] = float(line[3])
    Y[idx-1] = float(line[4])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

lrmodel = LinearRegression().fit(X_train, Y_train)

beta_0 = lrmodel.coef_[0]
beta_1 = lrmodel.coef_[1]
beta_2 = lrmodel.coef_[2]
beta_3 = lrmodel.intercept_

print("Scikit-learn의 결과물")
print("beta_0: %f" % beta_0)
print("beta_1: %f" % beta_1)
print("beta_2: %f" % beta_2)
print("beta_3: %f" % beta_3)

# print(X_test)
X_test_pred = np.array(lrmodel.predict(X_test))

print("Mean squared error: %.2f" % mean_squared_error(Y_test, X_test_pred))

print("Variance score: %.2f" % r2_score(Y_test, X_test_pred))

def expected_sales(tv, rd, newspaper, beta_0, beta_1, beta_2, beta_3):
   return (tv*beta_0 + rd*beta_1 + newspaper*beta_2 + beta_3)

print("TV: {}, Radio: {}, Newspaper: {} 판매량: {}".format(
   X_test[3][0],X_test[3][1],X_test[3][2],Y_test[3]))

print("예상 판매량: {}".format(expected_sales(
       float(X_test[3][0]),float(X_test[3][1]),float(X_test[3][2]), beta_0, beta_1, beta_2, beta_3)))


print("predict", lrmodel.predict(X_test)[3])

with open('model.clf', 'wb') as f:
   pickle.dump(lrmodel, f)