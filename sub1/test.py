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



reg = LinearRegression().fit(X_train, Y_train)

print('coef', reg.coef_)
print('intercept',reg.intercept_)
print('score', reg.score(X_train, Y_train))

print('predict', reg.predict(X_test))

# print('')
# print('residues', reg.residues_)
# print(reg)

# print(type(Y))


    # X[idx-1]
#     print(line)

#    X.append(np.array(['TV',line[1]]))
#    X.append(np.array(['Radio',line[2]])) 
#    X.append(np.array(['Newspaper',line[3]])) 
#    Y.append(np.array([line[4]]))
# f.close()

# print(X)
# print(Y)
# print(rdr)
# X = np.array(200, 3)
