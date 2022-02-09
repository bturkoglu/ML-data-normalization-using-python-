import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

#url ='https://raw.githubusercontent.com/cagriemreakin/Machine-Learning/master/1-%20Data%20Preprocessing/dataset.csv'
dataset = pd.read_csv('kanser2.csv', header=None, usecols=range(1,11))
dataset = dataset*1.0
print(dataset)

X = dataset.iloc[:].values

print("**************************************")
print(X)
"""
y = dataset.iloc[:, 9].values

print("**************************************")
print(y)



from sklearn.preprocessing import Imputer #Imputer Class' ının yüklenmesi
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features = [0])
X = ohe.fit_transform(X).toarray()
# Diğer kategorik değer ise, dependent variable olan Purchase sütunu
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
"""

#"""

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling (Özellik Ölçekleme)
from sklearn.preprocessing import MinMaxScaler
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(dataset)
#X_test = sc_X.transform(X_test)

print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
print(X_train)
#"""

for satir in X_train:
    for hucre in satir:
        print('{0:1.10f},'.format(hucre), end=' ')
    print()

for satir in X_train:
    line = ''
    for hucre in satir:
        line = line + '{0:1.10f}, '.format(hucre)
    print(line[:-2])
""""
ExportToFile = "aga7.csv"
with open(ExportToFile, 'w') as out:
    for satir in X_train:
        line = ''
        for hucre in satir:
            line += '{0:1.10f}, '.format(hucre)
        print(line[:-2], file=out)
"""
ExportToFile = "aga8.csv"
with open(ExportToFile, 'w') as out:
    for satir in X_train:
        line = ', '.join([str(i) for i in satir])
        print(line, file=out)

"""

dataset.to_csv(ExportToFile)


with open(ExportToFile, 'w', newline='\n') as out:
    writer = csv.writer(out, delimiter=',')
    writer.writerow(X_train)
out.close()
"""