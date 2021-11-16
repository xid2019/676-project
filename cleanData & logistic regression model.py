import pandas as pd
import numpy as np
# reading csv files
colTitles = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship',
             'race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
data = pd.read_csv('adult.data', sep=",",names=colTitles)
data = data.drop(columns=['fnlwgt','education-num','relationship','capital-gain','capital-loss'])

rows = data.shape[0]
for row in range(rows):
    # # age: 5 years per class
    # data.iloc[row,0] = data.iloc[row,0]//5
    # # hours-per-week: 10 hrs per class
    # data.iloc[row,7] = data.iloc[row,7]//10
    # # income: 1: >50k, 0:<= 50k
    if data.iloc[row,9] == ' <=50K':
        data.iloc[row,9] = 0
    else:
        data.iloc[row,9] = 1

# remove all spaces and remove rows that contains ?
for i in range(10):
    if i not in [0,7,9]:
        data.iloc[:,i] = data.iloc[:,i].str.strip()
    data = data[data.iloc[:,i] != '?']

# classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
sc = StandardScaler()
X[:,0:1] = sc.fit_transform(X[:,0:1])
X[:,7:8] = sc.fit_transform(X[:,7:8])

# One Hot encoding categorical data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1,2,3,4,5,6,8])], remainder='passthrough')
X = ct.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Training the Logistic Regression model on the Training set
y_train = y_train.astype('int')
y_test = y_test.astype('int')
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0, max_iter=1000)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_predict = classifier.predict(X_test)
np.set_printoptions(precision=2)
compare = np.concatenate((y_test.reshape(len(y_test),1), y_predict.reshape(len(y_predict),1)),1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_predict)
print('class 0, class 1')
print(cm)
score = accuracy_score(y_test, y_predict)
print('accuracy_score = ', score)