import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

data= pd.read_csv('/Users/sakshi/Documents/covid.csv')
len(data)

#If there is a zero we replace it with mean value
no_zero=['BloodOxy','Age','Temperature']
for col in no_zero:
    data[col]=data[col].replace(0,np.NaN)
    mean=int(data[col].mean(skipna=True))
    data[col]=data[col].replace(np.NaN,mean)

X=data.iloc[:,0:8]
Y=data.iloc[:,8]
X_train,X_test,y_train,y_test= train_test_split(X,Y,random_state=0,test_size=0.2)

#data fitment into a scaler all between 0 and 1 data 
scaler_x= StandardScaler()
X_train=scaler_x.fit_transform(X_train)
X_test=scaler_x.transform(X_test)

#Covid +ve or not is given by p
knn= KNeighborsClassifier(n_neighbors=11,p=2,metric='euclidean')
knn.fit(X_train,y_train)
y_predicts = knn.predict(X_test)
conf_matx=confusion_matrix(y_test,y_predicts)
print(conf_matx)
print(f1_score(y_test,y_predicts))
print(accuracy_score(y_test,y_predicts))



