# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph
6. Compare the graphs and hence we obtained the linear regression for the given data.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: G.Mithik jain
RegisterNumber: 212224240087

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_atest)
print(y_pred)
print(y_test)
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

*/
```

## Output:
Dataset

<img width="172" height="425" alt="image" src="https://github.com/user-attachments/assets/a01293c2-e7a9-4061-b331-908eaffbeeb4" />


Head values

<img width="235" height="137" alt="image" src="https://github.com/user-attachments/assets/e07768ca-c6d0-41b9-8e86-7108be031d44" />


Tail values

<img width="211" height="135" alt="image" src="https://github.com/user-attachments/assets/107f8292-ee6c-4e40-9872-a4a60db81e71" />


X and Y values

<img width="467" height="370" alt="image" src="https://github.com/user-attachments/assets/beb406cb-1e66-4bb9-bef6-70384e500cbb" />


Predication values of X and Y

<img width="462" height="41" alt="image" src="https://github.com/user-attachments/assets/13cc7022-02c9-4652-a694-d6132935ce3e" />


MSE,MAE and RMSE:

<img width="168" height="46" alt="image" src="https://github.com/user-attachments/assets/7e2bcf76-8b24-4b36-8181-2056a26e3ce2" />


Training Set

<img width="406" height="566" alt="image" src="https://github.com/user-attachments/assets/7b298c50-35ab-407b-8973-fadbca8fb135" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
