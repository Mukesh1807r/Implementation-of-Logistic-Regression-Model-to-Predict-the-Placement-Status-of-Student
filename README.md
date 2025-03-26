# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Mukesh R 
RegisterNumber: 212224240098  
```

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv("Placement_Data.csv")
dataset

dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)

dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset

X= dataset.iloc[:,:-1].values
Y= dataset.iloc[:,-1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
dataset.head()

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=0,solver='lbfgs',max_iter=1000).fit(X_train,Y_train)
clf.score(X_test,Y_test)

clf.predict([[0,87,0,95,0,2,78,2,0,0,1,0]])

clf.predict([[0,7,0,95,0,2,8,2,0,0,1,0]])

```

## Output:
![image](https://github.com/user-attachments/assets/c9a78742-ddc5-41d1-b7b3-f8280491a69f)

![image](https://github.com/user-attachments/assets/9e63fb7f-3c59-4a82-b348-6e6299783fe0)

![image](https://github.com/user-attachments/assets/236139aa-b1f7-4803-8a2a-22fa8b609ce5)

![image](https://github.com/user-attachments/assets/a8b61dc2-6d56-4926-a87e-a90ee4bc84e7)

![image](https://github.com/user-attachments/assets/36c1a5ee-c4bb-4a17-9ac4-cd61e30769c7)

![image](https://github.com/user-attachments/assets/372d009e-5990-43b5-a3c8-636654590b96)

![image](https://github.com/user-attachments/assets/583ed21f-47d3-48b2-8e12-eab4b2f139b5)








## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
