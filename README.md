# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the required packages.
2. Print the present data and placement data and salary data.
3. Using logistic regression find the predicted values of accuracy confusion matrices.
4. Display the results.

## Program:
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: Mukesh R

RegisterNumber: 212224240098

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('Placement_Data.csv')
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:, :-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = (y_test, y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test, y_pred)
print(classification_report1)
lr.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]])
```

## Output:

### Placement Data
![image](https://github.com/user-attachments/assets/d3e98296-8a8e-46b0-ad63-6e75d6d1f515)


### Checking the null() function
![image](https://github.com/user-attachments/assets/fa436e8d-4c7b-4cef-bb24-853f88a1f8e1)


### Print Data:
![image](https://github.com/user-attachments/assets/a082633b-ad76-44b2-a532-6e517781f35d)


### Y_prediction array
![image](https://github.com/user-attachments/assets/3ba9b8c1-7d36-4d4d-b1be-ad1f00adf346)


### Accuracy value
![image](https://github.com/user-attachments/assets/fad7f75d-206e-4c58-b009-2e002df23579)


### Confusion array
![image](https://github.com/user-attachments/assets/c0184e01-b15b-47a2-abaf-e03cb84cda6d)


### Classification Report
![image](https://github.com/user-attachments/assets/b5a09fde-7760-42cc-a788-dcfd241bd4ef)


### Prediction of LR
![image](https://github.com/user-attachments/assets/994d90fb-a128-4088-bdb5-899a74a00da7)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
