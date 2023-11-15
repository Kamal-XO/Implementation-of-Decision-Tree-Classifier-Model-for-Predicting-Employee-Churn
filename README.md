# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import pandas module and import the required data set.

2.Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values.

5.From sklearn.model_selection import train_test_split.

6.Assign the train dataset and test dataset.

7.From sklearn.tree import DecisionTreeClassifier.

8.Use criteria as entropy.

9.From sklearn import metrics.

10.Find the accuracy of our model and predict the require values.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Kamalesh SV
RegisterNumber:  212222240041
```
```

import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()

data.isnull().sum()

data["left"].value_counts

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

### Data.head():
![output](https://user-images.githubusercontent.com/118708624/275312260-9e04f714-a610-408d-b1a9-8dd85b2d26ca.png)

### Data.info():
![output](https://user-images.githubusercontent.com/118708624/275312397-93170cd3-0c4a-48a4-a5dc-9fa300988810.png)

### isnull() and sum():
![output](https://user-images.githubusercontent.com/118708624/275312509-672f410e-c623-42ca-a3bb-a7dc0dd56e57.png)

### Data Value Counts():
![output](https://user-images.githubusercontent.com/118708624/275312651-5b877215-afa9-4d24-8089-2666ad8bfe6e.png)

### Data.head() for salary:
![output](https://user-images.githubusercontent.com/118708624/275312740-2a426f24-026b-4cc4-a4f4-d06c79b51c01.png)

### x.head():
![output](https://user-images.githubusercontent.com/118708624/275312831-454aa3e2-941b-441b-811f-820ddda8e208.png)

### Accuracy Value:
![output](https://user-images.githubusercontent.com/118708624/275312924-7b327d26-5101-4ede-b8bc-7fd2bee7076a.png)

### Data Prediction:
![output](https://user-images.githubusercontent.com/118708624/275313004-d86df3aa-373b-4beb-a92a-b728bbf492db.png)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
