from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("salary_dataset.csv")
data.head()

x = data[['YearsExperience']]
y = data['Salary']

x_train, x_test, y_train, y_test = train_test_split( # fixed order of returning values
    x, y,
    test_size = 0.2,
    random_state = 42
)

plt.scatter(x_train, y_train)
plt.xlabel('YOE')
plt.ylabel('Salary')

model = LinearRegression()
model.fit(x_train, y_train) # this is the learning step

print(model.coef_) # gives m for our data (increase in salary after each year)
print(model.intercept_) # gives c for our data (predicted salary when YOE = 0 or x = 0)

model.predict(pd.DataFrame([[5]], columns=['YearsExperience'])) # input feature YOE = 5

y_pred = model.predict(x_test)

print(y_test)
print(y_pred)

print(model.score(x_test, y_test)) # 1 is perfect, 0 is worst

a = float(input("What is your YOE?: "))

prediction = model.predict(
    pd.DataFrame([[a]], columns=['YearsExperience'])
)

print("Predicted salary: ", prediction[0])



