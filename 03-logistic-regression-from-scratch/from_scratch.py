import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Pass_Fail_Data.csv")
data.head()

data = data.drop(columns = ["student_id", "attendance_pct", "homework_pct", "midterm_score"])
data.head()

x_train = data["study_hours_per_week"].values
y_train = data["pass"].values

plt.scatter(x_train, y_train) #.plot(connects the points), this doesn't
plt.xlabel("Hours Studied")
plt.ylabel("Pass/Fail")
plt.show()

def predict(x, m, c):
    line = -(x * m + c)
    denominator = 1 + np.exp(line)
    y_predicted = 1 / denominator
    return y_predicted

def cost_function(y_train, x_train, m, c):
    y_predicted = predict(x_train, m, c)
    first_half = y_train * np.log(y_predicted) # math.log() doesn't operate on arrays
    second_half = (1 - y_train) * np.log(1 - y_predicted)
    total = first_half + second_half
    cost = -np.sum(total) / len(y_train)
    return cost

def gradients(y_train, x_train, m, c):
    n = len(y_train)
    
    error_term = predict(x_train, m, c) - y_train
    
    gradient_m = np.sum(error_term * x_train) / n
    gradient_c = np.sum(error_term) / n
    
    return gradient_m, gradient_c

def gradient_descent(y_train, x_train, m, c, learning_rate):
    
    gradient_m, gradient_c = gradients(y_train, x_train, m, c)
    
    m = m - learning_rate * gradient_m
    c = c - learning_rate * gradient_c

    return m, c

m = 0
c = 0
learning_rate = 0.01
epochs = 4000

for i in range(epochs):
    m, c = gradient_descent(y_train, x_train, m, c, learning_rate)
    
    if i % 200 == 0:
        print(f"Iteration: {i} Cost: {cost_function(y_train, x_train, m, c)}")
        

print(f'Final m: {m}, Final c: {c}')


predictions = predict(x_train, m, c)

x_plot = np.linspace(min(x_train) - 5, max(x_train) + 5, 300)
y_plot = predict(x_plot, m, c)

plt.figure(figsize=(8, 5))
plt.scatter(x_train, y_train, color="blue", label="Actual Data")
plt.plot(x_plot, y_plot, color="red", linewidth=2.5, label="Logistic Curve")

plt.xlim(min(x_plot), max(x_plot))
plt.ylim(-0.05, 1.05)

plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

hours = float(input("How many hours did you study?: "))

prediction = predict(hours, m, c)

if(prediction >= 0.5):
    print("Pass")
else:
    print("Fail")