import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('time_vs_score.csv')

x_train = data['Hours_Studied'].values # .values on any pandas series returns a NumPy ndarray 
y_train = data['Exam_Score'].values # array of exam score

print('Array of hours studied:', x_train)
print('Array of exam scores:', y_train)

plt.scatter(x_train, y_train)
plt.title('Data')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.show()

def predict(x, m, c):
    y_predicted = m * x + c
    return y_predicted

print("For single x = 4.49:" , predict(4.49, 10, 5)) # return a single value
print("For the array x_train:", predict(x_train, 10, 5)) # returns an array of y_predicted wrt x_train

def cost_function(y_train, x_train, m, c):
    
    n = len(y_train)
    
    errors = y_train - predict(x_train, m, c) # array of all the errors

    sq_errors = errors ** 2 # squared error array is in scientific notation
                            # numpy does this to represent numbers in a compact form
                    
    total_sq_error = np.sum(sq_errors)

    cost = total_sq_error / n
    
    return cost
    
cost = cost_function(y_train, x_train, 0, 0)
print(cost)

def gradients(y_train, x_train, m, c):
    
    n = len(y_train)
    
    errors = y_train - predict(x_train, m, c)
    
    gradient_m = (-2 / n) * np.sum(errors * x_train)
    gradient_c = (-2 / n) * np.sum(errors)
    
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

plt.scatter(x_train, y_train)
plt.plot(x_train, predictions, color = 'red')
plt.title("Linear Regression using Gradient Descent")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.show()

hours = float(input("How many houes did you study?: "))

predicted_marks = predict(hours, m, c)

print(f"Yours expected marks: {predicted_marks:.2f}")

