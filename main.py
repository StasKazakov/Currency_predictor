"""This program predicts the expected euro exchange rate in Ukrainian hryvnias based
    on the dollar exchange rate. The neural network is used for calculation."""
import csv
import numpy as np

# Reading exchange rate data from csv file.
def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    return data

csv_file_path = 'usdeuro.csv' # Exchange rates taken from https://bank.gov.ua
csv_data = read_csv_file(csv_file_path)

# Creating lists with exchange rate for treining.
euro = []
usd = []
for row in csv_data:
    if row[3] == 'USD':
        usd.append(float(row[6]))
    else:
        euro.append(float(row[6]))
    
# Weight inicialization.
weights = 0.9 # Random value.

# Prediction function.
def prediction(u):
    global weights
    return weights * u

# Function for weights updating.
def weights_update(u, e, lr):
    global weights
    e_predict = prediction(u)
    error = e_predict - e
    weights -= lr * error * u
    return weights

# The best values have been selected experimentally.
learning_rate = 0.0001
epochs = 1000

for epoch in range(epochs):
    for u, e in zip(usd, euro):
        weights_update(u, e, learning_rate)

# Testing model.
u_test = np.array([27.4977, 27.5093, 27.7372, 28.4038, 28.3749, 28.9879, 28.9879, 28.3876, 28.2701, 27.9795])
e_test = prediction(u_test)

expected_outcome = np.array([31.1095, 31.1722, 31.7813, 32.1943, 32.0594, 32.3432, 32.3432, 32.1419, 31.9198, 31.996])


# Calculating the accuracy of the model.
def accuracy(e_test, expected_outcome):
    accur = (1 - abs(e_test - expected_outcome) / expected_outcome) * 100
    return np.mean(accur)


print(f'Accuracy of model equal: {round(accuracy(e_test, expected_outcome), 2)}')

# Interaction with the user.
predict_euro = input('Input data: ')
euro_rate = round(prediction(float(predict_euro)), 2)
print(f'Expected euro exchange rate: {euro_rate}')