from random import choice
from numpy import array, dot, random
import matplotlib.pyplot as plt 

def classify(number):
    return 1 if number < 0 else -1

def normalize(data, normalize_class):
    newData = []
    for i in data:
        if i[1] == normalize_class:
            newValue = i[0]*-1
        else:
            newValue = i[0]
        newData.append((newValue, i[1]))
    return newData

training_data = [ 
        (array([2, 7, 1]), -1),
        (array([8, 1, 1]), -1),
        (array([7, 5, 1]), -1),
        (array([6, 3, 1]), -1),
        (array([7, 8, 1]), -1),
        (array([5, 9, 1]), -1),
        (array([4, 5, 1]), -1),
        (array([4, 2, 1]), 1),
        (array([-1, -1, 1]), 1),
        (array([1, 3, 1]), 1),
        (array([3, -2, 1]), 1),
        (array([5, 3.25, 1]), 1),
        (array([2, 4, 1]), 1),
        (array([7, 1, 1]), 1)
        ]

new_training_data = normalize(training_data, 1)
weights = random.rand(3)
eta = 0.02
n = 100000
k = 0

print weights

for i in range(n):
    value = (k + 1) % len(training_data)
    yk = new_training_data[value][0]
    if dot(weights.T, yk) < 0:
        weights += yk
    k += 1

print weights

final_data = []
for value, _ in training_data:
    result = dot(weights.T, value)
    final_data.append((value, classify(result)))
    print value, result, classify(result)

a, b, c = weights
slope = -(a/b)
x11 = -2
y11 = (slope * x11) - c/b
x12 = 10
y12 = (slope * x12) - c/b

x1 = []
y1 = []
x2 = []
y2 = []
for i, j in training_data:
    if j == -1:
        x1.append(i[0])
        y1.append(i[1])
    else:
        x2.append(i[0])
        y2.append(i[1])
plt.scatter(x1, y1, c = 'red')
plt.scatter(x2, y2, c = 'blue')
plt.scatter([x11, x12], [y11, y12], c = 'green')
plt.plot([x11, x12], [y11, y12], c = 'green')
plt.show()
