import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

x_1 = [0, 1, 0, 1]
x_2 = [0, 0, 1, 1]
print(type(x_1))
x = np.shape(x_1)
teta = 1.5

print("論理積")
for i in range(x[0]):
    f = x_1[i] + x_2[i] - 1.5
    if f > 0:
     ans = 1
    else:
     ans = 0

    print(x_1[i],x_2[i],ans)

print()
print()
print("論理和")
for i in range(x[0]):
    f = x_1[i] + x_2[i] - 0.5
    if f > 0:
     ans = 1
    else:
     ans = 0

    print(x_1[i],x_2[i],ans)


print()
print()
print("Simple Perceptron Learning")


random.randint(0,1)
epoch = 30
eat = 0.1
Wow = 0.5
W = np.array([Wow, Wow])
for i in range(epoch):

    x_1 = random.randint(0,1)
    x_2 = random.randint(0,1)
    s = x_1 + x_2
    if s == 2:
        s = 1

    x = np.array([x_1, x_2])

    K = W * x

    f = K[0] + K[1] - teta

    if f > 0:
     ans = 1
    else:
     ans = 0

    if ans == s:
        W = W
    else:
        if ans < s:
            W = W + (eat * x)
        else:
            W = W - (eat * x)

    print(x_1, x_2, ans, "_________", W)


print()
print()
print("Perceptron XOR gate")


random.randint(0,1)
epoch = 30
eat = 0.1
Wow = 0.5
W = np.array([Wow, Wow])
for i in range(epoch):

    x_1 = random.randint(0,1)
    x_2 = random.randint(0,1)
    if x_1 == 1:
        X_1 = 0

    if x_1 == 0:
        X_1 = 1

    if x_2 == 1:
        X_2 = 0

    if x_2 == 0:
        X_2 = 1

    s = x_1 + x_2
    if s == 2:
        s = 1

    x1 = np.array([X_1, x_2])
    x2 = np.array([x_1, X_2])
    K1 = W * x1
    K2 = W * x2
    # f1 = X_1 + x_2 - 1.5
    # f2 = x_1 + X_2 - 1.5
    f1 = K1[0] + K1[1] - 1.5
    f2 = K2[0] + K2[1] - 1.5

    if f1 > 0:
     ans1 = 1
    else:
     ans1 = 0

    if f2 > 0:
     ans2 = 1
    else:
     ans2 = 0

    fin = ans1 + ans2 - 0.5
    if fin > 0:
     ans = 1
    else:
     ans = 0


    if ans1 == s:
        W = W
    else:
        if ans1 < s:
            W = W + (eat * x1)
        else:
            W = W - (eat * x1)

    print(x_1, x_2, "|", ans, "_________", W)