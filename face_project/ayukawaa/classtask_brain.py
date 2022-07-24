import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# x_1 = [0, 1, 0, 1]
# x_2 = [0, 0, 1, 1]
# print(type(x_1))
# x = np.shape(x_1)
# teta = 1.5
#
# print("論理積")
# for i in range(x[0]):
#     f = x_1[i] + x_2[i] - 1.5
#     if f > 0:
#      ans = 1
#     else:
#      ans = 0
#
#     print(x_1[i],x_2[i],ans)
#
# print()
# print()
# print("論理和")
# for i in range(x[0]):
#     f = x_1[i] + x_2[i] - 0.5
#     if f > 0:
#      ans = 1
#     else:
#      ans = 0
#
#     print(x_1[i],x_2[i],ans)
#
#
# print()
# print()
# print("Simple Perceptron Learning")


# random.randint(0,1)
# epoch = 30
# eat = 0.1
# Wow = 0.5
# W = np.array([Wow, Wow])
# for i in range(epoch):
#
#     x_1 = random.randint(0,1)
#     x_2 = random.randint(0,1)
#     s = x_1 + x_2
#     if s == 2:
#         s = 1
#
#     x = np.array([x_1, x_2])
#
#     # for n in range(2)
#
#     K = W * x
#     # print(K)
#     f = K[0] + K[1] - teta
#     # print(W*x)
#     # print(f)
#     # exit()
#     if f > 0:
#      ans = 1
#     else:
#      ans = 0
#
#     if ans == s:
#         W = W
#     else:
#         if ans < s:
#             W = W + (eat * x)
#         else:
#             W = W - (eat * x)
#
#     print(x_1,x_2,ans, "_________",W)





