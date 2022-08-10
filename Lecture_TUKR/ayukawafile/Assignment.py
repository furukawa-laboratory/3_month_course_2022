import numpy as np
import jax,jaxlib
import jax.numpy as jnp
from tqdm import tqdm #プログレスバーを表示させてくれる
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

sigma = 0.2


a = np.loadtxt('datastore/task/data_learn.txt')
b = np.loadtxt('datastore/task/data_test.txt')

B = np.sort(b)

xa1 = a[:,0]
ya1 = a[:,1]
xa = xa1.reshape(50,1)
ya = np.array(ya1).reshape(50, 1)
# print(xa1.shape)
# print(ya1.shape)
# print(xa.shape)
# print(ya.shape)
# exit()
bb = np.array(b).reshape(50)
# print(ya1.shape)
# print(xa.shape)
# exit()
Chi = ((bb[: , None] - xa1[None, :]) ** 2)
k = np.exp(-1 * (Chi / (2 * sigma ** 2)))
sumk = np.sum(k, axis=1, keepdims=True)
f = (k @ ya) / sumk

plt.scatter(a[:, 0],a[:, 1], c = "b")# 点を描画
plt.scatter(bb, f, c = "r")
f_1 = f.reshape(50)
print(b,f_1)
# plt.plot(b, f)
plt.show()
#
# d = (bb[:, None] - xa1[None, :])**2
# #カーネル関数
# k = np.exp(-1*(d/(2*sigma**2)))
# print(k.shape)
# nume = k@ya
# print(nume.shape)
# deno = np.sum(k, axis=1,keepdims = True)
# print(deno.shape)
# f = nume/deno
# print(f.shape)
#
# plt.scatter(xa1, ya1, c='r')
# plt.scatter(bb, f)
# plt.show()