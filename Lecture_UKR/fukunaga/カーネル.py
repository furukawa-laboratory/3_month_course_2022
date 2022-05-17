z_num=100
tikai=1
x_num=1
k_num=0.1
import numpy as np
# z=np.random.uniform(0,20,z_num)
z=np.linspace(0,10,z_num)
x=(3*z+4)[:,None]+(np.random.normal(0,15,(z_num,x_num)))
real_x=(3*z+4)[:,None]


import matplotlib.pyplot as plt

fig = plt.figure()
plt.scatter(z,x)

d = np.abs(z[:,None]-z[None,:])
H = -1*((d**2)/(2*k_num**2))
h = np.exp(H)

bunshi = h@x
bunbo = (np.sum(np.exp(H),axis = 1))

s = bunshi.reshape(-1)/bunbo

plt.scatter(z,x,color='r')
plt.plot(z,s,color='b')
plt.show()