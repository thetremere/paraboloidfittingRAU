import numpy as np

pts = np.add.accumulate(np.random.random((10,3)))
x,y,z = pts.T

A_xz = np.vstack((x, np.ones(len(x)))).T
m_xz, c_xz = np.linalg.lstsq(A_xz, z)[0]

A_yz = np.vstack((y, np.ones(len(y)))).T
m_yz, c_yz = np.linalg.lstsq(A_yz, z)[0]


def lin(z):
    x = (z - c_xz)/m_xz
    y = (z - c_yz)/m_yz
    return x,y


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = Axes3D(fig)
zz = np.linspace(0,5)
xx,yy = lin(zz)
ax.scatter(x, y, z)
ax.plot(xx,yy,zz)
plt.savefig('test.png')
plt.show()
