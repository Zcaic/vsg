import numpy as np
from sklearn import manifold

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','notebook','no-latex'])

def nor10(x:np.ndarray):
    shape=x.shape
    x=x.flatten()
    xl=np.min(x)
    xu=np.max(x)

    y=(x-xl)/(xu-xl)*20-10
    y=y.reshape(shape)

    return y

N=100
x=np.random.random((100,1))
x=nor10(x)
y=np.random.random((100,1))
y=nor10(y)
z=np.full((100,1),0.5)
data=np.hstack((x,y,z))

# tsne=manifold.TSNE(init='pca')
# trans_data=tsne.fit_transform(data,)
lle=manifold.LocallyLinearEmbedding(n_neighbors=6,n_components=2,method='hessian')
# lle=lle.fit(data)
trans_data=lle.fit_transform(data)

fig=plt.figure(0,layout='constrained')
ax=fig.add_subplot(111,projection='3d')
ax.plot(data[:,0],data[:,1],data[:,2],'ro')

fig=plt.figure(1,layout='constrained')
ax=fig.add_subplot(111)
ax.plot(trans_data[:,0],trans_data[:,1],'go')
plt.show()

