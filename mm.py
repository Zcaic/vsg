import numpy as np

x=np.ones((3,3))
index=np.array([1,2])
index=(slice(None),index)
y=x[index]
print(y)

    

    