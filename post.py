import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import scienceplots
plt.style.use(['science','notebook','no-latex'])
plt.rcParams['font.family']='Times New Roman'

with open('./p1_history.pkl','rb') as fin:
    data=pickle.load(fin)

F=data.optF
F1=[i[0] for i in F]

with open('./p2_history.pkl','rb') as fin:
    data=pickle.load(fin)

F=data.optF
F2=[i[0] for i in F]

with open('./p3_history.pkl','rb') as fin:
    data=pickle.load(fin)

F=data.optF
F3=[i[0] for i in F]

with open('./p0_history.pkl','rb') as fin:
    data=pickle.load(fin)

F=data.optF
F0=[i[0] for i in F]

fig=plt.figure(0,layout='constrained')
ax=fig.add_subplot(111)

ax.plot(F1,label='player1')
ax.plot(F2,label='player2')
ax.plot(F3,label='player3')
ax.plot(F0,label='Normal')

ax.legend(prop={'size':20})
ax.xaxis.set_major_locator(MultipleLocator(10))

plt.show()

