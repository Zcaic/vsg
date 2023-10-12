from stackelberg import Stackelberg,Player
from stackelberg.testfunctions import ZDT1,Rastrigin

import openmdao.api as om
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA


import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','notebook','no-latex'])
plt.rcParams['font.family']='Times New Roman'


def test_zdt1():
    npop=100
    discipline=Player(optimizer_type='External',optimizer=NSGA2(pop_size=npop),reports=None)
    model=discipline.model

    # indeps=om.IndepVarComp()
    # indeps.add_output('x',np.ones((30,)))
    # model.add_subsystem(name='indeps',subsys=indeps,promotes=['x'])

    zdt1=ZDT1(n_var=30,n_pop=npop)
    model.add_subsystem(name='zdt1',subsys=zdt1,promotes=['x','f1','f2'])

    model.add_design_var('x',lower=0,upper=1)
    model.add_objective('f1')
    model.add_objective('f2')
    # model.add_constraint('f1',equals=10,alias='eq0')
    # model.add_constraint('f2',upper=5,alias='ieq0')

    discipline.setup()
    res=discipline.run_External_driver(termination=('n_gen',100))
    # print(discipline['f1'])
    # print(discipline['f2'])

    fig=plt.figure(layout='constrained')
    ax=fig.add_subplot(111)
    ax.plot(res.F[:,0],res.F[:,1],linestyle='none',marker='o',label='pf')
    # ax.legend()
    plt.show()

    # om.n2(discipline)

def test_rastrigin():
    disciplie=Player(optimizer_type='External',optimizer=GA(pop_size=100))
    model=disciplie.model

    rastrigin=Rastrigin(n_var=2,pop_size=100)
    model.add_subsystem('ras',rastrigin,promotes=['x','f1'])

    model.add_design_var('x',lower=-5.12,upper=5.12)
    model.add_objective('f1')

    disciplie.setup()
    res=disciplie.run_External_driver(termination=('n_gen',5),prophen=np.zeros((100,2)))

    x=res.X
    f=res.F
    print(x)
    print(f)

    res=disciplie.run_External_driver(termination=('n_gen',5),prophen=np.ones((100,2)),restart=False)

    x=res.X
    f=res.F
    print(x)
    print(f)

if __name__=='__main__':
    test_rastrigin()
    # test_zdt1()