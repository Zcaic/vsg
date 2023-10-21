from stackelberg import Player,Game
from stackelberg.testfunctions import ZDT1,Rastrigin,Sphere

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
    discipline=Player(tag='zdt1',optimizer_type='External',optimizer=NSGA2(pop_size=npop),reports=None)
    model=discipline.model

    # indeps=om.IndepVarComp()
    # indeps.add_output('x',np.ones((30,)))
    # model.add_subsystem(name='indeps',subsys=indeps,promotes=['x'])

    zdt1=ZDT1(n_var=30,pop_size=npop)
    model.add_subsystem(name='zdt1',subsys=zdt1,promotes=['x','f1','f2'])

    model.add_design_var('x',lower=0,upper=1)
    model.add_objective('f1')
    model.add_objective('f2')
    # model.add_constraint('f1',equals=10,alias='eq0')
    # model.add_constraint('f2',upper=5,alias='ieq0')

    discipline.setup()
    res=discipline.run_External_driver(termination=('n_gen',200))
    # print(discipline['f1'])
    # print(discipline['f2'])

    fig=plt.figure(layout='constrained')
    ax=fig.add_subplot(111)
    ax.plot(res.F[:,0],res.F[:,1],linestyle='none',marker='o',label='pf')
    # ax.legend()
    plt.show()

    # om.n2(discipline)

def test_rastrigin():
    disciplie=Player(tag='Ras',optimizer_type='External',optimizer=GA(pop_size=100))
    model=disciplie.model

    rastrigin=Rastrigin(n_var=2,pop_size=100)
    model.add_subsystem('ras',rastrigin,promotes=['x','f1'])

    model.add_design_var('x',lower=-5.12,upper=5.12)
    model.add_objective('f1')

    disciplie.setup()
    res=disciplie.run_External_driver(termination=('n_gen',200),savehistory=True)

    his=disciplie.history
    optF=disciplie.history.optF
    optF=[i[:,0] for i in optF]
    fig=plt.figure(0,layout='constrained')
    ax=fig.add_subplot(111)
    ax.plot(np.arange(len(optF)),optF,'r-')
    plt.show()
    # x=res.X
    # f=res.F
    # print(x)
    # print(f)


def test_sphere():
    popsize=100
    disciplie=Player(tag='Sphere',optimizer_type='External',optimizer=GA(pop_size=popsize))
    model=disciplie.model

    sph=Sphere(n_var=2,pop_size=popsize)
    model.add_subsystem(name='sph',subsys=sph,promotes=['x','f1'])
    model.add_design_var('x',lower=-5.12,upper=5.12,indices=[1,])
    model.add_objective('f1')

    disciplie.setup()

    disciplie.set_srcval('x',6,indices=[0])
    # print(disciplie.get_val('x')[:5])
    res=disciplie.run_External_driver(termination=('n_gen',200),savehistory=True)
    print(disciplie.get_val('x')[:5])
    print(res.X,res.F)

    # his=disciplie.history
    # F=his.optF
    # F=[i[:,0] for i in F]

    # fig=plt.figure(0,layout='constrained')
    # ax=fig.add_subplot(111)
    # ax.plot(np.arange(len(F)),F,'r-')
    # plt.show()

def NashGame_ras():
    p1=Player(tag='p1',optimizer_type='External',optimizer=GA(pop_size=100))
    model=p1.model
    model.add_subsystem('ras',subsys=Rastrigin(n_var=2,pop_size=100),promotes=['x','f1'])
    model.add_objective('f1')
    model.add_design_var('x',lower=-5.12,upper=5.12,indices=[0,])
    p1.setup()

    p2=Player(tag='p2',optimizer_type='External',optimizer=GA(pop_size=100))
    model=p2.model
    model.add_subsystem('ras',subsys=Rastrigin(n_var=2,pop_size=100),promotes=['x','f1'])
    model.add_objective('f1')
    model.add_design_var('x',lower=-5.12,upper=5.12,indices=[1,])
    p2.setup()

    class Nash(Game):
        def __init__(self, Learders: list[Player] = None, Followers: list[Player] = None) -> None:
            super().__init__(Learders, Followers)

        def run_External_driver(self):
            self.comm_desvar=np.full(2,5.)
            p1ind=[0]
            p2ind=[1]
            p1=self.Leaders[0]
            p2=self.Followers[0]

            NITERS=20
            for i in range(NITERS):
                p1.set_srcval('x',self.comm_desvar)                
                p2.set_srcval('x',self.comm_desvar)
                p1.run_External_driver(termination=('n_gen',10),restart=False)
                p2.run_External_driver(termination=('n_gen',10),restart=False)
                p1X=p1.opt['X'][0]
                p2X=p2.opt['X'][0]
                self.comm_desvar[p1ind]=p1X
                self.comm_desvar[p2ind]=p2X


            
    nash=Nash()
    nash.add_Leader(p1)
    nash.add_Follower(p2)
    nash.setup()
    nash.run_External_driver()
    
    print(nash.comm_desvar)

if __name__=='__main__':
    # test_rastrigin()
    # test_zdt1()
    # test_sphere()
    NashGame_ras()