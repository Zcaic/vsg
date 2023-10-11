from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA,BGA
from pymoo.core.problem import Problem
from pymoo.core.population import Population
from pymoo.problems import get_problem
from pymoo.operators.sampling.lhs import LHS,sampling_lhs
import numpy as np

def prepop(prob:Problem,nind,prophet=None):
    if prophet is not None:
        prophet=np.atleast_2d(prophet)
        pre=np.vstack((prophet,LHS()._do(prob,nind)))[:nind]
    else:
        pre=LHS()._do(prob,nind)
    np.random.shuffle(pre)
    return pre

NIND=100
NVAR=10
udp = get_problem("ackley", n_var=NVAR)

# samp=prepop(udp,NIND)
# uda = GA(pop_size=100, eliminate_duplicates=True)
# ones=np.ones((10,10))
# pop=ones*(np.arange(1,11)[:,None])
uda = GA(
    pop_size=100,
    sampling=prepop(udp,NIND)
)

uda.setup(udp, termination=("n_gen", 110), seed=1, verbose=False)

while uda.has_next():
    pop:Population = uda.ask()
    uda.evaluator.eval(udp, pop)

    uda.tell(infills=pop)

    # print(uda.n_gen-1, uda.opt.get('F'))


res = uda.result()

# calculate a hash to show that all executions end with the same result
print(res.exec_time)
