from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival, binary_tournament

from pymoo.core.population import Population

# from pymoo.operators.crossover.sbx import SBX
# from pymoo.operators.mutation.pm import PM
# from pymoo.operators.sampling.rnd import FloatRandomSampling
# from pymoo.operators.selection.tournament import TournamentSelection
# from pymoo.core.mating import Mating
# from pymoo.operators.selection.tournament import TournamentSelection
# from pymoo.operators.crossover.sbx import SBX
# from pymoo.operators.mutation.pm import PM
# from pymoo.core.repair import NoRepair
# from pymoo.core.duplicate import DefaultDuplicateElimination
from pymoo.core.problem import Problem

from pymoo.optimize import minimize
from pymoo.problems import get_problem


from minisom import MiniSom

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import RegularPolygon
from pymoo.util.display.multi import MultiObjectiveOutput
import scienceplots

plt.style.use(["science", "notebook", "no-latex"])
plt.rcParams["font.family"] = "Times New Roman"

# class Mymat(Mating):
#     def __init__(self):
#         self.selection=TournamentSelection(func_comp=binary_tournament)
#         self.crossover=SBX(eta=15, prob=0.9)
#         self.mutation=PM(eta=20)
#         self.repair=NoRepair()
#         self.eliminate_duplicates=DefaultDuplicateElimination()
#         self.n_max_iterations=100


#     def _do(self, problem, pop, n_offsprings, parents=None, **kwargs):
#         print('it is my mating')
#         off1=super()._do(problem, pop, n_offsprings, parents, **kwargs)
#         return off1


class DT_NSGA2(NSGA2):
    def setup(self, problem, **kwargs):

        super().setup(problem, **kwargs)

    def _infill(self):
        noff = self.n_gen
        off = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)

        # if the mating could not generate any new offspring (duplicate elimination might make that happen)
        if len(off) == 0:
            self.termination.force_termination = True
            return

        # if not the desired number of offspring could be created
        elif len(off) < self.n_offsprings:
            if self.verbose:
                print(
                    "WARNING: Mating could not produce the required number of (unique) offsprings!"
                )

        return off
    

if __name__ == "__main__":
    udp = get_problem("zdt1")

    uda = DT_NSGA2(pop_size=100)
    # uda.setup(udp, D=9, seed=1, verbose=False)
    uda.setup(udp, verbose=False)

    for i in range(100):
        pop = uda.ask()
        uda.evaluator.eval(udp, pop)
        uda.tell(infills=pop)
        # fronts=uda.pop.get('rank')
        # fronts=set(fronts)

    res = uda.result()
    fronts = res.pop.get("rank")
    fronts_set = np.unique(fronts)

    rank_pop = []
    for i in fronts_set:
        index = fronts == i
        rank_pop.append(res.pop[index])

    fig = plt.figure(0, layout="constrained")
    ax = fig.add_subplot(111)

    # ax.plot(udp.pareto_front()[:,0],udp.pareto_front()[:,1],udp.pareto_front()[:,2],'ro',alpha=0.7)
    # ax.plot(res.F[:,0],res.F[:,1],res.F[:,2],'go',markerfacecolor='none')
    # ax.plot(res.X[:,0],res.X[:,1],res.X[:,2],'go',markerfacecolor='none')

    for i, vi in enumerate(rank_pop):
        objv = vi.get("F")
        ax.plot(objv[:, 0], objv[:, 1], marker="o", linestyle="none", label=f"rank_{i}")
    ax.legend(fontsize=20)
    plt.show()
