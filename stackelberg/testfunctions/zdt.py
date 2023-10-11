import openmdao.api as om
import numpy as np
from tqdm import tqdm
from mpire import WorkerPool
# from multiprocessing import Pool as npl

__all__=['ZDT1']

def zdt1_target(ix):
    n=len(ix)

    y1=ix[0]
    g = 1 + 9 / (n-1)*np.sum(ix[1:])
    y2=g*(1-np.sqrt(ix[0]/g))

    return y1,y2

class ZDT1(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(
            name="n_var", default=30, desc="The number of variables for the function"
        )
        self.options.declare(name='pop_size',default=100,desc="Number of populations")
        self.options.declare(name='mpi',default=None,desc="Number of processes/ off")
    def setup(self):
        m=self.options['n_var']
        n=self.options['pop_size']
        self.add_input(name="x", val=np.ones((n,m)))
        self.add_output(name="f1", val=np.ones((n,1)))
        self.add_output(name="f2", val=np.ones((n,1)))

    def setup_partials(self):
        self.declare_partials('*','*',method='cs')


    
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        x = inputs["x"]
        # n=self.options['n_var']
        n_pop=self.options['pop_size']
        mpi=self.options['mpi']
        if mpi is None:

            with tqdm(iterable=x,desc=f'iter: {self.iter_count+1}',ncols=150) as pbar:
                ac_results=[zdt1_target(ix) for ix in pbar]

            f1=[]
            f2=[]
            for i in ac_results:
                # rs=i.get()
                y1=i[0]
                y2=i[1]
                f1.append(y1)
                f2.append(y2)

        else:
            pool=WorkerPool(n_jobs=mpi)
            ac_results=pool.map(zdt1_target,[(ix,) for ix in x],iterable_len=n_pop,progress_bar=True)
            pool.stop_and_join()
            f1=[]
            f2=[]
            for i in ac_results:
                y1=i[0]
                y2=i[1]
                f1.append(y1)
                f2.append(y2)

        f1=np.array(f1).reshape((-1,1))
        f2=np.array(f2).reshape((-1,1))

        outputs['f1']=f1
        outputs['f2']=f2
