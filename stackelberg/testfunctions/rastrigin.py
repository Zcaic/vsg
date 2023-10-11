import openmdao.api as om
import numpy as np

from tqdm import tqdm

__all__=['Rastrigin']

def rastrigin_target(ix):
    y=20+np.sum(ix**2-10*np.cos(2*np.pi*ix))
    return y


class Rastrigin(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n_var',default=2,desc="number of variables")
        self.options.declare('pop_size',default=100,desc="Number of populations")
        
    def setup(self):
        m=self.options['pop_size']
        n=self.options['n_var']
        self.add_input('x',val=1,shape=(m,n))
        self.add_output('f1',val=1,shape=(m,1))

    def setup_partials(self):
        self.declare_partials(of=['*'],wrt=['*'],method='cs')
    
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        x=inputs['x']
        total=len(x)
        ac_res=[]

        pbar=tqdm(total=total,ncols=150)
        for ix in x:
            pbar.set_description(f'iter: {self.iter_count+1}')
            ac_res.append(rastrigin_target(ix))
            pbar.update(1)
        pbar.close()

        f1=np.array(ac_res).reshape((-1,1))
        outputs['f1']=f1
