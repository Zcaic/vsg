import openmdao.api as om
import numpy as np
from tqdm import tqdm

__all__=['Sphere']

def sphere_target(ix):
    f=np.sum(ix**2)
    return f

class Sphere(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(name='n_var',default=2,desc='Number of design vars')
        self.options.declare(name='pop_size',default=100,desc='Number of population')

    def setup(self):
        m=self.options['pop_size']
        n=self.options['n_var']
        self.add_input('x',val=10.,shape=(m,n))
        self.add_output('f1',val=1.,shape=(m,1))

    def setup_partials(self):
        self.declare_partials(of=['*',],wrt=['*',],method='cs')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        x=inputs['x']
        pop_size=len(x)

        pbar=tqdm(total=pop_size,desc=f'iter: {self.iter_count+1}',ncols=150)

        ac_result=[]
        for i in x:
            ac_result.append(sphere_target(i))
            pbar.update(1)
        pbar.close()

        ac_result=np.array(ac_result)
        ac_result=ac_result.reshape((-1,1))

        outputs['f1']=ac_result



if __name__=="__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.axes3d import Axes3D

    N=100
    x=np.linspace(-5.12,5.12,N)
    y=np.linspace(-5.12,5.12,N)
    x,y=np.meshgrid(x,y)

    # z=np.empty_like(x)
    udp=om.Problem()
    sph=Sphere(n_var=2,pop_size=N*N)
    udp.model.add_subsystem('sph',subsys=sph,promotes=['x','f1'])
    udp.setup()

    xx=x.reshape((-1,1))
    yy=y.reshape((-1,1))
    xy=np.hstack((xx,yy))
    
    udp.set_val(name='x',val=xy)
    udp.run_model()
    z=udp.get_val('f1')

    z=np.reshape(z,(N,N))

    fig=plt.figure()
    ax:Axes3D=fig.add_subplot(111,projection='3d')
    
    ax.plot_surface(x,y,z,cmap='inferno')
    plt.show()






        
        

