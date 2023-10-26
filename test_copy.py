import openmdao.api as om
import aerosandbox as asb
from stackelberg import Player, NaGame
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA

from tqdm import tqdm
from mpire import WorkerPool

import concurrent.futures

import time


# def udp_airfoil_target(baseline_lower_weights, baseline_upper_weights, iaoa, il, iu):
#     design_lower = baseline_lower_weights + il
#     design_upper = baseline_upper_weights + iu
#     design_airfoil = asb.KulfanAirfoil(
#         name="design", lower_weights=design_lower, upper_weights=design_upper
#     )
#     aero = design_airfoil.get_aero_from_neuralfoil(
#         alpha=iaoa, Re=1e6, mach=0.3, model_size="xxxlarge"
#     )
#     return aero["CL"], aero["CD"], aero["CM"], design_airfoil.max_thickness()


class udp_airfoil(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("pop_size", default=1, desc="The number of population")
        self.options.declare("baseairfoil", default="naca0012", desc="The base airfoil")
        self.options.declare("mpi", default=0, desc="The number of processes")

    def setup(self):
        pop_size = self.options["pop_size"]
        self.add_input("aoa", val=0.0, shape=(pop_size, 1), desc="attack angle")
        self.add_input(
            "kulfan_dl",
            val=0.0,
            shape=(pop_size, 8),
            desc=r"CST parameters' change of lower surface",
        )
        self.add_input(
            "kulfan_du",
            val=0.0,
            shape=(pop_size, 8),
            desc=r"CST parameters' change of upper surface",
        )
        self.add_output("CL", val=1.0, shape=(pop_size, 1))
        self.add_output("CD", val=1.0, shape=(pop_size, 1))
        self.add_output("CM", val=1.0, shape=(pop_size, 1))
        self.add_output("MAXTC", val=1.0, shape=(pop_size, 1))

        baseairfoil = self.options["baseairfoil"]
        self.baseline = asb.KulfanAirfoil(name=baseairfoil)
        # self.baseline.upper_weights

    def setup_partials(self):
        self.declare_partials(
            of=[
                "*",
            ],
            wrt=[
                "*",
            ],
            method="fd",
        )

    @staticmethod
    def _target(baseline_lower_weights, baseline_upper_weights, iaoa, il, iu):
        design_lower = baseline_lower_weights + il
        design_upper = baseline_upper_weights + iu
        design_airfoil = asb.KulfanAirfoil(
            name="design", lower_weights=design_lower, upper_weights=design_upper
        )
        aero = design_airfoil.get_aero_from_neuralfoil(
            alpha=iaoa, Re=1e6, mach=0.3, model_size="xxxlarge"
        )
        return aero["CL"], aero["CD"], aero["CM"], design_airfoil.max_thickness()

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        mpi = self.options["mpi"]
        if not mpi:
            aoa = inputs["aoa"]
            dl = inputs["kulfan_dl"]
            du = inputs["kulfan_du"]
            pop_size = dl.shape[0]

            pbar = tqdm(total=pop_size, desc=f"iter: {self.iter_count+1}", ncols=150)
            cl = []
            cd = []
            cm = []
            maxtc = []

            for i in range(pop_size):
                aero = udp_airfoil._target(
                    self.baseline.lower_weights,
                    self.baseline.upper_weights,
                    aoa[i],
                    dl[i],
                    du[i],
                )
                cl.append(aero[0])
                cd.append(aero[1])
                cm.append(aero[2])
                maxtc.append(aero[3])
                pbar.update(1)
            pbar.close()

            outputs["CL"] = np.array(cl).reshape(-1, 1)
            outputs["CD"] = np.array(cd).reshape(-1, 1)
            outputs["CM"] = np.array(cm).reshape(-1, 1)
            outputs["MAXTC"] = np.array(maxtc).reshape(-1, 1)

        else:
            aoa = inputs["aoa"]
            dl = inputs["kulfan_dl"]
            du = inputs["kulfan_du"]
            pop_size = dl.shape[0]

            with WorkerPool(n_jobs=mpi) as pool:
                mpirs = pool.map(
                    udp_airfoil._target,
                    zip(
                        np.tile(self.baseline.lower_weights, (pop_size, 1)),
                        np.tile(self.baseline.upper_weights, (pop_size, 1)),
                        aoa,
                        dl,
                        du,
                    ),
                    iterable_len=pop_size,
                    progress_bar=True,
                    progress_bar_options={
                        "desc": f"iter: {self.iter_count}",
                        "ncols": 150,
                    },
                )
            cl = []
            cd = []
            cm = []
            maxtc = []
            for i in mpirs:
                cl.append(i[0])
                cd.append(i[1])
                cm.append(i[2])
                maxtc.append(i[3])

            outputs["CL"] = np.array(cl).reshape(-1, 1)
            outputs["CD"] = np.array(cd).reshape(-1, 1)
            outputs["CM"] = np.array(cm).reshape(-1, 1)
            outputs["MAXTC"] = np.array(maxtc).reshape(-1, 1)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        ...

    def plot(self):
        import matplotlib.pyplot as plt
        import scienceplots

        plt.style.use(["science", "notebook", "no-latex"])
        plt.rcParams["font.family"] = "Times New Roman"

        fig = plt.figure(0, layout="constrained")
        ax = fig.add_subplot(111)

        ax.plot(
            self.baseline.coordinates[:, 0],
            self.baseline.coordinates[:, 1],
            "r-",
            label="baseline",
        )
        ax.legend(prop={"size": 20})
        ax.tick_params(which="both", top=False, right=False)
        plt.show()


def test_aero():
    pop = 100
    airPlayer = Player(tag="airfoil", optimizer_type="External")
    airPlayer.optimizer = GA(pop_size=pop)
    model = airPlayer.model
    air = model.add_subsystem(
        name="air",
        subsys=udp_airfoil(baseairfoil="rae2822", pop_size=pop, mpi=0),
        promotes_inputs=["*"],
        promotes_outputs=["*"],
    )

    model.add_design_var("aoa", lower=-5.0, upper=5.0)
    model.add_design_var("kulfan_dl", lower=-0.2, upper=0.0)
    model.add_design_var("kulfan_du", lower=0.0, upper=0.2)
    model.add_objective("CL", scaler=-1.0)
    model.add_constraint("MAXTC", lower=0.12)
    airPlayer.setup()

    airPlayer.run_External_driver(termination=("n_gen", 100),savetopkl=True,pklfile='./p0_history.pkl')

    print(airPlayer.opt)
    # airPlayer.run_model()

    # print(airPlayer["CL"], airPlayer["CD"], airPlayer["CM"], airPlayer["MAXTC"])


def test_nash():
    p1_pop = 100
    p2_pop = 100
    p3_pop = 100

    p1 = Player(tag="p1", optimizer_type="External")
    p1.optimizer = GA(pop_size=p1_pop)
    model = p1.model
    air = model.add_subsystem(
        "air",
        subsys=udp_airfoil(baseairfoil="rae2822", pop_size=p1_pop, mpi=0),
        promotes_inputs=["*"],
        promotes_outputs=["*"],
    )
    model.add_design_var("aoa", lower=-5.0, upper=5.0)
    model.add_objective("CL", scaler=-1.0)
    model.add_constraint("MAXTC", lower=0.12)
    p1.setup()

    p2 = Player(tag="p2", optimizer_type="External")
    p2.optimizer = GA(pop_size=p2_pop)
    model = p2.model
    air = model.add_subsystem(
        name="air",
        subsys=udp_airfoil(baseairfoil="rae2822", pop_size=p2_pop, mpi=0),
        promotes_inputs=["*"],
        promotes_outputs=["*"],
    )
    model.add_design_var("kulfan_dl", lower=-0.2, upper=0.0)
    model.add_objective("CL", scaler=-1.0)
    model.add_constraint("MAXTC", lower=0.12)
    p2.setup()

    p3 = Player(tag="p3", optimizer_type="External")
    p3.optimizer = GA(pop_size=p3_pop)
    model = p3.model
    air = model.add_subsystem(
        "air",
        subsys=udp_airfoil(baseairfoil="rae2822", pop_size=p3_pop, mpi=0),
        promotes_inputs=["*"],
        promotes_outputs=["*"],
    )
    model.add_design_var("kulfan_du", lower=0.0, upper=0.2)
    model.add_objective("CL", scaler=-1.0)
    model.add_constraint("MAXTC", lower=0.12)
    p3.setup()

    class Nash(NaGame):
        def __init__(self, players: list[Player] = ...) -> None:
            self.players = [p1, p2, p3]

        def run_External_driver(self):
            self.comm_aoa = np.zeros(1)
            self.comm_kulfandl = np.zeros(8)
            self.comm_kulfandu = np.zeros(8)

            Niters = 10
            for i in range(Niters):
                p1.set_srcval("kulfan_dl", self.comm_kulfandl)
                p1.set_srcval("kulfan_du", self.comm_kulfandu)

                p2.set_srcval("aoa", self.comm_aoa)
                p2.set_srcval("kulfan_du", self.comm_kulfandu)

                p3.set_srcval("aoa", self.comm_aoa)
                p3.set_srcval("kulfan_dl", self.comm_kulfandl)

                t1=time.time()

                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    futures1=executor.submit(p1.run_External_driver,termination=('n_gen',10),restart=False,savetopkl=True)
                    futures2=executor.submit(p2.run_External_driver,termination=('n_gen',10),restart=False,savetopkl=True)
                    futures3=executor.submit(p3.run_External_driver,termination=('n_gen',10),restart=False,savetopkl=True)
                    futures=[futures1,futures2,futures3]
                    concurrent.futures.wait(futures)
                
                # p1.run_External_driver(termination=('n_gen',10),restart=False,savetopkl=True)
                # p2.run_External_driver(termination=('n_gen',10),restart=False,savetopkl=True)
                # p3.run_External_driver(termination=('n_gen',10),restart=False,savetopkl=True)

                t2=time.time()
                print(f"execute in {t2-t1} seconds")

                opt_aoa=p1.opt['X']
                opt_dl=p2.opt['X']
                opt_du=p3.opt['X']

                self.comm_aoa[:]=opt_aoa
                self.comm_kulfandl[:]=opt_dl
                self.comm_kulfandu[:]=opt_du
    nash=Nash()
    nash.setup()
    nash.run_External_driver()


def test_SgE():
    ...

if __name__ == "__main__":
    test_aero()
