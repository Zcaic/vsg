import openmdao.api as om
import aerosandbox as asb
from stackelberg import Player
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA

from tqdm import tqdm


class udp_airfoil(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("pop_size", default=1, desc="The number of population")
        self.options.declare("baseairfoil", default="naca0012", desc="The base airfoil")

    def setup(self):
        pop_size = self.options["pop_size"]
        self.add_input("aoa", val=0.0, shape=(1, 1), desc="attack angle")
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
            method="cs",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        def _target(il, iu):
            design_lower = self.baseline.lower_weights + il
            design_upper = self.baseline.upper_weights + iu
            design_airfoil = asb.KulfanAirfoil(
                name="design", lower_weights=design_lower, upper_weights=design_upper
            )
            aero = design_airfoil.get_aero_from_neuralfoil(
                alpha=aoa.flatten(), Re=1e6, mach=0.3, model_size="xxxlarge"
            )
            return aero["CL"], aero["CD"], aero["CM"], design_airfoil.max_thickness()

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
            aero = _target(dl[i], du[i])
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


if __name__ == "__main__":
    airPlayer = Player(tag="airfoil", optimizer_type="External")
    model = airPlayer.model
    air = model.add_subsystem(
        name="air",
        subsys=udp_airfoil(baseairfoil="rae2822", pop_size=1),
        promotes_inputs=["*"],
        promotes_outputs=["*"],
    )

    airPlayer.setup()

    airPlayer.run_model()

    print(airPlayer["CL"], airPlayer["CD"], airPlayer["CM"], airPlayer["MAXTC"])
