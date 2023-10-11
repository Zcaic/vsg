import openmdao.api as om
import numpy as np


class Stackelberg(object):
    def __init__(self, leader: om.Problem, follower: om.Problem) -> None:
        self.leader = leader
        self.follower = follower

    def _setup(
        self,
    ):
        pass


if __name__ == "__main__":
    udp0 = om.Problem()
    model = udp0.model
    indeps: om.IndepVarComp = model.add_subsystem(
        "indeps", om.IndepVarComp(), promotes=["x"]
    )
    indeps.add_output("x", val=10)

    model.add_subsystem("f1", om.ExecComp("y=x+g", g=2), promotes=["x", "y"])
    model.add_subsystem("f2", om.ExecComp("x=(y-h)/2.", h=3), promotes=["x", "y"])

    # model.connect('indeps.x',['f1.x','f2.x'])
    model.add_objective()

    udp0.setup()
    udp0.run_model()
    print(udp0["y"])
    print(udp0["x"])
    print(udp0["f1.g"])
    print(udp0["f2.h"])
    # om.n2(udp0)
