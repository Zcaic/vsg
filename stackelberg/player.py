from typing import Literal
from typing_extensions import runtime
import numpy as np

# from mpire import WorkerPool

import openmdao.api as om
from openmdao.core.constants import _UNDEFINED
from openmdao.core.problem import PETScVector
from openmdao.vectors.default_vector import DefaultVector
from openmdao.vectors.petsc_vector import PETScVector

from pymoo.core.algorithm import Algorithm
from pymoo.core.problem import Problem

# from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.lhs import LHS
from pymoo.core.population import Population


INF = 1.0e30


class History:
    def __init__(self) -> None:
        self.popX = []
        self.popF = []
        self.optX = []
        self.optF = []


class Player(om.Problem):
    def __init__(
        self,
        tag: str,
        optimizer_type: Literal["External", "Internal"] = "External",
        optimizer: Algorithm = None,
        model=None,
        driver=None,
        comm=None,
        name=None,
        reports=None,
        **options
    ):
        """Nash Player.

        Parameters
        ----------
        optimizer_type : Literal['External','Internal'], optional
            The `optimizer_type` parameter is a string that specifies the type of optimizer to be used.
            It can have two possible values: 'External' or 'Internal'.
            If value is `Internal`, you must define optimizer in `openmado.problem`.
        optimizer : if `optimizer_type` is `External`, you must offer its value.

        """
        if not optimizer_type in ["External", "Internal"]:
            raise ValueError(r'optimizer_type must be "External" or "Internal"')
        else:
            self._type = optimizer_type
            self.optimizer = optimizer
        self.result = None
        self.n_evolutionary = 0
        # self.history = {}
        # self.history["X"] = []
        # self.history["F"] = []
        self.history = History()
        self.tag = tag

        super().__init__(model, driver, comm, name, reports, **options)

    def setup(
        self,
        check=False,
        logger=None,
        mode="auto",
        force_alloc_complex=False,
        distributed_vector_class=PETScVector,
        local_vector_class=DefaultVector,
        derivatives=True,
    ):
        super().setup(
            check,
            logger,
            mode,
            force_alloc_complex,
            distributed_vector_class,
            local_vector_class,
            derivatives,
        )
        self.setup_external_optimizer()

    def setup_external_optimizer(self):
        if self._type == "External" and self.optimizer is not None:
            design_vars, ndesvar, desvar_index, lb, ub = self._get_design_info()
            objv, nobjv, objv_index = self._get_objv_info()
            (
                eq_info,
                lower_info,
                upper_info,
                neq,
                nieq,
                eq_index,
                ieq_index,
            ) = self._get_cons_info()

            def init(ins):
                Problem.__init__(
                    ins,
                    n_var=ndesvar,
                    n_obj=nobjv,
                    n_eq_constr=neq,
                    n_ieq_constr=nieq,
                    xl=lb,
                    xu=ub,
                )

            # def _target(ins, x):
            #     for k, v in desvar_index.items():
            #         self.set_design(design_vars[k], x[slice(*v)])
            #     self.run_model()
            #     F = self.get_objv(objv)
            #     H, G = self.get_cons(eq_info, lower_info, upper_info)
            #     return F, G, H

            def evaluate(ins, x, out, *args, **kwargs):
                # out_objv = []
                # out_cons_eq = []
                # out_cons_ieq = []

                # for i in x:
                #     F, G, H = ins._target(i)
                #     out_objv.append(F)
                #     out_cons_ieq.append(G)
                #     out_cons_eq.append(H)
                for k, v in desvar_index.items():
                    self.set_design(design_vars[k], x[:, slice(*v)])
                self.run_model(reset_iter_counts=False)
                F = self.get_objv(objv)
                G, H = self.get_cons(eq_info, lower_info, upper_info)

                out["F"] = F
                if G:
                    out["G"] = G
                if H:
                    out["H"] = H

            Myproblem = type(
                "Myproblem",
                (Problem,),
                {"__init__": init, "_evaluate": evaluate},
            )
            self.udp = Myproblem()


    def _get_design_info(self):
        design_vars: dict = self.model.get_design_vars()
        if design_vars == {}:
            raise RuntimeError('must call "add_design_var" to add desin_var first')
        else:
            ndesvar = 0
            desvar_index = {}

            for k, v in design_vars.items():
                src_name = v["source"]
                npop, nvar = self.model._get_var_meta(src_name, "shape")
                idxs = v["indices"]
                if idxs is not None:
                    idxs = idxs._arr
                else:
                    idxs = slice(None)
                var = np.arange(nvar)[idxs]
                nvar = len(var)
                # size = v["size"]
                desvar_index[k] = (ndesvar, ndesvar + nvar)
                ndesvar += nvar

            lb = np.empty((ndesvar,))
            ub = np.empty((ndesvar,))

            for k, v in design_vars.items():
                i, j = desvar_index[k]
                lb[i:j] = v["lower"]
                ub[i:j] = v["upper"]

        return design_vars, ndesvar, desvar_index, lb, ub

    def _get_objv_info(self):
        objv: dict = self.model.get_objectives()
        if objv == {}:
            raise RuntimeError('must call "add_objective" to add objective first')
        else:
            nobjv = 0
            objv_index = {}
            for k, v in objv.items():
                src_name = v["source"]
                npop, no = self.model._get_var_meta(src_name, "shape")
                idxs = v["indices"]
                if idxs is not None:
                    idxs = idxs._arr
                else:
                    idxs = slice(None)
                no = np.arange(no)[idxs]
                no = len(no)
                # size = v["size"]
                objv_index[k] = (nobjv, nobjv + no)
                nobjv += no
        return objv, nobjv, objv_index

    def _get_cons_info(self):
        cons: dict = self.model.get_constraints()
        eq_info = {}
        lower_info = {}
        upper_info = {}
        neq = 0
        nieq = 0  # squence lower,upper
        eq_index = {}
        ieq_index = {}

        if cons == {}:
            ...
        else:
            for k, v in cons.items():
                src_name = v["source"]
                _, size = self.model._get_var_meta(src_name, "shape")
                if idxs is not None:
                    idxs = idxs._arr
                else:
                    idxs = slice(None)
                size = np.arange(size)[idxs]
                size = len(size)
                if v["equals"] is not None:
                    eq_index[k] = (neq, neq + size)
                    neq += size
                    eq_info[k] = v
                else:
                    lower = v["lower"]
                    upper = v["upper"]
                    lower = np.array(lower).reshape((-1, size))
                    upper = np.array(upper).reshape((-1, size))

                    isize = 0
                    if np.any(lower > -INF):
                        isize += size
                        lower_info[k] = v
                    if np.any(upper < INF):
                        isize += size
                        upper_info[k] = v

                    ieq_index[k] = (nieq, nieq + isize)
                    nieq += isize

        return eq_info, lower_info, upper_info, neq, nieq, eq_index, ieq_index

    def _get_value(self, meta):
        model = self.model
        ind = meta["indices"]
        src_name = meta["source"]

        if ind is None:
            val = model._abs_get_val(src_name).copy()
        else:
            val = model._abs_get_val(src_name)[:, ind.as_array()]

        adder = meta["total_adder"]
        if adder is not None:
            val += adder

        scaler = meta["total_scaler"]
        if scaler is not None:
            val *= scaler

        return val

    def get_objv(self, objv: dict, flat=True):
        ret = {}
        for k, v in objv.items():
            ret[k] = self._get_value(v)
        if flat:
            res = [v for v in ret.values()]
            res = np.hstack(res)
            return res
        else:
            return ret

    def get_cons(self, eq: dict, lower: dict, upper: dict, flat=True):
        ret_eq = {}
        ret_ieq = {}
        if eq != {}:
            for k, v in eq.items():
                ret_eq[k] = self._get_value(v) - v["equals"]
        if lower != {}:
            for k, v in lower.items():
                ret_ieq[k + "_lower"] = v["lower"] - self._get_value(v)
        if upper != {}:
            for k, v in upper.items():
                ret_ieq[k + "_upper"] = self._get_value(v) - v["upper"]
        if flat:
            if ret_eq != {}:
                ret_eq = [v for v in ret_eq.values()]
                ret_eq = np.hstack(ret_eq)
            if ret_ieq != {}:
                ret_ieq = [v for v in ret_ieq.values()]
                ret_ieq = np.hstack(ret_ieq)

        return ret_ieq, ret_eq

    def set_design(self, meta, value):
        # design_vars: dict = self.model.get_design_vars()
        # meta = design_vars[meta]
        src_name = meta["source"]

        desvar = self.model._abs_get_val(src_name)
        loc_idxs = meta["indices"]
        if loc_idxs is not None:
            loc_idxs = loc_idxs._arr
        else:
            loc_idxs = slice(None)
        desvar[:, loc_idxs] = value

        if not meta["total_scaler"] is None:
            desvar[:,loc_idxs] *= 1.0 / meta["total_scaler"]
        if not meta["total_adder"] is None:
            desvar[:,loc_idxs] -= meta["total_adder"]

        # print(self.get_val('x')[:5])
        # return

    def set_srcval(self, name, value, indices=slice(None)):
        src_name = self.model.get_source(name)
        setted_val = self.model._abs_get_val(src_name)
        setted_val[:, indices] = value

        # print(self.get_val('x'))
        # return

    @property
    def opt(self):
        opt_res = {}
        if self.optimizer.problem.n_obj == 1:
            opt_res["X"] = self.result.X
            opt_res["F"] = self.result.F
        else:
            raise RuntimeError("Now it dose not support Multi-objective problem!")
            # TODO

        return opt_res
        
    # def update_External_optimizer(self,optimizer):
    #     self.optimizer=optimizer
    #     self.setup_external_optimizer()

    def run_External_driver(
        self,
        termination=None,
        copy_algorithm=True,
        copy_termination=True,
        prophen: np.ndarray = None,
        sampling=FloatRandomSampling(),
        restart=True,
        savehistory=False,
        **kwargs
    ):
        """
        termination : :class:`~pymoo.core.termination.Termination` or tuple
            The termination criterion that is used to stop the algorithm.

        seed : integer
            The random seed to be used.

        verbose : bool
            Whether output should be printed or not.

        display : :class:`~pymoo.util.display.Display`
            Each algorithm has a default display object for printouts. However, it can be overwritten if desired.

        callback : :class:`~pymoo.core.callback.Callback`
            A callback object which is called each iteration of the algorithm.

        save_history : bool
            Whether the history should be stored or not.

        copy_algorithm : bool
            Whether the algorithm object should be copied before optimization.

        prophen : None | np.ndarray (two-dimensional)
            Prior knowledge
        """
        # if restart:
        #     if prophen is not None:
        #         prophen=np.unique(prophen,axis=0)
        #         pop_size=self.optimizer.pop_size
        #         pop=Population.new(X=prophen)
        #         if len(pop)<pop_size:
        #             pop_rest=sampling(self.udp,pop_size)
        #             pop=Population.merge(pop,pop_rest)
        #             pop=pop[:pop_size]

        #         self.optimizer.initialization.sampling=pop
        #     else:
        #         self.optimizer.initialization.sampling=sampling
        if self.optimizer is None:
            raise ValueError('Please offer a optimizer')
        else:
            if restart:
                self.n_evolutionary = 0

            if (not restart) and (self.optimizer.is_initialized):
                if prophen is not None:
                    prophen = np.unique(prophen, axis=0)
                    pop_size = self.optimizer.pop_size
                    pop = Population.new(X=prophen)
                    if len(pop) < pop_size:
                        pop_rest = self.optimizer.pop.get("X")
                        pop = Population.merge(pop, pop_rest)
                        pop = pop[:pop_size]
                else:
                    pop = self.optimizer.pop.get("X")
                self.optimizer.initialization.sampling = pop
            else:
                if prophen is not None:
                    prophen = np.unique(prophen, axis=0)
                    pop_size = self.optimizer.pop_size
                    pop = Population.new(X=prophen)
                    if len(pop) < pop_size:
                        pop_rest = sampling(self.udp, pop_size)
                        pop = Population.merge(pop, pop_rest)
                        pop = pop[:pop_size]

                    self.optimizer.initialization.sampling = pop
                else:
                    self.optimizer.initialization.sampling = sampling

            self.optimizer.setup(
                problem=self.udp,
                termination=termination,
                copy_algorithm=copy_algorithm,
                copy_termination=copy_termination,
            )
            self.optimizer.is_initialized = False
            # self.result = minimize(
            #     problem=...,
            #     algorithm=self.optimizer,
            #     copy_algorithm=copy_algorithm,

            #     copy_termination=copy_termination,
            #     **kwargs
            # )
            if savehistory:
                while self.optimizer.has_next():
                    self.optimizer.next()
                    self.n_evolutionary += 1
                    # self.history["X"].append(self.optimizer.pop.get("X"))
                    # self.history["F"].append(self.optimizer.pop.get("F"))
                    self.history.popX.append(np.atleast_2d(self.optimizer.pop.get("X")))
                    self.history.popF.append(np.atleast_2d(self.optimizer.pop.get("F")))
                    self.history.optX.append(np.atleast_2d(self.optimizer.opt.get("X")))
                    self.history.optF.append(np.atleast_2d(self.optimizer.opt.get("F")))
            else:
                while self.optimizer.has_next():
                    self.optimizer.next()
                    self.n_evolutionary += 1
                    # print(self.optimizer.pop.get('X'))

            self.result = self.optimizer.result()

            return self.result
