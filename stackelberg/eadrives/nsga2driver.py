import numpy as np
from openmdao.core.driver import Driver
from pymoo.algorithms.moo.nsga2 import NSGA2

class Nsga2Driver(Driver):
    """
    Driver for a simple genetic algorithm.

    Parameters
    ----------
    **kwargs : dict of keyword arguments
        Keyword arguments that will be mapped into the Driver options.

    Attributes
    ----------
    _problem_comm : MPI.Comm or None
        The MPI communicator for the Problem.
    _concurrent_pop_size : int
        Number of points to run concurrently when model is a parallel one.
    _concurrent_color : int
        Color of current rank when running a parallel model.
    _desvar_idx : dict
        Keeps track of the indices for each desvar, since GeneticAlgorithm sees an array of
        design variables.
    _ga : <GeneticAlgorithm>
        Main genetic algorithm lies here.
    _randomstate : np.random.RandomState, int
         Random state (or seed-number) which controls the seed and random draws.
    _nfit : int
         Number of successful function evaluations.
    """

    def __init__(self, **kwargs):
        """
        Initialize the SimpleGADriver driver.
        """
        if lhs is None:
            raise RuntimeError(f"{self.__class__.__name__} requires the 'pyDOE2' package, "
                               "which can be installed with one of the following commands:\n"
                               "    pip install openmdao[doe]\n"
                               "    pip install pyDOE2")

        super().__init__(**kwargs)

        # What we support
        self.supports['optimization'] = True
        self.supports['integer_design_vars'] = True
        self.supports['inequality_constraints'] = True
        self.supports['equality_constraints'] = True
        self.supports['multiple_objectives'] = True

        # What we don't support yet
        self.supports['two_sided_constraints'] = False
        self.supports['linear_constraints'] = False
        self.supports['simultaneous_derivatives'] = False
        self.supports['active_set'] = False
        self.supports['distributed_design_vars'] = False
        self.supports._read_only = True

        self._desvar_idx = {}
        self._ga = None

        # random state can be set for predictability during testing
        if 'SimpleGADriver_seed' in os.environ:
            self._randomstate = int(os.environ['SimpleGADriver_seed'])
        else:
            self._randomstate = None

        # Support for Parallel models.
        self._concurrent_pop_size = 0
        self._concurrent_color = 0

        self._nfit = 0  # Number of successful function evaluations

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('bits', default={}, types=(dict),
                             desc='Number of bits of resolution. Default is an empty dict, where '
                             'every unspecified variable is assumed to be integer, and the number '
                             'of bits is calculated automatically. If you have a continuous var, '
                             'you should set a bits value as a key in this dictionary.')
        self.options.declare('elitism', types=bool, default=True,
                             desc='If True, replace worst performing point with best from previous'
                             ' generation each iteration.')
        self.options.declare('gray', types=bool, default=False,
                             desc='If True, use Gray code for binary encoding. Gray coding makes'
                             ' the binary representation of adjacent integers differ by one bit.')
        self.options.declare('cross_bits', types=bool, default=False,
                             desc='If True, crossover swaps single bits instead the default'
                             ' k-point crossover.')
        self.options.declare('max_gen', default=100,
                             desc='Number of generations before termination.')
        self.options.declare('pop_size', default=0,
                             desc='Number of points in the GA. Set to 0 and it will be computed '
                             'as four times the number of bits.')
        self.options.declare('run_parallel', types=bool, default=False,
                             desc='Set to True to execute the points in a generation in parallel.')
        self.options.declare('procs_per_model', default=1, lower=1,
                             desc='Number of processors to give each model under MPI.')
        self.options.declare('penalty_parameter', default=10., lower=0.,
                             desc='Penalty function parameter.')
        self.options.declare('penalty_exponent', default=1.,
                             desc='Penalty function exponent.')
        self.options.declare('Pc', default=0.1, lower=0., upper=1.,
                             desc='Crossover rate.')
        self.options.declare('Pm', default=0.01, lower=0., upper=1., allow_none=True,
                             desc='Mutation rate.')
        self.options.declare('multi_obj_weights', default={}, types=(dict),
                             desc='Weights of objectives for multi-objective optimization.'
                             'Weights are specified as a dictionary with the absolute names'
                             'of the objectives. The same weights for all objectives are assumed, '
                             'if not given.')
        self.options.declare('multi_obj_exponent', default=1., lower=0.,
                             desc='Multi-objective weighting exponent.')
        self.options.declare('compute_pareto', default=False, types=(bool, ),
                             desc='When True, compute a set of non-dominated points based on all '
                             'given objectives and update it each generation. The multi-objective '
                             'weight and exponents are ignored because the algorithm uses all '
                             'objective values instead of a composite.')

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer to the containing problem.
        """
        super()._setup_driver(problem)

        # check design vars and constraints for invalid bounds
        for name, meta in self._designvars.items():
            lower, upper = meta['lower'], meta['upper']
            for param in (lower, upper):
                if param is None or np.all(np.abs(param) >= INF_BOUND):
                    msg = (f"Invalid bounds for design variable '{name}'. When using "
                           f"{self.__class__.__name__}, values for both 'lower' and 'upper' "
                           f"must be specified between +/-INF_BOUND ({INF_BOUND}), "
                           f"but they are: lower={lower}, upper={upper}.")
                    raise ValueError(msg)

        for name, meta in self._cons.items():
            equals, lower, upper = meta['equals'], meta['lower'], meta['upper']
            if ((equals is None or np.all(np.abs(equals) >= INF_BOUND)) and
               (lower is None or np.all(np.abs(lower) >= INF_BOUND)) and
               (upper is None or np.all(np.abs(upper) >= INF_BOUND))):
                msg = (f"Invalid bounds for constraint '{name}'. "
                       f"When using {self.__class__.__name__}, the value for 'equals', "
                       f"'lower' or 'upper' must be specified between +/-INF_BOUND "
                       f"({INF_BOUND}), but they are: "
                       f"equals={equals}, lower={lower}, upper={upper}.")
                raise ValueError(msg)

        model_mpi = None
        comm = problem.comm
        if self._concurrent_pop_size > 0:
            model_mpi = (self._concurrent_pop_size, self._concurrent_color)
        elif not self.options['run_parallel']:
            comm = None

        self._ga = GeneticAlgorithm(self.objective_callback, comm=comm, model_mpi=model_mpi)

    def _setup_comm(self, comm):
        """
        Perform any driver-specific setup of communicators for the model.

        Here, we generate the model communicators.

        Parameters
        ----------
        comm : MPI.Comm or <FakeComm> or None
            The communicator for the Problem.

        Returns
        -------
        MPI.Comm or <FakeComm> or None
            The communicator for the Problem model.
        """
        self._problem_comm = comm

        procs_per_model = self.options['procs_per_model']
        if MPI and self.options['run_parallel']:

            full_size = comm.size
            size = full_size // procs_per_model
            if full_size != size * procs_per_model:
                raise RuntimeError("The total number of processors is not evenly divisible by the "
                                   "specified number of processors per model.\n Provide a "
                                   "number of processors that is a multiple of %d, or "
                                   "specify a number of processors per model that divides "
                                   "into %d." % (procs_per_model, full_size))
            color = comm.rank % size
            model_comm = comm.Split(color)

            # Everything we need to figure out which case to run.
            self._concurrent_pop_size = size
            self._concurrent_color = color

            return model_comm

        self._concurrent_pop_size = 0
        self._concurrent_color = 0
        return comm

    def _setup_recording(self):
        """
        Set up case recording.
        """
        if MPI:
            run_parallel = self.options['run_parallel']
            procs_per_model = self.options['procs_per_model']

            for recorder in self._rec_mgr:
                if run_parallel:
                    # write cases only on procs up to the number of parallel models
                    # (i.e. on the root procs for the cases)
                    if procs_per_model == 1:
                        recorder.record_on_process = True
                    else:
                        size = self._problem_comm.size // procs_per_model
                        if self._problem_comm.rank < size:
                            recorder.record_on_process = True

                elif self._problem_comm.rank == 0:
                    # if not running cases in parallel, then just record on proc 0
                    recorder.record_on_process = True

        super()._setup_recording()

    def _get_name(self):
        """
        Get name of current Driver.

        Returns
        -------
        str
            Name of current Driver.
        """
        return "SimpleGA"

    def get_driver_objective_calls(self):
        """
        Return number of objective evaluations made during a driver run.

        Returns
        -------
        int
            Number of objective evaluations made during a driver run.
        """
        return self._nfit

    def get_driver_derivative_calls(self):
        """
        Return number of derivative evaluations made during a driver run.

        Returns
        -------
        int
            Number of derivative evaluations made during a driver run.
        """
        return 0

    def run(self):
        """
        Execute the genetic algorithm.

        Returns
        -------
        bool
            Failure flag; True if failed to converge, False is successful.
        """
        model = self._problem().model
        ga = self._ga

        ga.elite = self.options['elitism']
        ga.gray_code = self.options['gray']
        ga.cross_bits = self.options['cross_bits']
        pop_size = self.options['pop_size']
        max_gen = self.options['max_gen']
        user_bits = self.options['bits']
        compute_pareto = self.options['compute_pareto']

        Pm = self.options['Pm']  # if None, it will be calculated in execute_ga()
        Pc = self.options['Pc']

        self._check_for_missing_objective()
        self._check_for_invalid_desvar_values()

        if compute_pareto:
            self._ga.nobj = len(self._objs)

        # Size design variables.
        desvars = self._designvars
        desvar_vals = self.get_design_var_values()

        count = 0
        for name, meta in desvars.items():
            if name in self._designvars_discrete:
                val = desvar_vals[name]
                if np.ndim(val) == 0:
                    size = 1
                else:
                    size = len(val)
            else:
                size = meta['size']
            self._desvar_idx[name] = (count, count + size)
            count += size

        lower_bound = np.empty((count, ))
        upper_bound = np.empty((count, ))
        outer_bound = np.full((count, ), np.inf)
        bits = np.empty((count, ), dtype=np.int_)
        x0 = np.empty(count)

        # Figure out bounds vectors and initial design vars
        for name, meta in desvars.items():
            i, j = self._desvar_idx[name]
            lower_bound[i:j] = meta['lower']
            upper_bound[i:j] = meta['upper']
            x0[i:j] = desvar_vals[name]

        # Bits of resolution
        abs2prom = model._var_abs2prom['output']

        for name, meta in desvars.items():
            i, j = self._desvar_idx[name]

            if name in abs2prom:
                prom_name = abs2prom[name]
            else:
                prom_name = name

            if name in user_bits:
                val = user_bits[name]

            elif prom_name in user_bits:
                val = user_bits[prom_name]

            else:
                # If the user does not declare a bits for this variable, we assume they want it to
                # be encoded as an integer. Encoding requires a power of 2 in the range, so we need
                # to pad additional values above the upper range, and adjust accordingly. Design
                # points with values above the upper bound will be discarded by the GA.
                log_range = np.log2(upper_bound[i:j] - lower_bound[i:j] + 1)
                val = log_range  # default case -- no padding required
                mask = log_range % 2 > 0  # mask for vars requiring padding
                val[mask] = np.ceil(log_range[mask])
                outer_bound[i:j][mask] = upper_bound[i:j][mask]
                upper_bound[i:j][mask] = 2**np.ceil(log_range[mask]) - 1 + lower_bound[i:j][mask]

            bits[i:j] = val

        # Automatic population size.
        if pop_size == 0:
            pop_size = 4 * np.sum(bits)

        desvar_new, obj, self._nfit = ga.execute_ga(x0, lower_bound, upper_bound, outer_bound,
                                                    bits, pop_size, max_gen,
                                                    self._randomstate, Pm, Pc)

        if compute_pareto:
            # Just save the non-dominated points.
            self.desvar_nd = desvar_new
            self.obj_nd = obj

        else:
            # Pull optimal parameters back into framework and re-run, so that
            # framework is left in the right final state
            for name in desvars:
                i, j = self._desvar_idx[name]
                val = desvar_new[i:j]
                self.set_design_var(name, val)

            with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
                model.run_solve_nonlinear()
                rec.abs = 0.0
                rec.rel = 0.0
            self.iter_count += 1

        return False

    def objective_callback(self, x, icase):
        r"""
        Evaluate problem objective at the requested point.

        In case of multi-objective optimization, a simple weighted sum method is used:

        .. math::

           f = (\sum_{k=1}^{N_f} w_k \cdot f_k)^a

        where :math:`N_f` is the number of objectives and :math:`a>0` is an exponential
        weight. Choosing :math:`a=1` is equivalent to the conventional weighted sum method.

        The weights given in the options are normalized, so:

        .. math::

            \sum_{k=1}^{N_f} w_k = 1

        If one of the objectives :math:`f_k` is not a scalar, its elements will have the same
        weights, and it will be normed with length of the vector.

        Takes into account constraints with a penalty function.

        All constraints are converted to the form of :math:`g_i(x) \leq 0` for
        inequality constraints and :math:`h_i(x) = 0` for equality constraints.
        The constraint vector for inequality constraints is the following:

        .. math::

           g = [g_1, g_2  \dots g_N], g_i \in R^{N_{g_i}}

           h = [h_1, h_2  \dots h_N], h_i \in R^{N_{h_i}}

        The number of all constraints:

        .. math::

           N_g = \sum_{i=1}^N N_{g_i},  N_h = \sum_{i=1}^N N_{h_i}

        The fitness function is constructed with the penalty parameter :math:`p`
        and the exponent :math:`\kappa`:

        .. math::

           \Phi(x) = f(x) + p \cdot \sum_{k=1}^{N^g}(\delta_k \cdot g_k)^{\kappa}
           + p \cdot \sum_{k=1}^{N^h}|h_k|^{\kappa}

        where :math:`\delta_k = 0` if :math:`g_k` is satisfied, 1 otherwise

        .. note::

            The values of :math:`\kappa` and :math:`p` can be defined as driver options.

        Parameters
        ----------
        x : ndarray
            Value of design variables.
        icase : int
            Case number, used for identification when run in parallel.

        Returns
        -------
        float
            Objective value.
        bool
            Success flag, True if successful.
        int
            Case number, used for identification when run in parallel.
        """
        model = self._problem().model
        success = 1

        objs = self.get_objective_values()
        nr_objectives = len(objs)

        # Single objective, if there is only one objective, which has only one element
        if nr_objectives > 1:
            is_single_objective = False
        else:
            for obj in objs.items():
                is_single_objective = len(obj) == 1
                break

        obj_exponent = self.options['multi_obj_exponent']
        if self.options['multi_obj_weights']:  # not empty
            obj_weights = self.options['multi_obj_weights']
        else:
            # Same weight for all objectives, if not specified
            obj_weights = {name: 1. for name in objs.keys()}
        sum_weights = sum(obj_weights.values())

        for name in self._designvars:
            i, j = self._desvar_idx[name]
            self.set_design_var(name, x[i:j])

        # a very large number, but smaller than the result of nan_to_num in Numpy
        almost_inf = INF_BOUND

        # Execute the model
        with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
            self.iter_count += 1
            try:
                model.run_solve_nonlinear()

            # Tell the optimizer that this is a bad point.
            except AnalysisError:
                model._clear_iprint()
                success = 0

            obj_values = self.get_objective_values()
            if is_single_objective:  # Single objective optimization
                for i in obj_values.values():
                    obj = i  # First and only key in the dict

            elif self.options['compute_pareto']:
                obj = np.array([val for val in obj_values.values()]).flatten()

            else:  # Multi-objective optimization with weighted sums
                weighted_objectives = np.array([])
                for name, val in obj_values.items():
                    # element-wise multiplication with scalar
                    # takes the average, if an objective is a vector
                    try:
                        weighted_obj = val * obj_weights[name] / val.size
                    except KeyError:
                        msg = ('Name "{}" in "multi_obj_weights" option '
                               'is not an absolute name of an objective.')
                        raise KeyError(msg.format(name))
                    weighted_objectives = np.hstack((weighted_objectives, weighted_obj))

                obj = sum(weighted_objectives / sum_weights)**obj_exponent

            # Parameters of the penalty method
            penalty = self.options['penalty_parameter']
            exponent = self.options['penalty_exponent']

            if penalty == 0:
                fun = obj
            else:
                constraint_violations = np.array([])
                for name, val in self.get_constraint_values().items():
                    con = self._cons[name]
                    # The not used fields will either None or a very large number
                    if (con['lower'] is not None) and np.any(con['lower'] > -almost_inf):
                        diff = val - con['lower']
                        violation = np.array([0. if d >= 0 else abs(d) for d in diff])
                    elif (con['upper'] is not None) and np.any(con['upper'] < almost_inf):
                        diff = val - con['upper']
                        violation = np.array([0. if d <= 0 else abs(d) for d in diff])
                    elif (con['equals'] is not None) and np.any(np.abs(con['equals']) < almost_inf):
                        diff = val - con['equals']
                        violation = np.absolute(diff)
                    constraint_violations = np.hstack((constraint_violations, violation))
                fun = obj + penalty * sum(np.power(constraint_violations, exponent))
            # Record after getting obj to assure they have
            # been gathered in MPI.
            rec.abs = 0.0
            rec.rel = 0.0

        # print("Functions calculated")
        # print(x)
        # print(obj)
        return fun, success, icase

