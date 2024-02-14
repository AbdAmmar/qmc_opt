
import numpy as np
from numpy import (atleast_1d, eye, argmin, zeros, shape, squeeze,
                   asarray, sqrt)
import warnings

_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has '
                              'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.',
                   'nan': 'NaN result encountered.',
                   'out_of_bounds': 'The result is outside of the provided '
                                    'bounds.'}


def fmin_powell(func, x0, args=(), bounds=None, xtol=1e-4, ftol=1e-4, maxiter=None, maxfev=None,
                maxfun=None, full_output=0, disp=1, retall=0, callback=None,
                direc=None):
    """
    Minimize a function using modified Powell's method.

    This method only uses function values, not derivatives.

    Parameters
    ----------
    func : callable f(x,*args)
        Objective function to be minimized.
    x0 : ndarray
        Initial guess.
    args : tuple, optional
        Extra arguments passed to func.
    xtol : float, optional
        Line-search error tolerance.
    ftol : float, optional
        Relative error in ``func(xopt)`` acceptable for convergence.
    maxiter : int, optional
        Maximum number of iterations to perform.
    maxfun : int, optional
        Maximum number of function evaluations to make.
    full_output : bool, optional
        If True, ``fopt``, ``xi``, ``direc``, ``iter``, ``funcalls``, and
        ``warnflag`` are returned.
    disp : bool, optional
        If True, print convergence messages.
    retall : bool, optional
        If True, return a list of the solution at each iteration.
    callback : callable, optional
        An optional user-supplied function, called after each
        iteration.  Called as ``callback(xk)``, where ``xk`` is the
        current parameter vector.
    direc : ndarray, optional
        Initial fitting step and parameter order set as an (N, N) array, where N
        is the number of fitting parameters in `x0`. Defaults to step size 1.0
        fitting all parameters simultaneously (``np.eye((N, N))``). To
        prevent initial consideration of values in a step or to change initial
        step size, set to 0 or desired step size in the Jth position in the Mth
        block, where J is the position in `x0` and M is the desired evaluation
        step, with steps being evaluated in index order. Step size and ordering
        will change freely as minimization proceeds.

    Returns
    -------
    xopt : ndarray
        Parameter which minimizes `func`.
    fopt : number
        Value of function at minimum: ``fopt = func(xopt)``.
    direc : ndarray
        Current direction set.
    iter : int
        Number of iterations.
    funcalls : int
        Number of function calls made.
    warnflag : int
        Integer warning flag:
            1 : Maximum number of function evaluations.
            2 : Maximum number of iterations.
            3 : NaN result encountered.
            4 : The result is out of the provided bounds.
    allvecs : list
        List of solutions at each iteration.

    See also
    --------
    minimize: Interface to unconstrained minimization algorithms for
        multivariate functions. See the 'Powell' method in particular.

    Notes
    -----
    Uses a modification of Powell's method to find the minimum of
    a function of N variables. Powell's method is a conjugate
    direction method.

    The algorithm has two loops. The outer loop merely iterates over the inner
    loop. The inner loop minimizes over each current direction in the direction
    set. At the end of the inner loop, if certain conditions are met, the
    direction that gave the largest decrease is dropped and replaced with the
    difference between the current estimated x and the estimated x from the
    beginning of the inner-loop.

    The technical conditions for replacing the direction of greatest
    increase amount to checking that

    1. No further gain can be made along the direction of greatest increase
       from that iteration.
    2. The direction of greatest increase accounted for a large sufficient
       fraction of the decrease in the function value from that iteration of
       the inner loop.

    References
    ----------
    Powell M.J.D. (1964) An efficient method for finding the minimum of a
    function of several variables without calculating derivatives,
    Computer Journal, 7 (2):155-162.

    Press W., Teukolsky S.A., Vetterling W.T., and Flannery B.P.:
    Numerical Recipes (any edition), Cambridge University Press

    Examples
    --------
    >>> def f(x):
    ...     return x**2

    >>> from scipy import optimize

    >>> minimum = optimize.fmin_powell(f, -1)
    Optimization terminated successfully.
             Current function value: 0.000000
             Iterations: 2
             Function evaluations: 16
    >>> minimum
    array(0.0)

    """
    opts = {'xtol': xtol,
            'bounds': bounds,
            'ftol': ftol,
            'maxiter': maxiter,
            'maxfev': maxfun,
            'disp': disp,
            'direc': direc,
            'return_all': retall}

    callback = _wrap_callback(callback)
    res = _minimize_powell(func, x0, args, callback=callback, **opts)

    if full_output:
        retlist = (res['x'], res['fun'], res['direc'], res['nit'],
                   res['nfev'], res['status'])
        if retall:
            retlist += (res['allvecs'], )
        return retlist
    else:
        if retall:
            return res['x'], res['allvecs']
        else:
            return res['x']


def _call_callback_maybe_halt(callback, res):
    """Call wrapped callback; return True if algorithm should stop.

    Parameters
    ----------
    callback : callable or None
        A user-provided callback wrapped with `_wrap_callback`
    res : OptimizeResult
        Information about the current iterate

    Returns
    -------
    halt : bool
        True if minimization should stop

    """
    if callback is None:
        return False
    try:
        callback(res)
        return False
    except StopIteration:
        callback.stop_iteration = True
        return True


def _minimize_powell(func, x0, args=(), callback=None, bounds=None,
                     xtol=1e-4, ftol=1e-4, maxiter=None, maxfev=None,
                     disp=False, direc=None, return_all=False,
                     **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    modified Powell algorithm.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.

            ``fun(x, *args) -> float``

        where ``x`` is a 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where ``n`` is the number of independent variables.
    args : tuple, optional
        Extra arguments passed to the objective function and its
        derivatives (`fun`, `jac` and `hess` functions).
    method : str or callable, optional
        The present documentation is specific to ``method='powell'``, but other
        options are available. See documentation for `scipy.optimize.minimize`.
    bounds : sequence or `Bounds`, optional
        Bounds on decision variables. There are two ways to specify the bounds:

            1. Instance of `Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. None
               is used to specify no bound.

        If bounds are not provided, then an unbounded line search will be used.
        If bounds are provided and the initial guess is within the bounds, then
        every function evaluation throughout the minimization procedure will be
        within the bounds. If bounds are provided, the initial guess is outside
        the bounds, and `direc` is full rank (or left to default), then some
        function evaluations during the first iteration may be outside the
        bounds, but every function evaluation after the first iteration will be
        within the bounds. If `direc` is not full rank, then some parameters
        may not be optimized and the solution is not guaranteed to be within
        the bounds.

    options : dict, optional
        A dictionary of solver options. All methods accept the following
        generic options:

            maxiter : int
                Maximum number of iterations to perform. Depending on the
                method each iteration may use several function evaluations.
            disp : bool
                Set to True to print convergence messages.

        See method-specific options for ``method='powell'`` below.
    callback : callable, optional
        Called after each iteration. The signature is:

            ``callback(xk)``

        where ``xk`` is the current parameter vector.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully and
        ``message`` which describes the cause of the termination. See
        `OptimizeResult` for a description of other attributes.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    xtol : float
        Relative error in solution `xopt` acceptable for convergence.
    ftol : float
        Relative error in ``fun(xopt)`` acceptable for convergence.
    maxiter, maxfev : int
        Maximum allowed number of iterations and function evaluations.
        Will default to ``N*1000``, where ``N`` is the number of
        variables, if neither `maxiter` or `maxfev` is set. If both
        `maxiter` and `maxfev` are set, minimization will stop at the
        first reached.
    direc : ndarray
        Initial set of direction vectors for the Powell method.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    """
    _check_unknown_options(unknown_options)
    maxfun = maxfev
    retall = return_all

    x = asarray(x0).flatten()
    if retall:
        allvecs = [x]
    N = len(x)
    # If neither are set, then set both to default
    if maxiter is None and maxfun is None:
        maxiter = N * 1000
        maxfun = N * 1000
    elif maxiter is None:
        # Convert remaining Nones, to np.inf, unless the other is np.inf, in
        # which case use the default to avoid unbounded iteration
        if maxfun == np.inf:
            maxiter = N * 1000
        else:
            maxiter = np.inf
    elif maxfun is None:
        if maxiter == np.inf:
            maxfun = N * 1000
        else:
            maxfun = np.inf

    # we need to use a mutable object here that we can update in the
    # wrapper function
    fcalls, func = _wrap_scalar_function_maxfun_validation(func, args, maxfun)

    if direc is None:
        direc = eye(N, dtype=float)
    else:
        direc = asarray(direc, dtype=float)
        if np.linalg.matrix_rank(direc) != direc.shape[0]:
            warnings.warn("direc input is not full rank, some parameters may "
                          "not be optimized",
                          OptimizeWarning, stacklevel=3)

    if bounds is None:
        # don't make these arrays of all +/- inf. because
        # _linesearch_powell will do an unnecessary check of all the elements.
        # just keep them None, _linesearch_powell will not have to check
        # all the elements.
        lower_bound, upper_bound = None, None
    else:
        # bounds is standardized in _minimize.py.
        #lower_bound, upper_bound = bounds.lb, bounds.ub
        lower_bound, upper_bound = bounds["lb"], bounds["ub"]
        if np.any(lower_bound > x0) or np.any(x0 > upper_bound):
            warnings.warn("Initial guess is not within the specified bounds",
                          OptimizeWarning, stacklevel=3)

    fval = squeeze(func(x))
    x1 = x.copy()
    iter = 0
    while True:
        try:
            fx = fval
            bigind = 0
            delta = 0.0
            for i in range(N):
                direc1 = direc[i]
                fx2 = fval
                fval, x, direc1 = _linesearch_powell(func, x, direc1,
                                                     tol=xtol * 100,
                                                     lower_bound=lower_bound,
                                                     upper_bound=upper_bound,
                                                     fval=fval)
                if (fx2 - fval) > delta:
                    delta = fx2 - fval
                    bigind = i
            iter += 1
            if retall:
                allvecs.append(x)
            intermediate_result = OptimizeResult(x=x, fun=fval)
            if _call_callback_maybe_halt(callback, intermediate_result):
                break
            bnd = ftol * (np.abs(fx) + np.abs(fval)) + 1e-20
            if 2.0 * (fx - fval) <= bnd:
                break
            if fcalls[0] >= maxfun:
                break
            if iter >= maxiter:
                break
            if np.isnan(fx) and np.isnan(fval):
                # Ended up in a nan-region: bail out
                break

            # Construct the extrapolated point
            direc1 = x - x1
            x1 = x.copy()
            # make sure that we don't go outside the bounds when extrapolating
            if lower_bound is None and upper_bound is None:
                lmax = 1
            else:
                _, lmax = _line_for_search(x, direc1, lower_bound, upper_bound)
            x2 = x + min(lmax, 1) * direc1
            fx2 = squeeze(func(x2))

            if (fx > fx2):
                t = 2.0*(fx + fx2 - 2.0*fval)
                temp = (fx - fval - delta)
                t *= temp*temp
                temp = fx - fx2
                t -= delta*temp*temp
                if t < 0.0:
                    fval, x, direc1 = _linesearch_powell(
                        func, x, direc1,
                        tol=xtol * 100,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        fval=fval
                    )
                    if np.any(direc1):
                        direc[bigind] = direc[-1]
                        direc[-1] = direc1
        except _MaxFuncCallError:
            break

    warnflag = 0
    msg = _status_message['success']
    # out of bounds is more urgent than exceeding function evals or iters,
    # but I don't want to cause inconsistencies by changing the
    # established warning flags for maxfev and maxiter, so the out of bounds
    # warning flag becomes 3, but is checked for first.
    if bounds and (np.any(lower_bound > x) or np.any(x > upper_bound)):
        warnflag = 4
        msg = _status_message['out_of_bounds']
    elif fcalls[0] >= maxfun:
        warnflag = 1
        msg = _status_message['maxfev']
    elif iter >= maxiter:
        warnflag = 2
        msg = _status_message['maxiter']
    elif np.isnan(fval) or np.isnan(x).any():
        warnflag = 3
        msg = _status_message['nan']

    if disp:
        _print_success_message_or_warn(warnflag, msg, RuntimeWarning)
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % iter)
        print("         Function evaluations: %d" % fcalls[0])

    result = OptimizeResult(fun=fval, direc=direc, nit=iter, nfev=fcalls[0],
                            status=warnflag, success=(warnflag == 0),
                            message=msg, x=x)
    if retall:
        result['allvecs'] = allvecs
    return result


def _wrap_callback(callback, method=None):
    """Wrap a user-provided callback so that attributes can be attached."""
    if callback is None or method in {'tnc', 'slsqp', 'cobyla'}:
        return callback  # don't wrap

    sig = inspect.signature(callback)

    if set(sig.parameters) == {'intermediate_result'}:
        def wrapped_callback(res):
            return callback(intermediate_result=res)
    elif method == 'trust-constr':
        def wrapped_callback(res):
            return callback(np.copy(res.x), res)
    elif method == 'differential_evolution':
        def wrapped_callback(res):
            return callback(np.copy(res.x), res.convergence)
    else:
        def wrapped_callback(res):
            return callback(np.copy(res.x))

    wrapped_callback.stop_iteration = False
    return wrapped_callback

def _check_unknown_options(unknown_options):
    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        # Stack level 4: this is called from _minimize_*, which is
        # called from another function in SciPy. Level 4 is the first
        # level in user code.
        warnings.warn("Unknown solver options: %s" % msg, OptimizeWarning, stacklevel=4)


def _wrap_scalar_function_maxfun_validation(function, args, maxfun):
    # wraps a minimizer function to count number of evaluations
    # and to easily provide an args kwd.
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(x, *wrapper_args):
        if ncalls[0] >= maxfun:
            raise _MaxFuncCallError("Too many function calls")
        ncalls[0] += 1
        # A copy of x is sent to the user function (gh13740)
        fx = function(np.copy(x), *(wrapper_args + args))
        # Ideally, we'd like to a have a true scalar returned from f(x). For
        # backwards-compatibility, also allow np.array([1.3]),
        # np.array([[1.3]]) etc.
        if not np.isscalar(fx):
            try:
                fx = np.asarray(fx).item()
            except (TypeError, ValueError) as e:
                raise ValueError("The user-provided objective function "
                                 "must return a scalar value.") from e
        return fx

    return ncalls, function_wrapper


class _MaxFuncCallError(RuntimeError):
    pass

def _linesearch_powell(func, p, xi, tol=1e-3,
                       lower_bound=None, upper_bound=None, fval=None):
    """Line-search algorithm using fminbound.

    Find the minimum of the function ``func(x0 + alpha*direc)``.

    lower_bound : np.array.
        The lower bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded below, then ``lower_bound[i]``
        should be ``-np.inf``.
        Note ``np.shape(lower_bound) == (n,)``.
    upper_bound : np.array.
        The upper bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded above, then ``upper_bound[i]``
        should be ``np.inf``.
        Note ``np.shape(upper_bound) == (n,)``.
    fval : number.
        ``fval`` is equal to ``func(p)``, the idea is just to avoid
        recomputing it so we can limit the ``fevals``.

    """
    def myfunc(alpha):
        return func(p + alpha*xi)

    # if xi is zero, then don't optimize
    if not np.any(xi):
        return ((fval, p, xi) if fval is not None else (func(p), p, xi))
    elif lower_bound is None and upper_bound is None:
        # non-bounded minimization
        res = _recover_from_bracket_error(_minimize_scalar_brent,
                                          myfunc, None, tuple(), xtol=tol)
        alpha_min, fret = res.x, res.fun
        xi = alpha_min * xi
        return squeeze(fret), p + xi, xi
    else:
        bound = _line_for_search(p, xi, lower_bound, upper_bound)
        if np.isneginf(bound[0]) and np.isposinf(bound[1]):
            # equivalent to unbounded
            return _linesearch_powell(func, p, xi, fval=fval, tol=tol)
        elif not np.isneginf(bound[0]) and not np.isposinf(bound[1]):
            # we can use a bounded scalar minimization
            res = _minimize_scalar_bounded(myfunc, bound, xatol=tol / 100)
            xi = res.x * xi
            return squeeze(res.fun), p + xi, xi
        else:
            # only bounded on one side. use the tangent function to convert
            # the infinity bound to a finite bound. The new bounded region
            # is a subregion of the region bounded by -np.pi/2 and np.pi/2.
            bound = np.arctan(bound[0]), np.arctan(bound[1])
            res = _minimize_scalar_bounded(
                lambda x: myfunc(np.tan(x)),
                bound,
                xatol=tol / 100)
            xi = np.tan(res.x) * xi
            return squeeze(res.fun), p + xi, xi

def _line_for_search(x0, alpha, lower_bound, upper_bound):
    """
    Given a parameter vector ``x0`` with length ``n`` and a direction
    vector ``alpha`` with length ``n``, and lower and upper bounds on
    each of the ``n`` parameters, what are the bounds on a scalar
    ``l`` such that ``lower_bound <= x0 + alpha * l <= upper_bound``.


    Parameters
    ----------
    x0 : np.array.
        The vector representing the current location.
        Note ``np.shape(x0) == (n,)``.
    alpha : np.array.
        The vector representing the direction.
        Note ``np.shape(alpha) == (n,)``.
    lower_bound : np.array.
        The lower bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded below, then ``lower_bound[i]``
        should be ``-np.inf``.
        Note ``np.shape(lower_bound) == (n,)``.
    upper_bound : np.array.
        The upper bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded above, then ``upper_bound[i]``
        should be ``np.inf``.
        Note ``np.shape(upper_bound) == (n,)``.

    Returns
    -------
    res : tuple ``(lmin, lmax)``
        The bounds for ``l`` such that
            ``lower_bound[i] <= x0[i] + alpha[i] * l <= upper_bound[i]``
        for all ``i``.

    """
    # get nonzero indices of alpha so we don't get any zero division errors.
    # alpha will not be all zero, since it is called from _linesearch_powell
    # where we have a check for this.
    nonzero, = alpha.nonzero()
    lower_bound, upper_bound = lower_bound[nonzero], upper_bound[nonzero]
    x0, alpha = x0[nonzero], alpha[nonzero]
    low = (lower_bound - x0) / alpha
    high = (upper_bound - x0) / alpha

    # positive and negative indices
    pos = alpha > 0

    lmin_pos = np.where(pos, low, 0)
    lmin_neg = np.where(pos, 0, high)
    lmax_pos = np.where(pos, high, 0)
    lmax_neg = np.where(pos, 0, low)

    lmin = np.max(lmin_pos + lmin_neg)
    lmax = np.min(lmax_pos + lmax_neg)

    # if x0 is outside the bounds, then it is possible that there is
    # no way to get back in the bounds for the parameters being updated
    # with the current direction alpha.
    # when this happens, lmax < lmin.
    # If this is the case, then we can just return (0, 0)
    return (lmin, lmax) if lmax >= lmin else (0, 0)


def _linesearch_powell(func, p, xi, tol=1e-3,
                       lower_bound=None, upper_bound=None, fval=None):
    """Line-search algorithm using fminbound.

    Find the minimum of the function ``func(x0 + alpha*direc)``.

    lower_bound : np.array.
        The lower bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded below, then ``lower_bound[i]``
        should be ``-np.inf``.
        Note ``np.shape(lower_bound) == (n,)``.
    upper_bound : np.array.
        The upper bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded above, then ``upper_bound[i]``
        should be ``np.inf``.
        Note ``np.shape(upper_bound) == (n,)``.
    fval : number.
        ``fval`` is equal to ``func(p)``, the idea is just to avoid
        recomputing it so we can limit the ``fevals``.

    """
    def myfunc(alpha):
        return func(p + alpha*xi)

    # if xi is zero, then don't optimize
    if not np.any(xi):
        return ((fval, p, xi) if fval is not None else (func(p), p, xi))
    elif lower_bound is None and upper_bound is None:
        # non-bounded minimization
        res = _recover_from_bracket_error(_minimize_scalar_brent,
                                          myfunc, None, tuple(), xtol=tol)
        alpha_min, fret = res.x, res.fun
        xi = alpha_min * xi
        return squeeze(fret), p + xi, xi
    else:
        bound = _line_for_search(p, xi, lower_bound, upper_bound)
        if np.isneginf(bound[0]) and np.isposinf(bound[1]):
            # equivalent to unbounded
            return _linesearch_powell(func, p, xi, fval=fval, tol=tol)
        elif not np.isneginf(bound[0]) and not np.isposinf(bound[1]):
            # we can use a bounded scalar minimization
            res = _minimize_scalar_bounded(myfunc, bound, xatol=tol / 100)
            xi = res.x * xi
            return squeeze(res.fun), p + xi, xi
        else:
            # only bounded on one side. use the tangent function to convert
            # the infinity bound to a finite bound. The new bounded region
            # is a subregion of the region bounded by -np.pi/2 and np.pi/2.
            bound = np.arctan(bound[0]), np.arctan(bound[1])
            res = _minimize_scalar_bounded(
                lambda x: myfunc(np.tan(x)),
                bound,
                xatol=tol / 100)
            xi = np.tan(res.x) * xi
            return squeeze(res.fun), p + xi, xi

def _line_for_search(x0, alpha, lower_bound, upper_bound):
    """
    Given a parameter vector ``x0`` with length ``n`` and a direction
    vector ``alpha`` with length ``n``, and lower and upper bounds on
    each of the ``n`` parameters, what are the bounds on a scalar
    ``l`` such that ``lower_bound <= x0 + alpha * l <= upper_bound``.


    Parameters
    ----------
    x0 : np.array.
        The vector representing the current location.
        Note ``np.shape(x0) == (n,)``.
    alpha : np.array.
        The vector representing the direction.
        Note ``np.shape(alpha) == (n,)``.
    lower_bound : np.array.
        The lower bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded below, then ``lower_bound[i]``
        should be ``-np.inf``.
        Note ``np.shape(lower_bound) == (n,)``.
    upper_bound : np.array.
        The upper bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded above, then ``upper_bound[i]``
        should be ``np.inf``.
        Note ``np.shape(upper_bound) == (n,)``.

    Returns
    -------
    res : tuple ``(lmin, lmax)``
        The bounds for ``l`` such that
            ``lower_bound[i] <= x0[i] + alpha[i] * l <= upper_bound[i]``
        for all ``i``.

    """
    # get nonzero indices of alpha so we don't get any zero division errors.
    # alpha will not be all zero, since it is called from _linesearch_powell
    # where we have a check for this.
    nonzero, = alpha.nonzero()
    lower_bound, upper_bound = lower_bound[nonzero], upper_bound[nonzero]
    x0, alpha = x0[nonzero], alpha[nonzero]
    low = (lower_bound - x0) / alpha
    high = (upper_bound - x0) / alpha

    # positive and negative indices
    pos = alpha > 0

    lmin_pos = np.where(pos, low, 0)
    lmin_neg = np.where(pos, 0, high)
    lmax_pos = np.where(pos, high, 0)
    lmax_neg = np.where(pos, 0, low)

    lmin = np.max(lmin_pos + lmin_neg)
    lmax = np.min(lmax_pos + lmax_neg)

    # if x0 is outside the bounds, then it is possible that there is
    # no way to get back in the bounds for the parameters being updated
    # with the current direction alpha.
    # when this happens, lmax < lmin.
    # If this is the case, then we can just return (0, 0)
    return (lmin, lmax) if lmax >= lmin else (0, 0)


def _linesearch_powell(func, p, xi, tol=1e-3,
                       lower_bound=None, upper_bound=None, fval=None):
    """Line-search algorithm using fminbound.

    Find the minimum of the function ``func(x0 + alpha*direc)``.

    lower_bound : np.array.
        The lower bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded below, then ``lower_bound[i]``
        should be ``-np.inf``.
        Note ``np.shape(lower_bound) == (n,)``.
    upper_bound : np.array.
        The upper bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded above, then ``upper_bound[i]``
        should be ``np.inf``.
        Note ``np.shape(upper_bound) == (n,)``.
    fval : number.
        ``fval`` is equal to ``func(p)``, the idea is just to avoid
        recomputing it so we can limit the ``fevals``.

    """
    def myfunc(alpha):
        return func(p + alpha*xi)

    # if xi is zero, then don't optimize
    if not np.any(xi):
        return ((fval, p, xi) if fval is not None else (func(p), p, xi))
    elif lower_bound is None and upper_bound is None:
        # non-bounded minimization
        res = _recover_from_bracket_error(_minimize_scalar_brent,
                                          myfunc, None, tuple(), xtol=tol)
        alpha_min, fret = res.x, res.fun
        xi = alpha_min * xi
        return squeeze(fret), p + xi, xi
    else:
        bound = _line_for_search(p, xi, lower_bound, upper_bound)
        if np.isneginf(bound[0]) and np.isposinf(bound[1]):
            # equivalent to unbounded
            return _linesearch_powell(func, p, xi, fval=fval, tol=tol)
        elif not np.isneginf(bound[0]) and not np.isposinf(bound[1]):
            # we can use a bounded scalar minimization
            res = _minimize_scalar_bounded(myfunc, bound, xatol=tol / 100)
            xi = res.x * xi
            return squeeze(res.fun), p + xi, xi
        else:
            # only bounded on one side. use the tangent function to convert
            # the infinity bound to a finite bound. The new bounded region
            # is a subregion of the region bounded by -np.pi/2 and np.pi/2.
            bound = np.arctan(bound[0]), np.arctan(bound[1])
            res = _minimize_scalar_bounded(
                lambda x: myfunc(np.tan(x)),
                bound,
                xatol=tol / 100)
            xi = np.tan(res.x) * xi
            return squeeze(res.fun), p + xi, xi
def _line_for_search(x0, alpha, lower_bound, upper_bound):
    """
    Given a parameter vector ``x0`` with length ``n`` and a direction
    vector ``alpha`` with length ``n``, and lower and upper bounds on
    each of the ``n`` parameters, what are the bounds on a scalar
    ``l`` such that ``lower_bound <= x0 + alpha * l <= upper_bound``.


    Parameters
    ----------
    x0 : np.array.
        The vector representing the current location.
        Note ``np.shape(x0) == (n,)``.
    alpha : np.array.
        The vector representing the direction.
        Note ``np.shape(alpha) == (n,)``.
    lower_bound : np.array.
        The lower bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded below, then ``lower_bound[i]``
        should be ``-np.inf``.
        Note ``np.shape(lower_bound) == (n,)``.
    upper_bound : np.array.
        The upper bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded above, then ``upper_bound[i]``
        should be ``np.inf``.
        Note ``np.shape(upper_bound) == (n,)``.

    Returns
    -------
    res : tuple ``(lmin, lmax)``
        The bounds for ``l`` such that
            ``lower_bound[i] <= x0[i] + alpha[i] * l <= upper_bound[i]``
        for all ``i``.

    """
    # get nonzero indices of alpha so we don't get any zero division errors.
    # alpha will not be all zero, since it is called from _linesearch_powell
    # where we have a check for this.
    nonzero, = alpha.nonzero()
    lower_bound, upper_bound = lower_bound[nonzero], upper_bound[nonzero]
    x0, alpha = x0[nonzero], alpha[nonzero]
    low = (lower_bound - x0) / alpha
    high = (upper_bound - x0) / alpha

    # positive and negative indices
    pos = alpha > 0

    lmin_pos = np.where(pos, low, 0)
    lmin_neg = np.where(pos, 0, high)
    lmax_pos = np.where(pos, high, 0)
    lmax_neg = np.where(pos, 0, low)

    lmin = np.max(lmin_pos + lmin_neg)
    lmax = np.min(lmax_pos + lmax_neg)

    # if x0 is outside the bounds, then it is possible that there is
    # no way to get back in the bounds for the parameters being updated
    # with the current direction alpha.
    # when this happens, lmax < lmin.
    # If this is the case, then we can just return (0, 0)
    return (lmin, lmax) if lmax >= lmin else (0, 0)


def _linesearch_powell(func, p, xi, tol=1e-3,
                       lower_bound=None, upper_bound=None, fval=None):
    """Line-search algorithm using fminbound.

    Find the minimum of the function ``func(x0 + alpha*direc)``.

    lower_bound : np.array.
        The lower bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded below, then ``lower_bound[i]``
        should be ``-np.inf``.
        Note ``np.shape(lower_bound) == (n,)``.
    upper_bound : np.array.
        The upper bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded above, then ``upper_bound[i]``
        should be ``np.inf``.
        Note ``np.shape(upper_bound) == (n,)``.
    fval : number.
        ``fval`` is equal to ``func(p)``, the idea is just to avoid
        recomputing it so we can limit the ``fevals``.

    """
    def myfunc(alpha):
        return func(p + alpha*xi)

    # if xi is zero, then don't optimize
    if not np.any(xi):
        return ((fval, p, xi) if fval is not None else (func(p), p, xi))
    elif lower_bound is None and upper_bound is None:
        # non-bounded minimization
        res = _recover_from_bracket_error(_minimize_scalar_brent,
                                          myfunc, None, tuple(), xtol=tol)
        alpha_min, fret = res.x, res.fun
        xi = alpha_min * xi
        return squeeze(fret), p + xi, xi
    else:
        bound = _line_for_search(p, xi, lower_bound, upper_bound)
        if np.isneginf(bound[0]) and np.isposinf(bound[1]):
            # equivalent to unbounded
            return _linesearch_powell(func, p, xi, fval=fval, tol=tol)
        elif not np.isneginf(bound[0]) and not np.isposinf(bound[1]):
            # we can use a bounded scalar minimization
            res = _minimize_scalar_bounded(myfunc, bound, xatol=tol / 100)
            xi = res.x * xi
            return squeeze(res.fun), p + xi, xi
        else:
            # only bounded on one side. use the tangent function to convert
            # the infinity bound to a finite bound. The new bounded region
            # is a subregion of the region bounded by -np.pi/2 and np.pi/2.
            bound = np.arctan(bound[0]), np.arctan(bound[1])
            res = _minimize_scalar_bounded(
                lambda x: myfunc(np.tan(x)),
                bound,
                xatol=tol / 100)
            xi = np.tan(res.x) * xi
            return squeeze(res.fun), p + xi, xi



def _line_for_search(x0, alpha, lower_bound, upper_bound):
    """
    Given a parameter vector ``x0`` with length ``n`` and a direction
    vector ``alpha`` with length ``n``, and lower and upper bounds on
    each of the ``n`` parameters, what are the bounds on a scalar
    ``l`` such that ``lower_bound <= x0 + alpha * l <= upper_bound``.


    Parameters
    ----------
    x0 : np.array.
        The vector representing the current location.
        Note ``np.shape(x0) == (n,)``.
    alpha : np.array.
        The vector representing the direction.
        Note ``np.shape(alpha) == (n,)``.
    lower_bound : np.array.
        The lower bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded below, then ``lower_bound[i]``
        should be ``-np.inf``.
        Note ``np.shape(lower_bound) == (n,)``.
    upper_bound : np.array.
        The upper bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded above, then ``upper_bound[i]``
        should be ``np.inf``.
        Note ``np.shape(upper_bound) == (n,)``.

    Returns
    -------
    res : tuple ``(lmin, lmax)``
        The bounds for ``l`` such that
            ``lower_bound[i] <= x0[i] + alpha[i] * l <= upper_bound[i]``
        for all ``i``.

    """
    # get nonzero indices of alpha so we don't get any zero division errors.
    # alpha will not be all zero, since it is called from _linesearch_powell
    # where we have a check for this.
    nonzero, = alpha.nonzero()
    lower_bound, upper_bound = lower_bound[nonzero], upper_bound[nonzero]
    x0, alpha = x0[nonzero], alpha[nonzero]
    low = (lower_bound - x0) / alpha
    high = (upper_bound - x0) / alpha

    # positive and negative indices
    pos = alpha > 0

    lmin_pos = np.where(pos, low, 0)
    lmin_neg = np.where(pos, 0, high)
    lmax_pos = np.where(pos, high, 0)
    lmax_neg = np.where(pos, 0, low)

    lmin = np.max(lmin_pos + lmin_neg)
    lmax = np.min(lmax_pos + lmax_neg)

    # if x0 is outside the bounds, then it is possible that there is
    # no way to get back in the bounds for the parameters being updated
    # with the current direction alpha.
    # when this happens, lmax < lmin.
    # If this is the case, then we can just return (0, 0)
    return (lmin, lmax) if lmax >= lmin else (0, 0)


def _linesearch_powell(func, p, xi, tol=1e-3,
                       lower_bound=None, upper_bound=None, fval=None):
    """Line-search algorithm using fminbound.

    Find the minimum of the function ``func(x0 + alpha*direc)``.

    lower_bound : np.array.
        The lower bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded below, then ``lower_bound[i]``
        should be ``-np.inf``.
        Note ``np.shape(lower_bound) == (n,)``.
    upper_bound : np.array.
        The upper bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded above, then ``upper_bound[i]``
        should be ``np.inf``.
        Note ``np.shape(upper_bound) == (n,)``.
    fval : number.
        ``fval`` is equal to ``func(p)``, the idea is just to avoid
        recomputing it so we can limit the ``fevals``.

    """
    def myfunc(alpha):
        return func(p + alpha*xi)

    # if xi is zero, then don't optimize
    if not np.any(xi):
        return ((fval, p, xi) if fval is not None else (func(p), p, xi))
    elif lower_bound is None and upper_bound is None:
        # non-bounded minimization
        res = _recover_from_bracket_error(_minimize_scalar_brent,
                                          myfunc, None, tuple(), xtol=tol)
        alpha_min, fret = res.x, res.fun
        xi = alpha_min * xi
        return squeeze(fret), p + xi, xi
    else:
        bound = _line_for_search(p, xi, lower_bound, upper_bound)
        if np.isneginf(bound[0]) and np.isposinf(bound[1]):
            # equivalent to unbounded
            return _linesearch_powell(func, p, xi, fval=fval, tol=tol)
        elif not np.isneginf(bound[0]) and not np.isposinf(bound[1]):
            # we can use a bounded scalar minimization
            res = _minimize_scalar_bounded(myfunc, bound, xatol=tol / 100)
            xi = res.x * xi
            return squeeze(res.fun), p + xi, xi
        else:
            # only bounded on one side. use the tangent function to convert
            # the infinity bound to a finite bound. The new bounded region
            # is a subregion of the region bounded by -np.pi/2 and np.pi/2.
            bound = np.arctan(bound[0]), np.arctan(bound[1])
            res = _minimize_scalar_bounded(
                lambda x: myfunc(np.tan(x)),
                bound,
                xatol=tol / 100)
            xi = np.tan(res.x) * xi
            return squeeze(res.fun), p + xi, xi


def _line_for_search(x0, alpha, lower_bound, upper_bound):
    """
    Given a parameter vector ``x0`` with length ``n`` and a direction
    vector ``alpha`` with length ``n``, and lower and upper bounds on
    each of the ``n`` parameters, what are the bounds on a scalar
    ``l`` such that ``lower_bound <= x0 + alpha * l <= upper_bound``.


    Parameters
    ----------
    x0 : np.array.
        The vector representing the current location.
        Note ``np.shape(x0) == (n,)``.
    alpha : np.array.
        The vector representing the direction.
        Note ``np.shape(alpha) == (n,)``.
    lower_bound : np.array.
        The lower bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded below, then ``lower_bound[i]``
        should be ``-np.inf``.
        Note ``np.shape(lower_bound) == (n,)``.
    upper_bound : np.array.
        The upper bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded above, then ``upper_bound[i]``
        should be ``np.inf``.
        Note ``np.shape(upper_bound) == (n,)``.

    Returns
    -------
    res : tuple ``(lmin, lmax)``
        The bounds for ``l`` such that
            ``lower_bound[i] <= x0[i] + alpha[i] * l <= upper_bound[i]``
        for all ``i``.

    """
    # get nonzero indices of alpha so we don't get any zero division errors.
    # alpha will not be all zero, since it is called from _linesearch_powell
    # where we have a check for this.
    nonzero, = alpha.nonzero()
    lower_bound, upper_bound = lower_bound[nonzero], upper_bound[nonzero]
    x0, alpha = x0[nonzero], alpha[nonzero]
    low = (lower_bound - x0) / alpha
    high = (upper_bound - x0) / alpha

    # positive and negative indices
    pos = alpha > 0

    lmin_pos = np.where(pos, low, 0)
    lmin_neg = np.where(pos, 0, high)
    lmax_pos = np.where(pos, high, 0)
    lmax_neg = np.where(pos, 0, low)

    lmin = np.max(lmin_pos + lmin_neg)
    lmax = np.min(lmax_pos + lmax_neg)

    # if x0 is outside the bounds, then it is possible that there is
    # no way to get back in the bounds for the parameters being updated
    # with the current direction alpha.
    # when this happens, lmax < lmin.
    # If this is the case, then we can just return (0, 0)
    return (lmin, lmax) if lmax >= lmin else (0, 0)


def _linesearch_powell(func, p, xi, tol=1e-3,
                       lower_bound=None, upper_bound=None, fval=None):
    """Line-search algorithm using fminbound.

    Find the minimum of the function ``func(x0 + alpha*direc)``.

    lower_bound : np.array.
        The lower bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded below, then ``lower_bound[i]``
        should be ``-np.inf``.
        Note ``np.shape(lower_bound) == (n,)``.
    upper_bound : np.array.
        The upper bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded above, then ``upper_bound[i]``
        should be ``np.inf``.
        Note ``np.shape(upper_bound) == (n,)``.
    fval : number.
        ``fval`` is equal to ``func(p)``, the idea is just to avoid
        recomputing it so we can limit the ``fevals``.

    """
    def myfunc(alpha):
        return func(p + alpha*xi)

    # if xi is zero, then don't optimize
    if not np.any(xi):
        return ((fval, p, xi) if fval is not None else (func(p), p, xi))
    elif lower_bound is None and upper_bound is None:
        # non-bounded minimization
        res = _recover_from_bracket_error(_minimize_scalar_brent,
                                          myfunc, None, tuple(), xtol=tol)
        alpha_min, fret = res.x, res.fun
        xi = alpha_min * xi
        return squeeze(fret), p + xi, xi
    else:
        bound = _line_for_search(p, xi, lower_bound, upper_bound)
        if np.isneginf(bound[0]) and np.isposinf(bound[1]):
            # equivalent to unbounded
            return _linesearch_powell(func, p, xi, fval=fval, tol=tol)
        elif not np.isneginf(bound[0]) and not np.isposinf(bound[1]):
            # we can use a bounded scalar minimization
            res = _minimize_scalar_bounded(myfunc, bound, xatol=tol / 100)
            xi = res.x * xi
            return squeeze(res.fun), p + xi, xi
        else:
            # only bounded on one side. use the tangent function to convert
            # the infinity bound to a finite bound. The new bounded region
            # is a subregion of the region bounded by -np.pi/2 and np.pi/2.
            bound = np.arctan(bound[0]), np.arctan(bound[1])
            res = _minimize_scalar_bounded(
                lambda x: myfunc(np.tan(x)),
                bound,
                xatol=tol / 100)
            xi = np.tan(res.x) * xi
            return squeeze(res.fun), p + xi, xi



def _line_for_search(x0, alpha, lower_bound, upper_bound):
    """
    Given a parameter vector ``x0`` with length ``n`` and a direction
    vector ``alpha`` with length ``n``, and lower and upper bounds on
    each of the ``n`` parameters, what are the bounds on a scalar
    ``l`` such that ``lower_bound <= x0 + alpha * l <= upper_bound``.


    Parameters
    ----------
    x0 : np.array.
        The vector representing the current location.
        Note ``np.shape(x0) == (n,)``.
    alpha : np.array.
        The vector representing the direction.
        Note ``np.shape(alpha) == (n,)``.
    lower_bound : np.array.
        The lower bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded below, then ``lower_bound[i]``
        should be ``-np.inf``.
        Note ``np.shape(lower_bound) == (n,)``.
    upper_bound : np.array.
        The upper bounds for each parameter in ``x0``. If the ``i``th
        parameter in ``x0`` is unbounded above, then ``upper_bound[i]``
        should be ``np.inf``.
        Note ``np.shape(upper_bound) == (n,)``.

    Returns
    -------
    res : tuple ``(lmin, lmax)``
        The bounds for ``l`` such that
            ``lower_bound[i] <= x0[i] + alpha[i] * l <= upper_bound[i]``
        for all ``i``.

    """
    # get nonzero indices of alpha so we don't get any zero division errors.
    # alpha will not be all zero, since it is called from _linesearch_powell
    # where we have a check for this.
    nonzero, = alpha.nonzero()
    lower_bound, upper_bound = lower_bound[nonzero], upper_bound[nonzero]
    x0, alpha = x0[nonzero], alpha[nonzero]
    low = (lower_bound - x0) / alpha
    high = (upper_bound - x0) / alpha

    # positive and negative indices
    pos = alpha > 0

    lmin_pos = np.where(pos, low, 0)
    lmin_neg = np.where(pos, 0, high)
    lmax_pos = np.where(pos, high, 0)
    lmax_neg = np.where(pos, 0, low)

    lmin = np.max(lmin_pos + lmin_neg)
    lmax = np.min(lmax_pos + lmax_neg)

    # if x0 is outside the bounds, then it is possible that there is
    # no way to get back in the bounds for the parameters being updated
    # with the current direction alpha.
    # when this happens, lmax < lmin.
    # If this is the case, then we can just return (0, 0)
    return (lmin, lmax) if lmax >= lmin else (0, 0)


def _minimize_scalar_bounded(func, bounds, args=(),
                             xatol=1e-5, maxiter=500, disp=0,
                             **unknown_options):
    """
    Options
    -------
    maxiter : int
        Maximum number of iterations to perform.
    disp: int, optional
        If non-zero, print messages.
            0 : no message printing.
            1 : non-convergence notification messages only.
            2 : print a message on convergence too.
            3 : print iteration results.
    xatol : float
        Absolute error in solution `xopt` acceptable for convergence.

    """
    _check_unknown_options(unknown_options)
    maxfun = maxiter
    # Test bounds are of correct form
    if len(bounds) != 2:
        raise ValueError('bounds must have two elements.')
    x1, x2 = bounds

    if not (is_finite_scalar(x1) and is_finite_scalar(x2)):
        raise ValueError("Optimization bounds must be finite scalars.")

    if x1 > x2:
        raise ValueError("The lower bound exceeds the upper bound.")

    flag = 0
    header = ' Func-count     x          f(x)          Procedure'
    step = '       initial'

    sqrt_eps = sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - sqrt(5.0))
    a, b = x1, x2
    fulc = a + golden_mean * (b - a)
    nfc, xf = fulc, fulc
    rat = e = 0.0
    x = xf
    fx = func(x, *args)
    num = 1
    fmin_data = (1, xf, fx)
    fu = np.inf

    ffulc = fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1

    if disp > 2:
        print(" ")
        print(header)
        print("%5.0f   %12.6g %12.6g %s" % (fmin_data + (step,)))

    while (np.abs(xf - xm) > (tol2 - 0.5 * (b - a))):
        golden = 1
        # Check for parabolic fit
        if np.abs(e) > tol1:
            golden = 0
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = np.abs(q)
            r = e
            e = rat

            # Check for acceptability of parabola
            if ((np.abs(p) < np.abs(0.5*q*r)) and (p > q*(a - xf)) and
                    (p < q * (b - xf))):
                rat = (p + 0.0) / q
                x = xf + rat
                step = '       parabolic'

                if ((x - a) < tol2) or ((b - x) < tol2):
                    si = np.sign(xm - xf) + ((xm - xf) == 0)
                    rat = tol1 * si
            else:      # do a golden-section step
                golden = 1

        if golden:  # do a golden-section step
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean*e
            step = '       golden'

        si = np.sign(rat) + (rat == 0)
        x = xf + si * np.maximum(np.abs(rat), tol1)
        fu = func(x, *args)
        num += 1
        fmin_data = (num, x, fu)
        if disp > 2:
            print("%5.0f   %12.6g %12.6g %s" % (fmin_data + (step,)))

        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = x, fu

        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1

        if num >= maxfun:
            flag = 1
            break

    if np.isnan(xf) or np.isnan(fx) or np.isnan(fu):
        flag = 2

    fval = fx
    if disp > 0:
        _endprint(x, flag, fval, maxfun, xatol, disp)

    result = OptimizeResult(fun=fval, status=flag, success=(flag == 0),
                            message={0: 'Solution found.',
                                     1: 'Maximum number of function calls '
                                        'reached.',
                                     2: _status_message['nan']}.get(flag, ''),
                            x=xf, nfev=num, nit=num)

    return result

def is_finite_scalar(x):
    """Test whether `x` is either a finite scalar or a finite array scalar.

    """
    return np.size(x) == 1 and np.isfinite(x)



class _RichResult(dict):
    """ Container for multiple outputs with pretty-printing """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        order_keys = ['message', 'success', 'status', 'fun', 'funl', 'x', 'xl',
                      'col_ind', 'nit', 'lower', 'upper', 'eqlin', 'ineqlin',
                      'converged', 'flag', 'function_calls', 'iterations',
                      'root']
        order_keys = getattr(self, '_order_keys', order_keys)
        # 'slack', 'con' are redundant with residuals
        # 'crossover_nit' is probably not interesting to most users
        omit_keys = {'slack', 'con', 'crossover_nit', '_order_keys'}

        def key(item):
            try:
                return order_keys.index(item[0].lower())
            except ValueError:  # item not in list
                return np.inf

        def omit_redundant(items):
            for item in items:
                if item[0] in omit_keys:
                    continue
                yield item

        def item_sorter(d):
            return sorted(omit_redundant(d.items()), key=key)

        if self.keys():
            return _dict_formatter(self, sorter=item_sorter)
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


class OptimizeResult(_RichResult):
    """
    Represents the optimization result.

    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    fun, jac, hess: ndarray
        Values of objective function, its Jacobian and its Hessian (if
        available). The Hessians may be approximations, see the documentation
        of the function in question.
    hess_inv : object
        Inverse of the objective function's Hessian; may be an approximation.
        Not available for all solvers. The type of this attribute may be
        either np.ndarray or scipy.sparse.linalg.LinearOperator.
    nfev, njev, nhev : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    nit : int
        Number of iterations performed by the optimizer.
    maxcv : float
        The maximum constraint violation.

    Notes
    -----
    Depending on the specific solver being used, `OptimizeResult` may
    not have all attributes listed here, and they may have additional
    attributes not listed here. Since this class is essentially a
    subclass of dict with attribute accessors, one can see which
    attributes are available using the `OptimizeResult.keys` method.

    """
    pass

def _print_success_message_or_warn(warnflag, message, warntype=None):
    if not warnflag:
        print(message)
    else:
        warnings.warn(message, warntype or OptimizeWarning, stacklevel=3)
