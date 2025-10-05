"""Numerical solvers for finding steady-state solutions of ODE systems.

This module provides various numerical methods for solving nonlinear equations
of the form F(x) = 0, where F(x) represents the derivative function at steady-state
(i.e., dx/dt = 0).
"""

import typing
import numpy as np
import logging
from scipy.optimize import fsolve, root_scalar, root, least_squares
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class SteadyStateSolver(ABC):
    """Abstract base class for steady-state solvers."""

    @abstractmethod
    def solve(
        self,
        deriv_func: typing.Callable[[float, list[float], dict], tuple[float, ...]],
        initial_guess: list[float],
        kinetic_params: dict,
        **kwargs
    ) -> tuple[np.ndarray, bool]:
        """Find steady-state solution.

        Args:
            deriv_func: Derivative function that returns dx/dt
            initial_guess: Initial guess for species concentrations
            kinetic_params: Kinetic parameters for the model
            **kwargs: Additional solver-specific parameters

        Returns:
            Tuple of (solution, success_flag)
        """
        pass


class NewtonRaphsonSolver(SteadyStateSolver):
    """Newton-Raphson method with analytical or numerical Jacobian."""

    def __init__(
        self,
        max_iter: int = 100,
        tolerance: float = 1e-8,
        step_size: float = 1e-8,
        use_numerical_jacobian: bool = True
    ):
        """Initialize Newton-Raphson solver.

        Args:
            max_iter: Maximum number of iterations
            tolerance: Convergence tolerance for ||F(x)||
            step_size: Step size for numerical Jacobian computation
            use_numerical_jacobian: Whether to use numerical differentiation
        """
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.step_size = step_size
        self.use_numerical_jacobian = use_numerical_jacobian

    def _compute_numerical_jacobian(
        self,
        func: typing.Callable[[np.ndarray], np.ndarray],
        x: np.ndarray
    ) -> np.ndarray:
        """Compute Jacobian matrix using finite differences."""
        n = len(x)
        jacobian = np.zeros((n, n))
        f0 = func(x)

        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += self.step_size
            f_plus = func(x_plus)
            jacobian[:, i] = (f_plus - f0) / self.step_size

        return jacobian

    def solve(
        self,
        deriv_func: typing.Callable[[float, list[float], dict], tuple[float, ...]],
        initial_guess: list[float],
        kinetic_params: dict,
        **kwargs
    ) -> tuple[np.ndarray, bool]:
        """Solve using Newton-Raphson method."""

        def f(x):
            """Wrapper function that returns residuals."""
            return np.array(deriv_func(0.0, x.tolist(), kinetic_params))

        x = np.array(initial_guess)

        for iteration in range(self.max_iter):
            fx = f(x)
            norm_fx = np.linalg.norm(fx)

            if norm_fx < self.tolerance:
                logger.info(f"Newton-Raphson converged in {iteration} iterations")
                return x, True

            if self.use_numerical_jacobian:
                jacobian = self._compute_numerical_jacobian(f, x)
            else:
                raise NotImplementedError("Analytical Jacobian not implemented")

            try:
                delta_x = np.linalg.solve(jacobian, -fx)
                x = x + delta_x
            except np.linalg.LinAlgError:
                logger.warning("Singular Jacobian matrix encountered")
                return x, False

        logger.warning(f"Newton-Raphson failed to converge in {self.max_iter} iterations")
        return x, False


class ScipyRootSolver(SteadyStateSolver):
    """Wrapper for scipy.optimize root-finding methods."""

    def __init__(
        self,
        method: str = 'hybr',
        tolerance: float = 1e-8,
        max_iter: int = 1000
    ):
        """Initialize scipy root solver.

        Args:
            method: Scipy optimization method ('hybr', 'lm', 'broyden1', etc.)
            tolerance: Convergence tolerance
            max_iter: Maximum number of iterations
        """
        self.method = method
        self.tolerance = tolerance
        self.max_iter = max_iter

    def solve(
        self,
        deriv_func: typing.Callable[[float, list[float], dict], tuple[float, ...]],
        initial_guess: list[float],
        kinetic_params: dict,
        **kwargs
    ) -> tuple[np.ndarray, bool]:
        """Solve using scipy root-finding methods."""

        def f(x):
            """Wrapper function that returns residuals."""
            return np.array(deriv_func(0.0, x.tolist(), kinetic_params))

        try:
            result = root(
                f,
                initial_guess,
                method=self.method,
                options={
                    'xtol': self.tolerance,
                    'maxiter': self.max_iter
                }
            )

            if result.success:
                logger.info(f"Scipy {self.method} solver converged")
                return result.x, True
            else:
                logger.warning(f"Scipy {self.method} solver failed: {result.message}")
                return result.x, False

        except Exception as e:
            logger.error(f"Scipy solver error: {str(e)}")
            return np.array(initial_guess), False


class FsolveSolver(SteadyStateSolver):
    """Wrapper for scipy.optimize.fsolve."""

    def __init__(
        self,
        tolerance: float = 1e-8,
        max_fev: int = 1000
    ):
        """Initialize fsolve solver.

        Args:
            tolerance: Convergence tolerance
            max_fev: Maximum function evaluations
        """
        self.tolerance = tolerance
        self.max_fev = max_fev

    def solve(
        self,
        deriv_func: typing.Callable[[float, list[float], dict], tuple[float, ...]],
        initial_guess: list[float],
        kinetic_params: dict,
        **kwargs
    ) -> tuple[np.ndarray, bool]:
        """Solve using fsolve."""

        def f(x):
            """Wrapper function that returns residuals."""
            return np.array(deriv_func(0.0, x.tolist(), kinetic_params))

        try:
            solution, info, ier, msg = fsolve(
                f,
                initial_guess,
                xtol=self.tolerance,
                maxfev=self.max_fev,
                full_output=True
            )

            success = (ier == 1)
            if success:
                logger.info("fsolve converged successfully")
            else:
                logger.warning(f"fsolve failed: {msg}")

            return solution, success

        except Exception as e:
            logger.error(f"fsolve error: {str(e)}")
            return np.array(initial_guess), False


class BoundedLeastSquaresSolver(SteadyStateSolver):
    """Least-squares solver with bounds to enforce non-negative concentrations."""

    def __init__(
        self,
        tolerance: float = 1e-8,
        max_iter: int = 1000,
        lower_bound: float = 0.0,
        upper_bound: float = np.inf
    ):
        """Initialize bounded least squares solver.

        Args:
            tolerance: Convergence tolerance
            max_iter: Maximum number of iterations
            lower_bound: Lower bound for species concentrations (default: 0.0)
            upper_bound: Upper bound for species concentrations (default: inf)
        """
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def solve(
        self,
        deriv_func: typing.Callable[[float, list[float], dict], tuple[float, ...]],
        initial_guess: list[float],
        kinetic_params: dict,
        **kwargs
    ) -> tuple[np.ndarray, bool]:
        """Solve using bounded least squares method."""

        def f(x):
            """Wrapper function that returns residuals."""
            return np.array(deriv_func(0.0, x.tolist(), kinetic_params))

        try:
            # Set bounds for all species concentrations
            bounds = (self.lower_bound, self.upper_bound)

            result = least_squares(
                f,
                initial_guess,
                bounds=bounds,
                ftol=self.tolerance,
                max_nfev=self.max_iter,
                method='trf'  # Trust Region Reflective algorithm
            )

            if result.success:
                residual_norm = np.linalg.norm(result.fun)
                logger.info(f"Bounded least squares converged with residual: {residual_norm}")
                return result.x, True
            else:
                logger.warning(f"Bounded least squares failed: {result.message}")
                return result.x, False

        except Exception as e:
            logger.error(f"Bounded least squares error: {str(e)}")
            return np.array(initial_guess), False


class ContinuationSolver(SteadyStateSolver):
    """Simple continuation method for following steady-state branches."""

    def __init__(
        self,
        base_solver: SteadyStateSolver,
        parameter_name: str,
        parameter_values: list[float],
        tolerance: float = 1e-8
    ):
        """Initialize continuation solver.

        Args:
            base_solver: Base solver to use at each parameter value
            parameter_name: Name of parameter to vary
            parameter_values: List of parameter values to follow
            tolerance: Convergence tolerance
        """
        self.base_solver = base_solver
        self.parameter_name = parameter_name
        self.parameter_values = parameter_values
        self.tolerance = tolerance

    def solve(
        self,
        deriv_func: typing.Callable[[float, list[float], dict], tuple[float, ...]],
        initial_guess: list[float],
        kinetic_params: dict,
        **kwargs
    ) -> tuple[np.ndarray, bool]:
        """Solve using continuation method."""

        current_solution = np.array(initial_guess)
        current_params = kinetic_params.copy()

        for param_value in self.parameter_values:
            current_params[self.parameter_name] = param_value

            solution, success = self.base_solver.solve(
                deriv_func,
                current_solution.tolist(),
                current_params
            )

            if not success:
                logger.warning(f"Continuation failed at {self.parameter_name}={param_value}")
                return current_solution, False

            current_solution = solution

        logger.info("Continuation method completed successfully")
        return current_solution, True


class SteadyStateSolverFactory:
    """Factory for creating steady-state solvers."""

    @staticmethod
    def create_solver(
        solver_type: str = 'scipy',
        **kwargs
    ) -> SteadyStateSolver:
        """Create a steady-state solver.

        Args:
            solver_type: Type of solver ('newton', 'scipy', 'fsolve', 'bounded', 'continuation')
            **kwargs: Solver-specific parameters

        Returns:
            Configured solver instance
        """
        solver_type = solver_type.lower()

        if solver_type == 'newton':
            return NewtonRaphsonSolver(**kwargs)
        elif solver_type == 'scipy':
            return ScipyRootSolver(**kwargs)
        elif solver_type == 'fsolve':
            return FsolveSolver(**kwargs)
        elif solver_type == 'bounded':
            return BoundedLeastSquaresSolver(**kwargs)
        elif solver_type == 'continuation':
            base_solver = kwargs.pop('base_solver', ScipyRootSolver())
            return ContinuationSolver(base_solver=base_solver, **kwargs)
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")


def find_steady_state(
    deriv_func: typing.Callable[[float, list[float], dict], tuple[float, ...]],
    initial_guess: list[float],
    kinetic_params: dict,
    solver_type: str = 'scipy',
    **solver_kwargs
) -> tuple[np.ndarray, bool, str]:
    """Convenience function to find steady-state solution.

    Args:
        deriv_func: Derivative function that returns dx/dt
        initial_guess: Initial guess for species concentrations
        kinetic_params: Kinetic parameters for the model
        solver_type: Type of solver to use
        **solver_kwargs: Additional solver parameters

    Returns:
        Tuple of (solution, success_flag, message)
    """
    try:
        solver = SteadyStateSolverFactory.create_solver(solver_type, **solver_kwargs)
        solution, success = solver.solve(deriv_func, initial_guess, kinetic_params)

        if success:
            # Verify the solution
            residual = np.array(deriv_func(0.0, solution.tolist(), kinetic_params))
            residual_norm = np.linalg.norm(residual)

            if residual_norm > solver_kwargs.get('tolerance', 1e-6):
                return solution, False, f"Large residual norm: {residual_norm}"
            else:
                return solution, True, f"Converged with residual norm: {residual_norm}"
        else:
            return solution, False, "Solver failed to converge"

    except Exception as e:
        logger.error(f"Error in find_steady_state: {str(e)}")
        return np.array(initial_guess), False, f"Error: {str(e)}"
