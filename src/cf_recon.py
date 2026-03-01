import numpy as np
from src.pt import *
import copy

def reconstruct_cf(control: list, cf_0: np.ndarray, n: int = 10, project: bool = False, tol: float = 1e-8):
    """
    Reconstructs the counterfactual (CF) from the control and the initial CF (cf_0).

    Parameters:
    - control: list of control samples (each sample is a numpy array)
    - cf_0: initial counterfactual (numpy array)
    - n: approximation resolution
    - project: whether to project the tangent onto conservative vector fields (avoid if dimensionality is too high)
    - tol: tolerance for support alignment in parallel transport (increase for high-dimensional or noisy data)

    Returns:
    - cf_reconstructed: the reconstructed counterfactual curve
    """

    # Initialize the reconstructed CF with the initial CF
    cf_cur = EmpiricalMeasure(cf_0, weights=np.ones(len(cf_0)) / len(cf_0))
    control_measures = [EmpiricalMeasure(c, weights=np.ones(len(c)) / len(c)) for c in control]
    cf_curve = [copy.copy(cf_cur)]

    for i in range(len(control) - 1):
        # obtain current tangent vector field from control_i to control_{i+1}
        tangent_nu_i_nu_i_plus_1 = wasserstein_logmap(control_measures[i], control_measures[i+1])
        # Compute the PT update based on the control samples and the current CF
        cf_tan = tangent_nu_i_nu_i_plus_1.parallel_transport(control_measures[i], cf_cur, n=n, project=project, tol=tol)
        # Update the reconstructed CF
        cf_next = cf_tan.wasserstein_expmap()
        cf_curve.append(copy.copy(cf_next))
        cf_cur = cf_next

    return cf_curve