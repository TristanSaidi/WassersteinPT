import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.linalg as linalg
# change to latex font


def step_gauss_pt(A_t, a_t, B, M_t, sigma_0, stepsize):
    """Advance one Euler step of Gaussian parallel transport for Brenier maps.

    Parameters
    ----------
    A_t : np.ndarray
        Current symmetric matrix component of the tangent vector.
    a_t : np.ndarray
        Current vector component of the tangent (kept constant here).
    B : np.ndarray
        Brenier matrix of the geodesic at t=1.
    M_t : np.ndarray
        Matrix M_t for the geodesic at time t.
    sigma_0 : np.ndarray
        Base covariance at t=0.
    stepsize : float
        Euler step size for advancing A_t.

    Returns
    -------
    A_t_new : np.ndarray
        Updated symmetric matrix component.
    a_t_new : np.ndarray
        Updated vector component (unchanged in this model).
    """
    # intermediate objects
    Q_t = np.linalg.inv(M_t).T @ np.linalg.inv(sigma_0) @ np.linalg.inv(M_t)
    S_t = (np.eye(len(a_t)) - B)@np.linalg.inv(M_t)
    # solve \dot{A}Q_t + Q_t \dot{A} = S_t^TA_tQ_t + Q_tA_tS_t
    lyap = S_t.T @ A_t @ Q_t + Q_t @ A_t @ S_t
    # solve for \dot{A}
    dot_A = linalg.solve_sylvester(Q_t, Q_t, lyap)
    # update A_t using Euler's method
    A_t_new = A_t + dot_A * stepsize
    assert np.allclose(A_t_new, A_t_new.T), "A_t is not symmetric"
    a_t_new = a_t # a_t remains constant in this case
    return A_t_new, a_t_new

class Tangent:
    def __init__(self ):
        """Base Tangent class.

        Parameters
        ----------
        A : np.ndarray
            Matrix component of the tangent vector.
        a : np.ndarray
            Vector component of the tangent vector.
        """
        pass

    def pushforward_measure(self, t):
        """Compute the pushforward mean and covariance at time t."""
        pass

    def interpolate(self):
        pass

    def parallel_transport(self):
        pass


class BrenierGaussian(Tangent):
    def __init__(self, mean0, sigma0, A, a):
        """Brenier map for Gaussian measures along a geodesic.

        Parameters
        ----------
        mean0 : np.ndarray
            Base mean at t=0.
        sigma0 : np.ndarray
            Base covariance at t=0.
        A : np.ndarray
            Matrix component of the tangent vector.
        a : np.ndarray
            Vector component of the tangent vector.
        """
        self.mean0 = mean0
        self.sigma0 = sigma0
        self.a = a # tangent vector element 1
        self.A = A # tangent vector element 2
        self.brenier = np.eye(len(mean0)) + A # Brenier matrix =
        self.M_t = lambda t: np.eye(len(mean0)) + t * self.A # M_t = I + tA
        self.m_t = lambda t: self.mean0 + t * self.a # m_t = m_0 + ta

    def pushforward_measure(self, t):
        """Compute the pushforward mean and covariance at time t."""
        mean_t = self.m_t(t)
        sigma_t = self.M_t(t) @ self.sigma0 @ self.M_t(t).T
        return mean_t, sigma_t
    
    def interpolate(self, num_points):
        """Interpolate means and covariances along the geodesic.

        Parameters
        ----------
        num_points : int
            Number of interior points to sample (endpoints excluded).

        Returns
        -------
        interpolated_means : list[np.ndarray]
            Means along the path.
        interpolated_covariances : list[np.ndarray]
            Covariances along the path.
        """
        # exclude endpoints
        t_values = np.linspace(0, 1, num_points + 2)[1:-1]
        interpolated_means = []
        interpolated_covariances = []

        for t in t_values:
            mu_t, sigma_t = self.pushforward_measure(t)
            interpolated_means.append(mu_t)
            interpolated_covariances.append(sigma_t)

        return interpolated_means, interpolated_covariances
    
    def parallel_transport(self, geodesic_brenier, num_points):
        """Parallel transport the tangent vector along a geodesic.

        Parameters
        ----------
        geodesic_brenier : BrenierGaussian
            The geodesic defining the transport path.
        num_points : int
            Number of steps for the Euler discretization.

        Returns
        -------
        transported_As : list[np.ndarray]
            Transported matrix components over time.
        transported_as : list[np.ndarray]
            Transported vector components over time.
        """
        t_values = np.linspace(0, 1, num_points)
        a_0 = self.a # 
        A_0 = self.A # Brenier matrix at t=1
        B = geodesic_brenier.brenier # Brenier matrix at t=1
        sigma_0 = geodesic_brenier.sigma0
        # intermediate parallel transport results
        transported_as = []
        transported_As = []
        A_t = A_0.copy()
        for t in t_values:
            # Compute the parallel transport of the tangent vector along the geodesic
            M_t = geodesic_brenier.M_t(t)
            A_t, a_t = step_gauss_pt(A_t, a_0, B, M_t, sigma_0, 1/num_points)
            transported_as.append(a_t)
            transported_As.append(A_t)
        return transported_As, transported_as
