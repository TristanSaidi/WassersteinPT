import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.linalg as linalg
from scipy.spatial import cKDTree
import ot
import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
# import warnings
import warnings
# change to latex font

def _default_compact_kernel(t):
    """
    Compactly supported kernel on R_+ with support [0,1].
    Input: t = ||x_i-x_j||^2 / h^2.
    """
    t = np.asarray(t, dtype=float)
    out = 1.0 - t
    out[out < 0.0] = 0.0
    return out


PROJECT_ARGS_DEFAULTS = dict(
    h=None,
    h_r=None,
    kernel=_default_compact_kernel,
    alpha_reg=0.0,
    min_neighbors=15,
    k_bandwidth=25,
    mixing_param=1.0,
    # RKHS-specific
    rkhs=True,
    sigma=None,
    lam=1e-4,
    jitter=1e-10,
    n_rff=1000,
)

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



class EmpiricalMeasure:
    def __init__(self, locs, weights):
        """Empirical measure class.

        Parameters
        ----------
        locs : np.ndarray
            Locations of the empirical measure.
        weights : np.ndarray
            Weights of the empirical measure.
        """
        self.locs = locs
        self.weights = weights / weights.sum()


def interp_tan(tangent, t_0, t_1):
    # get the interpolated map along the geodesic described by tangent at time t_0 and t_1
    assert 0 <= t_0 <= 1 and 0 <= t_1 <= 1, "t_0 and t_1 should be in [0, 1]."
    if tangent.vels is not None and tangent.src_measure.locs is not None:
        return interp_tan_vel(tangent, t_0, t_1)
    elif tangent.coupling is not None and tangent.src_measure.locs is not None and tangent.locs_dst is not None:
        return interp_tan_coupling(tangent, t_0, t_1)

def interp_tan_vel(tangent: "W2EuclideanTangent", t_0, t_1):
    # tangent is vel-mode and represents the full geodesic from src to dst
    X0 = np.asarray(tangent.src_measure.locs)
    w0 = np.asarray(tangent.src_measure.weights)
    V  = np.asarray(tangent.vels)

    if V.shape != X0.shape:
        raise ValueError("tangent.vels must have same shape as src_measure.locs")

    # positions at time t0
    X_t0 = X0 + t_0 * V

    # incremental displacement from t0 to t1
    V_inc = (t_1 - t_0) * V

    src_t0 = EmpiricalMeasure(locs=X_t0, weights=w0.copy())
    return W2EuclideanTangent(src_measure=src_t0, vels=V_inc)



def interp_tan_coupling(tangent, t_0, t_1, eps=0.0):
    """
    Induce a coupling-tangent from time t_0 to t_1 along the displacement interpolation
    defined by (coupling, locs_src, locs_dst).

    Returns
    -------
    W2EuclideanTangent in coupling-mode:
        locs_src := support of mu_{t0}
        locs_dst := support of mu_{t1}
        coupling := a coupling matrix whose row/col sums equal the weights of mu_{t0}, mu_{t1}.
    """
    assert tangent.coupling is not None
    assert tangent.src_measure.locs is not None and tangent.locs_dst is not None
    assert 0.0 <= t_0 <= 1.0 and 0.0 <= t_1 <= 1.0

    P = tangent.coupling  # may be sparse or dense
    X = np.asarray(tangent.src_measure.locs)
    Y = np.asarray(tangent.locs_dst)

    n, m = P.shape
    if X.shape[0] != n:
        raise ValueError(f"locs_src has {X.shape[0]} points but coupling has {n} rows")
    if Y.shape[0] != m:
        raise ValueError(f"locs_dst has {Y.shape[0]} points but coupling has {m} cols")

    # Extract nonzero entries — works for both sparse and dense
    if sp.issparse(P):
        P_coo = P.tocoo()
        if eps > 0:
            mask = P_coo.data > eps
            ii = P_coo.row[mask].astype(int)
            jj = P_coo.col[mask].astype(int)
            masses = P_coo.data[mask]
        else:
            ii = P_coo.row.astype(int)
            jj = P_coo.col.astype(int)
            masses = P_coo.data.copy()
    else:
        P = np.asarray(P)
        if eps > 0:
            ii, jj = np.nonzero(P > eps)
        else:
            ii, jj = np.nonzero(P)
        masses = P[ii, jj]

    if ii.size == 0:
        # no mass - return empty
        d = X.shape[1]
        return W2EuclideanTangent(
            src_measure=EmpiricalMeasure(np.zeros((0, d)), np.zeros(0)),
            locs_dst=np.zeros((0, d)),
            coupling=sp.csr_matrix((0, 0)),
        )

    masses = np.asarray(masses, dtype=float)    # (K,)
    Xi = X[ii]                                  # (K,d)
    Yj = Y[jj]                                  # (K,d)

    # Expanded supports at t0 and t1
    Z0 = (1.0 - t_0) * Xi + t_0 * Yj            # (K,d)
    Z1 = (1.0 - t_1) * Xi + t_1 * Yj            # (K,d)

    # Compress duplicates separately on each side
    # NOTE: exact float equality is assumed here. If you expect roundoff noise, see note below.
    uniq0, inv0 = np.unique(Z0, axis=0, return_inverse=True)  # inv0 in {0..N0-1}
    uniq1, inv1 = np.unique(Z1, axis=0, return_inverse=True)  # inv1 in {0..N1-1}

    N0 = uniq0.shape[0]
    N1 = uniq1.shape[0]

    # Build the compressed coupling as sparse COO then CSR
    coup01 = sp.coo_matrix((masses, (inv0, inv1)), shape=(N0, N1)).tocsr()

    new_weights_0 = np.asarray(coup01.sum(axis=1)).ravel()  # (N0,)
    new_src_measure = EmpiricalMeasure(locs=uniq0, weights=new_weights_0)

    return W2EuclideanTangent(src_measure=new_src_measure, locs_dst=uniq1, coupling=coup01)

def cost_euclidean(src, dst):
    # compute the cost matrix between src and dst in the Euclidean space
    # ot.dist uses BLAS and avoids materialising the (n, m, d) broadcast tensor
    return ot.dist(src, dst, metric='euclidean')

# general case
def quadratic_coupling(src, dst, cost):
    # compute the optimal transport map from src to dst with cost function cost
    cost_sq = cost ** 2
    gamma = ot.emd(src, dst, M=cost_sq)
    # Convert to sparse immediately to avoid holding the full (n x m) dense
    # matrix downstream — the OT plan is typically supported on a map so
    # the sparse representation is O(n) rather than O(n*m).
    gamma_sparse = sp.csr_matrix(gamma)
    del gamma
    return gamma_sparse

def coupling_to_tan(coupling, src: EmpiricalMeasure, locs_dst):
    # if coupling supported on a map, return vector field. Otherwise, return coupling
    locs_src = src.locs
    # A coupling is supported on a map if each source location sends mass to at most one target
    if sp.issparse(coupling):
        cx = coupling.tocsr()
        row_nnz = np.diff(cx.indptr)
        if np.all(row_nnz <= 1):
            # For rows with no mass (row_nnz == 0), cx.indices won't have an entry;
            # find destination index per row safely via argmax on each row.
            dst_idx = np.array(cx.argmax(axis=1)).ravel()
            vels = locs_dst[dst_idx] - locs_src
            return W2EuclideanTangent(src_measure=src, vels=vels)
        return W2EuclideanTangent(src_measure=src, locs_dst=locs_dst, coupling=cx)
    else:
        row_nnz = np.count_nonzero(coupling, axis=1)
        if np.all(row_nnz <= 1):
            dst_idx = np.argmax(coupling, axis=1)
            vels = locs_dst[dst_idx] - locs_src
            return W2EuclideanTangent(src_measure=src, vels=vels)
        return W2EuclideanTangent(src_measure=src, locs_dst=locs_dst, coupling=coupling)

def wasserstein_logmap(src: EmpiricalMeasure, dst: EmpiricalMeasure):
    # compute the logarithmic map of dst at src in the Wasserstein space (for empirical measures)
    src_weights = src.weights
    dst_weights = dst.weights
    locs_src = src.locs
    locs_dst = dst.locs
    # compute the optimal transport map from src to dst
    cost = cost_euclidean(locs_src, locs_dst)
    coupling = quadratic_coupling(src_weights, dst_weights, cost)
    # compute the tangent vector from src to dst
    tangent = coupling_to_tan(coupling, src, locs_dst)
    return tangent

# tangents will be vector fields supported on the source
def expmap_euclidean(src, tangent):
    # compute the exponential map of tangent at src
    return src + tangent

def logmap_euclidean(src, dst):
    # compute the logarithmic map of dst at src
    return dst - src

def wasserstein_expmap_coupling(base: EmpiricalMeasure, tangent: Tangent = None):
    # compute the exponential map of tangent at src in the Wasserstein space (for empirical measures)
    locs_dst = tangent.locs_dst
    coupling = tangent.coupling
    dst_weights = np.asarray(coupling.sum(axis=0)).ravel()
    assert dst_weights.shape[0] == locs_dst.shape[0], "The number of destination locations must match the number of columns in the coupling matrix."
    return EmpiricalMeasure(locs_dst, dst_weights)

def align_by_nearest(X_from, X_to, tol=1e-7):
    """
    For each row in X_to, find nearest row in X_from.
    Returns indices into X_from. Raises if max dist > tol.
    """
    tree = cKDTree(np.asarray(X_from))
    dists, idx = tree.query(np.asarray(X_to), k=1)

    maxd = float(np.max(dists)) if dists.size else 0.0
    if maxd > tol:
        bad = int(np.argmax(dists))
        raise ValueError(
            f"Support alignment failed: max NN dist {maxd:.3e} exceeds tol={tol}. "
            f"Bad index={bad}."
        )
    return idx

def transport_field_under_coupling(vels_src, coupling):
    """
    vels_src : (n_src, d)
    coupling : (n_src, n_dst) with nonnegative entries — dense or sparse

    Returns
    -------
    vels_dst : (n_dst, d), where vels_dst[j] = sum_i coupling[i,j] vels_src[i] / sum_i coupling[i,j]
    dst_weights : (n_dst,), column sums of coupling
    """
    V = np.asarray(vels_src)
    if V.ndim != 2:
        raise ValueError("vels_src must be 2D (n_src, d)")

    if sp.issparse(coupling):
        Gamma = coupling.tocsr()
        if Gamma.shape[0] != V.shape[0]:
            raise ValueError("coupling rows must match vels_src length")
        dst_weights = np.asarray(Gamma.sum(axis=0)).ravel()  # (n_dst,)
        # sparse matmul: (n_dst, n_src) @ (n_src, d) — only touches nonzeros
        num = np.asarray(Gamma.T @ V)
    else:
        Gamma = np.asarray(coupling)
        if Gamma.ndim != 2:
            raise ValueError("coupling must be 2D")
        if Gamma.shape[0] != V.shape[0]:
            raise ValueError("coupling rows must match vels_src length")
        dst_weights = Gamma.sum(axis=0)
        num = Gamma.T @ V

    vels_dst = np.zeros((coupling.shape[1], V.shape[1]), dtype=V.dtype)
    mask = dst_weights > 0
    vels_dst[mask] = num[mask] / dst_weights[mask, None]
    return vels_dst, dst_weights

class W2EuclideanTangent(Tangent):
    def __init__(self, src_measure: EmpiricalMeasure, vels = None, coupling = None, locs_dst = None):
        """Tangent vector in the P_2(R^d) space.

        Parameters
        ----------
        vels : np.ndarray
            Velocities of the tangent vectors.
        locs : np.ndarray
            Locations of the tangent vectors.
        """
        super().__init__()
        # either (vels, locs_src), or (coupling, locs_src, locs_dst)
        if vels is not None and src_measure is not None:
            assert coupling is None and locs_dst is None, "If vels and locs_src are provided, coupling and locs_dst should be None."
        elif coupling is not None and src_measure is not None and locs_dst is not None:
            assert vels is None, "If coupling, locs_src and locs_dst are provided, vels should be None."
        else:
            raise ValueError("Either (vels, locs_src) or (coupling, locs_src, locs_dst) should be provided.")
        self.vels = vels
        self.src_measure = src_measure
        self.coupling = coupling
        self.locs_dst = locs_dst

    def pushforward_measure(self, t):
        """Compute the pushforward mean and covariance at time t."""
        # Compute the pushforward mean and covariance at time t
        

    def interpolate(self):
        pass


    def wasserstein_expmap(self):
        # compute the exponential map of tangent at src in the Wasserstein space (for empirical measures)
        # for now, we only implement the case where tangent is given by velocities at the support of src
        if self.vels is not None and self.src_measure.locs is not None:
            return wasserstein_expmap_vel(self.src_measure, self)
        elif self.coupling is not None and self.src_measure.locs is not None and self.locs_dst is not None:
            return wasserstein_expmap_coupling(self.src_measure, self)
        else:
            raise ValueError("Tangent must be given by either (vels, locs_src) or (coupling, locs_src, locs_dst).")




    def parallel_transport(self, src: EmpiricalMeasure, dst: EmpiricalMeasure, n=1, project=False, tol=1e-8, project_args=PROJECT_ARGS_DEFAULTS):
        if self.coupling is not None:
            print("Warning: coupling is being converted to velocity mode via barycentric projection for parallel transport.")
            tangent = barycentric_projection_tangent(self)
        else:
            tangent = self
        stepsize = 1.0 / n
        geodesic_tangent = wasserstein_logmap(src, dst)

        # parse project_args with defaults (graph-based params)
        h            = project_args.get('h',            PROJECT_ARGS_DEFAULTS['h'])
        h_r          = project_args.get('h_r',          PROJECT_ARGS_DEFAULTS['h_r'])
        kernel       = project_args.get('kernel',       PROJECT_ARGS_DEFAULTS['kernel'])
        alpha_reg    = project_args.get('alpha_reg',    PROJECT_ARGS_DEFAULTS['alpha_reg'])
        min_neighbors= project_args.get('min_neighbors',PROJECT_ARGS_DEFAULTS['min_neighbors'])
        k_bandwidth  = project_args.get('k_bandwidth',  PROJECT_ARGS_DEFAULTS['k_bandwidth'])
        rkhs         = project_args.get('rkhs', False)

        # parse RKHS-specific params
        sigma    = project_args.get('sigma',    None)
        lam      = project_args.get('lam',      1e-4)
        jitter   = project_args.get('jitter',   1e-10)
        n_rff    = project_args.get('n_rff',    None)
        rff_seed = project_args.get('rff_seed', None)

        # Pre-compute sigma and RFF features once before the PT loop.
        # The source support (and thus sigma) is fixed throughout transport.
        rff_features = None
        if project and rkhs and n_rff is not None:
            X_src = np.asarray(tangent.src_measure.locs, dtype=float)
            d_src = X_src.shape[1]
            if sigma is None:
                min_sigma = project_args.get('min_sigma', 1e-8)
                kq = min(max(2, k_bandwidth + 1), len(X_src))
                knn_dists, _ = cKDTree(X_src).query(X_src, k=kq)
                k0 = min(k_bandwidth, len(X_src) - 1)
                kth = knn_dists[:, k0]
                sigma = max(float(np.mean(kth[np.isfinite(kth)])), float(min_sigma))
            rng = np.random.default_rng(rff_seed)
            Omega_pre = rng.standard_normal((n_rff, d_src)) / float(sigma)
            b_pre     = rng.uniform(0.0, 2.0 * np.pi, n_rff)
            rff_features = (Omega_pre, b_pre)

        for iter in range(n):
            t_0 = iter * stepsize
            t_1 = (iter + 1) * stepsize

            incremental_tangent = interp_tan(geodesic_tangent, t_0, t_1)

            tangent = parallel_transport_incremental(
                tangent=tangent,
                incremental_tangent=incremental_tangent,
                tol=tol,
            )
            if project:
                tangent = project_tan_helper(
                    rkhs=rkhs,
                    tangent=tangent,
                    # graph-based
                    h=h, h_r=h_r, kernel=kernel,
                    alpha_reg=alpha_reg, min_neighbors=min_neighbors,
                    k_bandwidth=k_bandwidth,
                    # RKHS (passed through to project_tan_rkhs)
                    sigma=sigma, lam=lam, jitter=jitter,
                    n_rff=n_rff, rff_features=rff_features,
                )

        return tangent



def _solve_potential_with_gauge(L, g, ridge=0.0):
    """
    Solve L U = g with global mean-zero gauge 1^T U = 0.

    If the graph is disconnected, raise a warning and stop.
    """

    N = L.shape[0]
    g = np.asarray(g, dtype=float).reshape(-1)

    if g.shape[0] != N:
        raise ValueError("g must have shape (N,)")

    # Check connectivity
    n_comp, labels = connected_components(L, directed=False, connection="weak")

    if n_comp > 1:
        warnings.warn(
            f"Graph has {n_comp} connected components. "
            "Projection requires connected graph. Aborting.",
            RuntimeWarning,
        )
        raise RuntimeError("Graph is disconnected.")

    # Optional ridge (numerical stability only)
    if ridge > 0:
        L = (L + ridge * sp.eye(N, format="csr")).tocsr()

    # KKT system:
    # [L  1][U] = [g]
    # [1^T 0][λ]   [0]

    one = np.ones(N, dtype=float)
    one_col = sp.csr_matrix(one.reshape(-1, 1))

    top = sp.hstack([L, one_col], format="csr")
    bottom = sp.hstack([one_col.T, sp.csr_matrix((1, 1))], format="csr")
    KKT = sp.vstack([top, bottom], format="csr")

    rhs = np.zeros(N + 1, dtype=float)
    rhs[:N] = g
    rhs[N] = 0.0

    sol = spla.spsolve(KKT, rhs)
    U = np.asarray(sol[:N], dtype=float)

    # numerical cleanup
    U -= U.mean()

    return U

import numpy as np
import scipy.linalg as linalg
from scipy.spatial import cKDTree

def choose_sigma_knn(X, k_bandwidth=25, min_sigma=1e-8):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n <= 1:
        return 1.0
    tree = cKDTree(X)
    kq = min(max(2, k_bandwidth + 1), n)
    dists, _ = tree.query(X, k=kq)
    k0 = min(k_bandwidth, n - 1)
    kth = dists[:, k0]
    kth = kth[np.isfinite(kth)]
    if kth.size == 0:
        raise ValueError("Failed to choose sigma.")
    return max(float(np.mean(kth)), float(min_sigma))


class GaussianGradientRFFProjector:
    """
    Approximate gradient-RKHS projector using random Fourier features for the
    scalar Gaussian kernel.

    f_beta(x) = beta^T phi(x),
    grad f_beta(x) = J_phi(x)^T beta.
    """

    def __init__(self, sigma=None, n_rff=1000, k_bandwidth=25,
                 min_sigma=1e-8, seed=None):
        self.sigma = sigma
        self.n_rff = int(n_rff)
        self.k_bandwidth = k_bandwidth
        self.min_sigma = min_sigma
        self.seed = seed

        self.Omega = None   # (D, d)
        self.b = None       # (D,)
        self.beta = None    # (D,)
        self.d = None

    def _fit_features(self, X):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        self.d = d

        sigma = self.sigma
        if sigma is None:
            sigma = choose_sigma_knn(
                X, k_bandwidth=self.k_bandwidth, min_sigma=self.min_sigma
            )
        self.sigma = float(sigma)

        rng = np.random.default_rng(self.seed)
        self.Omega = rng.standard_normal((self.n_rff, d)) / self.sigma
        self.b = rng.uniform(0.0, 2.0 * np.pi, size=self.n_rff)

    def feature_jacobian_terms(self, X):
        """
        Returns S with shape (n, D), where
            S[i,r] = sqrt(2/D) * sin(omega_r^T x_i + b_r)

        Then
            grad phi_r(x_i) = -S[i,r] * omega_r.
        """
        X = np.asarray(X, dtype=float)
        Z = X @ self.Omega.T + self.b[None, :]
        S = np.sqrt(2.0 / self.n_rff) * np.sin(Z)
        return S

    def fit(self, X, V, lam=1e-4, sample_weight=None, jitter=1e-10):
        X = np.asarray(X, dtype=float)
        V = np.asarray(V, dtype=float)
        n, d = X.shape
        if V.shape != X.shape:
            raise ValueError("X and V must have same shape.")

        self._fit_features(X)
        D = self.n_rff

        if sample_weight is None:
            w = np.ones(n, dtype=float) / n
        else:
            w = np.asarray(sample_weight, dtype=float).reshape(-1)
            if w.shape[0] != n:
                raise ValueError("sample_weight must have length n.")
            if np.any(w < 0) or w.sum() <= 0:
                raise ValueError("sample_weight must be nonnegative and not all zero.")
            w = w / w.sum()

        S = self.feature_jacobian_terms(X)      # (n, D)
        Omega = self.Omega                      # (D, d)

        # G_i^T G_i contribution:
        #   G_i beta = - sum_r beta_r S[i,r] omega_r
        # Hence
        #   G_i^T G_i = (Omega Omega^T) \odot (S_i S_i^T)
        OOT = Omega @ Omega.T                   # (D, D)

        M = np.zeros((D, D), dtype=float)
        rhs = np.zeros(D, dtype=float)

        for i in range(n):
            si = S[i]                           # (D,)
            vi = V[i]                           # (d,)
            GiTGi = np.outer(si, si) * OOT
            M += w[i] * GiTGi

            # G_i^T v_i = - s_i \odot (Omega v_i)
            rhs += -w[i] * si * (Omega @ vi)

        M += (lam + jitter) * np.eye(D)

        try:
            self.beta = linalg.solve(M, rhs, assume_a='sym')
        except Exception:
            self.beta = np.linalg.lstsq(M, rhs, rcond=None)[0]

        return self

    def grad(self, X):
        X = np.asarray(X, dtype=float)
        S = self.feature_jacobian_terms(X)      # (n, D)
        # grad f(x_i) = - (S[i] * beta)^T Omega
        return -(S * self.beta[None, :]) @ self.Omega
    
def project_tan_rkhs(
    tangent,
    sigma=None,
    lam=1e-4,
    n_rff=1000,
    sample_weight=None,
    jitter=1e-10,
    k_bandwidth=25,
    min_sigma=1e-8,
    seed=None,
    return_projector=False,
):
    if tangent.coupling is not None:
        tangent_vel = barycentric_projection_tangent(tangent)
    else:
        tangent_vel = tangent

    X = np.asarray(tangent_vel.src_measure.locs, dtype=float)
    V = np.asarray(tangent_vel.vels, dtype=float)

    if sample_weight is None:
        sample_weight = np.asarray(tangent_vel.src_measure.weights, dtype=float)

    projector = GaussianGradientRFFProjector(
        sigma=sigma,
        n_rff=n_rff,
        k_bandwidth=k_bandwidth,
        min_sigma=min_sigma,
        seed=seed,
    )
    projector.fit(X, V, lam=lam, sample_weight=sample_weight, jitter=jitter)
    G = projector.grad(X)

    projected_tangent = W2EuclideanTangent(
        src_measure=tangent_vel.src_measure,
        vels=G,
    )

    if return_projector:
        return projected_tangent, projector
    return projected_tangent

_PROJECT_TAN_RKHS_KEYS = {'tangent', 'sigma', 'lam', 'sample_weight', 'jitter', 'center_potential', 'k_bandwidth', 'min_sigma', 'n_rff', 'rff_seed', 'debug'}
_PROJECT_TAN_KEYS = {'tangent', 'h', 'h_r', 'kernel', 'alpha_reg', 'min_neighbors', 'k_bandwidth', 'debug', 'debug_cond_samples', 'd_large_threshold', 'max_mean_degree'}

def project_tan_helper(**kwargs):
    if kwargs.pop('rkhs', False):
        filtered = {k: v for k, v in kwargs.items() if k in _PROJECT_TAN_RKHS_KEYS}
        return project_tan_rkhs(**filtered)
    else:
        filtered = {k: v for k, v in kwargs.items() if k in _PROJECT_TAN_KEYS}
        return project_tan(**filtered)

def project_tan(
    tangent,
    h=None,
    h_r=None,
    kernel=_default_compact_kernel,
    alpha_reg=0.0,
    min_neighbors=15,
    k_bandwidth=25,
    debug=False,
    debug_cond_samples=True,
    d_large_threshold=12,
    max_mean_degree=None,
):
    """
    Weighted Helmholtz-Hodge projection with optional debugging.

    Parameters
    ----------
    tangent : W2EuclideanTangent
        Input tangent field. If tangent.coupling is present, barycentric projection
        is applied first.

    h, h_r : float or None
        Graph and regression bandwidths. If None, chosen from the mean distance to
        the k_bandwidth-th nearest neighbor.

    kernel : callable
        Compactly supported kernel on R_+.

    alpha_reg : float
        Ridge regularization added to each local linear regression system.
        Required to be > 0 when d >= d_large_threshold.

    min_neighbors : int
        Minimum number of neighbors enforced through kNN fallback.

    k_bandwidth : int
        Used both for automatic bandwidth selection and for kNN fallback.

    debug : bool
        If True, emit diagnostic print statements.

    debug_cond_samples : bool
        If True and debug=True, print condition numbers of a few local systems.

    d_large_threshold : int
        If ambient dimension d >= d_large_threshold, require alpha_reg > 0.

    max_mean_degree : float or None
        If not None, raise an error when the radius graph mean degree exceeds this.
        Useful for catching near-complete graphs before memory blows up.
    """

    def dprint(*args, **kwargs):
        if debug:
            print(*args, **kwargs)

    if getattr(tangent, "coupling", None) is not None:
        tangent = barycentric_projection_tangent(tangent)

    X = np.asarray(tangent.src_measure.locs, dtype=float)
    V = np.asarray(tangent.vels, dtype=float)
    N, d = X.shape

    if d >= d_large_threshold and alpha_reg <= 0:
        raise ValueError(
            f"project_tan requires alpha_reg > 0 when d >= {d_large_threshold}. "
            f"Got d={d}, alpha_reg={alpha_reg}."
        )

    dprint("=" * 80)
    dprint("[project_tan] start")
    dprint(f"[project_tan] N={N}, d={d}")
    dprint(
        f"[project_tan] requested h={h}, h_r={h_r}, alpha_reg={alpha_reg}, "
        f"min_neighbors={min_neighbors}, k_bandwidth={k_bandwidth}"
    )

    # ---------------------------------------------------------
    # Basic diagnostics on X and V
    # ---------------------------------------------------------
    dprint(f"[project_tan] X finite? {np.isfinite(X).all()}")
    dprint(f"[project_tan] V finite? {np.isfinite(V).all()}")

    if np.isfinite(X).any():
        dprint(f"[project_tan] X min/max = {np.nanmin(X):.6g}, {np.nanmax(X):.6g}")
    else:
        dprint("[project_tan] X has no finite entries")

    if np.isfinite(V).any():
        dprint(f"[project_tan] V min/max = {np.nanmin(V):.6g}, {np.nanmax(V):.6g}")
    else:
        dprint("[project_tan] V has no finite entries")

    bad_X_rows = (~np.isfinite(X).all(axis=1)).sum()
    bad_V_rows = (~np.isfinite(V).all(axis=1)).sum()
    dprint(f"[project_tan] bad X rows = {bad_X_rows}")
    dprint(f"[project_tan] bad V rows = {bad_V_rows}")

    if debug:
        try:
            n_unique = np.unique(X, axis=0).shape[0]
            dprint(f"[project_tan] duplicate X rows = {N - n_unique}")
        except Exception as e:
            dprint(f"[project_tan] could not compute duplicate row count: {e}")

    if N <= min_neighbors:
        dprint("[project_tan] N <= min_neighbors, returning tangent unchanged")
        dprint("=" * 80)
        return tangent

    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite entries before building cKDTree.")
    if not np.isfinite(V).all():
        raise ValueError("V contains non-finite entries before projection.")

    tree = cKDTree(X)

    # ---------------------------------------------------------
    # Precompute kNN once
    # ---------------------------------------------------------
    kq = min(max(k_bandwidth, min_neighbors) + 1, N)
    dprint(f"[project_tan] querying kNN with k={kq}")
    knn_dists, knn_idx = tree.query(X, k=kq)

    dprint(f"[project_tan] knn_dists finite? {np.isfinite(knn_dists).all()}")
    if np.isfinite(knn_dists).any():
        dprint(
            f"[project_tan] knn_dists min/max = "
            f"{np.nanmin(knn_dists):.6g}, {np.nanmax(knn_dists):.6g}"
        )

    # ---------------------------------------------------------
    # Bandwidth selection
    # ---------------------------------------------------------
    k0 = min(k_bandwidth, N - 1)
    kdist = knn_dists[:, k0]
    finite_kdist = kdist[np.isfinite(kdist)]

    dprint(f"[project_tan] k0={k0}")
    dprint(f"[project_tan] finite k0-dist count = {finite_kdist.size} / {len(kdist)}")
    if finite_kdist.size > 0:
        dprint(
            f"[project_tan] k0-dist min/median/max = "
            f"{np.min(finite_kdist):.6g}, {np.median(finite_kdist):.6g}, {np.max(finite_kdist):.6g}"
        )

    if h is None:
        if finite_kdist.size == 0:
            raise ValueError("Cannot choose h: no finite kNN distances.")
        h = float(np.mean(finite_kdist))

    if h_r is None:
        if finite_kdist.size == 0:
            raise ValueError("Cannot choose h_r: no finite kNN distances.")
        h_r = float(np.mean(finite_kdist))

    dprint(f"[project_tan] chosen h   = {h}")
    dprint(f"[project_tan] chosen h_r = {h_r}")

    if not np.isfinite(h) or h <= 0:
        raise ValueError(f"Invalid bandwidth h={h}")
    if not np.isfinite(h_r) or h_r <= 0:
        raise ValueError(f"Invalid bandwidth h_r={h_r}")

    # ---------------------------------------------------------
    # Neighborhood diagnostics AFTER valid h, h_r are known
    # ---------------------------------------------------------
    dprint("[project_tan] querying radius neighborhoods")
    neigh = tree.query_ball_point(X, r=h)
    lens_raw = np.array([len(js) for js in neigh], dtype=int)
    lens = np.array([max(len(js) - 1, 0) for js in neigh], dtype=int)

    mean_deg = float(lens.mean())
    dprint(
        f"[project_tan] graph raw neighbor counts:"
        f" min={lens_raw.min()}, mean={lens_raw.mean():.3f}, median={np.median(lens_raw):.3f}, max={lens_raw.max()}"
    )
    dprint(
        f"[project_tan] graph excl-self neighbor counts:"
        f" min={lens.min()}, mean={mean_deg:.3f}, median={np.median(lens):.3f}, max={lens.max()}"
    )
    dprint(f"[project_tan] total graph neighbor refs (excl self) = {lens.sum()}")

    if max_mean_degree is not None and mean_deg > max_mean_degree:
        raise RuntimeError(
            f"Radius graph too dense: mean degree = {mean_deg:.3f}, "
            f"threshold = {max_mean_degree}, h = {h}."
        )

    neigh_r = tree.query_ball_point(X, r=h_r)
    lens_r_raw = np.array([len(js) for js in neigh_r], dtype=int)
    lens_r = np.array([max(len(js) - 1, 0) for js in neigh_r], dtype=int)

    mean_deg_r = float(lens_r.mean())
    dprint(
        f"[project_tan] reg raw neighbor counts:"
        f" min={lens_r_raw.min()}, mean={lens_r_raw.mean():.3f}, median={np.median(lens_r_raw):.3f}, max={lens_r_raw.max()}"
    )
    dprint(
        f"[project_tan] reg excl-self neighbor counts:"
        f" min={lens_r.min()}, mean={mean_deg_r:.3f}, median={np.median(lens_r):.3f}, max={lens_r.max()}"
    )
    dprint(f"[project_tan] total reg neighbor refs (excl self) = {lens_r.sum()}")

    if max_mean_degree is not None and mean_deg_r > max_mean_degree:
        raise RuntimeError(
            f"Regression radius graph too dense: mean degree = {mean_deg_r:.3f}, "
            f"threshold = {max_mean_degree}, h_r = {h_r}."
        )

    # ---------------------------------------------------------
    # Step 1: Build edge list exactly
    # ---------------------------------------------------------
    dprint("[project_tan] building edge list")
    edge_i = []
    edge_j = []

    for i, js in enumerate(neigh):
        for j in js:
            if j != i:
                a, b = (i, j) if i < j else (j, i)
                edge_i.append(a)
                edge_j.append(b)

    dprint(f"[project_tan] candidate radius edges (with duplicates) = {len(edge_i)}")

    # ensure >= min_neighbors via kNN fallback
    kneed = min(min_neighbors + 1, N)
    fallback_nodes = 0
    fallback_edges_added = 0

    for i in range(N):
        js = [int(j) for j in np.atleast_1d(knn_idx[i, :kneed]) if int(j) != i]
        if max(len(neigh[i]) - 1, 0) < min_neighbors:
            fallback_nodes += 1
            for j in js:
                a, b = (i, j) if i < j else (j, i)
                edge_i.append(a)
                edge_j.append(b)
                fallback_edges_added += 1

    dprint(f"[project_tan] fallback nodes = {fallback_nodes}")
    dprint(f"[project_tan] fallback edges added (pre-dedup) = {fallback_edges_added}")

    if len(edge_i) == 0:
        dprint("[project_tan] no edges found, returning tangent unchanged")
        dprint("=" * 80)
        return tangent

    dprint("[project_tan] deduplicating edges")
    edges = np.unique(np.column_stack([edge_i, edge_j]), axis=0)
    src = edges[:, 0]
    dst = edges[:, 1]
    M = len(src)

    dprint(f"[project_tan] unique undirected edges M = {M}")

    if M == 0:
        dprint("[project_tan] no unique edges after dedup, returning tangent unchanged")
        dprint("=" * 80)
        return tangent

    # ---------------------------------------------------------
    # Step 2: Edge weights and actions
    # ---------------------------------------------------------
    dprint("[project_tan] computing edge geometry")
    DX = X[dst] - X[src]
    dprint(f"[project_tan] DX shape = {DX.shape}, DX finite? {np.isfinite(DX).all()}")

    t_edge = np.einsum("ij,ij->i", DX, DX) / (h * h)
    dprint(f"[project_tan] t_edge finite? {np.isfinite(t_edge).all()}")
    if np.isfinite(t_edge).any():
        dprint(f"[project_tan] t_edge min/max = {np.nanmin(t_edge):.6g}, {np.nanmax(t_edge):.6g}")

    K_edge = kernel(t_edge)
    dprint(f"[project_tan] K_edge finite? {np.isfinite(K_edge).all()}")
    if np.isfinite(K_edge).any():
        dprint(f"[project_tan] K_edge min/max = {np.nanmin(K_edge):.6g}, {np.nanmax(K_edge):.6g}")

    weights = K_edge / (h ** d)
    weights[weights <= 0.0] = 1e-12 / (h ** d)

    dprint(f"[project_tan] weights finite? {np.isfinite(weights).all()}")
    if np.isfinite(weights).any():
        dprint(f"[project_tan] weights min/max = {np.nanmin(weights):.6g}, {np.nanmax(weights):.6g}")

    a_edge = 0.5 * np.einsum("ij,ij->i", V[src] + V[dst], DX)
    dprint(f"[project_tan] a_edge finite? {np.isfinite(a_edge).all()}")
    if np.isfinite(a_edge).any():
        dprint(f"[project_tan] a_edge min/max = {np.nanmin(a_edge):.6g}, {np.nanmax(a_edge):.6g}")

    wa = weights * a_edge
    dprint(f"[project_tan] wa finite? {np.isfinite(wa).all()}")

    # ---------------------------------------------------------
    # Step 3: Direct Laplacian assembly
    # ---------------------------------------------------------
    dprint("[project_tan] assembling Laplacian")
    diag = np.zeros(N, dtype=float)
    np.add.at(diag, src, weights)
    np.add.at(diag, dst, weights)

    g = np.zeros(N, dtype=float)
    np.add.at(g, src, -wa)
    np.add.at(g, dst, +wa)

    dprint(f"[project_tan] diag finite? {np.isfinite(diag).all()}")
    dprint(f"[project_tan] g finite? {np.isfinite(g).all()}")

    row = np.concatenate([src, dst, np.arange(N)])
    col = np.concatenate([dst, src, np.arange(N)])
    data = np.concatenate([-weights, -weights, diag])

    L = sp.coo_matrix((data, (row, col)), shape=(N, N)).tocsr()
    dprint(f"[project_tan] L shape = {L.shape}, nnz = {L.nnz}")
    dprint(f"[project_tan] L.data MB   = {L.data.nbytes / 1024**2:.3f}")
    dprint(f"[project_tan] L.indices MB= {L.indices.nbytes / 1024**2:.3f}")
    dprint(f"[project_tan] L.indptr MB = {L.indptr.nbytes / 1024**2:.3f}")

    dprint("[project_tan] solving potential")
    U = _solve_potential_with_gauge(L, g, ridge=0.0)
    dprint(f"[project_tan] U finite? {np.isfinite(U).all()}")
    if np.isfinite(U).any():
        dprint(f"[project_tan] U min/max = {np.nanmin(U):.6g}, {np.nanmax(U):.6g}")

    # ---------------------------------------------------------
    # Step 4: Local linear gradient extraction
    # ---------------------------------------------------------
    dprint("[project_tan] local linear gradient extraction")
    G = np.zeros((N, d), dtype=float)

    n_empty = 0
    n_fallback = 0
    n_lstsq = 0
    max_local_n = 0

    cond_sample_idx = set()
    if debug and debug_cond_samples and N > 0:
        cond_sample_idx.update(range(min(5, N)))
        cond_sample_idx.update({N // 4, N // 2, (3 * N) // 4, N - 1})

    for i in range(N):
        js = [int(j) for j in neigh_r[i] if j != i]

        if len(js) < min_neighbors:
            extra = [
                int(j) for j in np.atleast_1d(knn_idx[i, :min(min_neighbors + 1, N)])
                if int(j) != i
            ]
            js = list(set(js).union(extra))
            n_fallback += 1

        if not js:
            n_empty += 1
            continue

        js = np.asarray(js, dtype=int)
        max_local_n = max(max_local_n, len(js))

        dx = X[js] - X[i]
        t = np.einsum("ij,ij->i", dx, dx) / (h_r * h_r)
        w = kernel(t).astype(float)

        mask = w > 0
        if np.sum(mask) >= min_neighbors:
            js = js[mask]
            dx = dx[mask]
            w = w[mask]
        else:
            w = np.maximum(w, 1e-12)

        y = U[js]

        if debug:
            if not np.isfinite(dx).all():
                dprint(f"[project_tan] WARNING: non-finite dx at i={i}")
            if not np.isfinite(w).all():
                dprint(f"[project_tan] WARNING: non-finite w at i={i}")
            if not np.isfinite(y).all():
                dprint(f"[project_tan] WARNING: non-finite y at i={i}")

        s0 = np.sum(w)
        s1 = dx.T @ w
        S2 = dx.T @ (dx * w[:, None])

        t0 = np.dot(w, y)
        t1 = dx.T @ (w * y)

        Amat = np.empty((d + 1, d + 1), dtype=float)
        Amat[0, 0] = s0
        Amat[0, 1:] = s1
        Amat[1:, 0] = s1
        Amat[1:, 1:] = S2
        Amat += alpha_reg * np.eye(d + 1)

        rhs = np.empty(d + 1, dtype=float)
        rhs[0] = t0
        rhs[1:] = t1

        if debug and debug_cond_samples and i in cond_sample_idx:
            try:
                condA = np.linalg.cond(Amat)
                dprint(
                    f"[project_tan] i={i}, local_n={len(js)}, "
                    f"cond(Amat)={condA:.3e}, s0={s0:.3e}"
                )
            except Exception as e:
                dprint(f"[project_tan] i={i}, cond failed: {e}")

        try:
            beta = np.linalg.solve(Amat, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(Amat, rhs, rcond=None)[0]
            n_lstsq += 1

        if debug and not np.isfinite(beta).all():
            dprint(f"[project_tan] WARNING: non-finite beta at i={i}")

        G[i] = beta[1:]

    dprint("[project_tan] local regression summary:")
    dprint(f"    empty neighborhoods = {n_empty}")
    dprint(f"    fallback-used nodes = {n_fallback}")
    dprint(f"    lstsq fallbacks     = {n_lstsq}")
    dprint(f"    max local n         = {max_local_n}")
    dprint(f"[project_tan] G finite? {np.isfinite(G).all()}")
    if np.isfinite(G).any():
        dprint(f"[project_tan] G min/max = {np.nanmin(G):.6g}, {np.nanmax(G):.6g}")

    dprint("[project_tan] done")
    dprint("=" * 80)

    return W2EuclideanTangent(src_measure=tangent.src_measure, vels=G)


def deterministic_step_to_coupling(current: EmpiricalMeasure, incremental_tangent: W2EuclideanTangent):
    """
    Convert an incremental (locs_src, vels) step into a coupling (Gamma) using current.weights,
    compressing duplicate destinations.

    Returns
    -------
    locs_src : (n_src,d)  (should equal current.locs)
    locs_dst : (n_dst,d)  unique destination locations
    Gamma    : sparse (n_src,n_dst) deterministic coupling with row sums = current.weights
    next_measure : EmpiricalMeasure(locs_dst, dst_weights)
    """
    X = np.asarray(incremental_tangent.src_measure.locs)
    V = np.asarray(incremental_tangent.vels)
    if X.shape != current.locs.shape:
        raise ValueError("incremental_tangent.locs_src must match current.locs (same support ordering/size).")
    if V.shape != X.shape:
        raise ValueError("incremental_tangent.vels must have same shape as locs_src.")

    w = np.asarray(current.weights)
    if w.shape[0] != X.shape[0]:
        raise ValueError("current.weights length must match number of support points.")

    Y = X + V  # (n_src,d)

    # Merge duplicate Y locations
    uniqY, inv = np.unique(Y, axis=0, return_inverse=True)  # inv in {0..n_dst-1}
    n_src = X.shape[0]
    n_dst = uniqY.shape[0]

    # Build as sparse COO — deterministic map so exactly one nonzero per row
    Gamma = sp.csr_matrix((w, (np.arange(n_src), inv)), shape=(n_src, n_dst))

    dst_w = np.bincount(inv, weights=w, minlength=n_dst)
    next_measure = EmpiricalMeasure(uniqY, dst_w)
    return X, uniqY, Gamma, next_measure


def parallel_transport_incremental(tangent: W2EuclideanTangent,
                                  incremental_tangent: W2EuclideanTangent,
                                  tol=1e-8):
    if tangent.coupling is not None:
        raise ValueError("This routine transports velocity tangents only (tangent.coupling must be None).")
    if tangent.vels is None or tangent.src_measure.locs is None:
        raise ValueError("tangent must have (locs_src, vels).")

    current = tangent.src_measure
    # Case A: incremental step given as a coupling
    if incremental_tangent.coupling is not None:
        Gamma = incremental_tangent.coupling  # may be sparse — transport_field_under_coupling handles both
        X_step = np.asarray(incremental_tangent.src_measure.locs)
        Y_step = np.asarray(incremental_tangent.locs_dst)

        # Align tangent field from current support onto X_step
        # (handles permutation + floating drift + unique-compression differences)
        idx = align_by_nearest(current.locs, X_step, tol=tol)
        V_on_step = tangent.vels[idx]

        # Now apply the weighted conditional expectation under Gamma
        vels_dst, dst_w = transport_field_under_coupling(V_on_step, Gamma)

        next_measure = EmpiricalMeasure(Y_step, dst_w)
        new_tangent = W2EuclideanTangent(src_measure=next_measure, vels=vels_dst)
        return new_tangent

    # Case B: incremental step given as a velocity map -> construct implied coupling using current.weights
    elif incremental_tangent.vels is not None:
        # IMPORTANT: deterministic_step_to_coupling currently assumes exact locs_src match.
        # Make it tolerant too by aligning incremental locs_src to current.locs.
        X_inc = np.asarray(incremental_tangent.src_measure.locs)
        idx = align_by_nearest(current.locs, X_inc, tol=tol)

        # reorder current so it matches incremental ordering
        current_reindexed = EmpiricalMeasure(current.locs[idx], current.weights[idx])

        _, locs_dst_step, Gamma, next_measure = deterministic_step_to_coupling(current_reindexed, incremental_tangent)

        vels_dst, _ = transport_field_under_coupling(tangent.vels[idx], Gamma)
        new_tangent = W2EuclideanTangent(src_measure=next_measure, vels=vels_dst)
        return new_tangent

    else:
        raise ValueError("incremental_tangent must be either vel-mode or coupling-mode.")




def barycentric_projection_tangent(tan: W2EuclideanTangent, eps=1e-15):
    """
    Convert a coupling-mode W2EuclideanTangent into a velocity-mode tangent
    by barycentric projection (conditional expectation of Y given X).

    Parameters
    ----------
    tan : W2EuclideanTangent
        Must have coupling, locs_src, locs_dst.
    eps : float
        Small threshold to avoid division by zero for empty rows.

    Returns
    -------
    W2EuclideanTangent
        Velocity-mode tangent supported on tan.locs_src, with vels[i] = E[Y|X=x_i] - x_i.
    """
    if tan.coupling is None or tan.src_measure.locs is None or tan.locs_dst is None:
        raise ValueError("Input must be coupling-mode: need (coupling, locs_src, locs_dst).")

    if sp.issparse(tan.coupling):
        Gamma = tan.coupling.tocsr()
    else:
        Gamma = np.asarray(tan.coupling)        # (n,m)
    X = np.asarray(tan.src_measure.locs)            # (n,d)
    Y = np.asarray(tan.locs_dst)            # (m,d)

    if not sp.issparse(Gamma) and Gamma.ndim != 2:
        raise ValueError("coupling must be 2D")
    n, m = Gamma.shape
    if X.shape[0] != n:
        raise ValueError("coupling rows must match locs_src")
    if Y.shape[0] != m:
        raise ValueError("coupling cols must match locs_dst")
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("locs_src and locs_dst must be 2D arrays (n,d) and (m,d).")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("locs_src and locs_dst must have the same ambient dimension.")

    row_mass = np.asarray(Gamma.sum(axis=1)).ravel()   # (n,)

    # Compute barycenters: (n,d) = (n,m) @ (m,d) / row_mass
    bary = np.asarray(Gamma @ Y)            # (n,d)
    vels = np.zeros_like(X, dtype=bary.dtype)

    mask = row_mass > eps
    vels[mask] = bary[mask] / row_mass[mask, None] - X[mask]

    # If some rows have ~0 mass, vels stays 0 there (safe fallback).
    return W2EuclideanTangent(src_measure=tan.src_measure, vels=vels)

def wasserstein_expmap_vel(base: EmpiricalMeasure, tangent: "W2EuclideanTangent", tol=1e-8):
    Xb = np.asarray(base.locs)
    wb = np.asarray(base.weights)

    Xt = np.asarray(tangent.src_measure.locs)
    Vt = np.asarray(tangent.vels)

    if Xt.shape != Vt.shape:
        raise ValueError("tangent.src_measure.locs and tangent.vels must have the same shape.")
    if Xb.shape[1] != Xt.shape[1]:
        raise ValueError("Ambient dimension mismatch.")
    if wb.shape[0] != Xb.shape[0]:
        raise ValueError("base.weights length must match base.locs.")

    # tolerant matching to pull the right weights (since Xt may be permuted / drifted)
    tree = cKDTree(Xb)
    dists, idx = tree.query(Xt, k=1)
    if np.max(dists) > tol:
        bad = int(np.argmax(dists))
        raise ValueError(
            f"Support of tangent not contained in base within tol={tol}. "
            f"Max dist={np.max(dists):.3e} at tangent index {bad}."
        )

    w_tan = wb[idx]

    new_locs = Xt + Vt
    unique_locs, inv = np.unique(new_locs, axis=0, return_inverse=True)
    combined_weights = np.bincount(inv, weights=w_tan, minlength=unique_locs.shape[0])

    return EmpiricalMeasure(unique_locs, combined_weights)



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
