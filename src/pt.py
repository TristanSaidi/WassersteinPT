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
        self.weights = weights


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

    P = np.asarray(tangent.coupling)
    X = np.asarray(tangent.src_measure.locs)
    Y = np.asarray(tangent.locs_dst)

    if P.ndim != 2:
        raise ValueError(f"coupling must be 2D, got {P.shape}")
    n, m = P.shape
    if X.shape[0] != n:
        raise ValueError(f"locs_src has {X.shape[0]} points but coupling has {n} rows")
    if Y.shape[0] != m:
        raise ValueError(f"locs_dst has {Y.shape[0]} points but coupling has {m} cols")

    # Support of the original coupling
    if eps > 0:
        ii, jj = np.nonzero(P > eps)
    else:
        ii, jj = np.nonzero(P)

    if ii.size == 0:
        # no mass - return empty
        d = X.shape[1]
        return W2EuclideanTangent(
            locs_src=np.zeros((0, d)),
            locs_dst=np.zeros((0, d)),
            coupling=np.zeros((0, 0)),
        )

    masses = P[ii, jj]                          # (K,)
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

    # Build the compressed coupling: sum masses for identical (inv0, inv1) pairs
    coup01 = np.zeros((N0, N1), dtype=masses.dtype)
    np.add.at(coup01, (inv0, inv1), masses)

    new_weights_0 = coup01.sum(axis=1)  # (N0,)
    new_src_measure = EmpiricalMeasure(locs=uniq0, weights=new_weights_0)

    return W2EuclideanTangent(src_measure=new_src_measure, locs_dst=uniq1, coupling=coup01)

def cost_euclidean(src, dst):
    # compute the cost matrix between src and dst in the Euclidean space
    return np.linalg.norm(src[:, None, :] - dst[None, :, :], axis=-1)

# general case
def quadratic_coupling(src, dst, cost):
    # compute the optimal transport map from src to dst with cost function cost
    cost_sq = cost ** 2
    gamma = ot.emd(src, dst, M=cost_sq)
    return gamma

def coupling_to_tan(coupling, src: EmpiricalMeasure, locs_dst):
    # if coupling supported on a map, return vector field. Otherwise, return coupling
    locs_src = src.locs
    # A coupling is supported on a map if each source location sends mass to at most one target
    row_nnz = np.count_nonzero(coupling, axis=1)
    if np.all(row_nnz <= 1):
        # build velocities for each source location
        dst_idx = np.argmax(coupling, axis=1)
        vels = locs_dst[dst_idx] - locs_src
        return W2EuclideanTangent(src_measure=src, vels=vels)
    tangent = W2EuclideanTangent(src_measure=src, locs_dst=locs_dst, coupling=coupling)
    return tangent

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
    dst_weights = coupling.sum(axis=0)
    assert dst_weights.shape[0] == locs_dst.shape[0], "The number of destination locations must match the number of columns in the coupling matrix."
    return EmpiricalMeasure(locs_dst, dst_weights)

def align_by_nearest(X_from, X_to, tol=1e-8):
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
    coupling : (n_src, n_dst) with nonnegative entries

    Returns
    -------
    vels_dst : (n_dst, d), where vels_dst[j] = sum_i coupling[i,j] vels_src[i] / sum_i coupling[i,j]
    dst_weights : (n_dst,), column sums of coupling
    """
    Gamma = np.asarray(coupling)
    V = np.asarray(vels_src)

    if Gamma.ndim != 2:
        raise ValueError("coupling must be 2D")
    if V.ndim != 2:
        raise ValueError("vels_src must be 2D (n_src, d)")
    if Gamma.shape[0] != V.shape[0]:
        raise ValueError("coupling rows must match vels_src length")

    dst_weights = Gamma.sum(axis=0)  # (n_dst,)
    # Numerator: (n_dst, d) = (n_dst, n_src) @ (n_src, d)
    num = Gamma.T @ V

    vels_dst = np.zeros((Gamma.shape[1], V.shape[1]), dtype=V.dtype)
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


    def parallel_transport(self, src: EmpiricalMeasure, dst: EmpiricalMeasure, n=1, project=False, tol=1e-8):
        if self.coupling is not None:
            print("Warning: coupling is being converted to velocity mode via barycentric projection for parallel transport.")
            tangent = barycentric_projection_tangent(self)
        else:
            tangent = self
        stepsize = 1.0 / n
        geodesic_tangent = wasserstein_logmap(src, dst)

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
                tangent = project_tan(tangent) # project back onto gradient fields

        return tangent


def _default_compact_kernel(t):
    """
    Compactly supported kernel on R_+ with support [0,1].
    Input: t = ||x_i-x_j||^2 / h^2.
    """
    t = np.asarray(t, dtype=float)
    out = 1.0 - t
    out[out < 0.0] = 0.0
    return out

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


def project_tan(
    tangent: "W2EuclideanTangent",
    h: float = None,
    h_r: float = None,
    kernel=_default_compact_kernel,
    alpha_reg: float = 0.0,
    min_neighbors: int = 15,     # <---- new
    k_bandwidth: int = 30,      # <---- used when h/h_r are None
):
    """
    Weighted Helmholtz-Hodge projection aligned with the manuscript, with the
    extra guarantee that each node has at least `min_neighbors` neighbors.

    Graph weights: w_ij = (1/h^d) K(||xi-xj||^2 / h^2)
    Regression weights: K(||xi-x||^2 / h_r^2) (no 1/h_r^d needed; cancels)
    """

    # coupling -> velocity if needed
    if getattr(tangent, "coupling", None) is not None:
        tangent = barycentric_projection_tangent(tangent)

    X = np.asarray(tangent.src_measure.locs, float)
    V = np.asarray(tangent.vels, float)
    N, d = X.shape

    if N <= min_neighbors:
        # not enough points to guarantee min_neighbors; return as-is
        return tangent

    tree = cKDTree(X)

    # ---------------------------------------------------------
    # Choose bandwidths via kNN if not provided
    # ---------------------------------------------------------
    if h is None or h_r is None:
        k0 = min(k_bandwidth, N - 1)
        dists, _ = tree.query(X, k=k0 + 1)  # includes self at column 0
        h_knn = float(np.mean(dists[:, k0]))
        if h is None:
            h = h_knn
        if h_r is None:
            h_r = h_knn

    if h <= 0 or h_r <= 0:
        raise ValueError("h and h_r must be positive.")

    # ---------------------------------------------------------
    # Step 1: Build graph edges, ensuring >= min_neighbors per node
    # ---------------------------------------------------------
    # Start with radius neighbors (manuscript-style)
    neigh = tree.query_ball_point(X, r=h)

    # Track neighbor sets (undirected)
    nbrs = [set(int(j) for j in neigh[i] if j != i) for i in range(N)]

    # Augment with kNN edges where needed
    k_need = min_neighbors
    for i in range(N):
        if len(nbrs[i]) >= k_need:
            continue
        # query k_need nearest (plus self) and add them
        dists_i, idx_i = tree.query(X[i], k=min(k_need + 1, N))
        idx_i = [int(j) for j in np.atleast_1d(idx_i) if int(j) != i]
        for j in idx_i:
            nbrs[i].add(j)
            nbrs[j].add(i)

    # Build unique undirected edges i<j from nbrs
    src, dst, weights = [], [], []
    for i in range(N):
        for j in nbrs[i]:
            if j <= i:
                continue
            dx = X[j] - X[i]
            t = float(np.dot(dx, dx)) / (h * h)
            kij = float(kernel(np.array([t]))[0])
            if kij > 0.0:
                src.append(i)
                dst.append(j)
                weights.append(kij / (h ** d))
            else:
                # If the kernel is compact and t>1, weight is 0.
                # But we *still* want the connectivity guarantee; in that case,
                # add a tiny weight so the edge exists numerically.
                # This preserves the "at least 5 neighbors" requirement.
                # (If you prefer: increase h instead of adding tiny weight.)
                src.append(i)
                dst.append(j)
                weights.append(1e-12 / (h ** d))

    src = np.asarray(src, dtype=int)
    dst = np.asarray(dst, dtype=int)
    weights = np.asarray(weights, dtype=float)
    M = len(src)

    if M == 0:
        return tangent

    # ---------------------------------------------------------
    # Step 2: Edge actions
    # ---------------------------------------------------------
    DX = X[dst] - X[src]
    a_edge = 0.5 * np.einsum("ij,ij->i", V[src] + V[dst], DX)

    # ---------------------------------------------------------
    # Step 3: Potential fit via weighted Laplacian
    # ---------------------------------------------------------
    rows = np.arange(M)
    data = np.concatenate([-np.ones(M), np.ones(M)])
    cols = np.concatenate([src, dst])
    rows2 = np.concatenate([rows, rows])

    delta = sp.coo_matrix((data, (rows2, cols)), shape=(M, N)).tocsr()
    Wedge = sp.diags(weights, format="csr")

    L = (delta.T @ Wedge @ delta).tocsr()
    g = (delta.T @ (weights * a_edge))
    g = np.asarray(g, dtype=float).reshape(-1)

    try:
        U = _solve_potential_with_gauge(L, g, ridge=0.0)
    except Exception:
        U = _solve_potential_with_gauge(L, g, ridge=1e-10)

    # ---------------------------------------------------------
    # Step 4: Local linear gradient extraction with >= min_neighbors
    # ---------------------------------------------------------
    G = np.zeros((N, d), dtype=float)

    # Start with radius regression neighbors
    neigh_r = tree.query_ball_point(X, r=h_r)

    for i in range(N):
        js = [int(j) for j in neigh_r[i] if int(j) != i]

        # Ensure at least min_neighbors by kNN fallback
        if len(js) < min_neighbors:
            d_i, idx_i = tree.query(X[i], k=min(min_neighbors + 1, N))
            idx_i = [int(j) for j in np.atleast_1d(idx_i) if int(j) != i]
            js = list(set(js).union(idx_i))

        js = np.asarray(js, dtype=int)
        if js.size == 0:
            continue

        DX_loc = X[js] - X[i]
        t = np.sum(DX_loc * DX_loc, axis=1) / (h_r * h_r)
        K_loc = kernel(t)

        # Keep positives; if too few survive, fall back to kNN weights without truncation
        mask = K_loc > 0
        if np.sum(mask) < min_neighbors:
            # Use all js but with a soft weight floor for numerical stability
            K_loc = np.maximum(K_loc, 1e-12)
        else:
            js = js[mask]
            DX_loc = DX_loc[mask]
            K_loc = K_loc[mask]

        y = U[js]

        B = np.column_stack([np.ones(len(js)), DX_loc])  # (m, 1+d)
        w = K_loc.astype(float)
        # Build BtWB without forming dense diag:
        BtWB = B.T @ (B * w[:, None])
        BtWy = B.T @ (w * y)

        A = BtWB + alpha_reg * np.eye(d + 1)
        try:
            beta = np.linalg.solve(A, BtWy)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(A, BtWy, rcond=None)[0]

        G[i] = beta[1:]

    return W2EuclideanTangent(src_measure=tangent.src_measure, vels=G)

def deterministic_step_to_coupling(current: EmpiricalMeasure, incremental_tangent: W2EuclideanTangent):
    """
    Convert an incremental (locs_src, vels) step into a coupling (Gamma) using current.weights,
    compressing duplicate destinations.

    Returns
    -------
    locs_src : (n_src,d)  (should equal current.locs)
    locs_dst : (n_dst,d)  unique destination locations
    Gamma    : (n_src,n_dst) deterministic coupling with row sums = current.weights
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

    Gamma = np.zeros((n_src, n_dst), dtype=w.dtype)
    Gamma[np.arange(n_src), inv] = w

    dst_w = Gamma.sum(axis=0)
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
        Gamma = np.asarray(incremental_tangent.coupling)
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

    Gamma = np.asarray(tan.coupling)        # (n,m)
    X = np.asarray(tan.src_measure.locs)            # (n,d)
    Y = np.asarray(tan.locs_dst)            # (m,d)

    if Gamma.ndim != 2:
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

    row_mass = Gamma.sum(axis=1)            # (n,)

    # Compute barycenters: (n,d) = (n,m) @ (m,d) / row_mass
    bary = Gamma @ Y                        # (n,d)
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
