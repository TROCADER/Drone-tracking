import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist, pdist, squareform


def weighted_mean_shift(
    points: np.ndarray,
    weights: np.ndarray,
    bandwidth: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-4,
    cluster_threshold: float | None = None,
):
    """Vectorized Weighted Mean Shift mode-seeking.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, D) containing point coordinates.
    weights : np.ndarray
        Array of shape (N,) with certainty/confidence weights.
    bandwidth : float
        Gaussian kernel standard deviation (h) for spatial smoothing.
    max_iter : int
        Maximum gradient ascent steps allowed per point.
    tol : float
        Convergence tolerance on spatial shift distance.
    cluster_threshold : float, optional
        Maximum distance between converged modes to merge them.
        Defaults to `bandwidth / 2.0`.

    Returns
    -------
    cluster_centers : np.ndarray
        Array of shape (K, D) with merged cluster centers.
    cluster_weights : np.ndarray
        Array of shape (K,) with total aggregated weights per cluster.
    labels : np.ndarray
        Array of shape (N,) containing cluster assignments (0 to K-1).

    """
    X = np.asarray(points, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1, 1)

    if cluster_threshold is None:
        cluster_threshold = bandwidth / 2.0

    # Shift each point along the density gradient toward local modes
    shifted_points = np.copy(X)
    for i, p in enumerate(shifted_points):
        p_curr = p[np.newaxis, :]

        for _ in range(max_iter):
            dists = cdist(p_curr, X)[0]
            kernel_w = w * np.exp(-0.5 * (dists / bandwidth) ** 2)[:, np.newaxis]
            weight_sum = np.sum(kernel_w)

            if weight_sum < 1e-12:
                break

            p_next = np.sum(kernel_w * X, axis=0, keepdims=True) / weight_sum

            if np.linalg.norm(p_next - p_curr) < tol:
                p_curr = p_next
                break

            p_curr = p_next

        shifted_points[i] = p_curr[0]

    # Connect modes within threshold distance and find connected components
    condensed_dists = pdist(shifted_points)
    adj_matrix = squareform(condensed_dists) < cluster_threshold
    num_clusters, labels = connected_components(csr_matrix(adj_matrix), directed=False)

    # Compute final weighted centroids and aggregate total cluster weights
    refined_centers = np.zeros((num_clusters, X.shape[1]))
    aggregated_weights = np.zeros(num_clusters)

    for k in range(num_clusters):
        mask = labels == k
        member_points = X[mask]
        member_weights = w[mask]

        total_weight = np.sum(member_weights)
        refined_centers[k] = (
            np.sum(member_points * member_weights, axis=0) / total_weight
        )
        aggregated_weights[k] = total_weight

    return refined_centers, aggregated_weights, labels


def create_kf(
    init_pos: np.ndarray,
    state_uncertainty,
    measurement_noise,
    process_noise,
    dt: float = 1.0,
) -> KalmanFilter:
    kf = KalmanFilter(dim_x=6, dim_z=2)

    # State: [x, y, vx, vy, ax, ay]
    kf.x = np.array([init_pos[0], init_pos[1], 0.0, 0.0, 0.0, 0.0])
    dt2 = 0.5 * (dt**2)

    # State transition matrix F
    kf.F = np.array(
        [
            [1, 0, dt, 0, dt2, 0],
            [0, 1, 0, dt, 0, dt2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )

    # Measurement function H (still measuring 2D position [x, y])
    kf.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])

    # Initial state uncertainty (P), measurement noise (R), and process noise (Q)
    kf.P *= state_uncertainty
    kf.R *= measurement_noise  # Increase to smooth out noisy detections

    q_1d = Q_discrete_white_noise(dim=3, dt=dt, var=process_noise)
    kf.Q = block_diag(q_1d, q_1d)

    return kf


class KalmanPointTracker:

    def __init__(
        self,
        state_uncertainty,
        measurement_noise,
        process_noise,
        max_distance: float = 50.0,
        max_age: int = 5,
        dt: float = 1.0,
    ):
        self.state_uncertainty = state_uncertainty
        self.measurement_noise = measurement_noise
        self.process_noise = process_noise
        self.max_distance = max_distance
        self.max_age = max_age
        self.dt = dt
        self.next_id = 0
        self.tracks = {}

    def update(self, detections: np.ndarray) -> dict[int, dict]:
        detections = np.asarray(detections, dtype=np.float64)

        # Predict future position for active tracks
        for track in self.tracks.values():
            track["kf"].predict()

        track_ids = list(self.tracks.keys())

        if len(self.tracks) == 0:
            for det in detections:
                self.tracks[self.next_id] = {
                    "kf": create_kf(
                        det,
                        self.state_uncertainty,
                        self.measurement_noise,
                        self.process_noise,
                        self.dt,
                    ),
                    "age": 0,
                }
                self.next_id += 1
            return self._get_active_states()

        if len(detections) == 0:
            for tid in track_ids:
                self.tracks[tid]["age"] += 1
            self._purge_old_tracks()
            return self._get_active_states()

        # Cost matrix matching filter predictions against new detections
        predicted_positions = np.array(
            [self.tracks[tid]["kf"].x[:2] for tid in track_ids]
        )
        cost_matrix = cdist(predicted_positions, detections)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_tracks = set()
        assigned_dets = set()

        # Update matched tracks
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= self.max_distance:
                tid = track_ids[r]
                self.tracks[tid]["kf"].update(detections[c])
                self.tracks[tid]["age"] = 0
                assigned_tracks.add(tid)
                assigned_dets.add(c)

        # Increment missing age
        for r, tid in enumerate(track_ids):
            if tid not in assigned_tracks:
                self.tracks[tid]["age"] += 1

        # Spawn new tracks for unassigned detections
        for c in range(len(detections)):
            if c not in assigned_dets:
                self.tracks[self.next_id] = {
                    "kf": create_kf(
                        detections[c],
                        self.state_uncertainty,
                        self.measurement_noise,
                        self.process_noise,
                        self.dt,
                    ),
                    "age": 0,
                }
                self.next_id += 1

        self._purge_old_tracks()
        return self._get_active_states()

    def _purge_old_tracks(self):
        dead_ids = [tid for tid, t in self.tracks.items() if t["age"] > self.max_age]
        for tid in dead_ids:
            del self.tracks[tid]

    def _get_active_states(self) -> dict[int, dict]:
        active_states = {}
        for tid, t in self.tracks.items():
            if t["age"] == 0:
                state = t["kf"].x
                active_states[tid] = {
                    "pos": np.array([state[0], state[1]]),
                    "vel": np.array([state[2], state[3]]),
                }
        return active_states
