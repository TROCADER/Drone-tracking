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
    kf = KalmanFilter(dim_x=8, dim_z=4)

    # State: [x, x', y, y', w, w', h, h']
    kf.x = np.array(
        [init_pos[0], 0.0, init_pos[1], 0.0, init_pos[2], 0.0, init_pos[3], 0.0]
    )

    # State transition matrix F
    kf.F = np.array(
        [
            [1, dt, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, dt, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, dt, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )

    # Measurement function H
    kf.H = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ]
    )

    # Initial state uncertainty (P), measurement noise (R), and process noise (Q)
    kf.P *= state_uncertainty
    kf.R = block_diag(
        measurement_noise,
        measurement_noise,
        3 * measurement_noise,
        3 * measurement_noise,
    )  # Increase to smooth out noisy detections

    q_1d = Q_discrete_white_noise(dim=2, dt=dt, var=process_noise, order_by_dim=True)
    kf.Q = block_diag(q_1d, q_1d, q_1d, q_1d)

    return kf


import numpy as np
import torch
import torchvision.ops as ops
from scipy.optimize import linear_sum_assignment


def calculate_cost_matrix(
    boxes1_xywh: torch.Tensor, boxes2_xywh: torch.Tensor
) -> np.ndarray:
    if len(boxes1_xywh) == 0 or len(boxes2_xywh) == 0:
        return np.empty((len(boxes1_xywh), len(boxes2_xywh)))

    # Convert xywh to xyxy just for the torchvision GIoU calculation
    boxes1_xyxy = ops.box_convert(boxes1_xywh, in_fmt="xywh", out_fmt="xyxy")
    boxes2_xyxy = ops.box_convert(boxes2_xywh, in_fmt="xywh", out_fmt="xyxy")

    matrix = 1.0 - ops.complete_box_iou(boxes1_xyxy, boxes2_xyxy)
    return matrix.numpy()
    # Safely convert to numpy and strip any residual NaNs<


def _extract_predictions(self, track_ids: list) -> torch.Tensor:
    preds = [self.tracks[tid]["kf"].x[[0, 2, 4, 6]].reshape(-1) for tid in track_ids]
    preds_tensor = torch.tensor(preds, dtype=torch.float32)

    if len(preds_tensor) > 0:
        # Clamp width (index 2) and height (index 3) to be strictly positive.
        # This prevents ops.box_convert from generating invalid xyxy coordinates.
        preds_tensor[:, 2] = torch.maximum(preds_tensor[:, 2], torch.tensor(1e-4))
        preds_tensor[:, 3] = torch.maximum(preds_tensor[:, 3], torch.tensor(1e-4))

    return preds_tensor


class Tracker:
    def __init__(
        self,
        state_uncertainty: float,
        measurement_noise: float,
        process_noise: float,
        dt: float = 1.0,
        max_distance: float = 1.0,
        max_age: int = 5,
    ):
        self.tracks = {}
        self.next_id = 0
        self.state_uncertainty = state_uncertainty
        self.measurement_noise = measurement_noise
        self.process_noise = process_noise
        self.dt = dt
        self.max_distance = max_distance
        self.max_age = max_age

    def _init_track(self, bbox: np.ndarray) -> None:
        self.tracks[self.next_id] = {
            "kf": create_kf(
                bbox,
                self.state_uncertainty,
                self.measurement_noise,
                self.process_noise,
                self.dt,
            ),
            "age": 0,
        }
        self.next_id += 1

    def _extract_predictions(self, track_ids: list) -> torch.Tensor:
        preds = np.array(
            [self.tracks[tid]["kf"].x[[0, 2, 4, 6]].reshape(-1) for tid in track_ids]
        )
        return torch.tensor(preds, dtype=torch.float32)

    def _associate_detections(self, track_ids: list, det_bboxes: torch.Tensor):
        if len(track_ids) == 0 or len(det_bboxes) == 0:
            return [], set(track_ids), set(range(len(det_bboxes)))

        predicted_boxes = self._extract_predictions(track_ids)
        cost_matrix = calculate_cost_matrix(predicted_boxes, det_bboxes)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        unassigned_tracks = set(track_ids)
        unassigned_dets = set(range(len(det_bboxes)))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= self.max_distance:
                tid = track_ids[r]
                matches.append((tid, c))
                unassigned_tracks.remove(tid)
                unassigned_dets.remove(c)

        return matches, unassigned_tracks, unassigned_dets

    def update(self, detections: torch.Tensor | np.ndarray) -> dict:
        if not isinstance(detections, torch.Tensor):
            detections = torch.tensor(detections, dtype=torch.float32)

        det_bboxes = detections[:, :4] if len(detections) > 0 else torch.empty((0, 4))

        for track in self.tracks.values():
            track["kf"].predict()

        track_ids = list(self.tracks.keys())

        matches, unassigned_tracks, unassigned_dets = self._associate_detections(
            track_ids, det_bboxes
        )

        det_bboxes_np = det_bboxes.cpu().numpy()

        for tid, det_idx in matches:
            self.tracks[tid]["kf"].update(det_bboxes_np[det_idx])
            self.tracks[tid]["age"] = 0

        for tid in unassigned_tracks:
            self.tracks[tid]["age"] += 1

        for det_idx in unassigned_dets:
            self._init_track(det_bboxes_np[det_idx])

        self._purge_old_tracks()
        return self._get_active_states()

    def _purge_old_tracks(self) -> None:
        self.tracks = {
            tid: track
            for tid, track in self.tracks.items()
            if track["age"] <= self.max_age
        }

    def _get_active_states(self) -> dict:
        return {
            tid: track["kf"].x[[0, 2, 4, 6]].reshape(-1)
            for tid, track in self.tracks.items()
        }
