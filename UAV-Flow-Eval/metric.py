import json
import numpy as np
import os
from scipy.spatial.distance import cdist
import sys


def _print_table(headers, rows, align=None):
    """Print a clean ASCII table.

    Args:
        headers: List of header strings.
        rows: List of rows (iterables of cell strings).
        align: Optional list of 'l' or 'r' for left/right alignment per column.
    """
    headers = [str(h) for h in headers]
    str_rows = [["" if c is None else str(c) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for r in str_rows:
        for i, c in enumerate(r):
            if i >= len(widths):
                widths.append(len(c))
            else:
                widths[i] = max(widths[i], len(c))
    if align is None:
        align = ['l'] * len(widths)
    def fmt_cell(i, s):
        if align[i] == 'r':
            return s.rjust(widths[i])
        return s.ljust(widths[i])
    sep = "+" + "+".join(["-" * (w + 2) for w in widths]) + "+"
    # header
    print(sep)
    print("| " + " | ".join(fmt_cell(i, headers[i]) for i in range(len(widths))) + " |")
    print(sep)
    # rows
    for r in str_rows:
        print("| " + " | ".join(fmt_cell(i, r[i] if i < len(r) else "") for i in range(len(widths))) + " |")
    print(sep)


def get_gt_states_from_rule_log(gt_path):
    """Load preprocessed ground-truth states from a rule-based log JSON file.

    Args:
        gt_path: Path to the ground-truth JSON file.
    Returns:
        A list of 6D states under the key 'reference_path_preprocessed'.
    """
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['reference_path_preprocessed']


def get_sampled_state9d_from_model_rule(model_path, step=5, zero_pos=False):
    """Sample 9D states from a model trajectory log.

    The output vector per sample is [x, y, z, sin(roll), sin(yaw), sin(pitch),
    cos(roll), cos(yaw), cos(pitch)], where position can be zeroed when zero_pos
    is True. Using both sin and cos captures rotation direction (cos alone is
    symmetric and cannot distinguish left from right turns).

    Args:
        model_path: Path to the model trajectory JSON.
        step: Sampling stride over frames.
        zero_pos: If True, set positions to zeros; otherwise use positions/100.
    Returns:
        A list of np.ndarray vectors (length 9).
    """
    with open(model_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    result = []
    for idx, item in enumerate(data):
        if idx % step == 0:
            if zero_pos:
                pos = np.zeros(3)
            else:
                pos = np.array(item['state'][0]) / 100
            rot = np.array(item['state'][1])
            rot_rad = np.deg2rad(rot)
            rot_sin = np.sin(rot_rad)
            rot_cos = np.cos(rot_rad)
            vec = np.concatenate([pos, rot_sin, rot_cos])
            result.append(vec)
    return result


def get_sampled_state9d_from_gt_rule(gt_path, step=5, max_points=20, zero_pos=False):
    """Sample 9D states from ground-truth states.

    Args:
        gt_path: Path to the ground-truth JSON file.
        step: Sampling stride over frames.
        max_points: Maximum number of sampled points to return.
        zero_pos: If True, set positions to zeros; otherwise use positions/100.
    Returns:
        A list of np.ndarray vectors (length 9), truncated to max_points.
    """
    states = get_gt_states_from_rule_log(gt_path)
    result = []
    for idx, item in enumerate(states):
        if idx % step == 0:
            if zero_pos:
                pos = np.zeros(3)
            else:
                pos = np.array(item[:3]) / 100
            rot = np.array(item[3:6])
            rot_rad = np.deg2rad(rot)
            rot_sin = np.sin(rot_rad)
            rot_cos = np.cos(rot_rad)
            vec = np.concatenate([pos, rot_sin, rot_cos])
            result.append(vec)
    return result[:max_points]


def dtw_distance(vecs1, vecs2):
    """Compute DTW (Dynamic Time Warping) distance between two sequences.

    Args:
        vecs1: Sequence of vectors (list of np.ndarray) for path 1.
        vecs2: Sequence of vectors (list of np.ndarray) for path 2.
    Returns:
        A float DTW distance, or None if either sequence is empty.
    """
    if len(vecs1) == 0 or len(vecs2) == 0:
        return None
    dist_matrix = cdist(vecs1, vecs2, metric='euclidean')
    n, m = dist_matrix.shape
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = dist_matrix[i-1, j-1]
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    return dtw[n, m]


def path_length(points):
    """Compute the polyline length for a sequence of points.

    Args:
        points: List of np.ndarray points.
    Returns:
        The total Euclidean length as float.
    """
    if len(points) < 2:
        return 0
    length = 0
    for i in range(1, len(points)):
        length += np.linalg.norm(points[i] - points[i-1])
    return length


def ndtw(dtw_dist, gt_len, eta=1):
    """Compute normalized DTW (nDTW) score.

    nDTW = exp(- DTW / (eta * L_gt)).

    Args:
        dtw_dist: DTW distance value.
        gt_len: Ground-truth path length.
        eta: Normalization hyper-parameter (default 1).
    Returns:
        nDTW score in [0,1], or None if inputs are invalid.
    """
    if dtw_dist is None or gt_len == 0:
        return None
    return np.exp(-dtw_dist / (eta * gt_len))


def endpoint_error(vecs1, vecs2):
    """Compute Euclidean distance between the last points of two sequences.

    Uses only the position components [x, y, z] (first 3 elements).

    Args:
        vecs1: First sequence of vectors (list of np.ndarray).
        vecs2: Second sequence of vectors (list of np.ndarray).
    Returns:
        Float endpoint distance, or None if either sequence is empty.
    """
    if len(vecs1) == 0 or len(vecs2) == 0:
        return None
    return float(np.linalg.norm(vecs1[-1][:3] - vecs2[-1][:3]))


def split_pos_ori(vecs):
    """Split 9D vectors into position-only (3D) and orientation-only (6D) lists.

    Args:
        vecs: List of 9D np.ndarray [x, y, z, sin_r, sin_y, sin_p, cos_r, cos_y, cos_p].
    Returns:
        Tuple of (pos_vecs, ori_vecs) where pos is [x,y,z] and ori is [sin..., cos...].
    """
    pos = [v[:3] for v in vecs]
    ori = [v[3:] for v in vecs]
    return pos, ori


def check_success(class_name, gt_vecs, model_vecs, gt_raw_states):
    """Determine if a task was successful based on class-specific geometric heuristics.

    Uses the GT endpoint as the target. For position-based tasks, checks if the
    model's final position is within a threshold of the GT's final position.
    For rotation-based tasks, checks final yaw error.

    Args:
        class_name: Task class (e.g. "Turn", "Approach").
        gt_vecs: GT 9D vectors (sampled, scaled).
        model_vecs: Model 9D vectors (sampled, scaled).
        gt_raw_states: Raw GT reference_path_preprocessed (unscaled, degrees).
    Returns:
        True if successful, False otherwise, or None if not enough data.
    """
    if len(gt_vecs) == 0 or len(model_vecs) == 0:
        return None

    # Position endpoint error (in scaled units, /100)
    epe = np.linalg.norm(gt_vecs[-1][:3] - model_vecs[-1][:3])

    # Yaw endpoint error (degrees, shortest path)
    if len(gt_raw_states) > 0:
        gt_final_yaw = gt_raw_states[-1][4]  # yaw in degrees
        # Model stores [roll, yaw, pitch] — yaw is index 1 of state[1]
        # but in 9D vecs it's encoded as sin/cos, so use raw data
        # We need to get model's final yaw from the raw model data instead
        # Use the sin/cos from the 9D vec to recover the angle
        m_sin_yaw = model_vecs[-1][4]  # sin(yaw)
        m_cos_yaw = model_vecs[-1][7]  # cos(yaw)
        model_final_yaw = np.degrees(np.arctan2(m_sin_yaw, m_cos_yaw))
        yaw_err = abs((model_final_yaw - gt_final_yaw + 180) % 360 - 180)
    else:
        yaw_err = 180  # worst case

    # Per-class success criteria
    # Position thresholds are in scaled units (/100)
    # Calibrated from GT displacement ranges per class
    if class_name in ["Turn", "Rotate"]:
        # Rotation-only: yaw within 30 degrees of GT final yaw
        return yaw_err < 30.0
    elif class_name == "Move":
        # Short displacement (~0.9 scaled), threshold = 50% of typical displacement
        return epe < 0.5
    elif class_name == "Shift":
        # Medium displacement (~5 scaled)
        return epe < 2.5
    elif class_name == "Surround":
        # Small endpoint displacement (~0.15), circular path
        return epe < 1.0
    elif class_name == "Ascend/Descend":
        # Vertical movement (~7 scaled)
        return epe < 3.0
    elif class_name == "Approach":
        # Move toward target (~7 scaled)
        return epe < 3.0
    elif class_name == "Retreat":
        # Move away (~4.5 scaled)
        return epe < 2.5
    elif class_name == "Pass":
        # Long displacement (~13 scaled)
        return epe < 5.0
    elif class_name == "Land":
        # Descend to ground (~9 scaled)
        return epe < 3.0
    else:
        # Fallback: generic position threshold
        return epe < 3.0


def evaluate_by_classification(classified_json_path, model_dir, gt_rule_dir, default_step=5):
    """Evaluate trajectories grouped by classification.

    Computes per-class:
      Primary:    SR (success rate), nDTW (combined)
      Supporting: nDTW (position-only), nDTW (orientation-only), EPE

    Args:
        classified_json_path: Path to a JSON mapping class_name -> [file names].
        model_dir: Directory containing model JSON files.
        gt_rule_dir: Directory containing ground-truth JSON files.
        default_step: Default sampling stride if not class-specific.
    """
    with open(classified_json_path, 'r', encoding='utf-8') as f:
        class_dict = json.load(f)
    all_sr = []
    all_ndtw = []
    all_ndtw_pos = []
    all_ndtw_ori = []
    all_epe = []
    table_rows = []
    detail_rows = []

    for class_name, file_list in class_dict.items():
        sr_results = []
        ndtw_results = []
        ndtw_pos_results = []
        ndtw_ori_results = []
        epe_results = []
        zero_pos = class_name in ["Turn", "Rotate"]
        if class_name in ["Turn", "Move"]:
            step = 2
        else:
            step = default_step
        num_valid = 0
        for file_name in file_list:
            gt_path = os.path.join(gt_rule_dir, file_name)
            model_path = os.path.join(model_dir, file_name)
            if not os.path.exists(gt_path) or not os.path.exists(model_path):
                continue
            model_vecs = get_sampled_state9d_from_model_rule(model_path, step, zero_pos=zero_pos)
            gt_vecs = get_sampled_state9d_from_gt_rule(gt_path, step, max_points=20, zero_pos=zero_pos)

            # Load raw GT states for yaw-based SR checks
            gt_raw = get_gt_states_from_rule_log(gt_path)

            # Combined 9D nDTW
            dtw_dist = dtw_distance(gt_vecs, model_vecs)
            gt_len = path_length(gt_vecs)
            score = ndtw(dtw_dist, gt_len, eta=1)
            if score is not None:
                ndtw_results.append(score)
                all_ndtw.append(score)

            # Position-only nDTW (3D)
            gt_pos, gt_ori = split_pos_ori(gt_vecs)
            m_pos, m_ori = split_pos_ori(model_vecs)
            dtw_pos = dtw_distance(gt_pos, m_pos)
            gt_pos_len = path_length(gt_pos)
            score_pos = ndtw(dtw_pos, gt_pos_len, eta=1)
            if score_pos is not None:
                ndtw_pos_results.append(score_pos)
                all_ndtw_pos.append(score_pos)

            # Orientation-only nDTW (6D)
            dtw_ori = dtw_distance(gt_ori, m_ori)
            gt_ori_len = path_length(gt_ori)
            score_ori = ndtw(dtw_ori, gt_ori_len, eta=1)
            if score_ori is not None:
                ndtw_ori_results.append(score_ori)
                all_ndtw_ori.append(score_ori)

            # Endpoint error
            epe = endpoint_error(gt_vecs, model_vecs)
            if epe is not None:
                epe_results.append(epe)
                all_epe.append(epe)

            # Success rate
            success = check_success(class_name, gt_vecs, model_vecs, gt_raw)
            if success is not None:
                sr_results.append(1.0 if success else 0.0)
                all_sr.append(1.0 if success else 0.0)

            num_valid += 1

        def _fmt(vals):
            return "{:.4f}".format(np.mean(vals)) if len(vals) > 0 else "-"

        def _pct(vals):
            return "{:.1f}%".format(np.mean(vals) * 100) if len(vals) > 0 else "-"

        # Primary metrics table
        table_rows.append([
            class_name,
            str(len(file_list)),
            str(num_valid),
            _pct(sr_results),
            _fmt(ndtw_results),
        ])

        # Supporting metrics table
        detail_rows.append([
            class_name,
            _fmt(ndtw_pos_results),
            _fmt(ndtw_ori_results),
            _fmt(epe_results),
        ])

    # Primary metrics
    print("\nUAV-Flow Evaluation by Class")
    _print_table(
        headers=["Class", "#Tasks", "#Eval", "SR", "nDTW"],
        rows=table_rows,
        align=['l', 'r', 'r', 'r', 'r']
    )

    # Supporting metrics
    print("\nSupporting Metrics")
    _print_table(
        headers=["Class", "nDTW(pos)", "nDTW(ori)", "EPE"],
        rows=detail_rows,
        align=['l', 'r', 'r', 'r']
    )

    def _ofmt(vals):
        return "{:.4f}".format(np.mean(vals)) if len(vals) else "-"

    def _opct(vals):
        return "{:.1f}%".format(np.mean(vals) * 100) if len(vals) else "-"

    print("\nOverall Summary")
    overall_rows = [[
        str(len(all_ndtw)),
        _opct(all_sr),
        _ofmt(all_ndtw),
        _ofmt(all_ndtw_pos),
        _ofmt(all_ndtw_ori),
        _ofmt(all_epe),
    ]]
    _print_table(
        headers=["#Samples", "SR", "nDTW", "nDTW(pos)", "nDTW(ori)", "EPE"],
        rows=overall_rows,
        align=['r', 'r', 'r', 'r', 'r', 'r']
    )


if __name__ == '__main__':
    model_list = ['openvla']

    # Redirect print output to a file for reproducible logging
    log_file = f'./metric.txt'
    sys.stdout = open(log_file, 'w', encoding='utf-8')
    
    for model in model_list:
        print("\n\n=========================")
        print(f"Model: {model}")
        print("=========================")
        model_dir = './results/UnrealTrack-DowntownWest-ContinuousColor-v0/{}'.format(model)
        # Ground-truth directory
        gt_dir = './test_jsons'
        classified_json_path = './classified_instr.json'
        default_step = 5
        
        evaluate_by_classification(classified_json_path, model_dir, gt_dir, default_step=default_step)
        

    