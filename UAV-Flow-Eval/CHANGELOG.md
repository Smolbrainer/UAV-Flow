# UAV-Flow Eval Changelog

## Metric: 6D → 9D Angle Encoding

**Files changed:** `metric.py`

**Problem:** The old metric used `cos(angle)` only to encode roll/yaw/pitch into vectors for DTW comparison. Since `cos(30°) == cos(-30°)`, the metric couldn't distinguish left turns from right turns — a model turning the wrong direction got full credit.

**Fix:** Use both `sin(angle)` and `cos(angle)` (unit circle encoding). Each trajectory point is now 9D `[x, y, z, sin(roll), sin(yaw), sin(pitch), cos(roll), cos(yaw), cos(pitch)]` instead of 6D `[x, y, z, cos(roll), cos(yaw), cos(pitch)]`.

**Impact on scores (openvla):**

| Class          | 6D (cos only) | 9D (sin+cos) | Delta   |
|----------------|---------------|--------------|---------|
| Turn           | 0.2987        | 0.2392       | -0.0595 |
| Move           | 0.1237        | 0.1237       |  0.0000 |
| Shift          | 0.7135        | 0.7091       | -0.0044 |
| Rotate         | 0.3600        | 0.2635       | -0.0965 |
| Surround       | 0.7464        | 0.7480       | +0.0016 |
| Ascend/Descend | 0.7235        | 0.7232       | -0.0003 |
| Approach       | 0.2014        | 0.2014       |  0.0000 |
| Retreat        | 0.3144        | 0.3144       |  0.0000 |
| Pass           | 0.2193        | 0.2193       |  0.0000 |
| Land           | 0.1950        | 0.1949       | -0.0001 |
| **Overall**    | **0.3674**    | **0.3578**   | **-0.0096** |

**Key takeaway:** Turn and Rotate classes dropped the most (-6% and -10%) because these are rotation-heavy tasks where the cos-only metric was hiding directional errors. Translation-dominated tasks (Move, Approach, Pass) were unaffected. The overall drop of ~1% reflects a more honest measurement.

---

## Metric: EPE + nDTW Split (Position vs Orientation)

**Files changed:** `metric.py`

**What:** Added three new metrics alongside combined nDTW:
- **nDTW(pos)** — position-only (3D `[x,y,z]`), ignores rotation
- **nDTW(ori)** — orientation-only (6D `[sin,cos]` of angles), ignores position
- **EPE** — endpoint error (Euclidean distance between final model and GT positions)

**Why:** Combined nDTW mixes position and orientation, masking component-level failures. The split reveals which aspect the model struggles with. EPE catches trajectories that follow the right shape but stop too early or overshoot.

**Key findings (openvla, stop-and-infer):**

| Class          | nDTW  | nDTW(pos) | nDTW(ori) | EPE    |
|----------------|-------|-----------|-----------|--------|
| Shift          | 0.71  | 0.71      | **0.04**  | 0.50   |
| Surround       | 0.75  | 0.74      | 0.84      | 0.21   |
| Approach       | 0.20  | 0.20      | 0.28      | **4.25** |
| Pass           | 0.22  | 0.22      | 0.25      | **6.48** |
| **Overall**    | 0.36  | 0.37      | **0.30**  | 2.62   |

- Shift: great position, terrible orientation (model goes the right way but faces wrong direction)
- Approach/Pass/Land: huge EPE — model doesn't reach the target
- Overall orientation (0.30) is worse than position (0.37)

---

## GT Overlay on Trajectory Plots

**Files changed:** `batch_run_act_all.py`

**What:** 2D and 3D trajectory plots now show the ground truth reference path as an orange dashed line alongside the model's blue trajectory. The GT is loaded from `reference_path_preprocessed` in the task JSON.

**Why:** Previously you had to mentally compare model output against ground truth. Now divergence is immediately visible on the plot.

---

## Action Chunking Experiments (Tested and Reverted)

**Files changed:** `local_serve.py` (chunk prediction kept), `batch_run_act_all.py` (reverted to stop-and-infer)

### Context: What the paper actually describes

The UAV-Flow paper presents two types of models:
- **OpenVLA-UAV** — single-step prediction. One forward pass = one `[x, y, z, yaw]` action.
- **Pi-0-UAV** — native 10-step chunk prediction via flow matching. One forward pass = 10 future actions.

The paper's "Globally-Aligned Continuous Motion" scheme (section 3.1) — async inference, look-ahead filtering, global pose fusion — is designed for **Pi-0-UAV's native chunks**. The paper doesn't explicitly address how single-step models like OpenVLA-UAV should work with this pipeline. It presents the continuous motion framework as a general contribution, but in practice it only makes sense with models that natively output multi-step predictions.

### What we tried

We attempted to make OpenVLA-UAV work with continuous motion by faking chunks — running the single-step model N times auto-regressively on the same image with updated proprio. This is fundamentally different from Pi-0-UAV's native chunking, where the model is trained to predict 10 future steps in a single forward pass with awareness of the full trajectory.

### Attempt 1: Async chunking without blending
- Server runs model N times (same image, updated proprio each step)
- Client executes chunk, fires next inference, hard-switches to new chunk on arrival
- **Result:** Drone rubberbanded at chunk boundaries.

### Attempt 2: Async chunking with weighted blending (LeRobot-style)
- Same server-side approach but overlapping waypoints blended via exponentially-weighted average
- Inspired by [LeRobot's async inference](https://huggingface.co/docs/lerobot/en/async)
- **Result:** Smoother transitions but still degraded trajectory quality significantly.

**Both attempts failed for the same root cause:** running a single-step model N times on a stale image is not real action chunking. The model was never trained to predict multiple future steps without fresh visual feedback — it expects a new image every step. Steps 2-N of each "chunk" are the model predicting blind, and no amount of post-hoc blending fixes wrong inputs.

**Comparison (openvla):**

| Metric    | Stop-and-infer | Fake chunking (blended) | Delta   |
|-----------|---------------|--------------------------|---------|
| nDTW      | 0.3578        | 0.2525                   | -0.1053 |
| nDTW(pos) | 0.3719        | 0.2637                   | -0.1082 |
| nDTW(ori) | 0.3043        | 0.2491                   | -0.0552 |
| EPE       | 2.6178        | 2.7391                   | +0.12   |

Everything got worse. Biggest drops: Turn (0.24→0.06), Surround (0.75→0.14), Land (0.19→0.07).

### Conclusion

For OpenVLA-UAV, stop-and-infer is the only viable approach — the model architecturally can only predict one step at a time. The continuous motion pipeline we built (async queue, weighted blending, threshold-based refill) is correct infrastructure, but it needs a model that natively outputs action chunks (like Pi-0-UAV) to work properly. To make OpenVLA-UAV work with continuous motion, the model itself would need to be retrained/fine-tuned to predict multi-step action sequences.

**Current status:** `batch_run_act_all.py` is back to stop-and-infer. `local_serve.py` still supports `chunk_size` for when a chunk-native model (e.g. Pi-0-UAV) is swapped in.

---

## Chunk Prediction in Server

**Files changed:** `local_serve.py`

**What:** The `/predict` endpoint accepts an optional `chunk_size` parameter (default 1, max 20). Currently runs OpenVLA-UAV N times auto-regressively — this is a workaround, not true chunking. When a chunk-native model like Pi-0-UAV is used, this would be replaced with a single forward pass returning N actions.

**Status:** Implemented. Defaults to chunk_size=1 (single-step) for OpenVLA-UAV eval.

---

## Known Issues (Not Yet Fixed)

- **Yaw wraparound in early stopping:** `abs(179° - (-179°)) = 358°` instead of 2°. Can cause early stopping to never trigger for tasks near ±180° yaw.
- **Early stopping thresholds are hardcoded:** 3 units pos, 1° yaw, 10 steps. No per-task-class adaptation.
