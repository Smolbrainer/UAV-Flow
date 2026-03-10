# UAV-Flow Evaluation Flow

End-to-end guide to how the closed-loop simulation evaluation works.

---

## Overview

```mermaid
graph LR
    A[local_serve.py<br/>GPU Inference] <-->|HTTP /predict| B[batch_run_act_all.py<br/>Sim Control + Logging]
    B -->|trajectory JSONs| C[metric.py<br/>Scoring]
    A --- D[OpenVLA-UAV<br/>on GPU]
    B --- E[Unreal Engine<br/>via UnrealCV]
    C --- F[nDTW / EPE<br/>per class]
```

Three stages: **environment setup**, **task execution** (stop-and-infer control loop), and **metric computation**.

---

## Stage 1: Environment Setup

**File:** `batch_run_act_all.py`

```mermaid
graph TD
    A[gym.make env] --> B[TimeDilationWrapper<br/>10x sim speed]
    B --> C[ConfigUEWrapper<br/>256x256 resolution]
    C --> D[RandomPopulationWrapper<br/>2 agents: drone + other]
    D --> E[Set agent category = drone]
    E --> F[Create scene objects]
    F --> F1[BP_Character_21<br/>Person]
    F --> F2[BP_Character_22<br/>Car]
    F1 & F2 --> G[Disable physics<br/>set_phy = 0]
    G --> H[Ready for task loop]
```

1. **Create gym env** -- loads `DowntownWest.json` config and connects to Unreal Engine via UnrealCV over TCP.
2. **Wrap the env** in three layers for time dilation, render resolution, and agent population.
3. **Set agent category to drone** -- the env supports players/animals/cars but eval only controls the drone.
4. **Create two scene objects** -- a person and a car. These are reference objects that tasks mention ("fly toward the person"). Created once, repositioned per task.
5. **Disable physics** -- the drone is teleported to positions, not physically simulated.

---

## Stage 2: Task Execution

**Files:** `batch_run_act_all.py`, `local_serve.py`

### Task Loop

```mermaid
graph TD
    A[Load 272 task JSONs] --> B{Already<br/>completed?}
    B -->|yes| C[Skip]
    B -->|no| D[Reset model]
    D --> E[Load task JSON<br/>instruction, initial_pos,<br/>target_pos, obj_id]
    E --> F[Place objects<br/>person or car at target_pos]
    F --> G[Run control loop]
    G --> H[Save trajectory JSON]
    H --> I[Draw 2D plot<br/>blue=model, orange=GT]
    I --> J[Draw 3D plot]
    J --> K{More tasks?}
    K -->|yes| B
    K -->|no| L[Done]
```

For each task JSON:
- `instruction` -- natural language task (e.g. "Fly toward the person")
- `initial_pos` -- 6D start pose `[x, y, z, roll, yaw, pitch]`
- `target_pos` -- 6D target pose (for object placement and plotting)
- `obj_id`, `use_obj` -- which object to place and what appearance
- `reference_path_preprocessed` -- ground truth trajectory for plot overlay

### Control Loop (Stop-and-Infer)

The core evaluation loop. The drone stops, gets a prediction, moves, repeats.

```mermaid
graph TD
    A[Set drone to initial_pos<br/>current_pose = 0,0,0,0] --> B[Capture image<br/>from camera 0]
    B --> C[POST /predict<br/>image + proprio + instruction]
    C --> D[Receive action<br/>x, y, z, yaw]
    D --> E[Transform to world coords<br/>rotate by initial_yaw<br/>add initial_pos]
    E --> F[Teleport drone<br/>Sync camera<br/>Log trajectory]
    F --> G{Converged?<br/>pos delta < 3<br/>yaw delta < 1<br/>for 10 steps}
    G -->|yes| H[End task]
    F --> I{max_steps<br/>reached?}
    I -->|yes| H
    G -->|no| B
    I -->|no| B
```

**Key details:**

- **Local frame** -- the model works relative to the start pose. `current_pose = [0,0,0,0]` at the beginning. The server accumulates deltas internally (rotates by current yaw, adds to current position) and returns cumulative local-frame positions.

- **World transform** -- the control loop converts local-frame predictions to world coordinates: rotate `[x,y]` by `initial_yaw`, then add `[initial_x, initial_y, initial_z]`.

- **The -180 offset** -- `set_rotation(drone, yaw - 180)` because UE's forward direction convention differs from the task JSONs.

- **Camera sync** -- `set_cam()` is called after every teleport because the camera isn't auto-attached to the drone in UnrealCV. It reads the drone's position/rotation and manually sets camera 0 to match.

### Inference Server

**File:** `local_serve.py`

```mermaid
sequenceDiagram
    participant C as Control Loop
    participant S as local_serve.py
    participant M as OpenVLA-UAV

    C->>S: POST /predict<br/>{image, proprio, instr}
    S->>S: Decode base64 image
    S->>S: Build prompt<br/>"Current State: x,y,z,yaw<br/>What action should the uav take to..."
    S->>M: Forward pass (bfloat16)
    M-->>S: Raw delta [dx, dy, dz, dyaw]
    S->>S: Accumulate:<br/>rotate delta by current yaw<br/>add to current position
    S-->>C: {action: [[x,y,z,yaw]], done: false}
```

The server also supports `chunk_size > 1` for multi-step prediction (used in real-world deployment, not sim eval).

---

## Stage 3: Metric Computation

**File:** `metric.py`

Run separately after all tasks complete: `python metric.py`

### Pipeline

```mermaid
graph TD
    A[Load classified_instr.json<br/>10 classes] --> B[For each class]
    B --> C[For each task file]
    C --> D[Load model trajectory<br/>results/env/model/*.json]
    C --> E[Load GT trajectory<br/>test_jsons/*.json]
    D & E --> F[Sample at stride<br/>default=5, Turn/Move=2]
    F --> G[Convert to 9D vectors<br/>pos/100 + sin/cos of angles]
    G --> H[Compute DTW on full 9D]
    G --> I[Compute DTW on pos only 3D]
    G --> J[Compute DTW on ori only 6D]
    G --> K[Compute endpoint error]
    H & I & J & K --> L[nDTW = exp -DTW/eta*L_gt]
    L --> M[Per-class + overall table]
```

### 10 Task Classes

| Class          | Count | Description                     |
|----------------|-------|---------------------------------|
| Turn           | 15    | Yaw rotation in place           |
| Move           | 15    | Forward/backward movement       |
| Shift          | 49    | Lateral movement                |
| Rotate         | 15    | Rotation (similar to Turn)      |
| Surround       | 12    | Circular motion around target   |
| Ascend/Descend | 19    | Vertical movement               |
| Approach       | 42    | Move toward target              |
| Retreat        | 12    | Move away from target           |
| Pass           | 41    | Fly past target                 |
| Land           | 54    | Descend to ground               |

### 9D Vector Encoding

```mermaid
graph LR
    A[Raw state<br/>x,y,z,roll,yaw,pitch] --> B[Position<br/>x/100, y/100, z/100]
    A --> C[Orientation<br/>sin roll, sin yaw, sin pitch<br/>cos roll, cos yaw, cos pitch]
    B & C --> D[9D vector]
```

- Position divided by 100 to scale with rotation components
- Both sin and cos for angles (captures rotation direction -- cos alone can't distinguish left from right)
- Turn/Rotate classes: position zeroed out (orientation-only evaluation)

### Metrics

```mermaid
graph TD
    subgraph "Primary -- optimize this"
        A[nDTW combined<br/>DTW on full 9D vectors<br/>exp -DTW / eta * L_gt<br/>Score 0-1, higher = better]
    end
    subgraph "Diagnostic -- use when nDTW drops"
        B[nDTW pos<br/>Position only 3D<br/>Is the path right?]
        C[nDTW ori<br/>Orientation only 6D<br/>Is the heading right?]
        D[EPE<br/>Endpoint error<br/>Did it reach the goal?]
    end
    A --> B & C & D
```

### Output

Per-class table + overall summary printed to `metric.txt`:

```
+----------------+-------+--------+-----------+-----------+--------+
| Class          | #Eval |   nDTW | nDTW(pos) | nDTW(ori) |    EPE |
+----------------+-------+--------+-----------+-----------+--------+
| Turn           |    15 | 0.2392 |         - |    0.2392 | 0.0000 |
| ...            |       |        |           |           |        |
+----------------+-------+--------+-----------+-----------+--------+
```

**For the train/eval feedback loop:** optimize combined nDTW. Use the split and EPE for diagnosing what went wrong.

---

## File Map

```mermaid
graph TD
    subgraph "UAV-Flow-Eval/"
        A[batch_run_act_all.py<br/>Main eval runner]
        B[local_serve.py<br/>GPU inference server]
        C[metric.py<br/>Scoring]
        D[classified_instr.json<br/>Task class mapping]
        E[test_jsons/<br/>272 GT task JSONs]
        F[results/<br/>Output trajectories + plots]
        G[relative.py<br/>Pose helpers]
        H[gym_unrealcv/<br/>Gym env wrapping UnrealCV]
        I[CHANGELOG.md]
        J[EVAL_FLOW.md<br/>This file]
    end

    A <-->|HTTP| B
    A -->|writes| F
    A -->|reads| E
    C -->|reads| F
    C -->|reads| E
    C -->|reads| D
    A -->|uses| H
    A -->|uses| G
```

---

## Running the Eval

```bash
# 1. Start Unreal Engine (Windows side if on WSL)

# 2. Start inference server
python local_serve.py --port 5007

# 3. Run evaluation
python batch_run_act_all.py

# 4. Compute metrics
python metric.py
```

Results go to `results/<env>/openvla/` (trajectory JSONs + plots) and `metric.txt` (scores).
