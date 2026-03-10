"""
Local inference server for OpenVLA-UAV on your GPU.

Usage:
    python local_serve.py                          # defaults: port 5007, model from HF
    python local_serve.py --port 5007
    python local_serve.py --model_path /path/to/local/weights
"""

import argparse
import base64
import logging
import time
from io import BytesIO

import numpy as np
import torch
from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODEL_ID = "wangxiangyu0814/OpenVLA-UAV"
UNNORM_KEY = "sim"


def create_app(model, processor, device):
    app = Flask(__name__)

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            data = request.json
            img_bytes = base64.b64decode(data["image"])
            image = Image.open(BytesIO(img_bytes)).convert("RGB")

            instruction = data.get("instr", "")
            proprio = np.array(data.get("proprio", [0, 0, 0, 0]), dtype=np.float32)
            chunk_size = int(data.get("chunk_size", 1))
            chunk_size = max(1, min(chunk_size, 20))  # clamp to [1, 20]

            current_pos = proprio[0:3].copy()
            current_yaw = np.deg2rad(proprio[-1])

            all_actions = []
            t0 = time.time()

            for step in range(chunk_size):
                proprio_str = ",".join([str(round(x, 1)) for x in
                                        [current_pos[0], current_pos[1], current_pos[2],
                                         float(np.rad2deg(current_yaw))]])
                prompt = f"In: Current State: {proprio_str}, What action should the uav take to {instruction}?\nOut:"

                inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
                with torch.inference_mode():
                    raw_action = model.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)

                # Accumulate: rotate delta by current yaw, add to current position
                cos_yaw, sin_yaw = np.cos(current_yaw), np.sin(current_yaw)
                R = np.array([[cos_yaw, -sin_yaw, 0],
                              [sin_yaw, cos_yaw, 0],
                              [0, 0, 1]])

                delta_pos = R @ raw_action[0:3]
                current_pos = current_pos + delta_pos
                current_yaw = current_yaw + raw_action[-1]

                all_actions.append([float(current_pos[0]), float(current_pos[1]),
                                    float(current_pos[2]), float(current_yaw)])

            log.info(f"Chunk inference: {chunk_size} steps in {time.time()-t0:.3f}s")

            return jsonify({
                "action": all_actions,
                "done": False,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.route("/reset", methods=["POST"])
    def reset():
        return jsonify({"status": "ok"})

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "model": MODEL_ID, "device": str(device)})

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5007)
    parser.add_argument("--model_path", type=str, default=MODEL_ID)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    log.info(f"Loading {args.model_path} on {device} ({torch.cuda.get_device_name(args.gpu)})...")

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    action_dim = model.get_action_dim(UNNORM_KEY)
    log.info(f"Model loaded. action_dim={action_dim}")

    app = create_app(model, processor, device)
    log.info(f"Serving on http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port)
