"""
Modal deployment for OpenVLA-UAV inference server.

Usage:
    # First-time setup
    pip install modal
    modal setup

    # Dev mode (temporary URL, hot-reload)
    modal serve modal_serve.py

    # Production deployment (persistent URL)
    modal deploy modal_serve.py

The server exposes /predict and /reset endpoints compatible with batch_run_act_all.py.
Pass the printed URL to the eval script via --server_url.
"""

import modal

MODEL_ID = "wangxiangyu0814/OpenVLA-UAV"
UNNORM_KEY = "sim"  # UAV simulation dataset (4D action: x, y, z, yaw)


def download_model():
    """Pre-download model weights into the container image for fast cold starts."""
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "transformers>=4.40",
        "accelerate",
        "Pillow",
        "numpy",
        "timm",
        "sentencepiece",
        "huggingface_hub",
    )
    .run_function(download_model)
)

app = modal.App("openvla-uav", image=image)


@app.cls(gpu="A10G", container_idle_timeout=300)
class Inference:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        print(f"Loading {MODEL_ID} ...")
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, trust_remote_code=True
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to("cuda")

        action_dim = self.model.get_action_dim(UNNORM_KEY)
        print(f"Model loaded. unnorm_key='{UNNORM_KEY}', action_dim={action_dim}")

    @modal.asgi_app()
    def serve(self):
        import base64
        import traceback
        from io import BytesIO

        import torch
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
        from PIL import Image

        web_app = FastAPI()

        @web_app.post("/predict")
        async def predict(request: Request):
            try:
                data = await request.json()

                # Decode base64 PNG image
                img_bytes = base64.b64decode(data["image"])
                image = Image.open(BytesIO(img_bytes)).convert("RGB")

                # Build OpenVLA prompt
                instruction = data.get("instr", "")
                prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

                # Run inference
                inputs = self.processor(prompt, image).to(
                    "cuda", dtype=torch.bfloat16
                )
                with torch.inference_mode():
                    action = self.model.predict_action(
                        **inputs, unnorm_key=UNNORM_KEY, do_sample=False
                    )

                # Action is [x, y, z, yaw] — wrap in list for eval compatibility
                return {"action": [action.tolist()], "done": False}

            except Exception as e:
                traceback.print_exc()
                return JSONResponse(status_code=500, content={"error": str(e)})

        @web_app.post("/reset")
        async def reset():
            return {"status": "ok"}

        @web_app.get("/health")
        async def health():
            return {"status": "ok", "model": MODEL_ID}

        return web_app
