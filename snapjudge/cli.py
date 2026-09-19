"""Command line entry point: snapjudge-serve --model <path or Hugging Face id>."""

from __future__ import annotations

import argparse
import os

import uvicorn


def main():
    ap = argparse.ArgumentParser(description="Serve typed decisions from a local Qwen model on MLX.")
    ap.add_argument("--model", default=os.environ.get("SO_MODEL", "mlx-community/Qwen3.5-4B-MLX-4bit"),
                    help="local path or Hugging Face id of a Qwen 3.5-family MLX model")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8724)
    ap.add_argument("--name", help="model name reported in responses (default: last path component)")
    ap.add_argument("--api-key", help="require this bearer token")
    ap.add_argument("--calibration", help="JSON file with per-type temperatures")
    ap.add_argument("--adapter", help="LoRA adapter directory (adapters.safetensors + adapter_config.json)")
    args = ap.parse_args()

    os.environ["SO_MODEL"] = args.model
    for var, value in (("SO_NAME", args.name), ("SO_API_KEY", args.api_key), ("SO_CALIBRATION", args.calibration), ("SO_ADAPTER", args.adapter)):
        if value:
            os.environ[var] = value
    print(f"Game: http://{args.host}:{args.port}/game/  ·  API: POST http://{args.host}:{args.port}/v1/systemone")
    uvicorn.run("snapjudge.server:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
