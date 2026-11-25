"""Example stereo inference pipeline with a placeholder PyTorch predictor."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def predict_disparity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Replace this function with your model. Return a [B, H, W] tensor in pixels."""
    if left.shape != right.shape:
        raise ValueError("Rectified stereo inputs must have the same shape")
    batch, _, height, width = left.shape
    # Stand-in prediction corresponding to a fronto-parallel plane.
    return left.new_full((batch, height, width), 32.0, dtype=torch.float32)


def disparity_to_depth(disparity_px: torch.Tensor, fx_px: float, baseline_m: float) -> torch.Tensor:
    """Accept an H×W or batched tensor. Keep the result on the input device."""
    valid = torch.isfinite(disparity_px) & (disparity_px > 0)
    safe_disparity = torch.where(valid, disparity_px, 1.0)
    return (fx_px * baseline_m / safe_disparity).masked_fill(~valid, torch.nan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("depth_m.npy"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    args = parser.parse_args()

    calibration = json.loads(args.calibration.read_text(encoding="utf-8-sig"))
    width, height = calibration["image_size_wh_px"]
    fx_px = float(calibration["views"]["left"]["K"][0][0])
    baseline_m = float(calibration["baseline_m"])
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    with torch.inference_mode():
        # Replace these tensors with your rectified stereo pair and model preprocessing.
        left = torch.zeros((1, 1, height, width), device=device)
        right = torch.zeros_like(left)
        disparity_px = predict_disparity(left, right)
        if disparity_px.shape != (left.shape[0], height, width) or disparity_px.device != left.device:
            raise ValueError("Return [B, H, W] disparity at the calibrated resolution and on the input device")
        depth_m = disparity_to_depth(disparity_px.float(), fx_px, baseline_m)
        # Copy to CPU only when saving the depth image.
        depth_image = depth_m[0].cpu().numpy()

    with args.output.open("wb") as file:
        np.save(file, depth_image, allow_pickle=False)
    print(f"{device}: saved {width}×{height} float32 depth in metres to {args.output}")


if __name__ == "__main__":
    main()

