#!/usr/bin/env python3
"""Convert the keyword spotting ONNX model to a full UINT8 TFLite model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[3]
WEIGHTS_DIR = ROOT / "src" / "catkin_ws" / "src" / "spot_keyword_spotting" / "keyword_spotting" / "weights"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ONNX keyword model to UINT8 TFLite.")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=WEIGHTS_DIR / "checkpoint.onnx",
        help="Input ONNX model path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=WEIGHTS_DIR / "checkpoint_uint8.tflite",
        help="Output UINT8 TFLite model path.",
    )
    parser.add_argument(
        "--saved-model-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "onnx_saved_model",
        help="Temporary TensorFlow SavedModel output from onnx2tf.",
    )
    parser.add_argument(
        "--representative-data",
        type=Path,
        required=True,
        help="Calibration samples in .npy or .npz format. Shape must match the model input.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of representative samples yielded per calibration step.",
    )
    return parser.parse_args()


def convert_onnx_to_saved_model(onnx_path: Path, saved_model_dir: Path) -> None:
    try:
        import onnx2tf
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install `onnx2tf` before running this script.") from exc

    saved_model_dir.mkdir(parents=True, exist_ok=True)
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(saved_model_dir),
        non_verbose=True,
    )


def load_representative_data(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        data = np.load(path)
    elif path.suffix == ".npz":
        loaded = np.load(path)
        key = loaded.files[0]
        data = loaded[key]
    else:
        raise ValueError("Representative data must be a .npy or .npz file.")

    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 0:
        raise ValueError("Representative data is empty or scalar.")
    return data


def representative_dataset(data: np.ndarray, batch_size: int) -> Iterable[list[np.ndarray]]:
    batch_size = max(1, int(batch_size))
    for start in range(0, len(data), batch_size):
        yield [data[start : start + batch_size].astype(np.float32)]


def convert_saved_model_to_uint8_tflite(
    saved_model_dir: Path,
    output_path: Path,
    calibration_data: np.ndarray,
    batch_size: int,
) -> None:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(calibration_data, batch_size)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(converter.convert())


def main() -> None:
    args = parse_args()
    if not args.onnx.exists():
        raise FileNotFoundError(f"ONNX model not found: {args.onnx}")

    calibration_data = load_representative_data(args.representative_data)
    convert_onnx_to_saved_model(args.onnx, args.saved_model_dir)
    convert_saved_model_to_uint8_tflite(
        args.saved_model_dir,
        args.out,
        calibration_data,
        args.batch_size,
    )
    print(f"Saved UINT8 TFLite model: {args.out}")


if __name__ == "__main__":
    main()
