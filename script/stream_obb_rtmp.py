#!/usr/bin/env python3
"""Pull an RTMP stream, run YOLO OBB inference, and restream annotated video."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from infer_obb import (
    DEFAULT_MIN_CONE_LONG_SIDE_RATIO,
    _draw_detections,
    _extract_detections,
)


DEFAULT_SOURCE = "rtmp://218.3.126.49:30861/live/_1581F7K3C253B00D1JZY"
DEFAULT_OUTPUT = "rtmp://10.8.30.29/live/traffic_cone_obb"


@dataclass(frozen=True)
class StreamInfo:
    width: int
    height: int
    fps: float


def probe_stream(source: str) -> StreamInfo:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        source,
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(completed.stdout).get("streams", [])
    if not streams:
        raise RuntimeError("No video stream was found")
    stream = streams[0]
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "25/1"
    fps = float(Fraction(rate)) if rate != "0/0" else 25.0
    if fps <= 0:
        fps = 25.0
    return StreamInfo(int(stream["width"]), int(stream["height"]), fps)


def decoder_command(source: str) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-analyzeduration",
        "1000000",
        "-probesize",
        "1000000",
        "-i",
        source,
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]


def encoder_command(output: str, info: StreamInfo) -> list[str]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{info.width}x{info.height}",
        "-r",
        f"{info.fps:.6f}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(max(1, round(info.fps * 2))),
    ]
    if output.startswith("rtmp://"):
        command.extend(["-f", "flv", output])
    else:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        command.extend(["-movflags", "+faststart", "-y", output])
    return command


def read_exact(stream, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            return b""
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def stop_process(process: subprocess.Popen, close_stdout: bool = False) -> None:
    if close_stdout and process.stdout is not None:
        process.stdout.close()
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--weights",
        type=Path,
        default=project_root
        / "script/runs/traffic_cone_uav_20260805_3class_ft/weights/selected.pt",
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--min-cone-long-side-ratio",
        type=float,
        default=DEFAULT_MIN_CONE_LONG_SIDE_RATIO,
        help=(
            "Draw but ignore cone OBBs below this fraction of the frame's "
            "shorter side; 0 disables the filter"
        ),
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-reconnects", type=int, default=0, help="0 means unlimited")
    parser.add_argument("--reconnect-delay", type=float, default=2.0)
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Decode and infer without starting the output encoder",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights do not exist: {args.weights}")
    if not 0.0 <= args.min_cone_long_side_ratio < 1.0:
        raise ValueError("--min-cone-long-side-ratio must be in [0, 1)")

    import cv2
    import numpy as np
    from ultralytics import YOLO

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    info = probe_stream(args.source)
    logging.info("Source video: %dx%d %.3f FPS", info.width, info.height, info.fps)
    model = YOLO(str(args.weights), task="obb")
    encoder = None
    if not args.no_push:
        encoder = subprocess.Popen(encoder_command(args.output, info), stdin=subprocess.PIPE)
        logging.info("Publishing annotated stream to %s", args.output)

    frame_size = info.width * info.height * 3
    frame_count = 0
    reconnects = 0
    started = time.monotonic()
    decoder = None
    try:
        while not args.max_frames or frame_count < args.max_frames:
            decoder = subprocess.Popen(decoder_command(args.source), stdout=subprocess.PIPE)
            assert decoder.stdout is not None
            while not args.max_frames or frame_count < args.max_frames:
                raw = read_exact(decoder.stdout, frame_size)
                if not raw:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(info.height, info.width, 3)
                result = model.predict(
                    frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    device=args.device,
                    verbose=False,
                )[0]
                (
                    _,
                    _,
                    detections,
                    _,
                    small_cone_suppressed,
                ) = _extract_detections(
                    result,
                    "cone",
                    None,
                    image_size=(info.width, info.height),
                    min_cone_long_side_ratio=args.min_cone_long_side_ratio,
                    primary_confidence=args.conf,
                )
                annotated = _draw_detections(
                    cv2, np, frame.copy(), detections
                )
                frame_count += 1
                elapsed = max(time.monotonic() - started, 1e-6)
                cv2.putText(
                    annotated,
                    f"frames={frame_count} avg={frame_count / elapsed:.1f} FPS",
                    (18, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if encoder is not None:
                    if encoder.poll() is not None or encoder.stdin is None:
                        raise RuntimeError(f"Output FFmpeg stopped with code {encoder.returncode}")
                    encoder.stdin.write(annotated.tobytes())
                if frame_count == 1 or frame_count % 100 == 0:
                    detection_count = len(result.obb) if result.obb is not None else 0
                    logging.info(
                        "Processed frame %d, detections=%d, small_skipped=%d",
                        frame_count,
                        detection_count,
                        small_cone_suppressed,
                    )

            stop_process(decoder, close_stdout=True)
            decoder = None
            if args.max_frames and frame_count >= args.max_frames:
                break
            reconnects += 1
            if args.max_reconnects and reconnects > args.max_reconnects:
                raise RuntimeError("Source stream ended and reconnect limit was reached")
            logging.warning("Source ended; reconnecting in %.1f seconds", args.reconnect_delay)
            time.sleep(args.reconnect_delay)
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        if decoder is not None and decoder.poll() is None:
            stop_process(decoder, close_stdout=True)
        if encoder is not None:
            if encoder.stdin is not None:
                encoder.stdin.close()
            encoder.wait(timeout=15)

    logging.info("Finished after %d frames", frame_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
