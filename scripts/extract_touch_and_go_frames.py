import argparse
from pathlib import Path
from typing import Optional

import cv2


def count_frames(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def extract_video(video_path: Path, output_dir: Path, limit: Optional[int] = None) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[skip] failed to open {video_path}")
        cap.release()
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    while True:
        if limit is not None and written >= limit:
            break

        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_path = output_dir / f"{written:010d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        written += 1

    cap.release()
    return written


def clear_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return

    for frame_path in output_dir.glob("*.jpg"):
        frame_path.unlink()


def process_sample(sample_dir: Path, overwrite: bool = False) -> None:
    video_path = sample_dir / "video.mp4"
    gelsight_path = sample_dir / "gelsight.mp4"
    video_out = sample_dir / "video_frame"
    gelsight_out = sample_dir / "gelsight_frame"

    has_video = video_path.exists()
    has_gelsight = gelsight_path.exists()

    if not has_video and not has_gelsight:
        print(f"[skip] {sample_dir}: no video.mp4 or gelsight.mp4")
        return

    if not overwrite and (video_out.exists() or gelsight_out.exists()):
        print(f"[skip] {sample_dir}: frame folder already exists")
        return

    if has_gelsight:
        if overwrite:
            clear_output_dir(gelsight_out)
        gelsight_count = count_frames(gelsight_path)
        gelsight_written = extract_video(gelsight_path, gelsight_out)
        print(
            f"[ok] {sample_dir}: wrote {gelsight_written} gelsight frames "
            f"(metadata count: {gelsight_count})"
        )

    if has_video:
        if overwrite:
            clear_output_dir(video_out)
        video_count = count_frames(video_path)
        video_written = extract_video(video_path, video_out)
        print(
            f"[ok] {sample_dir}: wrote {video_written} video frames "
            f"(metadata count: {video_count})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Touch-and-Go frames even when some samples only contain one video stream."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/touch_and_go/dataset"),
        help="Root directory containing sample folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract frames even if output folders already exist.",
    )
    args = parser.parse_args()

    if not args.root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {args.root}")

    sample_dirs = sorted(path for path in args.root.iterdir() if path.is_dir())
    for sample_dir in sample_dirs:
        process_sample(sample_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
