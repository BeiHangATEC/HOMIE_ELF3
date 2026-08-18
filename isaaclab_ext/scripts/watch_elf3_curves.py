"""Live ELF3 training curves in the terminal.

Reads the TensorBoard event files a run writes and plots them in-place, so you
can watch progress over ssh or inside screen without a browser.

    python isaaclab_ext/scripts/watch_elf3_curves.py --run-dir runs/<run>

Run it inside its own screen window and detach with Ctrl-A d:

    screen -S elf3-curves -dm python .../watch_elf3_curves.py --run-dir <run>
    screen -r elf3-curves
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path


DEFAULT_TAGS = ("Train/mean_episode_length", "Train/mean_reward")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="run directory, or a stage subdirectory holding events.out.tfevents.*",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=list(DEFAULT_TAGS),
        help="scalar tags to plot, one chart each",
    )
    parser.add_argument(
        "--interval", type=float, default=15.0, help="refresh seconds"
    )
    parser.add_argument(
        "--height", type=int, default=18, help="rows per chart"
    )
    parser.add_argument(
        "--once", action="store_true", help="draw once and exit (no watch loop)"
    )
    parser.add_argument(
        "--list-tags", action="store_true", help="print available tags and exit"
    )
    return parser.parse_args()


def find_event_files(run_dir: Path) -> list[str]:
    """Collect event files from the run dir or any stage subdirectory."""
    patterns = [
        str(run_dir / "events.out.tfevents.*"),
        str(run_dir / "*" / "events.out.tfevents.*"),
    ]
    files: list[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    # Chronological, so later chunks overwrite earlier ones at the same step.
    return sorted(set(files), key=os.path.getmtime)


def load_scalars(files: list[str], tags: list[str]) -> dict[str, dict[int, float]]:
    from tensorboard.backend.event_processing import event_accumulator

    series: dict[str, dict[int, float]] = {tag: {} for tag in tags}
    for path in files:
        accumulator = event_accumulator.EventAccumulator(
            path, size_guidance={"scalars": 0}
        )
        try:
            accumulator.Reload()
        except Exception:
            continue  # a file still being written can be momentarily unreadable
        available = set(accumulator.Tags().get("scalars", ()))
        for tag in tags:
            if tag not in available:
                continue
            for event in accumulator.Scalars(tag):
                series[tag][event.step] = event.value
    return series


def available_tags(files: list[str]) -> list[str]:
    from tensorboard.backend.event_processing import event_accumulator

    tags: set[str] = set()
    for path in files:
        accumulator = event_accumulator.EventAccumulator(
            path, size_guidance={"scalars": 0}
        )
        try:
            accumulator.Reload()
        except Exception:
            continue
        tags.update(accumulator.Tags().get("scalars", ()))
    return sorted(tags)


def _trend(values: list[float], window: int = 50) -> str:
    """Compare the last window against the one before it."""
    if len(values) < 2 * window:
        window = max(1, len(values) // 2)
    if len(values) < 2:
        return ""
    recent = sum(values[-window:]) / window
    earlier = sum(values[-2 * window : -window]) / window
    delta = recent - earlier
    arrow = "up" if delta > 0 else ("down" if delta < 0 else "flat")
    return f"last {window}: {recent:.2f} vs prior {window}: {earlier:.2f} ({arrow} {delta:+.2f})"


def draw(series: dict[str, dict[int, float]], height: int) -> None:
    import plotext as plt

    plt.clf()
    populated = [(tag, data) for tag, data in series.items() if data]
    if not populated:
        print("no scalar data yet; waiting for the first logged iteration")
        return

    plt.subplots(len(populated), 1)
    for index, (tag, data) in enumerate(populated, start=1):
        steps = sorted(data)
        values = [data[s] for s in steps]
        subplot = plt.subplot(index, 1)
        subplot.plot(steps, values, marker="braille")
        subplot.title(f"{tag}    latest {values[-1]:.3f}    {_trend(values)}")
        subplot.xlabel("iteration")
        subplot.plotsize(None, height)
    plt.show()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"error: {run_dir} is not a directory")
        return 1

    if args.list_tags:
        files = find_event_files(run_dir)
        if not files:
            print(f"no event files under {run_dir}")
            return 1
        for tag in available_tags(files):
            print(tag)
        return 0

    while True:
        files = find_event_files(run_dir)
        if not files:
            print(f"waiting for event files under {run_dir} ...", flush=True)
        else:
            series = load_scalars(files, args.tags)
            # Clear only once the new frame is ready, to avoid a visible flicker.
            print("\033[H\033[J", end="")
            print(f"{run_dir}    {len(files)} event file(s)    {time.strftime('%H:%M:%S')}")
            missing = [tag for tag, data in series.items() if not data]
            draw(series, args.height)
            if missing:
                print(f"no data for: {', '.join(missing)} (try --list-tags)")
            print("Ctrl-A d detaches screen; Ctrl-C stops this viewer", flush=True)
        if args.once:
            return 0
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
