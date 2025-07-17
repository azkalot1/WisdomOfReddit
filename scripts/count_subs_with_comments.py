#!/usr/bin/env python3
"""
Parallel count of submissions & comments per date & subreddit.

For each date-folder under base_dir, spins up a worker to:
  - parse all JSONs in that folder
  - count submissions and comments per subreddit
  - write <date>_counts.csv in the out_dir

Usage:
    python count_subs_parallel.py \
        --base-dir ./reddit_comments \
        --out-dir ./date_counts \
        [--workers 4]
"""

import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import multiprocessing
import dotenv
dotenv.load_dotenv()

def get_comment_count(data: Dict[str, Any]) -> int:
    """
    Count comments in the loaded JSON object.
    - If 'comments' is a list, returns its length.
    - If it's a non-empty scalar, returns 1.
    - Else returns 0.
    """
    comments = data.get("comments", [])
    if isinstance(comments, list):
        return len(comments)
    return 1 if comments else 0


def process_date(args: Tuple[str, List[Path], str]) -> str:
    """
    Worker function: for one date, parse all its files, tally counts,
    and write a CSV.

    Args:
      args: (date, list_of_file_paths, out_dir)

    Returns:
      the path of the written CSV
    """
    date, paths, out_dir = args
    # subreddit → {'n_submission': int, 'n_comments': int}
    stats = defaultdict(lambda: {"n_submission": 0, "n_comments": 0})

    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        sub = data.get("subreddit")
        if not sub:
            continue
        stats[sub]["n_submission"] += 1
        stats[sub]["n_comments"] += get_comment_count(data)

    # write per-date CSV
    out_path = Path(out_dir) / f"{date}_counts.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "subreddit", "n_submission", "n_comments"])
        for sub, meters in sorted(stats.items()):
            writer.writerow([date, sub, meters["n_submission"], meters["n_comments"]])

    return str(out_path)


def main(base_dir: str, out_dir: str, max_workers: int):
    base = Path(base_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Group JSON filepaths by date-folder name
    date_to_files: Dict[str, List[Path]] = defaultdict(list)
    for pj in base.rglob("*.json"):
        # assume parent folder name is the date (e.g. '20250708')
        date = pj.parent.name
        date_to_files[date].append(pj)

    if not date_to_files:
        print(f"No JSON files found under {base_dir!r}.")
        return

    # 2) Launch a pool, submit one task per date
    tasks = [
        (date, paths, str(out))
        for date, paths in date_to_files.items()
    ]
    print(f"Dispatching {len(tasks)} dates over {max_workers} workers…")

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(process_date, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            date = futures[fut]
            try:
                path = fut.result()
                print(f"[{date}] wrote {path}")
            except Exception as e:
                print(f"[{date}] ERROR: {e}")

    print("All done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--base-dir",  required=True,
                   help="Root folder of your date-named subfolders")
    p.add_argument("--out-dir",   default="date_counts",
                   help="Where to write each <date>_counts.csv")
    p.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                   help="Max parallel workers (default: CPU count)")
    args = p.parse_args()

    main(args.base_dir, args.out_dir, args.workers)
