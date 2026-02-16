from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import databento as db
import numpy as np

from scalpr_zen.types import ContractPeriod


def discover_dbn_files(data_dir: str, start_date: str, end_date: str) -> list[Path]:
    data_path = Path(data_dir)
    files = sorted(data_path.glob("glbx-mdp3-*.trades.dbn.zst"))
    result = []
    for f in files:
        date_str = f.name.split("-")[2].replace(".", "")[:8]
        file_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        if start_date <= file_date < end_date:
            result.append(f)
    return result


def active_instrument_id(date_str: str, schedule: list[ContractPeriod]) -> int | None:
    for cp in schedule:
        if cp.start_date <= date_str < cp.end_date:
            return int(cp.instrument_id)
    return None


def _load_one_day(
    file_path: Path,
    instrument_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    store = db.DBNStore.from_file(str(file_path))
    df = store.to_df()
    mask = df["instrument_id"] == instrument_id
    filtered = df.loc[mask]
    if filtered.empty:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
    prices = filtered["price"].to_numpy(dtype=np.float64).astype(np.float32)
    timestamps = filtered["ts_event"].astype(np.int64).to_numpy()
    return prices, timestamps


def preprocess_to_cache(
    data_dir: str,
    cache_path: str,
    schedule: list[ContractPeriod],
    start_date: str,
    end_date: str,
) -> int:
    files = discover_dbn_files(data_dir, start_date, end_date)
    if not files:
        return 0

    # Build (file, instrument_id) pairs
    tasks: list[tuple[Path, int]] = []
    for f in files:
        date_str_raw = f.name.split("-")[2][:8]
        file_date = f"{date_str_raw[:4]}-{date_str_raw[4:6]}-{date_str_raw[6:8]}"
        iid = active_instrument_id(file_date, schedule)
        if iid is not None:
            tasks.append((f, iid))

    all_prices: list[np.ndarray] = []
    all_timestamps: list[np.ndarray] = []
    total_ticks = 0

    workers = min(os.cpu_count() or 4, 8)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_load_one_day, fpath, iid): idx
            for idx, (fpath, iid) in enumerate(tasks)
        }
        # Collect results preserving original order
        results_by_idx: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        done_count = 0
        for future in as_completed(futures):
            idx = futures[future]
            results_by_idx[idx] = future.result()
            done_count += 1
            if done_count % 20 == 0 or done_count == len(tasks):
                ticks_so_far = sum(len(r[0]) for r in results_by_idx.values())
                print(f"  Loaded {done_count}/{len(tasks)} days... {ticks_so_far:,} ticks so far")

    rollover_indices: list[int] = []
    prev_iid: int | None = None
    for idx in range(len(tasks)):
        p, t = results_by_idx[idx]
        if len(p) > 0:
            current_iid = tasks[idx][1]
            if prev_iid is not None and current_iid != prev_iid:
                rollover_indices.append(total_ticks)
            prev_iid = current_iid
            all_prices.append(p)
            all_timestamps.append(t)
            total_ticks += len(p)

    if total_ticks == 0:
        return 0

    prices = np.concatenate(all_prices)
    timestamps = np.concatenate(all_timestamps)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(
        cache_path,
        prices=prices,
        timestamps=timestamps,
        rollover_indices=np.array(rollover_indices, dtype=np.int64),
    )
    return total_ticks


def load_cache(cache_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(cache_path)
    rollover_indices = data["rollover_indices"] if "rollover_indices" in data else np.array([], dtype=np.int64)
    return data["prices"], data["timestamps"], rollover_indices
