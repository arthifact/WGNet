"""OpenFWI velocity models, deterministic splits, and the simulated shot-gather cache.

Splits: for each family, the OpenFWI `train` files are pooled and permuted with a fixed seed; the first
`train_per_family` models form the training split and the next `val_per_family` the validation split.
The test split is drawn the same way from OpenFWI's own `test` files, so it is never seen in training.

Wave cache: every model is simulated with the native wgpu engine (5 shots, 15 Hz Ricker, 0.5 ms for
2001 steps) and stored decimated to 4 ms (251 samples) as float16: `[N, 5, 251, 70]`, 176 KB per model.
The physics stage re-simulates full-rate observed data on the fly for the few models it needs.
"""

from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import wave_gauss_rs

DATASET_SEED = 20261020
N = 70
H = 10.0
SHOTS = 5
NT = 2001
DT = 0.0005
PEAK_HZ = 15.0
DECIMATE = 8
NT_DEC = (NT - 1) // DECIMATE + 1  # 251
SPLITS = ("train", "val", "test")


def make_engine(shots: int = SHOTS, nt: int = NT) -> wave_gauss_rs.PyWaveEngine:
    return wave_gauss_rs.PyWaveEngine(shots, nt, DT, PEAK_HZ, 20, 1000.0, 5000.0)


class OpenFWIData:
    def __init__(
        self,
        root_dir: Path | str,
        train_per_family: int = 2000,
        val_per_family: int = 200,
        test_per_family: int = 500,
        families: tuple[str, ...] | None = None,
        seed: int = DATASET_SEED,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.openfwi_dir = self.root_dir / "data" / "openfwi"
        with open(self.openfwi_dir / "manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.families = [fam for fam in manifest["families"] if families is None or fam["id"] in families]
        if families is not None and len(self.families) != len(families):
            missing = set(families) - {f["id"] for f in self.families}
            raise ValueError(f"Families not in manifest: {sorted(missing)}")
        self.family_ids = [f["id"] for f in self.families]
        self.per_family = {"train": train_per_family, "val": val_per_family, "test": test_per_family}
        self.seed = seed
        tag = f"s{seed}_{train_per_family}_{val_per_family}_{test_per_family}_" + hashlib.blake2b(
            ",".join(self.family_ids).encode(), digest_size=4
        ).hexdigest()
        self.cache_dir = self.root_dir / "data" / f"cache_{tag}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.fields_km: dict[str, np.ndarray] = {}
        self.family_idx: dict[str, np.ndarray] = {}
        self._build_or_load_splits()

    # ------------------------------------------------------------------------------------------
    def _build_or_load_splits(self) -> None:
        fields_path = self.cache_dir / "fields.npz"
        if fields_path.exists():
            loaded = np.load(fields_path)
            for k in SPLITS:
                self.fields_km[k] = loaded[f"fields_{k}"]
                self.family_idx[k] = loaded[f"family_{k}"]
            return

        rng = np.random.default_rng(self.seed)
        out = {k: [] for k in SPLITS}
        fam_out = {k: [] for k in SPLITS}
        hashes: set[bytes] = set()
        manifest_rows = []

        for fi, fam in enumerate(self.families):
            pools = {}
            for split_name in ("train", "test"):
                files = [x for x in fam["files"] if x["split"] == split_name]
                arrays = [np.load(self.openfwi_dir / x["path"], mmap_mode="r").reshape(-1, N, N) for x in files]
                pools[split_name] = (files, arrays, np.cumsum([0] + [len(a) for a in arrays]))

            def take(pool_name: str, count: int, perm: np.ndarray, start: int, dest: str) -> int:
                files, arrays, offsets = pools[pool_name]
                chosen = np.sort(perm[start : start + count])
                for gidx in chosen:
                    fidx = int(np.searchsorted(offsets, gidx, side="right") - 1)
                    local = int(gidx - offsets[fidx])
                    m = np.ascontiguousarray(arrays[fidx][local], dtype=np.float32)
                    h = hashlib.blake2b(m.tobytes(), digest_size=16).digest()
                    if h in hashes:
                        raise ValueError(f"Duplicate model {fam['id']}/{files[fidx]['path']}[{local}] across splits")
                    hashes.add(h)
                    out[dest].append(m / 1000.0)
                    fam_out[dest].append(fi)
                    manifest_rows.append({"split": dest, "family": fam["id"], "file": files[fidx]["path"], "index": local})
                return start + count

            n_train_pool = int(pools["train"][2][-1])
            n_test_pool = int(pools["test"][2][-1])
            need_dev = self.per_family["train"] + self.per_family["val"]
            if need_dev > n_train_pool or self.per_family["test"] > n_test_pool:
                raise ValueError(
                    f"{fam['id']}: requested {need_dev} train+val of {n_train_pool} and "
                    f"{self.per_family['test']} test of {n_test_pool}"
                )
            perm_dev = rng.permutation(n_train_pool)
            pos = take("train", self.per_family["train"], perm_dev, 0, "train")
            take("train", self.per_family["val"], perm_dev, pos, "val")
            take("test", self.per_family["test"], rng.permutation(n_test_pool), 0, "test")

        for k in SPLITS:
            self.fields_km[k] = np.stack(out[k], axis=0).astype(np.float32) if out[k] else np.zeros((0, N, N), np.float32)
            self.family_idx[k] = np.asarray(fam_out[k], dtype=np.int64)
        np.savez(
            fields_path,
            **{f"fields_{k}": self.fields_km[k] for k in SPLITS},
            **{f"family_{k}": self.family_idx[k] for k in SPLITS},
        )
        with open(self.cache_dir / "split_manifest.json", "w", encoding="utf-8") as f:
            json.dump({"seed": self.seed, "families": self.family_ids, "per_family": self.per_family, "rows": manifest_rows}, f)

    # ------------------------------------------------------------------------------------------
    @property
    def train_mean_km(self) -> np.ndarray:
        return self.fields_km["train"].mean(axis=0).astype(np.float32)

    def wave_path(self, split: str) -> Path:
        return self.cache_dir / f"waves_{split}.f16.bin"

    def ensure_wave_cache(self, splits: tuple[str, ...] = SPLITS, engine=None) -> dict[str, np.memmap]:
        """Simulate (resumably) and open the decimated float16 shot-gather cache for each split."""
        if engine is None:
            engine = make_engine()
        memmaps = {}
        for split in splits:
            n_models = len(self.fields_km[split])
            shape = (n_models, SHOTS, NT_DEC, N)
            bin_path = self.wave_path(split)
            done_path = self.cache_dir / f"waves_{split}.done.json"
            prog_path = self.cache_dir / f"waves_{split}.progress.json"
            if bin_path.exists() and done_path.exists():
                memmaps[split] = np.memmap(bin_path, dtype=np.float16, mode="r", shape=shape)
                continue

            start, sq_sum = 0, 0.0
            if bin_path.exists() and prog_path.exists():
                with open(prog_path, "r", encoding="utf-8") as f:
                    prog = json.load(f)
                start, sq_sum = int(prog["completed"]), float(prog["sq_sum"])
            mm = np.memmap(bin_path, dtype=np.float16, mode="r+" if start > 0 else "w+", shape=shape)
            t0 = time.perf_counter()
            print(f"[waves/{split}] simulating {n_models - start} of {n_models} models on wgpu", flush=True)
            for i in range(start, n_models):
                vel = np.ascontiguousarray(self.fields_km[split][i].reshape(-1) * 1000.0, dtype=np.float32)
                traces = engine.simulate(vel)[:, ::DECIMATE, :]
                sq_sum += float(np.square(traces, dtype=np.float64).sum())
                mm[i] = traces.astype(np.float16)
                done = i + 1
                if done % 500 == 0 or done == n_models:
                    mm.flush()
                    with open(prog_path, "w", encoding="utf-8") as f:
                        json.dump({"completed": done, "sq_sum": sq_sum}, f)
                    rate = (done - start) / max(1e-6, time.perf_counter() - t0)
                    print(f"[waves/{split}] {done}/{n_models} ({rate:.1f} models/s, ETA {(n_models - done) / max(rate, 1e-6):.0f}s)", flush=True)
            rms = float(np.sqrt(sq_sum / max(1, n_models * SHOTS * NT_DEC * N)))
            with open(done_path, "w", encoding="utf-8") as f:
                json.dump({"split": split, "count": n_models, "rms": rms}, f, indent=2)
            prog_path.unlink(missing_ok=True)
            del mm
            memmaps[split] = np.memmap(bin_path, dtype=np.float16, mode="r", shape=shape)
        return memmaps

    def rms_scale(self) -> float:
        """Global RMS of the training gathers (the network's amplitude normalization)."""
        done_path = self.cache_dir / "waves_train.done.json"
        if not done_path.exists():
            self.ensure_wave_cache(("train",))
        with open(done_path, "r", encoding="utf-8") as f:
            return max(float(json.load(f)["rms"]), 1e-6)


def source_positions_m(shots: int = SHOTS) -> np.ndarray:
    """`[shots]` source x positions in meters matching the engine's default layout."""
    xmax = (N - 1) * H
    return np.array([xmax / 2.0], np.float32) if shots == 1 else np.linspace(0.0, xmax, shots, dtype=np.float32)
