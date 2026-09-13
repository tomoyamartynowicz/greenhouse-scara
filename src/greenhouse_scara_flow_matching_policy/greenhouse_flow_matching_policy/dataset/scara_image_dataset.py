from __future__ import annotations

import copy
import warnings
import re
from pathlib import Path
from typing import Dict, Mapping, Optional

import h5py
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from greenhouse_flow_matching_policy.dataset.base_dataset import BaseImageDataset
from greenhouse_flow_matching_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from greenhouse_scara_common.data import (image_specs, validate_episode, action_dataset, read_images)


_EPISODE_PATTERN = re.compile(r"episode_(\d+)\.hdf5$")


def _episode_number(path: Path) -> int:
    match = _EPISODE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Invalid episode filename: {path.name}")
    return int(match.group(1))


def _get_val_mask(n_episodes: int, val_ratio: float, seed: int) -> np.ndarray:
    """Use the upstream validation split, implemented locally without numba."""
    mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return mask
    n_validation = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    validation_indices = np.random.default_rng(seed).choice(
        n_episodes,
        size=n_validation,
        replace=False,
    )
    mask[validation_indices] = True
    return mask


def _downsample_mask(
    mask: np.ndarray,
    max_n: Optional[int],
    seed: int,
) -> np.ndarray:
    """Use the upstream episode downsampling rule."""
    if max_n is None or int(mask.sum()) <= int(max_n):
        return mask
    candidates = np.flatnonzero(mask)
    selected = np.random.default_rng(seed).choice(
        candidates,
        size=int(max_n),
        replace=False,
    )
    result = np.zeros_like(mask)
    result[selected] = True
    return result


class ScaraImageDataset(BaseImageDataset):
    """Lazy multimodal HDF5 windows; derive missing next-qpos actions in memory."""

    def __init__(
        self,
        shape_meta: Mapping,
        dataset_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: Optional[int] = None,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
        action_source="auto", action_offset=1, allow_single_episode=False,
    ) -> None:
        super().__init__()

        self.dataset_path = Path(dataset_path).expanduser().resolve()
        if not self.dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_path}")

        self.horizon = int(horizon)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.n_obs_steps = self.horizon if n_obs_steps is None else int(n_obs_steps)
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if not 1 <= self.n_obs_steps <= self.horizon:
            raise ValueError("n_obs_steps must be between 1 and horizon")
        if not 0 <= self.pad_before < self.horizon:
            raise ValueError("pad_before must be in [0, horizon)")
        if not 0 <= self.pad_after < self.horizon:
            raise ValueError("pad_after must be in [0, horizon)")
        if not 0.0 <= float(val_ratio) < 1.0:
            raise ValueError("val_ratio must be in [0, 1)")

        self.shape_meta = shape_meta
        self.specs = image_specs(shape_meta)
        self.rgb_keys = list(self.specs)  # Visual keys, including depth.
        self.lowdim_keys = [k for k, v in shape_meta['obs'].items() if v.get('type', 'low_dim') == 'low_dim']
        if self.lowdim_keys != ['qpos'] or list(shape_meta['obs']['qpos']['shape']) != [4]:
            raise ValueError('Greenhouse policies require qpos with shape [4]')
        self.obs_shapes = {k: tuple(v['shape']) for k,v in shape_meta['obs'].items()}
        self.action_shape = tuple(shape_meta['action']['shape'])
        if self.action_shape != (4,): raise ValueError('Action shape must be [4]')
        self.action_source = action_source
        self.action_offset = action_offset
        self.allow_single_episode = allow_single_episode

        episode_paths = [
            path
            for path in self.dataset_path.glob("episode_*.hdf5")
            if _EPISODE_PATTERN.fullmatch(path.name)
        ]
        self.episode_paths = sorted(episode_paths, key=_episode_number)
        if not self.episode_paths:
            raise FileNotFoundError(f"No episode_*.hdf5 files found in {self.dataset_path}")

        if len(self.episode_paths) == 1 and self.val_ratio_requested(val_ratio):
            if not allow_single_episode:
                raise ValueError('Need at least two demos for held-out validation; set allow_single_episode=true only for a smoke test')
            warnings.warn('Single-demo smoke test: training and validation use the SAME episode', stacklevel=2)

        self.episode_lengths = np.empty(len(self.episode_paths), dtype=np.int64)
        self._validate_episodes()

        self.val_mask = _get_val_mask(
            n_episodes=len(self.episode_paths),
            val_ratio=float(val_ratio),
            seed=int(seed),
        )
        self.train_mask = _downsample_mask(
            mask=~self.val_mask,
            max_n=max_train_episodes,
            seed=int(seed),
        )
        if not np.any(self.train_mask):
            raise ValueError("The train split contains no episodes")

        self._normalizer_mask = self.train_mask.copy()
        self._normalizer_data: Optional[dict[str, np.ndarray]] = None
        self.active_mask = self.train_mask.copy()
        self.samples = self._make_samples(self.active_mask)
        if not self.samples: raise ValueError("Episodes are too short for this horizon/padding")

    @staticmethod
    def val_ratio_requested(val_ratio):
        return float(val_ratio) > 0

    def _validate_episodes(self):
        for index, path in enumerate(self.episode_paths):
            with h5py.File(path, 'r') as episode:
                self.episode_lengths[index] = validate_episode(
                    episode, self.shape_meta, self.action_source, self.action_offset)

    def _make_samples(self, episode_mask: np.ndarray) -> list[tuple[int, int]]:
        samples: list[tuple[int, int]] = []
        for episode_index in np.flatnonzero(episode_mask):
            length = int(self.episode_lengths[episode_index])
            min_start = -self.pad_before
            max_start = length - self.horizon + self.pad_after
            samples.extend(
                (int(episode_index), start)
                for start in range(min_start, max_start + 1)
            )
        return samples

    @staticmethod
    def _read_padded(dataset: h5py.Dataset, start: int, length: int) -> np.ndarray:
        source_start = max(start, 0)
        source_end = min(start + length, int(dataset.shape[0]))
        if source_start >= source_end:
            raise IndexError("Requested sequence does not overlap the episode")

        data = np.asarray(dataset[source_start:source_end])
        left = source_start - start
        right = start + length - source_end
        parts = []
        if left:
            parts.append(np.repeat(data[:1], left, axis=0))
        parts.append(data)
        if right:
            parts.append(np.repeat(data[-1:], right, axis=0))
        return np.concatenate(parts, axis=0) if len(parts) > 1 else data

    def get_validation_dataset(self) -> "ScaraImageDataset":
        validation = copy.copy(self)
        validation.active_mask = self.val_mask.copy()
        if len(self.episode_paths) == 1 and self.allow_single_episode:
            validation.active_mask[:] = True
        validation.samples = validation._make_samples(validation.active_mask)
        return validation

    def _get_normalizer_data(self) -> dict[str, np.ndarray]:
        if self._normalizer_data is None:
            parts: dict[str, list[np.ndarray]] = {
                "action": [],
                **{key: [] for key in self.lowdim_keys},
            }
            for episode_index in np.flatnonzero(self._normalizer_mask):
                with h5py.File(self.episode_paths[int(episode_index)], "r") as episode:
                    parts["action"].append(np.asarray(action_dataset(episode, self.action_source, self.action_offset)[:], dtype=np.float32))
                    for key in self.lowdim_keys:
                        parts[key].append(
                            np.asarray(episode[f"observations/{key}"], dtype=np.float32)
                        )
            self._normalizer_data = {
                key: np.concatenate(values, axis=0) for key, values in parts.items()
            }
        return self._normalizer_data

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer.fit(
            data=self._get_normalizer_data(),
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        for key in self.rgb_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self._get_normalizer_data()["action"])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        episode_index, start = self.samples[index]
        with h5py.File(self.episode_paths[episode_index], "r") as episode:
            obs: dict[str, torch.Tensor] = {}
            for key, spec in self.specs.items():
                obs[key] = torch.from_numpy(read_images(episode, key, spec, start, self.n_obs_steps))
            for key in self.lowdim_keys:
                values = self._read_padded(
                    episode[f"observations/{key}"],
                    start=start,
                    length=self.n_obs_steps,
                )
                obs[key] = torch.from_numpy(
                    np.ascontiguousarray(values, dtype=np.float32)
                )
            action = self._read_padded(
                action_dataset(episode, self.action_source, self.action_offset),
                start=start,
                length=self.horizon,
            )

        return {
            "obs": obs,
            "action": torch.from_numpy(np.ascontiguousarray(action, dtype=np.float32)),
        }
