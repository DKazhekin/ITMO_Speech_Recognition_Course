"""Data-pipeline modules: audio-side augmentation and SpecAugment."""

from gp1.data.audio_aug import AudioAugmenter
from gp1.data.spec_aug import SpecAugmenter

# Wave-2 additions
from gp1.data.collate import DynamicBucketSampler, collate_fn
from gp1.data.dataset import SpokenNumbersDataset
from gp1.data.manifest import (
    build_manifest,
    leave_n_speakers_out_split,
    read_jsonl,
    write_jsonl,
)

__all__ = [
    "AudioAugmenter",
    "SpecAugmenter",
    "DynamicBucketSampler",
    "collate_fn",
    "SpokenNumbersDataset",
    "build_manifest",
    "leave_n_speakers_out_split",
    "read_jsonl",
    "write_jsonl",
]
