import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from tqdm import tqdm
import re

import torch
import torchaudio


def encode_dataset(args):
    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Encoding dataset at {in_dir}")
    for in_path in tqdm(sorted(list(in_dir.rglob("*.flac")))):
        wav, sr = torchaudio.load(in_path)
        assert sr == 16000

        wav = wav.view(1,-1)
        x = torchaudio.compliance.kaldi.fbank(wav, num_mel_bins=40)
        x = (StandardScaler().fit_transform(x)).astype('float32')
        #codes = kmeans.predict(x)

        relative_path = in_path.relative_to(in_dir)
        out_path = out_dir / relative_path.with_suffix("")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(out_path.with_suffix(".npy"), x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract log mel feature for an audio dataset."
    )
    parser.add_argument("in_dir", type=Path, help="Path to the directory to encode.")
    parser.add_argument("out_dir", type=Path, help="Path to the output directory.")
    args = parser.parse_args()
    encode_dataset(args)
