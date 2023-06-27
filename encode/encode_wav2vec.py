import argparse
import fairseq
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from tqdm import tqdm
import re
from torchaudio.models.wav2vec2.utils import import_fairseq_model

import torch
import torchaudio


def encode_dataset(args):
    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoints")
    bundle = torchaudio.pipelines.WAV2VEC2_BASE

    # Build the model and load pretrained weight.
    model = bundle.get_model().cuda()

    #kmeans = torch.hub.load("bshall/cpc:main", "kmeans50")

    print(f"Encoding dataset at {in_dir}")
    for in_path in tqdm(sorted(list(in_dir.rglob("*.flac")))):
        wav, sr = torchaudio.load(in_path)
        assert sr == 16000

        wav = wav.cuda()
        features, _ = model.extract_features(wav)
        x = features[6].squeeze().detach().cpu().numpy()
        #codes = kmeans.predict(x)

        relative_path = in_path.relative_to(in_dir)
        out_path = out_dir / relative_path.with_suffix("")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(out_path.with_suffix(".npy"), x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode an audio dataset using CPC-big (with speaker normalization and discretization)."
    )
    parser.add_argument("in_dir", type=Path, help="Path to the directory to encode.")
    parser.add_argument("out_dir", type=Path, help="Path to the output directory.")
    args = parser.parse_args()
    encode_dataset(args)
