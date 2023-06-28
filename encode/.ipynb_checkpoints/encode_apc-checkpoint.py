import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from tqdm import tqdm
import re
from apc_model import APCModel
from apc_utils import RNNConfig


import torch
import torchaudio


def encode_dataset(args):
    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoints")
    model = APCModel(mel_dim=40, prenet_config=None, rnn_config = RNNConfig(40, 512, 3, 0, True, 'LSTM')).cuda()
    model.load_state_dict(torch.load("models/apc.pth.tar", map_location=torch.device('cpu'))['model'])
    model.eval()

    print(f"Encoding dataset at {in_dir}")
    for in_path in tqdm(sorted(list(in_dir.rglob("*.flac")))):
        wav, sr = torchaudio.load(in_path)
        assert sr == 16000

        wav = wav.view(1,-1)
        x = torchaudio.compliance.kaldi.fbank(wav, num_mel_bins=40)
        x = (StandardScaler().fit_transform(x)).astype('float32')
        x = torch.tensor(x).cuda()
        l = x.shape[0]
        x = x.reshape(1, l, -1)
        l = torch.tensor([l]).cuda()
        
        _, internal_reps = model(x, [l])
        apc_feat = internal_reps[-1][0].detach().cpu().numpy()

        relative_path = in_path.relative_to(in_dir)
        out_path = out_dir / relative_path.with_suffix("")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(out_path.with_suffix(".npy"), apc_feat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode an audio dataset using APC."
    )
    parser.add_argument("in_dir", type=Path, help="Path to the directory to encode.")
    parser.add_argument("out_dir", type=Path, help="Path to the output directory.")
    args = parser.parse_args()
    encode_dataset(args)

