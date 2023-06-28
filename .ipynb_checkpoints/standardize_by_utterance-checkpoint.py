import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from joblib import load
from tqdm import tqdm
import re
import argparse


def map_dataset(args):
    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Encoding dataset at {in_dir}")
    for in_path in tqdm(sorted(list(in_dir.rglob("*.npy")))):
        x = np.load(in_path)
        x = StandardScaler(with_std=args.with_rescale).fit_transform(x)
        relative_path = in_path.relative_to(in_dir)
        out_path = out_dir / relative_path.with_suffix("")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(out_path.with_suffix(".npy"), x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform utterance-level centering/standardization"
    )
    parser.add_argument("in_dir", type=Path, help="Path to the directory to encode.")
    parser.add_argument("out_dir", type=Path, help="Path to the output directory.")
    parser.add_argument('--rescale', dest='with_rescale', action='store_const',
                    const=True, default=False)
    args = parser.parse_args()
    map_dataset(args)