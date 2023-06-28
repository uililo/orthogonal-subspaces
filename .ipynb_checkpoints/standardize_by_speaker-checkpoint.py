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
    spk_means = load(in_dir / "spk_means")
    spk_std = load(in_dir / "spk_std")

    print(f"Encoding dataset at {in_dir}")
    for in_path in tqdm(sorted(list(in_dir.rglob("*.npy")))):
        spk_id = re.match('%s/([0-9]*)/*'%(str(in_dir)), str(in_path)).group(1)
        x = np.load(in_path)
        x = x - spk_means[spk_id]
        if args.rescale:
            x = x / spk_std[spk_id]
        relative_path = in_path.relative_to(in_dir)
        out_path = out_dir / relative_path.with_suffix("")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(out_path.with_suffix(".npy"), x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform speaker-level centering/standardization"
    )
    parser.add_argument("in_dir", type=Path, help="Path to the directory to encode.")
    parser.add_argument("out_dir", type=Path, help="Path to the output directory.")
    parser.add_argument('--rescale', dest='rescale', action='store_const',
                    const=True, default=False)
    args = parser.parse_args()
    map_dataset(args)