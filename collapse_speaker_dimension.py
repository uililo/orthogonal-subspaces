import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from joblib import load
from tqdm import tqdm
import re
import argparse

def explained_var_hyperpar(pca, t=0.95):
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    if t == 1:
        return len(cumsum)
    elif t < 1 and t > 0:
        for i in range(len(cumsum)):
            if cumsum[i] > t:
                return i+1, cumsum[i]
        
def collapse_dimensions(feat, pca, idx):
    return feat - np.dot(np.dot(feat, pca.components_[:idx].transpose()), pca.components_[:idx])

def collapse_dimensions_for_dataset(args):
    in_dir, pca_dir, out_dir = Path(args.in_dir), Path(args.pca_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spk_pca = load(pca_dir)
    dim_idx, sum_var = explained_var_hyperpar(spk_pca, args.threshold)
    
    with open(out_dir / 'log.txt', 'w') as fo:
        fo.write('removing %d speaker dimensions (%.1f percent variance)' % (dim_idx, sum_var*100))

    print(f"Encoding dataset at {in_dir}")
    all_files = sorted(list(in_dir.rglob("*.npy")))
    for in_path in tqdm(all_files):
        x = np.load(in_path)
        x = collapse_dimensions(x, spk_pca, dim_idx)
        relative_path = in_path.relative_to(in_dir)
        out_path = out_dir / relative_path.with_suffix("")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path.with_suffix(".npy"), x)
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode an audio dataset using CPC-big (with speaker normalization and discretization)."
    )
    parser.add_argument("in_dir", type=Path, help="Path to the directory to encode.")
    parser.add_argument("pca_dir", type=Path, help="Filename of speaker PCA")
    parser.add_argument("out_dir", type=Path, help="Path to the output directory.")
    parser.add_argument("threshold", type=float, help="threshold for cumulative value of explained variance")
    args = parser.parse_args()
    collapse_dimensions_for_dataset(args)