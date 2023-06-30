from utils import *
from joblib import dump, load 
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm


def compute_speaker_mean_std(args):
    ali = pd.read_csv(args.alignment_dir, delimiter=' ')
    ali['spk_id'] = list(map(lambda x: re.match('([0-9]+)-*',x).group(1), ali.utt_id.values))
    ali['utt_only'] = list(map(lambda x: re.search('-([0-9]+)-*',x).group(1), ali.utt_id.values))

    spk_list = list(set(ali['spk_id']))
    spk_means = dict()
    spk_std = dict()
    for i, spk in enumerate(tqdm(spk_list)):
        x_features, x_phones = aggregate_feat_phone(spk, ali, args.feature_dir, frame_rate=args.frame_rate)
        spk_means[spk] = np.mean(x_features, axis=0)
        spk_std[spk] = np.std(x_features, axis=0)
    dump(spk_means, args.feature_dir+'/spk_means')
    dump(spk_std, args.feature_dir+'/spk_std')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute speaker mean & standard deviation"
    )
    parser.add_argument("feature_dir", type=str, help="Path to the directory to encode.")
    parser.add_argument("alignment_dir", type=Path)
    parser.add_argument("--fr", dest='frame_rate', type=int, default=100)
    
    args = parser.parse_args()
    compute_speaker_mean_std(args)