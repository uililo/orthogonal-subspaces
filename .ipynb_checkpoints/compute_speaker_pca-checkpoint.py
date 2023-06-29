from utils import *
from joblib import dump, load 
import sys

from tqdm import tqdm
import numpy as np
import sklearn
import re
import argparse
import pandas as pd
from collections import Counter

ali = pd.read_csv('LibriSpeech/forced_alignment/dev-clean.ali', delimiter=' ')
ali['spk_id'] = list(map(lambda x: re.match('([0-9]+)-*',x).group(1), ali.utt_id.values))
ali['utt_only'] = list(map(lambda x: re.search('-([0-9]+)-*',x).group(1), ali.utt_id.values))

def find_spk_dims(direc, row_norm, frame_rate):
    spk_vecs = []
    for spk in tqdm(clean_spk):
        x_features, x_phones = aggregate_feat_phone(spk, ali, direc, frame_rate)
        ph_id=np.array([i in only_ph for i in x_phones]).astype(bool)
        spk_vecs.append(np.mean(x_features,axis=0))
    spk_vecs = np.array(spk_vecs)
    if row_norm:
        spk_vec_n = spk_vecs - np.mean(spk_vecs, axis=1)[:,None]
    else:
        spk_vec_n = spk_vecs
    pca_spk = PCA(n_components=len(set(ali.spk_id.values)))
    pca_spk.fit(spk_vec_n)
    return pca_spk

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Identify speaker directions"
    )
    parser.add_argument("feature_dir", type=Path, help="Path to the directory to encode.")
    parser.add_argument('--row_norm', action='store_const', const=True, default=False)
    parser.add_argument("--fr", dest='frame_rate', type=int, default=100)
    args = parser.parse_args()
    spk_pca = find_spk_dims(args.feature_dir, args.row_norm, args.frame_rate)
    dump(spk_pca, args.feature_dir+'/spk_pca')