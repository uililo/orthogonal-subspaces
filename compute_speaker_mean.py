from utils import *
from joblib import dump, load 
import sys

import umap
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import umap.plot
import sklearn, re
import pandas as pd
from collections import Counter
plt.figure(figsize=(15, 15), dpi=80)
font = {
        'size'   : 22}
matplotlib.rc('font', **font)

ali = pd.read_csv('LibriSpeech/dev-other.ali', delimiter=' ')
ali['spk_id'] = list(map(lambda x: re.match('([0-9]+)-*',x).group(1), ali.utt_id.values))
ali['utt_only'] = list(map(lambda x: re.search('-([0-9]+)-*',x).group(1), ali.utt_id.values))

ph_list = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH','IY','OW', 'OY', 'UH', 'UW', 'M', 'N', 'NG', 'R', 'L', 'Y', 'W', 'P', 'B', 'T', 'D', 'K', 'G', 'JH',  'HH', 'F', 'V', 'S', 'Z', 'DH', 'SH', 'CH', 'ZH', 'TH', 'SIL', 'SPN']
two_letter = ['AA', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH','IY','OW', 'OY', 'UH', 'UW', 'NG', 'SH', 'DH', 'SIL', 'EH', 'SPN', 'AY', 'AW', 'AE', 'CH' 'ZH' 'JH', 'TH', 'HH']
only_ph = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH','IY','OW', 'OY', 'UH', 'UW', 'M', 'N', 'NG', 'R', 'L', 'Y', 'W', 'P', 'B', 'T', 'D', 'K', 'G', 'JH',  'HH', 'F', 'V', 'S', 'Z', 'DH', 'SH', 'CH', 'ZH', 'TH']
consonants = [ 'M', 'N', 'NG', 'R', 'L', 'Y', 'W', 'P', 'B', 'T', 'D', 'K', 'G', 'JH',  'HH', 'F', 'V', 'S', 'Z', 'DH', 'SH', 'CH', 'ZH', 'TH']
vowels = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH','IY','OW', 'OY', 'UH', 'UW']
clean_spk = ['1272', '174', '2078', '2086', '2428', '251', '2803', '2902', '3000', '3170', '3752', '422', '5536', '5694', '6241', '6295', '652', '777', '8297', '7976', '1462', '1673', '1919', '1988', '1993', '2035', '2277', '2412', '3081', '3536', '3576', '3853', '5338', '5895', '6313', '6319', '6345', '7850', '84', '8842']
other_spk = ['116', '1255', '1585', '1630', '1650', '1651', '1686', '1701', '2506', '3660', '3663', '3915', '4153', '4323', '4515', '4570', '4572', '4831', '5543', '5849', '6123', '6267', '6455', '6467', '6599', '6841', '700', '7601', '7641', '7697', '8173', '8254', '8288']

direc = sys.argv[1]

spk_means = dict()
spk_var = dict()
for i, spk in enumerate(other_spk):
    print(spk)
    x_features, x_phones = aggregate_feat_phone(spk, ali, direc,frame_rate=100)
    # print(x_features.shape)
    spk_means[spk] = np.mean(x_features, axis=0)
    spk_var[spk] = np.var(x_features, axis=0)
dump(spk_means, direc+'/spk_means')
dump(spk_var, direc+'/spk_var')