import pandas as pd
import numpy as np
import scipy
import scipy.spatial as sp
from scipy.stats import rankdata
from sklearn.decomposition import PCA
import sklearn

import matplotlib
import matplotlib.pyplot as plt
import seaborn
import umap
import umap.plot

from collections import Counter, defaultdict
import re
import os

ph_list = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH','IY','OW', 'OY', 'UH', 'UW', 'M', 'N', 'NG', 'R', 'L', 'Y', 'W', 'P', 'B', 'T', 'D', 'K', 'G', 'JH',  'HH', 'F', 'V', 'S', 'Z', 'DH', 'SH', 'CH', 'ZH', 'TH', 'SIL', 'SPN']
two_letter = ['AA', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH','IY','OW', 'OY', 'UH', 'UW', 'NG', 'SH', 'DH', 'SIL', 'EH', 'SPN', 'AY', 'AW', 'AE', 'CH' 'ZH' 'JH', 'TH', 'HH']
only_ph = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH','IY','OW', 'OY', 'UH', 'UW', 'M', 'N', 'NG', 'R', 'L', 'Y', 'W', 'P', 'B', 'T', 'D', 'K', 'G', 'JH',  'HH', 'F', 'V', 'S', 'Z', 'DH', 'SH', 'CH', 'ZH', 'TH']
consonants = [ 'M', 'N', 'NG', 'R', 'L', 'Y', 'W', 'P', 'B', 'T', 'D', 'K', 'G', 'JH',  'HH', 'F', 'V', 'S', 'Z', 'DH', 'SH', 'CH', 'ZH', 'TH']
vowels = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH','IY','OW', 'OY', 'UH', 'UW']
clean_spk = ['1272', '174', '2078', '2086', '2428', '251', '2803', '2902', '3000', '3170', '3752', '422', '5536', '5694', '6241', '6295', '652', '777', '8297', '7976', '1462', '1673', '1919', '1988', '1993', '2035', '2277', '2412', '3081', '3536', '3576', '3853', '5338', '5895', '6313', '6319', '6345', '7850', '84', '8842']
other_spk = ['116', '1255', '1585', '1630', '1650', '1651', '1686', '1701', '2506', '3660', '3663', '3915', '4153', '4323', '4515', '4570', '4572', '4831', '5543', '5849', '6123', '6267', '6455', '6467', '6599', '6841', '700', '7601', '7641', '7697', '8173', '8254', '8288']
test_spk = ['1188', '260', '5142', '1995', '4970', '1221', '121', '1320', '61', '7127', '7176', '1580', '2830', '7729', '1089', '1284', '2300', '3729', '8230', '6829', '3570', '5639', '237', '8224', '4992', '5683', '8463', '4507', '672', '2094', '6930', '908', '5105', '7021', '2961', '3575', '4077', '8455', '4446', '8555']
train_spk = ['374','7800','2514','3240','1088','5456','5750','1246','8238','1263','7505','587','226','1743','4214','5789','7635','5390','307','7447','4362','6529','233','3242','1624','4297','6181','6367','3723','8123','6563','403','5778','3112','7312','7367','7078','32','5322','3214','6818','481','5104','6385','5192','8226','3830','2989','8324','163','150','6476','1069','3983','1183','4788','426','311','2196','103','446','1502','8975','8770','1992','5678','8014','2182','7178','201','1034','5703','1363','250','6836','3168','1553','5163','89','1334','19','5393','4481','4160','8312','6415','87','7067','5688','2843','909','40','322','8797','2764','6848','3947','4014','6531','3664','3259','4441','7794','5463','5049','4018','4088','4853','7226','4859','78','7113','3440','460','2893','4680','302','4830','2518','4898','7780','1926','1963','1841','3526','254','1970','6209','458','7148','831','6147','839','8425','200','1723','2416','6019','4813','1455','2391','2910','6000','7302','2817','445','8468','2384','8630','4267','26','118','328','1867','3374','5022','8108','6081','8095','5514','8838','2007','2002','196','248','198','4340','5339','6454','4051','3982','6078','3857','1098','5867','2159','83','730','1235','8629','696','289','1116','5808','8063','8465','6272','6064','412','3607','1594','7278','625','2836','7859','3807','1355','332','8580','911','6880','8051','8088','3436','887','3879','39','3235','211','5652','2136','4406','27','1737','7059','125','3486','2911','7190','6437','2092','7517','6925','8747','7402','8609','2691','2952','1040','1081','2289','298','4397','7264','1578','60','229','3699','8419','4137','405','2436','1898','7511','4195','669','5561','1447','441','8098','4640']


monophthongs = ['AA', 'AE', 'AH', 'AO', 'EH', 'ER','IH','IY', 'UH', 'UW']
diphthongs = ['AW', 'AY', 'EY', 'OW', 'OY']
approximants = ['R', 'L', 'Y', 'W']
fricatives = ['HH', 'F', 'V', 'S', 'Z', 'DH', 'SH', 'ZH', 'TH']
affricates = ['JH', 'CH']
nasals = [ 'M', 'N', 'NG' ]
plosives = ['P', 'B', 'T', 'D', 'K', 'G']

def remove_bie(label):
    return re.sub('[0-9]','',label.split('_')[0])

def create_phone_labels(spk_ch_utt, feat_len, ali, frame_rate=100):
    selected_ali = ali[ali.utt_id == spk_ch_utt]
    phone_dur = selected_ali.phone_dur.values * frame_rate
    phone = selected_ali.phone.values
    phone = list(map(remove_bie, phone))

    phone_labels = []
    for dur, ph in zip(phone_dur, phone):
        phone_labels.extend([ph]*int(dur))
    if feat_len > len(phone_labels):
        pad_len = feat_len - len(phone_labels)
        phone_labels.extend(['SIL']*pad_len)
    elif feat_len < len(phone_labels):
        print('discrepant lengths - feature length: %d, label length: %d'%(feat_len, len(phone_labels)))
    return phone_labels

def aggregate_feat_phone(spk, ali, direc, frame_rate=100):
    feat = []
    phone = []
    for utt in set(ali[ali.spk_id == str(spk)].utt_id.values):
        utt_only = re.search('-([0-9]+)-*',utt).group(1)
        x = np.load(direc+'/%s/%s/%s.npy'%(str(spk),utt_only,utt))
        feat.append(x)
        phone.extend(create_phone_labels(utt, len(x), ali, frame_rate))
    return np.concatenate(feat, axis=0), phone

def compute_phone_centroid(x_feat, x_phone):
    x_phone_emb = defaultdict(list)
    x_phone_centroid = dict()
    for feat, phone in zip(x_feat, x_phone):
        x_phone_emb[phone].append(feat)
    phone_occurred = set(x_phone)
    for ph in ph_list:
        if ph in phone_occurred:
            x_phone_centroid[ph] = np.mean(np.array(x_phone_emb[ph]), axis=0)
        # print(x_phone_centroid[0].shape)
    return x_phone_centroid

def plot_cos_sim(cos_sim, ph_list):
    fig, ax = plt.subplots(figsize=(15, 15), dpi=80)
    im = ax.imshow(cos_sim, vmin=-0.3, vmax=1)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(ph_list)))
    ax.set_yticks(np.arange(len(ph_list))) 
    ax.set_xticklabels(ph_list)
    ax.set_yticklabels(ph_list)
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")
    im.set_cmap("Blues_r")
    plt.colorbar(im)
    plt.show()
    
def procrustes_mapping(source, target):
    R = scipy.linalg.orthogonal_procrustes(source, target)[0]
    return np.dot(source, R), R

def cos_sim(x_feat, y_feat):
    return 1 - sp.distance.cdist(x_feat, y_feat, 'cosine')

def compare_cos_sim(cs_a, cs_b, ph_list):
    a = [cs_a[i, i] for i in range(len(cs_a))]
    b = [cs_b[i, i] for i in range(len(cs_a))]
    for i, ph in enumerate(ph_list):
        print(ph, a[i], b[i])
        
def present_cos_sim(src1, tgt1, src2, tgt2, ph_list):
    mapped1, _ = procrustes_mapping(src1, tgt1)
    cs_1 = cos_sim(mapped1, tgt1)
    cs_2 = cos_sim(src2, tgt2)
    
    compare_cos_sim(cs_1, cs_2, ph_list)
    
def cos_sim_summary(cs_1, cs_2, ph_list):
    for i, ph in enumerate(ph_list):
        print(ph, ph_list[np.argmax(cs_1[i])],np.max(cs_1[i]), ph_list[np.argsort(cs_1[i])[-2]], sorted(cs_1[i])[-2], ph_list[np.argmax(cs_2[i])],np.max(cs_2[i]))

def avg_diagonal_sim(x_feat, y_feat):
    cs = cos_sim(x_feat, y_feat)
    diag_sim = [cs[i,i] for i in range(len(x_feat)-1)]
    # print(len(diag_sim))
    return np.mean(diag_sim)

def umap_emb(x_feat, y_feat):
    combined = np.concatenate([x_feat, y_feat])
    mapper = umap.UMAP().fit(combined)
    emb = umap.plot._get_embedding(mapper)
    x_len = len(x_feat)
    x_emb = emb[:x_len]
    y_emb = emb[x_len:]
    return x_emb, y_emb, mapper
        
def plot_phone_vec(x_feat, x_vector, y_feat, y_vector):
    x_emb, y_emb, mapper_n = umap_emb(x_feat, y_feat) 
    x_vector_emb = mapper_n.transform(x_vector)
    y_vector_emb = mapper_n.transform(y_vector)
    plt.figure(figsize=(15, 15), dpi=80)
    for i, ph in enumerate(ph_list):
        if ph in two_letter:
            plt.scatter(x_vector_emb[i,1], x_vector_emb[i,0], alpha=0.5, s=500, color='r', marker='$%s$'%ph)
            plt.scatter(y_vector_emb[i,1], y_vector_emb[i,0], alpha=0.5, s=500, color='g', marker='$%s$'%ph)
        else:
            plt.scatter(x_vector_emb[i,1], x_vector_emb[i,0], alpha=0.5, s=200, color='r', marker='$%s$'%ph)
            plt.scatter(y_vector_emb[i,1], y_vector_emb[i,0], alpha=0.5, s=200, color='g', marker='$%s$'%ph)

def umap_emb(x_feat, y_feat):
    combined = np.concatenate([x_feat, y_feat])
    mapper = umap.UMAP().fit(combined)
    emb = umap.plot._get_embedding(mapper)
    x_len = len(x_feat)
    x_emb = emb[:x_len]
    y_emb = emb[x_len:]
    return x_emb, y_emb, mapper

def pca_map(x_feat, y_feat, x_centroid, y_centroid, n_comp):
    pca_x = PCA(n_components=n_comp)
    pca_x.fit(x_feat)
    x_pc = pca_x.components_

    pca_y = PCA(n_components=n_comp)
    pca_y.fit(y_feat)
    y_pc = pca_y.components_
    x_n = pca_x.transform(x_centroid_n)
    y_n = pca_y.transform(y_centroid_n)
    _, R = procrustes_mapping(x_n, y_n)
    x_centroid_mapped = np.dot(x_n, R)
    cos_sim_mapped = 1 - sp.distance.cdist(x_centroid_mapped, y_n, 'cosine')
    # cos_sim = 1 - sp.distance.cdist(x_n, y_n, 'cosine')
    # present_cos_sim(x_centroid_mapped, y_n, x_n, y_n)
    # plot_cos_sim(cos_sim(x_n,y_n))
    # plot_cos_sim(cos_sim_mapped)
    diag_sim = np.mean([cos_sim_mapped[i,i] for i in range(len(ph_list)-1)])
    # print(diag_sim)
    return pca_x, pca_y, R

def shared_phones(a_phone, t_phone):
    shared = []
    for ph in ph_list:
        if ph in a_phone and ph in t_phone:
            shared.append(ph)
    return shared

def normalise_ph_vecs(x):
    return x / np.linalg.norm(x,axis=1)[:,None]