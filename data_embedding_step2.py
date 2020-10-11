# coding: utf-8
import pandas as pd
import numpy as np
import pickle
import tqdm
from itertools import chain

ind_ct_topics = ['betos', 'rx', 'submcc']
ind_topics = ['bh', 'cms', 'cmsd2', 'cons', 'hedis', 'lab', 'phy', 'prov', 'rev', 'smoker', 'ccsp']  # Special ind: hedis, cons
dense_topics = ['med', 'pdc', 'total', 'credit']
keep_cols = ['est_age', 'hcc_weighted_sum', 'cms_ma_risk_score_nbr', 'cms_partd_ra_factor_amt', 'cms_risk_adj_payment_rate_a_amt', 'cms_risk_adj_payment_rate_b_amt', 'cms_risk_adjustment_factor_a_amt', 'cms_rx_risk_score_nbr', 'cms_tot_ma_payment_amt', 'cms_tot_partd_payment_amt']
score_cols = ['cci_score', 'dcsi_score', 'fci_score']
category_cols = ['zip_cd', 'cons_cmys', 'cons_hhcomp', 'cons_homstat', 'cons_hcaccprf', 'lang_spoken_cd', 'mabh_seg', 'rucc_category', 'sex_cd', 'cms_ra_factor_type_cd']
drop_cols = ['lab_abn_result_ind', 'rx_overall_pmpm_ct', 'src_platform_cd']

test_data = pd.read_csv('../data/cleaned_test_no_one_hot.csv', low_memory=False)


train_data = pd.read_csv('../data/cleaned_train_no_one_hot.csv', low_memory=False)

all_data = pd.concat([train_data, test_data], sort=True)\nall_data = all_data.reset_index(drop=True)


embedding_data = pd.DataFrame()

for topic in chain(ind_ct_topics, ind_topics, score_cols, category_cols, dense_topics):
    arr = pickle.load(open('../data/{}.pkl'.format(topic), 'rb'))
    for i in range(arr.shape[1]):
        embedding_data[topic + '_' + str(i)] = arr[:,i]


embedding_data[keep_cols] = all_data[keep_cols]


embedding_data['transportation_issues'] = all_data['transportation_issues']


train_data = embedding_data[:train_data.shape[0]]
test_data = embedding_data[train_data.shape[0]:]

train_data.info()

train_data.to_csv('../data/embedding_train.csv', index=False)

test_data = test_data.drop(columns=['transportation_issues'])\ntest_data.to_csv('../data/embedding_test.csv', index=False)

