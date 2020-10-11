# coding: utf-8
import pandas as pd
import numpy as np
import pickle
import tqdm
from sklearn.decomposition import PCA

train_data = pd.read_csv('../data/cleaned_train.csv', low_memory=False)

test_data = pd.read_csv('../data/cleaned_test.csv', low_memory=False)

col_embedding = pickle.load(open('../data/col_embedding.pkl', 'rb'))

all_data = pd.concat([train_data, test_data], sort=True)\nall_data = all_data.reset_index(drop=True)
del train_data
del test_data


ind_ct_topics = ['betos', 'rx', 'submcc']
ind_topics = ['bh', 'cms', 'cmsd2', 'cons', 'hedis', 'lab', 'phy', 'prov', 'rev', 'smoker', 'ccsp']  # Special ind: hedis, cons
dense_topics = ['med', 'pdc', 'total', 'credit']
keep_cols = ['est_age', 'hcc_weighted_sum', 'cms_ma_risk_score_nbr', 'cms_partd_ra_factor_amt', 'cms_risk_adj_payment_rate_a_amt', 'cms_risk_adj_payment_rate_b_amt', 'cms_risk_adjustment_factor_a_amt', 'cms_rx_risk_score_nbr', 'cms_tot_ma_payment_amt', 'cms_tot_partd_payment_amt']
score_cols = ['cci_score', 'dcsi_score', 'fci_score']
category_cols = ['zip_cd', 'cons_cmys', 'cons_hhcomp', 'cons_homstat', 'cons_hcaccprf', 'lang_spoken_cd', 'mabh_seg', 'rucc_category', 'sex_cd', 'cms_ra_factor_type_cd']
drop_cols = ['lab_abn_result_ind', 'rx_overall_pmpm_ct', 'src_platform_cd']

ind_ct_dict = {}
for topic in ind_ct_topics:
    topic_cols = {}
    for col in all_data.columns:
        if col.split('_')[0] == topic and col.split('_')[-1] == 'ind':
            topic_cols[col] = '_'.join(col.split('_')[:-1]) + '_pmpm_ct'
    ind_ct_dict[topic] = topic_cols

ind_dict = {}
for topic in ind_topics:
    if topic in ['hedis', 'cons']:
        continue
    topic_cols = []
    for col in all_data.columns:
        if col.split('_')[0] == topic and col.split('_')[-1] == 'ind' and col not in drop_cols:
            topic_cols.append(col)
    ind_dict[topic] = topic_cols

for topic in ['hedis', 'cons']:
    topic_cols = []
    for col in all_data.columns:
        if col.split('_')[0] == topic and col not in category_cols:
            topic_cols.append(col)
    ind_dict[topic] = topic_cols


dense_dict = {}
for topic in dense_topics:
    topic_cols = []
    for col in all_data.columns:
        if col.split('_')[0] == topic and col.split('_')[-1] != 'ind' and col not in keep_cols and col not in category_cols:
            topic_cols.append(col)
    dense_dict[topic] = topic_cols

def embedding_zip_cd(zip_cd):
    if zip_cd == 'other':
           return col_embedding['other_zip_cd']
    return col_embedding[zip_cd]


def embedding_cons_cmys(cons_cmys):
    if cons_cmys == '0':
           return col_embedding['unknown education level']
    if cons_cmys == '2':
           return col_embedding['less than 12th grade']
    if cons_cmys == '3':
           return col_embedding['high school diploma']
    if cons_cmys == '4':
           return col_embedding['some college']
    if cons_cmys == '5':
           return col_embedding['associate degree']
    return col_embedding['bachelors degree']


def embedding_cons_hhcomp(cons_hhcomp):
    if cons_hhcomp == 'U':
           return col_embedding['composition unknown']
    if cons_hhcomp == 'B':
           return col_embedding['married with no children']
    if cons_hhcomp == 'L':
           return col_embedding['female householder with no children']
    if cons_hhcomp == 'A':
           return col_embedding['married with children']
    if cons_hhcomp == 'J':
           return col_embedding['male householder with no children']
    if cons_hhcomp == 'D':
           return col_embedding['marital status unknown with no children']
    if cons_hhcomp == 'K':
           return col_embedding['female householder with children']
    if cons_hhcomp == 'G':
           return col_embedding['female householder with one or more other persons of any gender with children']
    if cons_hhcomp == 'O':
           return col_embedding['other marital status']
    if cons_hhcomp == 'C':
           return col_embedding['marital status unknown with children']
    if cons_hhcomp == 'H':
           return col_embedding['female householder with one or more other persons of any gender with no children']
    if cons_hhcomp == 'E':
           return col_embedding['male householder with one or more other persons of any gender with children']
    return col_embedding['male householder with children']


def embedding_cons_homstat(cons_homstat):
    if cons_homstat == '1':
           return col_embedding['homeowner']
    if cons_homstat == 'U':
           return col_embedding['unknown home status']
    if cons_homstat == 'P':
           return col_embedding['probable homeowner']
    if cons_homstat == 'R':
           return col_embedding['renter']
    return col_embedding['probable renter']



def embedding_cons_hcaccprf(cons_hcaccprf):
    if cons_hcaccprf == 'cons_hcaccprf_p':
           return col_embedding['personal doctor or personal care physician']
    if cons_hcaccprf == 'cons_hcaccprf_o':
           return col_embedding['other healthcare treatment preference']
    if cons_hcaccprf == 'cons_hcaccprf_c':
           return col_embedding['community health clinic or health clinic in retail setting']
    return col_embedding['hospital or standalone emergency room or urgent care center']


def embedding_lang_spoken_cd(lang_spoken_cd):
    if lang_spoken_cd == 'ENG':
           return col_embedding['english']
    if lang_spoken_cd == 'SPA':
           return col_embedding['spanish']


def embedding_mabh_seg(mabh_seg):
    if mabh_seg == 'H2':
           return col_embedding['healthy auto-pilot participator']
    if mabh_seg == 'UNK':
           return col_embedding['unknown health status']
    if mabh_seg == 'H6':
           return col_embedding['healthy skeptical control seeker A']
    if mabh_seg == 'H1':
           return col_embedding['healthy self engaged optimist']
    if mabh_seg == 'C4':
           return col_embedding['chronic auto-pilot participator']
    if mabh_seg == 'C2':
           return col_embedding['chronic self engaged optimist']
    if mabh_seg == 'H7':
           return col_embedding['healthy skeptical control seeker B']
    if mabh_seg == 'C5':
           return col_embedding['chronic simplicity seeking follower']
    if mabh_seg == 'H4':
           return col_embedding['healthy self sustainer']
    if mabh_seg == 'C3':
           return col_embedding['chronic health services maximizer']
    if mabh_seg == 'H3':
           return col_embedding['healthy health services maximizer']
    if mabh_seg == 'H8':
           return col_embedding['healthy overwhelmed and reluctant reactor']
    if mabh_seg == 'H5':
           return col_embedding['healthy simplicity seeking follower']
    if mabh_seg == 'C6':
           return col_embedding['chronic skeptical control seeker']
    if mabh_seg == 'C1':
           return col_embedding['chronic support seeking participator']
    return col_embedding['chronic overwhelmed and reluctant reactor']




def embedding_cms_ra_factor_type_cd(cms_ra_factor_type_cd):
    if cms_ra_factor_type_cd == 'CN':
           return col_embedding['risk adjustment factor community non-dual']
    if cms_ra_factor_type_cd == 'CP':
           return col_embedding['risk adjustment factor community partial dual']
    if cms_ra_factor_type_cd == 'E':
           return col_embedding['risk adjustment factor new enrollee']
    if cms_ra_factor_type_cd == 'CF':
           return col_embedding['risk adjustment factor community full dual']
    if cms_ra_factor_type_cd == 'D':
           return col_embedding['risk adjustment factor dialysis']
    if cms_ra_factor_type_cd == 'C2':
           return col_embedding['risk adjustment factor community post graft']
    if cms_ra_factor_type_cd == 'I':
           return col_embedding['risk adjustment factor institutional']
    return col_embedding['risk adjustment factor new enrollee chronic care']



def embedding_sex_cd(sex_cd):
    if int(sex_cd) == 0:
           return col_embedding['female']
    return col_embedding['male']



def embedding_rucc_category(rucc_category):
    if rucc_category == '1-Metro':
           return col_embedding['metro counties in metro areas of one-million population or more']
    if rucc_category == '2-Metro':
           return col_embedding['metro counties in metro areas of 250000 to one-million population']
    if rucc_category == '3-Metro':
           return col_embedding['metro counties in metro areas of fewer than 250000 population']
    if rucc_category == '4-Nonmetro':
           return col_embedding['nonmetro urban population of 20000 or more adjacent to a metro area']
    if rucc_category == '5-Nonmetro':
           return col_embedding['nonmetro urban population of 20000 or more not adjacent to a metro area']
    if rucc_category == '6-Nonmetro':
           return col_embedding['nonmetro urban population of 2500 to 19999 adjacent to a metro area']
    if rucc_category == '7-Nonmetro':
           return col_embedding['nonmetro urban population of 2500 to 19999 not adjacent to a metro area']
    if rucc_category == '8-Nonmetro':
           return col_embedding['nonmetro completely rural or less than 2500 urban population adjacent to a metro area']
    return col_embedding['nonmetro completely rural or less than 2500 urban population not adjacent to a metro area']



embedding_data = pd.DataFrame()



def embedding_ind_ct(row):
    return_list = []
    for topic, pairs in list(ind_ct_dict.items()):
        topic_list = []
        for ind, ct in ind_ct_dict[topic].items():
            if row[ind] * row[ct] == 0:
                continue
            topic_list.append(col_embedding[ind] * row[ind] * row[ct])
        if not topic_list:
            topic_list.append(np.zeros(300,))
        return_list.append(np.average(topic_list, axis=0))
    return return_list




embedding_data[list(ind_ct_dict.keys())] = all_data.apply(embedding_ind_ct, axis=1, result_type="expand")



print(embedding_data['betos'][0].shape)
print(embedding_data['rx'][0].shape)
print(embedding_data['submcc'][0].shape)


for topic in ind_ct_topics:
    topic_array = np.stack(embedding_data[topic])
    print('{}: Finished Stack'.format(topic))
    embedding_data.drop(columns=[topic], inplace=True)
    print('{}: Finished Dropping'.format(topic))
    pca = PCA(n_components='mle')
    print('{}: Starting PCA'.format(topic))
    pca.fit(topic_array)
    print('{}: Finished Fit'.format(topic))
    topic_array = pca.transform(topic_array)
    print(topic, '\t', topic_array.shape)
    pickle.dump(topic_array, open("../data/{}.pkl".format(topic), "wb" ))
    del topic_array
    print('{}: Finished PCA'.format(topic))


def embedding_ind(row):
    return_list = []
    for topic, cols in list(ind_dict.items()):
        topic_list = []
        for ind in cols:
            if row[ind] == 0:
                continue
            topic_list.append(col_embedding[ind] * row[ind])
        if not topic_list:
            topic_list.append(np.zeros(300,))
        return_list.append(np.average(topic_list, axis=0))
    return return_list


# In[27]:


embedding_data[list(ind_dict.keys())] = all_data.apply(embedding_ind, axis=1, result_type="expand")


for topic in ind_topics:
    topic_array = np.stack(embedding_data[topic])
    print('{}: Finished Stack'.format(topic))
    embedding_data.drop(columns=[topic], inplace=True)
    print('{}: Finished Dropping'.format(topic))
    pca = PCA(n_components='mle')
    print('{}: Starting PCA'.format(topic))
    pca.fit(topic_array)
    print('{}: Finished Fit'.format(topic))
    topic_array = pca.transform(topic_array)
    print(topic, '\t', topic_array.shape)
    pickle.dump(topic_array, open("../data/{}.pkl".format(topic), "wb" ))
    del topic_array
    print('{}: Finished PCA'.format(topic))


def embedding_dense_dict(row):
    return_list = []
    for topic, cols in list(dense_dict.items()):
        topic_list = []
        for ind in cols:
            if row[ind] == 0:
                continue
            topic_list.append(col_embedding[ind] * row[ind])
        if not topic_list:
            topic_list.append(np.zeros(300,))
        return_list.append(np.average(topic_list, axis=0))
    return return_list


embedding_data[list(dense_dict.keys())] = all_data.apply(embedding_dense_dict, axis=1, result_type="expand")



for topic in dense_topics:
    topic_array = np.stack(embedding_data[topic])
    print('{}: Finished Stack'.format(topic))
    embedding_data.drop(columns=[topic], inplace=True)
    print('{}: Finished Dropping'.format(topic))
    pca = PCA(n_components='mle')
    print('{}: Starting PCA'.format(topic))
    pca.fit(topic_array)
    print('{}: Finished Fit'.format(topic))
    topic_array = pca.transform(topic_array)
    print(topic, '\t', topic_array.shape)
    pickle.dump(topic_array, open("../data/{}.pkl".format(topic), "wb" ))
    del topic_array
    print('{}: Finished PCA'.format(topic))


def embedding_score_cols(row):
    return_list = []
    for col in score_cols:
        return_list.append(col_embedding[col] * row[col])
    return return_list


embedding_data[score_cols] = all_data.apply(embedding_score_cols, axis=1, result_type="expand")

for topic in score_cols:
    topic_array = np.stack(embedding_data[topic])
    print('{}: Finished Stack'.format(topic))
    embedding_data.drop(columns=[topic], inplace=True)
    print('{}: Finished Dropping'.format(topic))
    pca = PCA(n_components='mle')
    print('{}: Starting PCA'.format(topic))
    pca.fit(topic_array)
    print('{}: Finished Fit'.format(topic))
    topic_array = pca.transform(topic_array)
    print(topic, '\t', topic_array.shape)
    pickle.dump(topic_array, open("../data/{}.pkl".format(topic), "wb" ))
    del topic_array
    print('{}: Finished PCA'.format(topic))

def embedding_category_cols(row):
    return_list = []
    for col in category_cols:
        return_list.append(eval('embedding_' + col)(row[col]))
    return return_list


embedding_data[category_cols] = all_data.apply(embedding_category_cols, axis=1, result_type="expand")

for topic in category_cols:
    topic_array = np.stack(embedding_data[topic])
    print('{}: Finished Stack'.format(topic))
    embedding_data.drop(columns=[topic], inplace=True)
    print('{}: Finished Dropping'.format(topic))
    pca = PCA(n_components='mle')
    print('{}: Starting PCA'.format(topic))
    pca.fit(topic_array)
    print('{}: Finished Fit'.format(topic))
    topic_array = pca.transform(topic_array)
    print(topic, '\t', topic_array.shape)
    pickle.dump(topic_array, open("../data/{}.pkl".format(topic), "wb" ))
    del topic_array
    print('{}: Finished PCA'.format(topic))
