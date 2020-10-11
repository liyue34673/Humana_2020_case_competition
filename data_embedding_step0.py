# coding: utf-8
import pandas as pd
import numpy as np
import pickle
from gensim.models import KeyedVectors

df = pd.read_csv('../data/glove.840B.300d.txt', sep=" ", quoting=3, header=None, index_col=0)
glove = {key: val.values for key, val in df.T.items()}

train_data = pd.read_csv('../data/cleaned_train_no_one_hot.csv', low_memory=False)



test_data = pd.read_csv('../data/cleaned_test_no_one_hot.csv', low_memory=False)



doc = pd.read_excel('../docs/2020_Competition_Data_Documentation.xlsx', sheet_name='Transporation model_dictionary', index_col='VARIABLE_NAME', usecols=['VARIABLE_NAME', 'English'])\ndoc = doc.dropna()



all_data = pd.concat([train_data, test_data], sort=True)\nall_data = all_data.reset_index(drop=True)


ind_ct_topics = ['betos', 'rx', 'submcc']
ind_topics = ['bh', 'ccsp', 'cms', 'cmsd2', 'cons', 'hedis', 'lab', 'phy', 'prov', 'rev', 'smoker']  # Special ind: hedis
dense_topics = ['cms', 'credit', 'med', 'pdc', 'total']
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


col_embedding = {}
for idx in doc.index:
    words = doc.loc[idx, 'English'].split()
    embedding = np.array(list(map(lambda s: glove[s], words))).mean(axis=0)
    col_embedding[idx] = embedding


def embedding_zip_code(zip_cd):
    if zip_cd == 'other':
           return col_embedding['other_zip_cd']
    return col_embedding[zip_cd]
col_embedding['other_zip_cd'] = np.array(list(map(lambda s: glove[s], ['other', 'postcode']))).mean(axis=0)


for zip_cd in all_data.zip_cd:
    col_embedding[str(zip_cd)] = glove[str(zip_cd)]


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



col_embedding['unknown education level'] = np.array(list(map(lambda s: glove[s], 'unknown education level'.split()))).mean(axis=0)
col_embedding['less than 12th grade'] = np.array(list(map(lambda s: glove[s], 'less than 12th grade'.split()))).mean(axis=0)
col_embedding['high school diploma'] = np.array(list(map(lambda s: glove[s], 'high school diploma'.split()))).mean(axis=0)
col_embedding['some college'] = np.array(list(map(lambda s: glove[s], 'some college'.split()))).mean(axis=0)
col_embedding['associate degree'] = np.array(list(map(lambda s: glove[s], 'associate degree'.split()))).mean(axis=0)
col_embedding['bachelors degree'] = np.array(list(map(lambda s: glove[s], 'bachelors degree'.split()))).mean(axis=0)



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



col_embedding['composition unknown'] = np.array(list(map(lambda s: glove[s], 'composition unknown'.split()))).mean(axis=0)
col_embedding['married with no children'] = np.array(list(map(lambda s: glove[s], 'married with no children'.split()))).mean(axis=0)
col_embedding['female householder with no children'] = np.array(list(map(lambda s: glove[s], 'female householder with no children'.split()))).mean(axis=0)
col_embedding['married with children'] = np.array(list(map(lambda s: glove[s], 'married with children'.split()))).mean(axis=0)
col_embedding['male householder with no children'] = np.array(list(map(lambda s: glove[s], 'male householder with no children'.split()))).mean(axis=0)
col_embedding['marital status unknown with no children'] = np.array(list(map(lambda s: glove[s], 'marital status unknown with no children'.split()))).mean(axis=0)
col_embedding['female householder with children'] = np.array(list(map(lambda s: glove[s], 'female householder with children'.split()))).mean(axis=0)
col_embedding['female householder with one or more other persons of any gender with children'] = np.array(list(map(lambda s: glove[s], 'female householder with one or more other persons of any gender with children'.split()))).mean(axis=0)
col_embedding['other marital status'] = np.array(list(map(lambda s: glove[s], 'other marital status'.split()))).mean(axis=0)
col_embedding['marital status unknown with children'] = np.array(list(map(lambda s: glove[s], 'marital status unknown with children'.split()))).mean(axis=0)
col_embedding['female householder with one or more other persons of any gender with no children'] = np.array(list(map(lambda s: glove[s], 'female householder with one or more other persons of any gender with no children'.split()))).mean(axis=0)
col_embedding['male householder with one or more other persons of any gender with children'] = np.array(list(map(lambda s: glove[s], 'male householder with one or more other persons of any gender with children'.split()))).mean(axis=0)
col_embedding['male householder with children'] = np.array(list(map(lambda s: glove[s], 'male householder with children'.split()))).mean(axis=0)


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


col_embedding['homeowner'] = np.array(list(map(lambda s: glove[s], 'homeowner'.split()))).mean(axis=0)
col_embedding['unknown home status'] = np.array(list(map(lambda s: glove[s], 'unknown home status'.split()))).mean(axis=0)
col_embedding['probable homeowner'] = np.array(list(map(lambda s: glove[s], 'probable homeowner'.split()))).mean(axis=0)
col_embedding['renter'] = np.array(list(map(lambda s: glove[s], 'renter'.split()))).mean(axis=0)
col_embedding['probable renter'] = np.array(list(map(lambda s: glove[s], 'probable renter'.split()))).mean(axis=0)



def embedding_cons_hcaccprf(cons_hcaccprf):
    if cons_hcaccprf == 'cons_hcaccprf_p':
           return col_embedding['personal doctor or personal care physician']
    if cons_hcaccprf == 'cons_hcaccprf_o':
           return col_embedding['other healthcare treatment preference']
    if cons_hcaccprf == 'cons_hcaccprf_c':
           return col_embedding['community health clinic or health clinic in retail setting']
    return col_embedding['hospital or standalone emergency room or urgent care center']



col_embedding['personal doctor or personal care physician'] = np.array(list(map(lambda s: glove[s], 'personal doctor or personal care physician'.split()))).mean(axis=0)
col_embedding['other healthcare treatment preference'] = np.array(list(map(lambda s: glove[s], 'other healthcare treatment preference'.split()))).mean(axis=0)
col_embedding['community health clinic or health clinic in retail setting'] = np.array(list(map(lambda s: glove[s], 'community health clinic or health clinic in retail setting'.split()))).mean(axis=0)
col_embedding['hospital or standalone emergency room or urgent care center'] = np.array(list(map(lambda s: glove[s], 'hospital or standalone emergency room or urgent care center'.split()))).mean(axis=0)


def embedding_lang_spoken_cd(lang_spoken_cd):
    if lang_spoken_cd == 'ENG':
           return col_embedding['english']
    if lang_spoken_cd == 'SPA':
           return col_embedding['spanish']



col_embedding['english'] = np.array(list(map(lambda s: glove[s], 'english'.split()))).mean(axis=0)
col_embedding['spanish'] = np.array(list(map(lambda s: glove[s], 'spanish'.split()))).mean(axis=0)



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



col_embedding['healthy auto-pilot participator'] = np.array(list(map(lambda s: glove[s], 'healthy auto-pilot participator'.split()))).mean(axis=0)
col_embedding['unknown health status'] = np.array(list(map(lambda s: glove[s], 'unknown health status'.split()))).mean(axis=0)
col_embedding['healthy skeptical control seeker A'] = np.array(list(map(lambda s: glove[s], 'healthy skeptical control seeker A'.split()))).mean(axis=0)
col_embedding['healthy self engaged optimist'] = np.array(list(map(lambda s: glove[s], 'healthy self engaged optimist'.split()))).mean(axis=0)
col_embedding['chronic auto-pilot participator'] = np.array(list(map(lambda s: glove[s], 'chronic auto-pilot participator'.split()))).mean(axis=0)
col_embedding['chronic self engaged optimist'] = np.array(list(map(lambda s: glove[s], 'chronic self engaged optimist'.split()))).mean(axis=0)
col_embedding['healthy skeptical control seeker B'] = np.array(list(map(lambda s: glove[s], 'healthy skeptical control seeker B'.split()))).mean(axis=0)
col_embedding['chronic simplicity seeking follower'] = np.array(list(map(lambda s: glove[s], 'chronic simplicity seeking follower'.split()))).mean(axis=0)
col_embedding['healthy self sustainer'] = np.array(list(map(lambda s: glove[s], 'healthy self sustainer'.split()))).mean(axis=0)
col_embedding['chronic health services maximizer'] = np.array(list(map(lambda s: glove[s], 'chronic health services maximizer'.split()))).mean(axis=0)
col_embedding['healthy health services maximizer'] = np.array(list(map(lambda s: glove[s], 'healthy health services maximizer'.split()))).mean(axis=0)
col_embedding['healthy overwhelmed and reluctant reactor'] = np.array(list(map(lambda s: glove[s], 'healthy overwhelmed and reluctant reactor'.split()))).mean(axis=0)
col_embedding['healthy simplicity seeking follower'] = np.array(list(map(lambda s: glove[s], 'healthy simplicity seeking follower'.split()))).mean(axis=0)
col_embedding['chronic skeptical control seeker'] = np.array(list(map(lambda s: glove[s], 'chronic skeptical control seeker'.split()))).mean(axis=0)
col_embedding['chronic support seeking participator'] = np.array(list(map(lambda s: glove[s], 'chronic support seeking participator'.split()))).mean(axis=0)
col_embedding['chronic overwhelmed and reluctant reactor'] = np.array(list(map(lambda s: glove[s], 'chronic overwhelmed and reluctant reactor'.split()))).mean(axis=0)



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


col_embedding['risk adjustment factor community non-dual'] = np.array(list(map(lambda s: glove[s], 'risk adjustment factor community non-dual'.split()))).mean(axis=0)
col_embedding['risk adjustment factor community partial dual'] = np.array(list(map(lambda s: glove[s], 'risk adjustment factor community partial dual'.split()))).mean(axis=0)
col_embedding['risk adjustment factor new enrollee'] = np.array(list(map(lambda s: glove[s], 'risk adjustment factor new enrollee'.split()))).mean(axis=0)
col_embedding['risk adjustment factor community full dual'] = np.array(list(map(lambda s: glove[s], 'risk adjustment factor community full dual'.split()))).mean(axis=0)
col_embedding['risk adjustment factor dialysis'] = np.array(list(map(lambda s: glove[s], 'risk adjustment factor dialysis'.split()))).mean(axis=0)
col_embedding['risk adjustment factor community post graft'] = np.array(list(map(lambda s: glove[s], 'risk adjustment factor community post graft'.split()))).mean(axis=0)
col_embedding['risk adjustment factor institutional'] = np.array(list(map(lambda s: glove[s], 'risk adjustment factor institutional'.split()))).mean(axis=0)
col_embedding['risk adjustment factor new enrollee chronic care'] = np.array(list(map(lambda s: glove[s], 'risk adjustment factor new enrollee chronic care'.split()))).mean(axis=0)



def embedding_sex_cd(sex_cd):
    if int(sex_cd) == 0:
           return col_embedding['female']
    return col_embedding['male']



col_embedding['female'] = np.array(list(map(lambda s: glove[s], 'female'.split()))).mean(axis=0)
col_embedding['male'] = np.array(list(map(lambda s: glove[s], 'male'.split()))).mean(axis=0)


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



col_embedding['metro counties in metro areas of one-million population or more'] = np.array(list(map(lambda s: glove[s], 'metro counties in metro areas of one-million population or more'.split()))).mean(axis=0)
col_embedding['metro counties in metro areas of 250000 to one-million population'] = np.array(list(map(lambda s: glove[s], 'metro counties in metro areas of 250000 to one-million population'.split()))).mean(axis=0)
col_embedding['metro counties in metro areas of fewer than 250000 population'] = np.array(list(map(lambda s: glove[s], 'metro counties in metro areas of fewer than 250000 population'.split()))).mean(axis=0)
col_embedding['nonmetro urban population of 20000 or more adjacent to a metro area'] = np.array(list(map(lambda s: glove[s], 'nonmetro urban population of 20000 or more adjacent to a metro area'.split()))).mean(axis=0)
col_embedding['nonmetro urban population of 20000 or more not adjacent to a metro area'] = np.array(list(map(lambda s: glove[s], 'nonmetro urban population of 20000 or more not adjacent to a metro area'.split()))).mean(axis=0)
col_embedding['nonmetro urban population of 2500 to 19999 adjacent to a metro area'] = np.array(list(map(lambda s: glove[s], 'nonmetro urban population of 2500 to 19999 adjacent to a metro area'.split()))).mean(axis=0)
col_embedding['nonmetro urban population of 2500 to 19999 not adjacent to a metro area'] = np.array(list(map(lambda s: glove[s], 'nonmetro urban population of 2500 to 19999 not adjacent to a metro area'.split()))).mean(axis=0)
col_embedding['nonmetro completely rural or less than 2500 urban population adjacent to a metro area'] = np.array(list(map(lambda s: glove[s], 'nonmetro completely rural or less than 2500 urban population adjacent to a metro area'.split()))).mean(axis=0)
col_embedding['nonmetro completely rural or less than 2500 urban population not adjacent to a metro area'] = np.array(list(map(lambda s: glove[s], 'nonmetro completely rural or less than 2500 urban population not adjacent to a metro area'.split()))).mean(axis=0)




pickle.dump(col_embedding, open("../data/col_embedding.pkl", "wb" ) )


