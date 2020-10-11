# -*- coding: utf-8 -*-
import pandas as pd

pdc_na = {'pdc_ast': 1.1, 'pdc_cvd': 1.1, 'pdc_dep': 1.1, 'pdc_dia': 1.1, 'pdc_hf': 1.1, 'pdc_ht': 1.1, 'pdc_lip': 1.1,
          'pdc_ost': 1.1}
train_data = pd.read_csv('../data/2020_Competition_Training.csv', low_memory=False, na_values=pdc_na)
test_data = pd.read_csv('../data/2020_Competition_Holdout.csv', low_memory=False, na_values=pdc_na)

all_data = pd.concat([train_data, test_data], sort=True)
all_data = all_data.reset_index(drop=True)

drop_cols = []
for col in test_data.columns:
    if test_data[col].nunique() == 1:
        drop_cols.append(col)
for col in train_data.columns:
    if train_data[col].nunique() == 1:
        drop_cols.append(col)
drop_cols = list(set(drop_cols))

info_data = pd.read_excel('../docs/2020_Competition_Data_Documentation.xlsx', sheet_name='Transporation '
                                                                                         'model_dictionary')
nan_methods = info_data[~info_data.METHOD.isna()].reset_index(drop=True)

all_data['cons_hcaccprf_o'] = 0
all_data.loc[(all_data['cons_hcaccprf_h'].isna()) & (all_data['cons_hcaccprf_p'].isna()), 'cons_hcaccprf_o'] = 1
all_data.loc[~(all_data['cons_hcaccprf_h'].isna()) & (all_data['cons_hcaccprf_p'].isna()), 'cons_hcaccprf_o'] = 1

train_data['cons_hcaccprf_c'] = ((train_data['cons_hcaccprf_p'].astype(int)) | (train_data['cons_hcaccprf_o']) | (
    train_data['cons_hcaccprf_h'].astype(int))) == 0
train_data['cons_hcaccprf_c'] = train_data['cons_hcaccprf_c'].astype(int)
reverse_dict = {'cons_hcaccprf': ['cons_hcaccprf_c', 'cons_hcaccprf_h', 'cons_hcaccprf_p', 'cons_hcaccprf_o']}
train_data['cons_hcaccprf'] = train_data[reverse_dict['cons_hcaccprf']].idxmax(1)
train_data.drop(columns=reverse_dict['cons_hcaccprf'], inplace=True)

for i in range(len(nan_methods)):
    col_name = nan_methods.loc[i, 'VARIABLE_NAME']
    method = nan_methods.loc[i, 'METHOD']
    if col_name in ['cons_hcaccprf_h', 'cons_hcaccprf_p']:
        all_data[col_name].fillna(0, inplace=True)
        continue
    if method == 'drop':
        all_data = all_data.drop(columns=[col_name])
    elif method == 'Mean':
        all_data[col_name].fillna(all_data[col_name].mean(), inplace=True)
    else:
        all_data[col_name].fillna(method, inplace=True)

all_data.replace({'Y': 1, 'N': 0, 'M': 1, 'F': 0}, inplace=True)
all_data.lang_spoken_cd.replace({'E': 'ENG'}, inplace=True)
all_data.cms_ra_factor_type_cd.replace({'*': 'CN'}, inplace=True)
all_data.cons_cmys.replace({'*': '0'}, inplace=True)
all_data = all_data.drop(columns=drop_cols)

train_data = all_data[:train_data.shape[0]]
test_data = all_data[train_data.shape[0]:]

train_data.to_csv('../data/cleaned_train.csv', index=False)

test_data = test_data.drop(columns=['transportation_issues'])
test_data.to_csv('../data/cleaned_test.csv', index=False)
