import pickle,os
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import trange
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def score(data,model):
    y_pred=model.predict_proba(data[feat])[:, 1]
    return roc_auc_score(data.Label, y_pred)

train = pd.read_csv('../data/train.csv')
train_y = pd.read_csv('../data/train_label.csv').Label
test = pd.read_csv('../data/test.csv')

cat_feat = ['登记机关', '行业代码', '行业门类', '企业类型']
feat = list(set(train.columns)-set(train.select_dtypes(object))-set(['Label', 'ID'])-set(cat_feat))
remove_col = []
for col in feat:
    if train[col].nunique() < 2:
        remove_col.append(col)
feat = list(set(feat) - set(remove_col))

class Feat:
    def __init__(self, config):
        self.config = config
    
    def fit(self, x, y):
        pass

    def transform(self, x):
        pass
    
    def fit_transform(self, x, y):
        self.fit(x, y)
        self.transform(x)


class CatCount(Feat):
    def transform(self, x):
        for col in self.config['cat_columns']:
            df_count = x[col].value_counts()
            x[f'{col}_catcount'] = x[col].map(df_count)

class CatCountRank(Feat):
    def fit(self, x, y):
        self.fit_dict = {}
        for col in self.config['cat_columns']:
            counter =  Counter(x[col]).most_common()
            self.fit_dict[col] = {k: i for (i, (k, v)) in enumerate(counter)}

    def transform(self, x):
        for col in self.config['cat_columns']:
            x[f'{col}_countrank'] = x[col].map(self.fit_dict[col])
            
config = {}
config['cat_columns'] = cat_feat

if os.path.exists('../trans_data/train_1000_10.pkl'):
    train = pickle.load(open('../trans_data/train_1000_10.pkl', 'rb'))
    test = pickle.load(open('../trans_data/test_1000_10.pkl', 'rb'))
else:
    d={'add':'+', 'sub':'-', 'mul':'*', 'div':'/'}
    feat0 = feat.copy()
    for i in trange(len(feat)):
        df_temp=train[feat0].copy()
        for j in range(i+1,len(feat)):
            df_temp['%s|%s|add'%(feat[i],feat[j])] = train[feat[i]]+train[feat[j]]
            df_temp['%s|%s|sub'%(feat[i],feat[j])] = train[feat[i]]-train[feat[j]]
            df_temp['%s|%s|mul'%(feat[i],feat[j])] = train[feat[i]]*train[feat[j]]
            df_temp['%s|%s|div'%(feat[i],feat[j])] = train[feat[i]]/train[feat[j]]
        model = LGBMClassifier(n_estimators=1000, learning_rate=0.08, max_depth=7, subsample=0.8, colsample_bytree=0.6, n_jobs=4)
        model.fit(df_temp.values, train_y)
        qq = pd.Series(model.feature_importances_, index=df_temp.columns).sort_values()
        for col in set(qq.loc[qq>10].index)-set(feat0):
            f0, f1, f2 = col.split('|')
            train[col] = df_temp[col]
            test[col] = eval("test['%s']%stest['%s']"%(f0,d[f2],f1))
        feat0.extend(list(set(qq.loc[qq>10].index)-set(feat0)))
    pickle.dump(train, open('../trans_data/train_1000_10.pkl','wb'))
    pickle.dump(test, open('../trans_data/test_1000_10.pkl','wb'))

def gen_feat(data):
    for col in cat_feat:
        data[col] = data[col].fillna('empty').astype(str)
    for col in data.columns:
        if '年' not in col and '|' not in col and data[col].isna().sum()>0:
            data['%s_na'%col] = data[col].isna().astype(int)
            
    featgen = CatCount(config)
    featgen.transform(data)
    
gen_feat(train)
gen_feat(test)

featgen = CatCountRank(config)
featgen.fit_transform(train, train_y)
featgen.transform(test)


feat0 = list(set(train.columns)-set(train.select_dtypes(object))-set(['Label','ID'])-set(cat_feat))
remove_col = []
for col in feat0:
    if train[col].nunique() < 2:
        remove_col.append(col)
feat0 = list(set(feat0) - set(remove_col))

kf = StratifiedKFold(5,True,random_state=1)
prob = np.zeros(len(train))
test_prob = np.zeros(len(test))
test_data = test[feat0].values
for idx, (train_index, valid_index) in enumerate(kf.split(train, train_y)):
    train_data = train.loc[train_index][feat0].values
    valid_data = train.loc[valid_index][feat0].values
    model = LGBMClassifier(n_estimators=1000, learning_rate=0.08, num_leaves=15, subsample=0.8, colsample_bytree=0.6, n_jobs=4)
    model.fit(train_data, train_y.loc[train_index], 
              eval_set=(valid_data, train_y.loc[valid_index]), early_stopping_rounds=50)
    prob[valid_index] = model.predict_proba(valid_data)[:, 1]
    test_prob += model.predict_proba(test_data)[:, 1]/5

train['lgb_prob'] = prob
test['lgb_prob'] = test_prob

kf = StratifiedKFold(5, True, random_state=1)
prob = np.zeros(len(train))
test_prob = np.zeros(len(test))
feat1=list(set(feat0 + cat_feat + ['lgb_prob']))
test_data=test[feat1].values
for idx, (train_index, valid_index) in enumerate(kf.split(train, train_y)):
    train_data = train.loc[train_index][feat1]
    valid_data = train.loc[valid_index][feat1]
    model = CatBoostClassifier(iterations=1000, learning_rate=0.08, depth=7, cat_features=cat_feat)
    model.fit(train_data, train_y.loc[train_index], 
              eval_set=(valid_data, train_y.loc[valid_index]), early_stopping_rounds=50)
    prob[valid_index] = model.predict_proba(valid_data)[:,1]
    test_prob += model.predict_proba(test_data)[:,1]/5

test['Label'] = test_prob
test[['ID', 'Label']].to_csv('../output/1120_count_rank.csv', index=False)