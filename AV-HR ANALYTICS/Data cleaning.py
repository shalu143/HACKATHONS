from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler
from sklearn.model_selection import cross_val_score,GridSearchCV,StratifiedKFold,RepeatedStratifiedKFold,RandomizedSearchCV
from sklearn.metrics import accuracy_score,auc,recall_score,precision_score,precision_recall_curve,confusion_matrix,roc_auc_score,f1_score
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator

import re
import seaborn as sns
from matplotlib import pyplot as plt
sns.set(style="darkgrid")

path = "AV hackathon"

train = pd.read_csv(f'{path}/train_jqd04QH.csv',dtype = {'enrollee_id':str})
test = pd.read_csv(f'{path}/test_KaymcHn.csv',dtype = {'enrollee_id':str})

f'train dimensions are : {train.shape}'
f'test dimensions are : {test.shape}'

train.head(2)
test.head(2)

def data_info(df = train):
    df_info = pd.DataFrame(df.isna().sum(),columns = ['Null_count'])
    df_info['Non_Null_count'] = df_info.index.map(df.notna().sum())
    df_info['N_unique'] = df_info.index.map(df.nunique())
    df_info['D_types'] = df_info.index.map(df.dtypes)
    df_info['Blank_count'] = df_info.index.map((df=='').sum())
    return df_info

data_info(train)
data_info(test)
 
###checking event rate
from collections import Counter
Counter(train.target)
train.target.value_counts(normalize=True)


##creating NA indicator for all the columns containing NAs
mindicator = MissingIndicator(missing_values=np.nan,error_on_new=False)
z = mindicator.fit_transform(train.drop('target',axis = 1))
cols_na_ind = [x+'_na_ind' for x in train.columns[mindicator.features_]]
train = pd.concat([train,pd.DataFrame(1*z,columns = cols_na_ind)],axis = 1)
train.head(1)
z = mindicator.transform(test)
cols_na_ind = [x+'_na_ind' for x in test.columns[mindicator.features_]]
test = pd.concat([test,pd.DataFrame(1*z,columns = cols_na_ind)],axis = 1)
test.head(1)

## Treating Null values
var = 'gender'
#f'count of NULLs in {var} : {train[[var]].isna().sum()[0]}'
train[var].value_counts(dropna = False,normalize = True)
pd.crosstab(index = train[var].fillna('Nan'), columns = train.target,margins = True,normalize='index',)

var = 'enrolled_university'
train[var].value_counts(dropna = False,normalize=True)
pd.crosstab(index = train[var].fillna('Nan'), columns = train.target,margins = True,normalize='index',)

var = 'education_level'
train[var].value_counts(dropna=False,normalize=True)
pd.crosstab(index = train[var].fillna('Nan'), columns = train.target,margins = True,normalize=True,dropna=False)

var = 'major_discipline'
train[var].value_counts(dropna=False,normalize=True)
pd.crosstab(index = train[var].fillna('Nan'), columns = train.target,margins = True,dropna = False,normalize='index')

var = 'experience'
#train[var].value_counts(dropna=False,normalize=True)
pd.crosstab(index = train[var].fillna('Nan'), columns = train.target,margins = True,dropna = False,normalize='index',)


na_cols = train.isna().sum()[train.isna().sum()>0].index
fillna_vals = dict(zip(na_cols,len(na_cols)*['missing']))
fillna_vals

train.fillna(fillna_vals,inplace = True)
test.fillna(fillna_vals,inplace = True)

data_info(train.loc[:,~train.columns.str.contains('_na_ind')])

train.to_csv(f'{path}/train_null_rem_1.csv',index= False)
test.to_csv(f'{path}/test_null_rem_1.csv',index= False)

###....Removed Null values.....

##creating event rate columns
train['city_event_rate'] = train.city.map(train.groupby('city').target.mean())
test['city_event_rate'] = test.city.map(train.groupby('city').target.mean())

var = 'major_discipline'
temp = train[var].value_counts(normalize=True)
temp = pd.Series(temp[temp> 0.03].index)
#temp
train[var] = np.where(train[var].isin(temp), train[var],'low_pop')
train[var].value_counts(normalize=True)
test[var] = np.where(test[var].isin(temp), test[var],'low_pop')

var = 'company_size'
temp = train[var].value_counts(normalize=True)
temp = pd.Series(temp[temp> 0.05].index)
#temp
train[var] = np.where(train[var].isin(temp), train[var],'low_pop')
train[var].value_counts(normalize=True)
test[var] = np.where(test[var].isin(temp), test[var],'low_pop')

var = 'company_type'
temp = train[var].value_counts(normalize=True)
temp = pd.Series(temp[temp> 0.05].index)
#temp
train[var] = np.where(train[var].isin(temp), train[var],'low_pop')
train[var].value_counts(normalize=True)
test[var] = np.where(test[var].isin(temp), test[var],'low_pop')

var = 'last_new_job'
temp = train[var].value_counts(normalize=True)
temp = pd.Series(temp[temp> 0.05].index)
#temp
train[var] = np.where(train[var].isin(temp), train[var],'low_pop')
train[var].value_counts(normalize=True)
test[var] = np.where(test[var].isin(temp), test[var],'low_pop')

## check the info of the train data after modification
data_info(train)

var = 'experience'
df = pd.crosstab(index = train[var], columns = train.target,margins = True).reset_index()

df.experience = pd.to_numeric(df.experience.replace({'<1': '0', '>20' : '21', 'All' : '100'}))
df = df.rename({0 :'Non_Event', 1:'Event'}, axis = 1)
df['Non_Event_%'] =  df.Non_Event/df.All
df['Event_%'] =  df.Event/df.All
df['woe'] = np.log(df['Event_%']/df['Non_Event_%'])
df['IV'] = (df['Event_%'] - df['Non_Event_%'])*df['woe']
df.sort_values('experience',inplace = True)#.reset_index(drop = True)
df.head(1)


train['experience_band'] = pd.to_numeric(train.experience.replace({'<1': '0', '>20' : '21', 'All' : '100'}))
train['experience_band'] = np.where(train.experience_band <= 7 , 'low_exp',np.where(train.experience_band >= 16, 'high_exp', 'med_exp'))

test['experience_band'] = pd.to_numeric(test.experience.replace({'<1': '0', '>20' : '21', 'All' : '100'}))
test['experience_band'] = np.where(test.experience_band <= 7 , 'low_exp',np.where(test.experience_band >= 16, 'high_exp', 'med_exp'))



# 5 - DIct Vectorizer on train and transforming it on test
def dict_vec(train_set,cols,is_test, test_set):
    """
    returns dict vectorizer on train set or train & test set for chosen columns
    train_set: Dataset on which DV is to be fit
    cols: List of columns of train_set which are to be considered for DV
    is_test: Boolean, If DV is to be transformed on test too
    test_set: Test set on which DV is to be transformed
    """
    from sklearn.feature_extraction import DictVectorizer
    import pandas as pd
    dvec = DictVectorizer(sparse=False)
    if not is_test:
        train_dvec = dvec.fit_transform(train_set[cols].transpose().to_dict().values())
        train_dvec = pd.DataFrame(train_dvec, index = train_set.index, columns = dvec.get_feature_names())
        train_df = pd.concat([train_set.drop(cols, axis = 1),train_dvec], axis = 1)
        return train_df,pd.DataFrame(),dvec
    else:
        train_dvec = dvec.fit_transform(train_set[cols].transpose().to_dict().values())
        train_dvec = pd.DataFrame(train_dvec, index = train_set.index, columns = dvec.get_feature_names())
        train_df = pd.concat([train_set.drop(cols, axis = 1),train_dvec], axis = 1)
        test_dvec = dvec.transform(test_set[cols].transpose().to_dict().values())
        test_dvec = pd.DataFrame(test_dvec, index = test_set.index, columns = dvec.get_feature_names())
        test_df = pd.concat([test_set.drop(cols, axis = 1),test_dvec], axis = 1)
        return train_df, test_df,dvec

obj_cols = train.drop(['enrollee_id','city', 'experience'],axis = 1).select_dtypes('object').columns
#obj_cols = [x for x in train.columns if x != 'enrollee_id' or x != 'city']
obj_cols 

df_train,df_test,dv = dict_vec(train, obj_cols,is_test = True,test_set =test)

df_train.shape; df_test.shape
df_train.columns

fin_cols = pd.Series(df_train.columns)
#fin_cols
df_train.columns = fin_cols.apply(lambda x : re.sub('>', 'grtr',x)).apply(lambda x : re.sub('<', 'smlr',x))

fin_cols_test = pd.Series(df_test.columns)
#fin_cols
df_test.columns = fin_cols_test.apply(lambda x : re.sub('>', 'grtr',x)).apply(lambda x : re.sub('<', 'smlr',x))

###writing the file after EDA
df_train.to_csv(f'{path}/train_for_model.csv',index= False)
df_test.to_csv(f'{path}/test_for_model.csv',index= False)
