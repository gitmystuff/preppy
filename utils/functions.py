import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler

def identify_consts(df):
  constant_features = [
      feat for feat in df.columns if len(df[feat].unique()) == 1
  ]
  return constant_features

def identify_quasi_consts(df, thresh=0.95):
  # quasi constant values
  quasi_consts = []
  for val in df.columns.sort_values():
      if (len(df[val].unique()) < 3 and max(df[val].value_counts(normalize=True)) > thresh):
          quasi_consts.append(val)

  return quasi_consts

def check_row_duplicates(df):
  # duplicate rows
  return len(df[df.duplicated(keep=False)])

def check_col_duplicates(df):
  # check of duplicate columns
  duplicate_features = []
  for i in range(0, len(df.columns)):
      orig = df.columns[i]

      for dupe in df.columns[i + 1:]:
          if df[orig].equals(df[dupe]):
              duplicate_features.append(dupe)

  return duplicate_features

def do_OHE(df):
  cat_features = []
  for feat in df.select_dtypes(include=['object', 'category']):
    if len(df[feat].value_counts()) < 3:
      df[feat] = df[feat].map({df[feat].value_counts().index[0]: 0, df[feat].value_counts().index[1]: 1})
      df[feat] = df[feat].astype(int)
    elif 2 < len(df[feat].value_counts()) < 6:
      cat_features.append(feat)
    elif len(df[feat].value_counts()) > 5:
      freq = df.groupby(feat, observed=False).size()/len(df)
      df[feat] = df[feat].map(freq)

  ohe = OneHotEncoder(categories='auto', drop='first', sparse_output=False, handle_unknown='ignore')
  ohe_df = ohe.fit_transform(df[cat_features])
  ohe_df = pd.DataFrame(ohe_df, columns=ohe.get_feature_names_out(cat_features))
  df.index = df.index
  df = df.join(ohe_df)
  df.drop(cat_features, axis=1, inplace=True)
  return df

def handle_missing_values(df):
  df_categorical_features = df.select_dtypes(include=['category', 'object']).columns
  dfx = df.copy()
  for feat in df.columns[df.isnull().sum() > 1]:
    if feat in df_categorical_features:
      dfx[feat] = df[feat].fillna(df[feat].mode()[0])
    else:
      if abs(df[feat].skew()) < .8:
        dfx[feat] = df[feat].fillna(round(df[feat].mean(), 2))
      else:
        dfx[feat] = df[feat].fillna(df[feat].median())

  return dfx

def handle_standard_scaler(df):
  feat = str(df._get_numeric_data().idxmax(1)[0])
  scaler = StandardScaler()
  df[feat] = scaler.fit_transform(df[[feat]].values)
  return df

def handle_minmax_scaler(df):
  feat = str(df._get_numeric_data().idxmax(1)[0])
  scaler = MinMaxScaler()
  df[feat] = scaler.fit_transform(df[[feat]].values)
  return df

def handle_outliers(df):
  feat = str(df._get_numeric_data().idxmax(1)[0])
  scaler = RobustScaler()
  df[feat] = scaler.fit_transform(df[[feat]].values)
  return df
