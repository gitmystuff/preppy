def hello_4050():
  return 'Hello 4050' 

def identify_consts(df):
  constant_features = [
      feat for feat in df.columns if len(df[feat].unique()) == 1
  ]
  return constant_features

def identify_quasi_consts(df):
  # quasi constant values
  quasi_consts = []
  for val in df.columns.sort_values():
      if (len(df[val].unique()) < 3 and max(df[val].value_counts(normalize=True)) > .98):
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