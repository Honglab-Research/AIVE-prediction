import pandas as pd
import numpy as np
import re

file_path = 'data.txt'

data = pd.read_csv(file_path, sep = '\t')

def sort(AA_mutation):
  AA_mutation = str(AA_mutation)
  AA_mutation = AA_mutation.split(',')
  i = 0
  mut_AA_length = len(AA_mutation)
  while i < mut_AA_length:
    if AA_mutation[i].startswith('S:') == False:
      AA_mutation.remove(AA_mutation[i])
      if i == len(AA_mutation):
        break
    else:
      i += 1
      if i == len(AA_mutation):
        break
  mut = []
  for mutation in AA_mutation:
    mut_list = mutation.replace('S:','')
    mut.append(mut_list)
  mut = ','.join(mut)
  return mut

def aa_sort(AA_mutation):
  AA_mutation = str(AA_mutation)
  AA_mutation = AA_mutation.split(',')
  mut = []
  for mutation in AA_mutation:
    if mutation == '':
      mut.append(mutation)
    elif mutation == 'nan':
      mutation = ''
      mut.append(mutation)
    else:
      pos = int(mutation[1:-1])
      if pos in range(437,509):
        mut.append(mutation)
  mut = ','.join(mut)
  if "*" in mut:
    mut = np.NAN
  return mut

def delete_json(clade):
  if '(' in clade:
    info = clade.split("(")
    cla = info[0]
  else:
    cla = clade
  return cla

data = data.dropna(subset=['deletions','insertions'], how='any')
data = data.drop(columns=['deletions','insertions'])

data['Nextstrain_clade'] = data.apply(lambda x: delete_json(x['Nextstrain_clade']), axis=1)
data['aaSubstitutions'] = data.apply(lambda x: sort(x['aaSubstitutions']), axis=1)
data['aaSubstitutions'] = data.apply(lambda x: aa_sort(x['aaSubstitutions']), axis=1)

data = data.dropna(subset=['aaSubstitutions'])

pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')

def is_valid_date(date_str):
    if not isinstance(date_str, str):
        return False
    return bool(pattern.match(date_str))

date_data = data[data['date'].apply(is_valid_date)]
df = date_data
reference_date = pd.to_datetime("2019-12-22")

df.loc[:,'date'] = pd.to_datetime(df['date'], errors='coerce')
df.loc[:,'days'] = (df['date'] - reference_date).dt.days
df = df.sort_values(by='days', ascending=True)

use_host = ['Homo sapiens','Homo']
human_data = df[df['host'].isin(use_host)]

def delete_question(lineage):
  if lineage == '' :
    lin = np.NaN
  elif lineage == '?' :
    lin = np.NaN
  else:
    lin = lineage
  return lin

human_data.loc[:,'pango_lineage'] = human_data.apply(lambda x: delete_question(x['pango_lineage']), axis=1)

human_data = human_data.dropna(axis=0)


human_data.to_csv('filtered_data.txt', sep='\t', mode='w', index=False)
