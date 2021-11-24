import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols
from functions import Stats_func
from first_analysis import First_analysis
from second_analysis import Second_analysis
from third_analysis import Third_analysis

### ============================================= Preparing the data ==================================================
df = pd.read_csv('dataset/loan_data.csv')

def find_extremes(data, col):
    data_ext = np.sort(data[col])
    q1 = np.percentile(data_ext, 25)
    q3 = np.percentile(data_ext, 75)
    dq = q3 - q1
    bbp = q1 - 1.5*dq
    bap = q3 + 1.5*dq
    index_bbp = [i for i, d in enumerate(data_ext) if d < bbp]
    index_bap = [i for i, d in enumerate(data_ext) if d > bap]
    index =  index_bbp + index_bap
    return index

columns_to_clean = ['int.rate', 'installment', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util']
for column in columns_to_clean:
  index = find_extremes(df, column)
  df = df.drop(index, axis=0)

df['purpose'] = df['purpose'].str.replace('_', ' ')

df['annual.income']= np.exp(df['log.annual.inc'])

df['debt'] = df['dti']*df['annual.income']
logged_debt = []

for debt in df['debt']:
    if debt == 0:
        logged_debt.append(0)
    else:
        logged_debt.append(np.log(debt))
df['log.debt'] = logged_debt

### ============================================= Preparing the data ==================================================

stats_func = Stats_func(df)

### =============================================== First analysis ====================================================
first_analysis = First_analysis(df)

_ = first_analysis.ecdf_annual_inc()

_ = first_analysis.ecdf_debt()

_ = first_analysis.correlation_between_debt_income()

### =============================================== First analysis ====================================================


### =============================================== Second analysis ===================================================

second_analysis = Second_analysis(df)

_ = second_analysis.late_debt_likelihood()

_ = second_analysis.not_fully_paid_likelihood()

_ = second_analysis.percentage_pays_late()

_ = second_analysis.percentage_not_fully_paid()


### ================================================ Third analysis ===================================================

third_analysis = Third_analysis(df)

_ = third_analysis.data_model_distributions()

_ = third_analysis.data_simulations()


### ================================================ Third analysis ===================================================

