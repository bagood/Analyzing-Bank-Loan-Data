import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols

# ------------------------------ Defining DataFrame and Many Functions  ------------------------------ 

df = pd.read_csv('loan_data.csv')


### define an ECDF function 
def ecdf(data):
    x_val = np.sort(data)
    length = len(x_val)
    y_val = np.array([r/length for r in range(1,length+1)])
    returns = (x_val, y_val)
    return returns

### define a function to plot the data alongside its supposed normal distributions
def normal_dist(data, col):
    mean = np.mean(data[col])
    std = np.std(data[col])
    samples = np.random.normal(mean, std, size=1000)
    x_data, y_data = ecdf(data[col])
    x_model, y_model = ecdf(samples)
    returns = (x_data, y_data, x_model, y_model)
    return returns

### define a function to plot the data alongside its supposed exponential distributions
def exponential_dist(data, col):
    mean = np.mean(data[col])
    samples = np.random.exponential(mean, size=1000)
    x_data, y_data = ecdf(data[col])
    x_model, y_model = ecdf(samples)
    returns = (x_data, y_data, x_model, y_model)
    return returns

### define a function to make a best fit regression line
def line_regres(x_data, y_data):
    sample = pd.DataFrame(dict(x_val=x_data, y_val=y_data))
    model_fit = ols(formula='y_val ~ x_val', data=sample).fit()
    x_model = x_data
    y_model = model_fit.predict(sample)
    intercept = model_fit.params['Intercept']
    slope = model_fit.params['x_val']
    results = (x_model, y_model, intercept, slope)
    return results

### define a function to plot a data based on it's purposes
def plot_each_purpose(data, col, func, suptitle, supxlabel, supylabel):
    purpose_list = list(df['purpose'].unique())
    row = np.sort(np.array([r if r < 4 else r - 4 for r in list(range(8))]))
    column = [c % 2 for c in list(range(8))]
    combined = [(p, r, c) for p, r, c in zip(purpose_list, row, column)]

    fig, ax = plt.subplots(4, 2)
    for c in combined:
        purpose, row, column = c[0], c[1], c[2]
        x_data, y_data, x_model, y_model = func(data, col)
        ax[row, column].plot(x_data, y_data, marker='.', linestyle='none')
        ax[row, column].plot(x_model, y_model)
        ax[row, column].set_xlabel(purpose)
    fig.delaxes(ax[3,1])
    fig.suptitle(suptitle)
    fig.supxlabel(supxlabel)
    fig.supylabel(supylabel)
    return plt.show()

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

### define a function to create an array of the column for a certain variable
def gen_arrays_criteria(data, col_plot, col_count, cond1, comm1, comm2=None):
    sum = 0
    data = df.sort_values(col_plot)
    store = []
    for i, d in enumerate(data[col_count], 1):
        if d == cond1:
            sum += comm1
        else:
            if comm2 == None:
                sum -= d
            else:
                sum -= comm2
        mean = sum / i
        store.append(mean)
    results = (data[col_plot], store)
    return results

def plot_all_criteria(data, col_count, cond1, comm1, comm2=None, suptitle=None, supxlabel=None, supylabel=None):
    fig, ax = plt.subplots(3, 3)

    col_to_plot = ['int.rate', 'installment', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'log.annual.inc', 'log.debt']
    rows = np.sort(np.array([r if r < 3 else r % 3 for r in list(range(0, 9))]))
    columns = np.array([c % 3 for c in list(range(9))])
    combined = [(col, r, c) for col, r, c in zip(col_to_plot, rows, columns)]

    for comb in combined:
        col_plot, row, column = comb
        data_x, arrays = gen_arrays_criteria(data, col_plot, col_count, cond1, comm1, comm2)
        ax[row, column].plot(data_x, arrays)
        ax[row, column].set_xlabel(col_plot)
    fig.suptitle(suptitle)
    fig.supxlabel(supxlabel)
    fig.supylabel(supylabel)
    return plt.show()

def bar_plot_purposes(data, column, cond1, comm1, comm2=None, title=None, xlabel=None, ylabel=None):    
    purpose_list = list(data['purpose'].unique())
    for purpose in purpose_list:
        data_purpose = data[data['purpose'] == purpose]
        good = 0
        bad = 0
        for d in data_purpose[column]:
            if d == cond1:
                good += comm1
            else:
                if comm2 == None:
                    bad += d
                else:
                    bad += 1
        bad_perc = bad / good * 100
        good_perc = 100 - bad_perc
        plt.bar(purpose, good_perc)
        plt.bar(purpose, bad_perc, bottom=good_perc)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    return plt.show()

# ------------------------------ Defining DataFrame and Many Functions  ------------------------------ 

# ------------------------------ Clean and manipulates the data ------------------------------

### calculate each person's debt and insert into a new column on the dataframe
df['annual.income']= np.exp(df['log.annual.inc'])
df['debt'] = df['dti']*df['annual.income']
logged_debt = []

for debt in df['debt']:
    if debt == 0:
        logged_debt.append(0)
    else:
        logged_debt.append(np.log(debt))

df['log.debt'] = logged_debt
columns = ['int.rate', 'installment', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'annual.income', 'debt']

for column in columns:
    index = find_extremes(df, column)
    df = df.drop(index, axis=0)

data = df['inq.last.6mths']
mean = np.mean(data)
list = []
for r in range(1000):
    sample = np.random.exponential(data.unique(), size=9)
    mean_sample = np.mean(sample)
    list.append(mean_sample)

array = np.array(list)
array = np.sort(array)

# data_bar = data.value_counts()
plt.plot(array)
plt.plot(data)
plt.show()
# ------------------------------ Clean and manipulates the data ------------------------------ 

# ------------------------------ First Analysis ------------------------------ 
# ## plots the count of people's purposes of getting a loan 
# sns.countplot(x='purpose', data=df)
# plt.xticks(rotation=45)
# plt.xlabel('Purposes')
# plt.ylabel('Count')
# plt.show()

# ## plots the actual-annual-income for each borrower's purposes alongside it's exponential distributions
# suptitle = 'ECDF plot of annual income and it\'s exponential distributions model'
# supxlabel = 'Annual Income'
# supylabel = 'ECDF Plot'
# plot_each_purpose(df, 'annual.income', exponential_dist, suptitle, supxlabel, supylabel)

# ## plots the debt-to-income ratio for each borrower's purposes alongside it's normal distributions
# suptitle = 'ECDF plot of Debt-to-income and it\'s normal distributions model'
# supxlabel = 'Debt-to-income'
# supylabel = 'ECDF Plot'
# plot_each_purpose(df, 'dti', normal_dist, suptitle, supxlabel, supylabel)

# ## plots the debt for each borrower's purposes alongside it's exponential distributions
# suptitle = 'ECDF plot of the debts and it\'s exponential distributions model'
# supxlabel = 'Debt-to-income'
# supylabel = 'ECDF Plot'
# plot_each_purpose(df, 'debt', exponential_dist, suptitle, supxlabel, supylabel)

# ## makes a line regression and the relations plot of annual-income and it debts
# x_model, y_model, intercept, slope = line_regres(df['debt'], df['annual.income'])
# sns.relplot(x='debt', y='annual.income', data=df, kind='scatter', alpha=0.2)
# plt.plot(x_model, y_model, color='red')
# plt.annotate('Intercept : {0:.3f}\nSlope : {1:.4f}'.format(intercept, slope), xy=(np.max(df['debt'])*1/8,np.max(df['annual.income'])*3/4))
# plt.title('Relations between annual-income and debt')
# plt.xlabel('Debt')
# plt.ylabel('Annual income')
# plt.show()

# ------------------------------ First Analysis ------------------------------ 

# ------------------------------ Second Analysis ------------------------------ 

# ## create a line plot for each criteria to determine whether or not a person will pay their debt past due date
# suptitle = 'Line plot for the number of time a person pays their debt 30+ days past due date based on a certain criteria'
# supxlabel = 'Criteria'
# supylabel = 'Number of times a person pays their debt 30+ days past due date'
# _ = plot_all_criteria(df, 'delinq.2yrs', 0, 1, None, suptitle, supxlabel, supylabel)

# ## create a line plot for each criteria to determine whether or not a person will pay back their debt in full
# suptitle = 'Line plot for the distributions of people who doesn\'t pay their debt in full based on a certain criteria'
# supxlabel = 'Criteria'
# supylabel = 'The distributions of people who doesn\'t pay their debt in full'
# _ = plot_all_criteria(df, 'not.fully.paid', 0, 1, 1, suptitle, supxlabel, supylabel)

# ## create a bar plot to compare the percentage of people who pay their debts late for each purposes
# title = 'Bar plot for each purposes percentage of paying their debt on time and past due date'
# xlabel = 'Purposes'
# ylabel = 'Percentage'
# _ = bar_plot_purposes(df, 'delinq.2yrs', 0, 1, None, title, xlabel, ylabel)

# ## create a bar plot to compare the pecentage of people who pay their debt in full or not
# title = 'Bar plot for each purposes percentage of paying their debt in full or not'
# xlabel = 'Purposes'
# ylabel = 'Percentage'
# _ = bar_plot_purposes(df, 'not.fully.paid', 0, 1, 1, title, xlabel, ylabel)

# ------------------------------ Second Analysis ------------------------------ 

# ------------------------------ Third Analysis ------------------------------ 
# purpose_list = list(df['purpose'].unique())
# row = np.sort(np.array([r % 4 for r in list(range(8))]))
# column = [c % 2 for c in list(range(8))]
# combined = [(p, r, c) for p, r, c in zip(purpose_list, row, column)]

### plots the inquiries in the last 6 months and its exponential distributions for each purposes 
# fig, ax = plt.subplots(4, 2)

# for c in combined:
#     purpose, row, column = c[0], c[1], c[2]
#     data = df[df['purpose'] == purpose]
#     x_data, y_data, x_model, y_model = exponential_dist(df, 'inq.last.6mths')
#     ax[row, column].bar(x_data, y_data)
#     ax[row, column].plot(x_model, y_model, color='red')
#     ax[row, column].set_xlabel(purpose)
# fig.delaxes(ax[3, 1])
# fig.suptitle('Bar plot of the inquiries in the last 6 months and its exponential distributions for each puposes')
# fig.supxlabel('Purposes')
# fig.supylabel('Number of time of inquiries in the last 6 months')
# plt.show()


# fig, ax = plt.subplots(4, 2)

# for c in combined:
#     purpose, row, column = c[0], c[1], c[2]
#     data = df[df['purpose'] == purpose]['inq.last.6mths']
#     bs_replicates = np.empty(1000)
#     for i in range(1000):
#         bs_replicates[i] = np.mean(np.random.choice(data, len(data)))
#     percentile = np.percentile(bs_replicates, [2.5, 97.5])
#     ax[row, column].hist(bs_replicates, bins=20)
#     ax[row, column].axvline(percentile[0], color = 'red')
#     ax[row, column].axvline(percentile[1], color='red')
# fig.delaxes(ax[3, 1])
# plt.show()