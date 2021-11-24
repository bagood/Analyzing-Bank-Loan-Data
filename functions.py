import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.formula.api import ols

class Stats_func:
    def __init__(self, data):
        self.data = data

    def ecdf(self, dataframe):
        length = len(dataframe)
        x_val = np.sort(dataframe)
        y_val = np.array([r/length for r in range(1,length+1)])
        coordinates = (x_val, y_val)
        return coordinates

    def normal_dist(self, column):
        mean = np.mean(self.data[column])
        std = np.std(self.data[column])
        samples = np.random.normal(mean, std, size=1000)
        x_data, y_data = self.ecdf(self.data[column])
        x_model, y_model = self.ecdf(samples)
        data_model_coordinates = (x_data, y_data, x_model, y_model)
        return data_model_coordinates

    def exponential_dist(self, column):
        mean = np.mean(self.data[column])
        samples = np.random.exponential(mean, size=1000)
        x_data, y_data = self.ecdf(self.data[column])
        x_model, y_model = self.ecdf(samples)
        data_model_coordinates = (x_data, y_data, x_model, y_model)
        return data_model_coordinates

    def line_regres(self, x_data, y_data):
        sample = pd.DataFrame(dict(x_val=x_data, y_val=y_data))
        model_fit = ols(formula='y_val ~ x_val', data=sample).fit()
        x_model = x_data
        y_model = model_fit.predict(sample)
        intercept = model_fit.params['Intercept']
        slope = model_fit.params['x_val']
        model_intercept_slope_coor = (x_model, y_model, intercept, slope)
        return model_intercept_slope_coor

    def plot_each_purpose(self, col_agg, func, suptitle, supxlabel, supylabel):
        purpose_list = list(self.data['purpose'].unique())
        rows = np.sort(np.array([r if r < 4 else r - 4 for r in list(range(8))]))
        columns = [c % 2 for c in list(range(8))]
        plt.style.use('bmh')
        fig, ax = plt.subplots(4, 2)
        for purpose, row, column  in zip(purpose_list, rows, columns):
            x_data, y_data, x_model, y_model = func(col_agg)
            ax[row, column].plot(x_data, y_data, marker='.', linestyle='none')
            ax[row, column].plot(x_model, y_model)
            ax[row, column].set_xlabel(purpose)
        fig.set_size_inches(18.5, 15.5)
        fig.delaxes(ax[3,1])
        fig.suptitle(suptitle)
        fig.supxlabel(supxlabel)
        fig.supylabel(supylabel)
        return plt.show()

    def gen_arrays_criteria(self, col_plot, col_count, cond1, comm1, comm2=None):
        data = self.data.sort_values(col_plot)
        sum = 0
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
        graph_coordinates = (data[col_plot], store)
        return graph_coordinates

    def plots_all_criteria(self, col_count, cond1, comm1, comm2=None, suptitle=None, supxlabel=None, supylabel=None):
        col_to_plot = ['int.rate', 'installment', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'log.annual.inc', 'log.debt']
        rows = np.sort(np.array([r if r < 3 else r % 3 for r in list(range(0, 9))]))
        columns = np.array([c % 3 for c in list(range(9))])

        plt.style.use('bmh')
        fig, ax = plt.subplots(3, 3)
        for col_plotted, row, column in zip(col_to_plot, rows, columns):
            data_x, arrays = self.gen_arrays_criteria(col_plotted, col_count, cond1, comm1, comm2)
            ax[row, column].plot(data_x, arrays)
            ax[row, column].set_xlabel(col_plotted)
        fig.set_size_inches(18.5, 10.5)
        fig.suptitle(suptitle)
        fig.supxlabel(supxlabel)
        fig.supylabel(' '*20 + f'{supylabel}\n\n \n\n <---- more likely' + ' '*100 + 'less likely ---->')
        return plt.show()
    
    def bar_plot_purposes(self, column, cond1, comm1, comm2=None, title=None, xlabel=None, ylabel=None,  bott_bar=None, top_bar=None):    
        purpose_list = list(self.data['purpose'].unique())
        plt.style.use('bmh')
        plt.figure(figsize=(10,10))
        for purpose in purpose_list:
            data_purpose = self.data[self.data['purpose'] == purpose]
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
            plt.bar(purpose, good_perc, color='cornflowerblue')
            plt.bar(purpose, bad_perc, bottom=good_perc, color='red')
        plt.legend((bott_bar, top_bar))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        return plt.show()
