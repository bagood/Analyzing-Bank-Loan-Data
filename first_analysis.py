import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from functions import Stats_func

class First_analysis:
    def __init__(self, data):
        self.data = data
        self.func = Stats_func(self.data)
    
    def ecdf_annual_inc(self):
        suptitle = 'ECDF plot of the annual income and it\'s exponential distribution model'
        supxlabel = 'Annual Income'
        supylabel = 'ECDF Plot'
        return self.func.plot_each_purpose('annual.income', self.func.exponential_dist, suptitle, supxlabel, supylabel)
    
    def ecdf_debt(self):
        suptitle = 'ECDF plot of the debt and it\'s exponential distribution model'
        supxlabel = 'Debt'
        supylabel = 'ECDF Plot'
        return self.func.plot_each_purpose('debt', self.func.exponential_dist, suptitle, supxlabel, supylabel)

    def correlation_between_debt_income(self):
        x_model, y_model, intercept, slope = self.func.line_regres(self.data['debt'], self.data['annual.income'])
        plt.figure(figsize=(15,15))
        sns.relplot(x='debt', y='annual.income', data=self.data, kind='scatter', alpha=0.2)
        plt.plot(x_model, y_model, color='red')
        plt.annotate('Model intercept : {0:.3f}\nModel slope : {1:.4f}'.format(intercept, slope), xy=(np.max(self.data['debt'])*1/8, np.max(self.data['annual.income'])*3/4))
        plt.title('Relations between annual-income and debt')
        plt.xlabel('Debt')
        plt.ylabel('Annual income')
        plt.legend(('Relation plot', 'Data linear model'))
        return plt.show()