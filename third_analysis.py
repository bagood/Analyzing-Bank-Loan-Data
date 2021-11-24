import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from functions import Stats_func

class Third_analysis:
    def __init__(self, data):
        self.data = data
        self.func = Stats_func(self.data)
        self.purposes = list(self.data['purpose'].unique())
        self.rows = np.sort(np.array([r % 4 for r in list(range(8))]))
        self.columns = [c % 2 for c in list(range(8))]

    def data_model_distributions(self):
        fig, ax = plt.subplots(4, 2)
        fig.set_size_inches(18.5, 16.5)
        plt.style.use('bmh')
        for purpose, row, column in zip(self.purposes, self.rows, self.columns):
            data = self.data[self.data['purpose'] == purpose]
            x_data, y_data, x_model, y_model = self.func.exponential_dist('inq.last.6mths')
            ax[row, column].bar(x_data, y_data)
            ax[row, column].plot(x_model, y_model, color='red')
            ax[row, column].set_xlabel(purpose)
        fig.delaxes(ax[3, 1])
        fig.suptitle('ECDF plot of the inquiries in the last 6 months and its exponential distribution for each puposes')
        fig.supxlabel('Number of time of inquiries in the last 6 months')
        fig.supylabel('ECDF')
        return plt.show()
    
    def data_simulations(self):
        fig, ax = plt.subplots(4, 2)
        fig.set_size_inches(18.5, 16.5)
        plt.style.use('bmh')
        for purpose, row, column in zip(self.purposes, self.rows, self.columns):
            data = self.data[self.data['purpose'] == purpose]['inq.last.6mths']
            bs_replicates = np.empty(1000)
            for i in range(1000):
                bs_replicates[i] = np.mean(np.random.choice(data, len(data)))
            percentile = np.percentile(bs_replicates, [2.5, 97.5])
            ax[row, column].hist(bs_replicates, bins=20)
            ax[row, column].axvline(percentile[0], color = 'red')
            ax[row, column].axvline(percentile[1], color='red')
            ax[row, column].set_xlabel(purpose)
        fig.suptitle('Simulates the event of getting a response from customers for 1000 times and count the mean for each of the simulations')
        fig.supxlabel('Mean of getting a feedback from customers')
        fig.supylabel('Count of times getting the mean')
        fig.delaxes(ax[3, 1])
        return plt.show()