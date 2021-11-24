import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from functions import Stats_func

class Second_analysis:
    def __init__(self, data):
        self.data = data
        self.func = Stats_func(self.data)
    
    def late_debt_likelihood(self):
        suptitle = 'Line plot of the likelihood that people pay their debt 30+ days past due date based on a certain criteria'
        supxlabel = 'Criteria'
        supylabel = 'The likelihood that the person pays their debt 30+ days past due date'
        return self.func.plots_all_criteria('delinq.2yrs', 0, 1, None, suptitle, supxlabel, supylabel)        
    
    def not_fully_paid_likelihood(self):
        suptitle = 'Line plot of the likelihood that people doesn\'t pay their debt in full based on a certain criteria'
        supxlabel = 'Criteria'
        supylabel = 'The likelihood that people doesn\'t pay their debt in full'
        return self.func.plots_all_criteria('not.fully.paid', 0, 1, 1, suptitle, supxlabel, supylabel)
    
    def percentage_pays_late(self):
        title = 'Bar plot for each purposes percentage of paying their debt on time and past due date'
        xlabel = 'Purposes'
        ylabel = 'Percentage (%)'
        bott_bar = 'Pays on time'
        top_bar = 'Pays past due date'
        return self.func.bar_plot_purposes('delinq.2yrs', 0, 1, None, title, xlabel, ylabel, bott_bar, top_bar)
    
    def percentage_not_fully_paid(self):
        title = 'Bar plot for each purposes percentage of paying their debt in full or not'
        xlabel = 'Purposes'
        ylabel = 'Percentage (%)'
        bott_bar = 'Debts paid in full'
        top_bar = 'Debts not paid in full'
        return self.func.bar_plot_purposes('not.fully.paid', 0, 1, 1, title, xlabel, ylabel, bott_bar, top_bar)