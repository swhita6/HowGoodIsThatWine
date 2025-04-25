import pandas as pandas
import numpy as np
from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plot
  
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
  
# metadata 
print(wine_quality.metadata) 
  
# variable information 
print(wine_quality.variables) 

red_data = pandas.read_csv("datasets/winequality-red.csv", sep=';')
white_data = pandas.read_csv("datasets/winequality-white.csv", sep=';')