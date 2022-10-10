import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# Load the diabetes dataset
diabetes = datasets.load_diabetes()
array1 = np.array(diabetes.data)
array2 = np.array(diabetes.target)
array3 = np.column_stack((array1, array2))

listName=diabetes.feature_names
listName.append("Target")

max=array3.shape[1]


import seaborn as sns
sns.set_theme(style="white")
df = pd.DataFrame(diabetes.data, columns= ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'])
df['target'] = diabetes.target
g = sns.PairGrid(df, diag_sharey=False)
g.map_upper(sns.scatterplot, s=15)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=2)
plt.savefig("seaborn.png")
plt.show()

