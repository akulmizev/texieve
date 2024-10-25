import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

results = pd.read_csv("./sib200_results.csv")
results = np.stack(results.loc[:, results.columns != "Wiki-ID"].to_numpy())
results = np.corrcoef(results, rowvar=False)

sns.heatmap(results)
plt.show()
