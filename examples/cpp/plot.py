import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# to be used with notebook
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
#sns.set_style("whitegrid")

df = pd.read_csv('results.txt', delimiter=r'\s+', comment='#', quotechar='"', error_bad_lines = False)

# coerces TBD to Nan and treated as number
df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
df['RMSE'] = pd.to_numeric(df['RMSE'], errors='coerce')

# dropping TBD values, can be added when info is available
df.dropna(inplace=True)
df.set_index('algo', inplace = True)

for task in ["AlphaLambda", "Single"]:
  mydf = df.loc[df["task"]==task]
  ax = mydf.plot.barh(figsize = (15,10), width = 0.7, logx = True, color = ['dodgerblue', 'lightcoral'])

  for i in ax.patches:
      ax.text(i.get_width()+1, i.get_y()+.1, \
              str(int((i.get_width()))), fontsize=16, color='k')

  ax.legend(ncol=3, loc="best", frameon=True)

  if (task=="AlphaLambda"):
    what="(for 1600 models: alpha/lambda search)"
  else:
    what="(for a single model)"
  ax.set_title('GLM Elastic Net\nvalidation error and training time ' + what, fontsize = 16)
  ax.set_xlabel('time/sec (log-scale)', fontsize = 16)

  # # axis can be inverted using below line
  #ax.invert_yaxis()

  plt.savefig("results" + task + ".png")
  plt.close()
