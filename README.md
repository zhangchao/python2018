# python2018
python sample code 

# Install enviroment 

1. bash Miniconda3-latest-MacOSX-x86_64.sh (https://conda.io/miniconda.html)
2. conda create -n PDSH python=3.5 --file requirements.txt
3. source activate PDSH

```
conda install pandas

import pandas as pd
data = pd.read_csv('data/president_heights.csv')
heights = np.array(data['height(cm)'])
print(heights)
```
