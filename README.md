# python2018
python sample code

# Install enviroment

1. bash Miniconda3-latest-MacOSX-x86_64.sh (https://conda.io/miniconda.html)
2. conda create -n MLDEMO python=3.6 --file requirements.txt
3. source activate MLDEMO

```
conda install pandas

pip install --user --upgrade git+https://github.com/jpmml/sklearn2pmml.git

import pandas as pd
data = pd.read_csv('data/president_heights.csv')
heights = np.array(data['height(cm)'])
print(heights)
```
