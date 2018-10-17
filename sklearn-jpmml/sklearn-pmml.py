import numpy as np
import pandas as pd
from sklearn import tree
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml

import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files/Java/jdk1.8.0_171/bin'

X=[[1,2,3,1],[2,4,1,5],[7,8,3,6],[4,8,4,7],[2,5,6,9]]
y=[0,1,0,2,1]
pipeline = PMMLPipeline([("classifier", tree.DecisionTreeClassifier(random_state=9))]);
pipeline.fit(X,y)

# pip install --user --upgrade git+https://github.com/jpmml/sklearn2pmml.git
sklearn2pmml(pipeline, "demo.pmml", with_repr = True)

from sklearn.externals import joblib
joblib.dump(pipeline, "pipeline.pkl.z", compress = 9)

# java -jar target/jpmml-sklearn-executable-1.5-SNAPSHOT.jar --pkl-input pipeline.pkl.z --pmml-output pipeline.pmml
