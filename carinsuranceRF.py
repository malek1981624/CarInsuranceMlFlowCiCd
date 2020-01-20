#learning_job.py
#!/usr/bin/python3
import pandas as pd
import sklearn.datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

# parameter

n_estimators = 15
max_depth = 3
random_state = 71


# iris dataを分類してみる

path='./data/'
dftrain = pd.read_csv(path+'carInsurance_train.csv', encoding='utf8', sep=',')
#dftest = pd.read_csv(path+'carInsurance_test.csv', encoding='utf8', sep=',')

dftrainenc = pd.read_csv(path+'train_LabelEncoding.csv', encoding='utf8', sep=',')
dftestenc = pd.read_csv(path+'test_LabelEncoding.csv', encoding='utf8', sep=',')
df03= dftrain['CarInsurance']

#labels = np.array(df.pop('CarInsurance'))

# 30% examples in test data
'''
train, test, train_labels, test_labels = train_test_split(dftrainenc,
                                         df03, 
                                         stratify = df03,
                                         test_size = 0.3, 
                                         random_state = RSEED)
'''
X_train, X_test, y_train, y_test = train_test_split(dftrainenc,df03, test_size=0.3, random_state=random_state)

# RandomForestClassifierを利用して学習


model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
model.fit(X_train, y_train)

# 予測結果確認
print(model.predict(X_test))

# 精度情報確認

score = model.score(X_test, y_test)

# MLflowに記録
mlflow.set_tracking_uri('http://localhost:5000')
with mlflow.start_run():
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)
    mlflow.log_metric("score", score)
    mlflow.sklearn.log_model(model, "ml_models")
mlflow.end_run()