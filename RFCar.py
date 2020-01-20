#learning_job.py
#!/usr/bin/python3
import inline as inline
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
#%matplotlib inline
from sklearn import metrics
from sklearn.model_selection import train_test_split,cross_val_score,KFold,cross_val_predict
from sklearn.metrics import accuracy_score, f1_score,classification_report, precision_score, recall_score,confusion_matrix,precision_recall_curve,roc_curve

import sklearn.datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split,cross_val_score,KFold,cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score,confusion_matrix,precision_recall_curve,roc_curve
from sklearn.model_selection import GridSearchCV
#from sklearn.grid_search import GridSearchCV
import mlflow
import mlflow.sklearn
import sys
# parameter
RSEED=42
parameters={'n_estimators':[100],
            'max_depth':[10],
            'max_features':[13,14],
            'min_samples_split':[11]}

#n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 10
#max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 5
#random_state = int(sys.argv[3]) if len(sys.argv) > 3 else 71

n_estimators=100
max_depth=10
max_features=15

path='./data/'
dftrain = pd.read_csv(path+'carInsurance_train.csv', encoding='utf8', sep=',')
#dftest = pd.read_csv(path+'carInsurance_test.csv', encoding='utf8', sep=',')

dftrainenc = pd.read_csv(path+'train.csv', encoding='utf8', sep=',')
dftestenc = pd.read_csv(path+'test.csv', encoding='utf8', sep=',')
df03= dftrain['CarInsurance']

#labels = np.array(df.pop('CarInsurance'))

# 30% examples in test data
'''
train, test, train_labels, test_labels = train_test_split(dftrainenc,df03,stratify = df03,test_size = 0.3, random_state = RSEED)
'''
X_train, X_test, y_train, y_test = train_test_split(dftrainenc,df03, test_size=0.3, random_state=RSEED)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# RandomForestClassifierを利用して学習

#modelrf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=RSEED)
modelrf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, oob_score = True, n_jobs = -1, random_state = RSEED, max_features = "auto", min_samples_leaf = 50)
modelrf = modelrf.fit(X_train, y_train)
score2 = modelrf.score(X_test, y_test)
y_pred = modelrf.predict(X_test)

#model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
#model.fit(X_train, y_train)
print(modelrf.predict(X_test))

# Create la fonction de cross validation
def get_best_model(estimator, params_grid={}):
    model = GridSearchCV(estimator = estimator,param_grid = params_grid,cv=3, scoring="accuracy", n_jobs= -1)
    model.fit(X_train,y_train)
    print('\n--- Meilleurs Parametres -----------------------------')
    print(model.best_params_)
    mlflow.log_param("Best_Params_du_modele", model.best_params_)

    print('\n--- Meilleur Modele -----------------------------')
    best_model = model.best_estimator_
    mlflow.log_param("Best_Model", best_model)
    print(best_model)
    return best_model


# Créer la fonction de fitting de modele
def model_fit(model, feature_imp=True, cv=5):
    # model fit
    clf = model.fit(X_train, y_train)

    # model prediction
    y_pred = clf.predict(X_test)

    # model report
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=class_names, title='Matrice de Confusion')
    plt.savefig('Matrice de Confusion.png')

    #mlflow.log_artifact('Matrice de Confusion.png')

    print('\n--- Ensemble de Train -----------------------------')
    print('Accuracy: %.5f +/- %.4f' % (
    np.mean(cross_val_score(clf, X_train, y_train, cv=cv)), np.std(cross_val_score(clf, X_train, y_train, cv=cv))))
    #mlflow.log_metric("Accuracy Train", np.mean(cross_val_score(clf, X_train, y_train, cv=cv)),np.std(cross_val_score(clf, X_train, y_train, cv=cv)))

    print('AUC: %.5f +/- %.4f' % (np.mean(cross_val_score(clf, X_train, y_train, cv=cv, scoring='roc_auc')),
                                  np.std(cross_val_score(clf, X_train, y_train, cv=cv, scoring='roc_auc'))))

    #mlflow.log_metric("AUC Train", np.mean(cross_val_score(clf, X_train, y_train, cv=cv, scoring='roc_auc')),
    #                              np.std(cross_val_score(clf, X_train, y_train, cv=cv, scoring='roc_auc')))

    print('\n--- Ensemble de Validation -----------------------------')
    print('Accuracy: %.5f +/- %.4f' % (np.mean(cross_val_score(clf, X_test, y_test, cv=cv)), np.std(cross_val_score(clf, X_test, y_test, cv=cv))))

    #mlflow.log_metric("Accuracy test ensemble Validation", np.mean(cross_val_score(clf, X_train, y_train, cv=cv)),
    #                  np.std(cross_val_score(clf, X_train, y_train, cv=cv)))

    print('AUC: %.5f +/- %.4f' % (np.mean(cross_val_score(clf, X_test, y_test, cv=cv, scoring='roc_auc')),
                                  np.std(cross_val_score(clf, X_test, y_test, cv=cv, scoring='roc_auc'))))
    #mlflow.log_metric("AUC test ensemble Validation", np.mean(cross_val_score(clf, X_test, y_test, cv=cv, scoring='roc_auc')),
    #                              np.std(cross_val_score(clf, X_test, y_test, cv=cv, scoring='roc_auc')))

    print('-----------------------------------------------')

    # features importants
    if feature_imp:
        feat_imp = pd.Series(clf.feature_importances_, index=dftrainenc.columns)
        feat_imp = feat_imp.nlargest(15).sort_values()
        plt.figure()
        feat_imp.plot(kind="barh", figsize=(6, 8), title="Les Features les plus Important")
        plt.savefig('CAR Insurance_roc_auc_curve.png')
        #mlflow.log_artifact('CAR Insurance_roc_auc_curve.png')

# Afficher la matrice de confusion
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Matrice de Confusion',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Vrai label')
    plt.xlabel('Label prédit')
    plt.savefig('Matrice_confusion.png')
    mlflow.log_artifact('Matrice_confusion.png')


class_names = ['Success', 'Failure']

#set the artifact_path to location where experiment artifacts will be saved
#log model params
#mlflow.set_tracking_uri('http://localhost:5000')
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
MAE=metrics.mean_absolute_error(y_test, y_pred)
MSE= metrics.mean_squared_error(y_test, y_pred)
RMSE=np.sqrt(metrics.mean_squared_error(y_test, y_pred))
#accuracy = ( modelrf.score ( X_train , y_train ))

############################################
errors = (y_pred != y_test).sum()
mlflow.log_metric("Nbre_Erreurs_Classification", errors)
acc = accuracy_score(y_pred, y_test) * 100
mlflow.log_metric("accuracy_score", acc)
ps = precision_score(y_pred, y_test) * 100
mlflow.log_metric("precision_score", ps)
rs = recall_score(y_pred, y_test) * 100
mlflow.log_metric("recall_score", rs)
f1 = f1_score(y_pred, y_test) * 100
mlflow.log_metric("f1_score", f1)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
#mlflow.log_metric("confusion_matrix", confmat)
print("nombre Erreurs de Classification   : {0:.2f}".format(errors))
print("Accuracy est de  : {0:.2f}".format(acc))
print("Precision  : {0:.2f}".format(ps))
print("Recall     : {0:.2f}".format(rs))
print("F1 Score   : {0:.2f}".format(f1))
print(confmat)
##############################################
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))
cm=confusion_matrix(y_test,y_pred)
print("*"*100)
print(classification_report(y_test,y_pred))
cr=classification_report(y_test,y_pred)
#print(accuracy_score(y_test, y_pred))

accur=accuracy_score(y_test, modelrf.predict(X_test))
accuracy = metrics.accuracy_score(y_test, y_pred)
#KFold=modelrf.KFold()
#Cross_val_prediction=modelrf.cross_val_predict()
#Accuracy=modelrf.accuracy_score()
#ClassificationReport=modelrf.classification_report()
#Precision=modelrf.precision_score()
#Recall=modelrf.recall_score()
#F1Score= modelrf.f1_score()
#mlflow.sklearn.log_model()
mlflow.log_param("n_estimators2", n_estimators)
mlflow.log_param("max_depth2", max_depth)
mlflow.log_param("random_state2", RSEED)
mlflow.log_metric("score2", score2)
#mlflow.log_metric("Cross_val_prediction",Cross_val_prediction)
#mlflow.log_metric("KFold",KFold)
#mlflow.log_metric("cross_val_predict",ClassificationReport)
mlflow.log_metric("accuracy_score",accuracy)
mlflow.log_metric("accur",accur)
mlflow.log_metric('MAE Mean Absolute Error', MAE)
mlflow.log_metric('MSE Mean Squared Error', MSE)
mlflow.log_metric('RMSE Root Mean Squared Error', RMSE)

#mlflow.log_metric("classification_report",ClassificationReport)
#mlflow.log_metric("precision_score",Precision)
#mlflow.log_metric("recall_score",Recall)
#mlflow.log_metric("F1Score",F1Score)


#confusion_matrix
#precision_recall_curve
#roc_curve
#mlflow.log_metric("RMSE", modelrf.rmse())
#log model
mlflow.sklearn.log_model(modelrf, "rf_models")
from sklearn.metrics import precision_recall_fscore_support
#precision_recall_fscore_support(y_train, y_pred, average='micro')
#rf = RandomForestClassifier(random_state=42)
'''
#parameters={'n_estimators':[100],
            'max_depth':[10],
            'max_features':[14,15],
            'min_samples_split':[11]}
'''
#
#clf_rf= get_best_model(rf,parameters)
#model_fit(model=clf_rf, feature_imp=True)
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib
import matplotlib.pyplot as plt
rf_prediction_proba = modelrf.predict_proba(X_test)[:, 1]
def roc_curve_and_score(y_test, pred_proba):
    fpr, tpr, _ = roc_curve(y_test.ravel(), pred_proba.ravel())
    roc_auc = roc_auc_score(y_test.ravel(), pred_proba.ravel())
    return fpr, tpr, roc_auc


plt.figure(figsize=(8, 6))
matplotlib.rcParams.update({'font.size': 14})
plt.grid()
fpr, tpr, roc_auc = roc_curve_and_score(y_test, rf_prediction_proba)
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='ROC AUC={0:.3f}'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.legend(loc="lower right")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC_TUVE.png')
plt.savefig('ROC_TUVE.png')
plt.show()
mlflow.log_artifact('ROC_TUVE.png')

##################################################################

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import recall_score, precision_score


def evaluate_model(predictions, probs, train_predictions, train_probs):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""

    baseline = {}
    baseline['Accuracy'] = accuracy_score(y_test,
                                          [1 for _ in range(len(y_test))])

    baseline['Recall'] = recall_score(y_test,
                                      [1 for _ in range(len(y_test))])
    baseline['Precision'] = precision_score(y_test,
                                            [1 for _ in range(len(y_test))])
    baseline['ROC'] = 0.5

    results = {}
    results['Accuracy'] = accuracy_score(y_test, predictions)
    results['Recall'] = recall_score(y_test, predictions)
    results['Precision'] = precision_score(y_test, predictions)
    results['ROC'] = roc_auc_score(y_test, probs)

    train_results = {}
    train_results['Accuracy'] = accuracy_score(y_train, train_predictions)
    train_results['Recall'] = recall_score(y_train, train_predictions)
    train_results['Precision'] = precision_score(y_train, train_predictions)
    train_results['ROC'] = roc_auc_score(y_train, train_probs)

    for metric in ['Accuracy', 'Recall', 'Precision', 'ROC']:
        print(
            f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')

    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])
    model_fpr, model_tpr, _ = roc_curve(y_test, probs)

    plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 16

    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label='baseline')
    plt.plot(model_fpr, model_tpr, 'r', label='model')
    plt.legend();
    plt.title('roc_auc_curve.png')
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate');
    plt.title('ROC Curves');
    plt.savefig('roc_auc_curve.png')
    mlflow.log_artifact('roc_auc_curve.png')

    plt.show();

# Training predictions (to demonstrate overfitting)
train_rf_predictions = modelrf.predict(X_train)
train_rf_probs = modelrf.predict_proba(X_train)[:, 1]
# Testing predictions (to determine performance)
rf_predictions = modelrf.predict(X_test)
rf_probs = modelrf.predict_proba(X_test)[:, 1]

evaluate_model(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs)

##########################################

from sklearn.metrics import confusion_matrix
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=12)
    plt.yticks(tick_marks, classes, size=12)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size=14)
    plt.xlabel('Predicted label', size=14)


# Confusion matrix
cm = confusion_matrix(y_test, rf_predictions)
plot_confusion_matrix(cm, classes=['Car Insurance', 'No Car Insurance'],
                      title='Car_Insurance_Confusion Matrix')

plt.savefig('Car_Insurance_Confusion_Matrix.png')
mlflow.log_artifact('Car_Insurance_Confusion_Matrix.png')

#####################################################################

mse              = metrics.mean_squared_error(y_test, y_pred)
logloss_test    = metrics.log_loss(y_test, y_pred)
accuracy_test   = metrics.accuracy_score(y_test, y_pred)
accuracy_test2  = modelrf.score(X_test, y_test)
F1_test         = metrics.f1_score(y_test, y_pred)
precision_test  = precision_score(y_test, y_pred, average='binary')
precision_test2 = metrics.precision_score(y_test, y_pred)
recall_test     = recall_score(y_test, y_pred, average='binary')
auc_test        = metrics.roc_auc_score(y_test, y_pred)
r2_test         = metrics.r2_score(y_test, y_pred)

mlflow.log_metric('mse22',mse)
mlflow.log_metric('logloss_test',logloss_test)
mlflow.log_metric('accuracy_test',accuracy_test)
mlflow.log_metric('accuracy_test',accuracy_test)
mlflow.log_metric('F1_test',F1_test)
mlflow.log_metric('precision_test',precision_test)
mlflow.log_metric('precision_test2',precision_test2)
mlflow.log_metric('recall_test',recall_test)
mlflow.log_metric('auc_test',auc_test)
mlflow.log_metric('r2_test',r2_test)

print(mse ,logloss_test,accuracy_test ,accuracy_test2 ,F1_test,precision_test,precision_test2,recall_test ,auc_test,r2_test)