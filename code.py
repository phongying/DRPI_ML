import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, classification_report

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import roc_auc_score, recall_score
from sklearn.metrics import precision_score, roc_curve, recall_score, f1_score, roc_auc_score, accuracy_score
import xgboost as xgb
from scipy import stats
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

data = pd.read_csv('data.csv')
y = data.iloc[:, -1]
x_og = data.iloc[:, :-1]
xtr, xte, ytr, yte = train_test_split(x_og, y, test_size=0.2, random_state=1)
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_new_train_normal, y_new_train_normal = ros.fit_resample(xtr, ytr)
print(sorted(Counter(y_new_train_normal).items()))
print(sorted(Counter(ytr).items()))
xtr = X_new_train_normal
ytr = y_new_train_normal
x = pd.concat([xtr, xte], axis=0)
y = pd.concat([ytr, yte], axis=0)
# X_train, Y_train, X_test, Y_test = xtr, ytr, xte, yte
X_new_train, y_new_train, X_new_test, y_new_test = xtr, ytr, xte, yte




from sklearn.linear_model import LogisticRegression  
from sklearn.linear_model import SGDClassifier  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.svm import SVC  
from sklearn.tree import DecisionTreeClassifier  #
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import GridSearchCV  
from sklearn.model_selection import StratifiedKFold  
from collections import Counter
from sklearn.metrics import confusion_matrix,  f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier  
from xgboost import XGBClassifier  

classifiers = {
    'LogisticRegression': LogisticRegression(),  
    "SVC": SVC(),  
    "KNN": KNeighborsClassifier(), 
    'DT': DecisionTreeClassifier(),  
    'RFC': RandomForestClassifier(),  
    'Bagging': BaggingClassifier(), 
    'SGD': SGDClassifier(),  
    'GBC': GradientBoostingClassifier(),  
    'xgb': XGBClassifier() 
}



# 1 LR
def LR_gs(X_train, y_train):
    # LR
    LR_param = {
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 1, 10]
    }

    LR_gs = GridSearchCV(LogisticRegression(), param_grid=LR_param, n_jobs=-1, scoring='roc_auc')
    LR_gs.fit(X_train, y_train)

    LR_estimators = LR_gs.best_estimator_  

    return LR_estimators


# 2 KNN
def KNN_gs(X_train, y_train):
    KNN_param = {
        'n_neighbors': list(range(2, 5, 1)),
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

    KNN_gs = GridSearchCV(KNeighborsClassifier(), param_grid=KNN_param, n_jobs=-1, scoring='roc_auc')
    KNN_gs.fit(X_train, y_train)

    KNN_estimators = KNN_gs.best_estimator_  

    return KNN_estimators


# 3 SVC
def SVC_gs(X_train, y_train):
    SVC_param = {
        'C': [0.5, 0.7, 0.9, 1, 5, 10, 15, 20, 25, 30, 50],
        'kernel': ['rfb', 'poly', 'sigmod', 'linear']
    }

    SVC_gs = GridSearchCV(SVC(probability=True), param_grid=SVC_param, n_jobs=-1, scoring='roc_auc')
    SVC_gs.fit(X_train, y_train)

    SVC_estimators = SVC_gs.best_estimator_ 

    return SVC_estimators


# 4 DT
def DT_gs(X_train, y_train):
    DT_param = {
        'criterion': ['gini', 'entropy'],  
        'max_depth': list(range(2, 10, 1)),  
        'min_samples_leaf': list(range(3, 10, 1))  
    }

    DT_gs = GridSearchCV(DecisionTreeClassifier(), param_grid=DT_param, n_jobs=-1, scoring='roc_auc')
    DT_gs.fit(X_train, y_train)

    DT_estimators = DT_gs.best_estimator_  
    return DT_estimators


# 5 RFC
def RFC_gs(X_train, y_train):
    RFC_param = {
        'n_estimators': [*range(1, 300)],  
        'criterion': ['gini', 'entropy'],  
        'max_depth': list(range(2, 5, 1)),  
    }

    RFC_gs = GridSearchCV(RandomForestClassifier(), param_grid=RFC_param, n_jobs=-1, scoring='roc_auc')
    RFC_gs.fit(X_train, y_train)

    RFC_estimators = RFC_gs.best_estimator_

    return RFC_estimators


# 6 Bag
def BAG_gs(X_train, y_train):
    BAG_param = {
        'n_estimators': [*range(1, 20)]  
    }

    BAG_gs = GridSearchCV(BaggingClassifier(), param_grid=BAG_param, n_jobs=-1, scoring='roc_auc')
    BAG_gs.fit(X_train, y_train)

    BAG_estimators = BAG_gs.best_estimator_

    return BAG_estimators


# 7 SGD
def SGD_gs(X_train, y_train):
    SGD_param = {
        'penalty': ['l2', 'l1'],
        'max_iter': [1000, 1500, 2000]
    }

    SGD_gs = GridSearchCV(SGDClassifier(), param_grid=SGD_param, n_jobs=-1, scoring='roc_auc')
    SGD_gs.fit(X_train, y_train)

    SGD_estimators = SGD_gs.best_estimator_

    return SGD_estimators


# 8 xgb
def XGB_gs(X_train, y_train):
    XGB_gs = xgbo(X_train, y_train)

    XGB_estimators = XGB_gs.fit(X_train, y_train)

    return XGB_estimators


LR_best_estimator = LR_gs(X_new_train, y_new_train)

KNN_best_estimator = KNN_gs(X_new_train, y_new_train)

SVC_best_estimator = SVC_gs(X_new_train, y_new_train)

DT_best_estimator = DT_gs(X_new_train, y_new_train)

RFC_best_estimator = RFC_gs(X_new_train, y_new_train)

BAG_best_estimator = BAG_gs(X_new_train, y_new_train)

SGD_best_estimator = SGD_gs(X_new_train, y_new_train)

XGB_best_estimator = XGB_gs(X_new_train, y_new_train)


# In[12]:



from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

result_df = pd.DataFrame(columns=['Accuracy', 'F1-score', 'Recall', 'Precision', 'AUC_ROC'],
                         index=['LR', 'KNN', 'DT', 'RFC', 'Bagging', 'SVC', 'XGB'])


#,'SGD'

def caculate(models, X_test, y_test):
    accuracy_results = []
    F1_score_results = []
    Recall_results = []
    Precision_results = []
    AUC_ROC_results = []
    cm_r = []

    for model in models:
        y_pred = model.predict(X_test)
        y_test_pred = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)  # 计算准确度
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred,
                                                                         average='binary')  
        AUC_ROC = roc_auc_score(y_test, y_test_pred)  
        # AUC_ROC=cross_val_score(model, X_test, y_test, cv=5, n_jobs=-1, scoring='roc_auc').mean()
        cm = confusion_matrix(y_test, y_pred)
        accuracy_results.append(accuracy)
        F1_score_results.append(f1_score)
        Recall_results.append(recall)
        AUC_ROC_results.append(AUC_ROC)
        Precision_results.append(precision)
        cm_r.append(cm)

    return accuracy_results, F1_score_results, Recall_results, AUC_ROC_results, Precision_results, cm_r


best_models = [LR_best_estimator, KNN_best_estimator, DT_best_estimator, RFC_best_estimator,
               BAG_best_estimator, SVC_best_estimator, XGB_best_estimator]

accuracy_results, F1_score_results, Recall_results, AUC_ROC_results, Precision_results, cm_r = caculate(best_models,
                                                                                                        X_new_test,
                                                                                                        y_new_test)

result_df['Accuracy'] = accuracy_results
result_df['F1-score'] = F1_score_results
result_df['Recall'] = Recall_results
result_df['Precision'] = Precision_results
result_df['AUC_ROC'] = AUC_ROC_results
result_df['cm'] = cm_r
result_df



accuracy_results, F1_score_results, Recall_results, AUC_ROC_results, Precision_results, cm_r = caculate(best_models,
                                                                                                        X_new_train,
                                                                                                        y_new_train)
result_df1 = pd.DataFrame(columns=['Accuracy', 'F1-score', 'Recall', 'Precision', 'AUC_ROC'],
                          index=['LR', 'KNN', 'DT', 'RFC', 'Bagging', 'SVC', 'XGB'])
# 将各项值放入到DataFrame中
result_df1['Accuracy'] = accuracy_results
result_df1['F1-score'] = F1_score_results
result_df1['Recall'] = Recall_results
result_df1['Precision'] = Precision_results
result_df1['AUC_ROC'] = AUC_ROC_results
result_df1['cm'] = cm_r
result_df1



result_df_test = pd.DataFrame(
    columns=['AUC_ROC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1-score', 'CM', 'cross_val_score_ACC',
             'cross_val_score_AUC'])
result_df_test.columns.name = 'test'
result_df_train = pd.DataFrame(
    columns=['AUC_ROC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1-score', 'CM', 'cross_val_score_ACC',
             'cross_val_score_AUC'])
result_df_train.columns.name = 'train'


def add_result_col(name, model, xtr, ytr, xte, yte, x=x, y=y, cv=5):
    yp = model.predict(xte)
    yp_r = model.predict(xtr)
    y_test_pred = model.predict_proba(xte)[:, 1]
    y_train_pred = model.predict_proba(xtr)[:, 1]
    precision_e, recall_e, f1_e, _ = precision_recall_fscore_support(yte, yp, average='binary')
    result_df_test.loc[name] = [roc_auc_score(yte, y_test_pred), accuracy_score(yte, yp), recall_e,
                                confusion_matrix(yte, yp)[0, 0] / (
                                        confusion_matrix(yte, yp)[0, 1] + confusion_matrix(yte, yp)[0, 0]), f1_e,
                                confusion_matrix(yte, yp), cross_val_score(model, x, y, cv=cv, n_jobs=-1).mean(),
                                cross_val_score(model, x, y, cv=cv, n_jobs=-1, scoring='roc_auc').mean()]
    precision_r, recall_r, f1_r, _ = precision_recall_fscore_support(ytr, yp_r, average='binary')
    result_df_train.loc[name] = [roc_auc_score(ytr, y_train_pred), accuracy_score(ytr, yp_r), recall_r,
                                 confusion_matrix(ytr, yp_r)[0, 0] / (
                                         confusion_matrix(ytr, yp_r)[0, 1] + confusion_matrix(ytr, yp_r)[0, 0]),
                                 f1_r,
                                 confusion_matrix(ytr, yp_r)
        , cross_val_score(model, x, y, cv=cv, n_jobs=-1).mean(),
                                 cross_val_score(model, x, y, cv=cv, n_jobs=-1, scoring='roc_auc').mean()]
    return result_df_test, result_df_train


best_models = [LR_best_estimator, KNN_best_estimator, DT_best_estimator, RFC_best_estimator,
               BAG_best_estimator, SVC_best_estimator, XGB_best_estimator]
#SGD_best_estimator,
for i in best_models:
    result_df_test, result_df_train = add_result_col("{}".format(i), i, xtr, ytr, xte, yte, x=x, y=y, cv=5)


result_df_test.to_csv('test_roc.csv')
result_df_train.to_csv('train_roc.csv')

# XGB_best_estimator
XGB_best_estimator= XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1.0, eval_metric='auc',
              gamma=0, gpu_id=-1, importance_type='gain',
              interaction_constraints='', learning_rate=0.3, max_delta_step=0,
              max_depth=8, min_child_weight=1,
              monotone_constraints='()', n_estimators=80, n_jobs=16,
              num_parallel_tree=1, random_state=420, reg_alpha=0.3,
              reg_lambda=0.7, scale_pos_weight=1, subsample=1.0,
              tree_method='exact', use_label_encoder=False,
              validate_parameters=1, verbosity=None).fit(xtr, ytr)

reg = XGB_best_estimator
import shap
f = plt.figure()
shap_values = shap.TreeExplainer(reg).shap_values(xtr)
shap.summary_plot(shap_values, X_train, plot_type="bar",max_display=10)

import matplotlib.pyplot as plt
g = plt.figure()
shap.summary_plot(shap_values, xtr,max_display=10)

# reg = XGB_best_estimator
f = plt.figure()
explainer = shap.Explainer(reg, xtr)
shap_values = explainer(xtr[:])
shap.plots.scatter(shap_values[:,'Length of stay'],xmax=100,show=False,color="#CC2B71")
color_bar="#008AFA"
plt.axhline(y=0, xmin=0, xmax=100,
            color = color_bar
            )
plt.axvline(x=16, ymin=-2, ymax=5,
            color = color_bar
            )
import matplotlib.pyplot as pl

h1 = pl.gcf()
# h1.savefig("dependence_plot_AGE.tif", bbox_inches='tight', dpi=600)
h1.savefig("dependence_plot_days.pdf", bbox_inches='tight', dpi=600)
# shap.dependence_plot('materia_kinds', shap_values, xtr, show=False)
shap.plots.scatter(shap_values[:,'Instrument type'],show=False,color="#02A0A2")
plt.axhline(y=0, xmin=0, xmax=100,
            color = color_bar
            )

j1 = pl.gcf()
# j1.savefig("dependence_plot_GFR.tif", bbox_inches='tight', dpi=600)
j1.savefig("dependence_plot_materia_type.pdf", bbox_inches='tight', dpi=600)
# shap.dependence_plot('emergency', shap_values, xtr, show=False)
shap.plots.scatter(shap_values[:,'Emergency admissions'],show=False,color="#E81B22")
plt.axhline(y=0, xmin=0, xmax=100,
            color = color_bar
            )
import matplotlib.pyplot as pl

p1 = pl.gcf()
# p1.savefig("dependence_plot_ALP.tif", bbox_inches='tight', dpi=600)
p1.savefig("dependence_plot_emergency.pdf", bbox_inches='tight', dpi=600)



f = plt.figure()
explainer = shap.TreeExplainer(reg, xte)
shap_values = explainer(xte)
# shap_values=shap_values[0]

shap.plots.heatmap(shap_values,max_display=8)
# f.savefig("summary_plot2.tif", bbox_inches='tight', dpi=600)
f.savefig("summary_plot2.pdf", bbox_inches='tight', dpi=600)




import matplotlib.pyplot as pl

reg = XGB_best_estimator
explainer = shap.TreeExplainer(reg)
shap_values = explainer.shap_values(X_test)
shap.initjs()
i = 36
shap.force_plot(explainer.expected_value, shap_values[i, :], X_test.iloc[i, :], show=False, matplotlib=True)
k = pl.gcf()
# plt.savefig("force_plot_1.tif", bbox_inches='tight', dpi=600)
plt.savefig("force_plot_1.pdf", bbox_inches='tight', dpi=600)
plt.close()




yte.iloc[36]
pd.DataFrame(X_test.iloc[36:37, ])
reg.predict(pd.DataFrame(X_test.iloc[36:37, ]))



import matplotlib.pyplot as pl

i = 43
shap.force_plot(explainer.expected_value, shap_values[i,:], X_test.iloc[i,:],show=False,matplotlib=True)
k = pl.gcf()
plt.savefig("force_plot_2.pdf", bbox_inches='tight', dpi=600)
# plt.savefig("force_plot_2.tif", bbox_inches='tight', dpi=600)
plt.close()

yte.iloc[i]
pd.DataFrame(X_test.iloc[i:i+1, ])
reg.predict(pd.DataFrame(X_test.iloc[i:i+1, ]))


reg = XGB_best_estimator
f = plt.figure()
explainer = shap.TreeExplainer(reg, xtr)
shap_values = explainer.shap_values(xtr)
for name in xtr.columns:
    shap.dependence_plot(name, shap_values, xtr)



def plot_auc_curve_model(y, prob):
    fpr, tpr, threshold = roc_curve(y, prob)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    plt.plot(fpr, tpr,
             lw=lw, label='{0}-ROC curve (area = {1:0.3f})'.format(list_model[i], roc_auc))  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.legend(fontsize=fontsize_legend, loc=4)

fontsize_legend = 21.5
fontsize_label = 25
list_model = {XGB_best_estimator: "XGBoost",SVC_best_estimator: "SVM",LR_best_estimator: "LogisticsRegression", RFC_best_estimator: "RandomForestClassification",DT_best_estimator: "DecisionTreeClassification", KNN_best_estimator: "KNeighborsClassification" }
plt.figure(figsize=(15, 15), dpi=600)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.00])
plt.xlabel('1-Specificity', size=fontsize_label)
plt.ylabel('Sensitivity', size=fontsize_label)
plt.tick_params(labelsize=fontsize_label)
# plt.title('Receiver operating characteristic',size=fontsize_label)
sns.set_theme(style="white")
# plt.legend(fontsize= 50)

lw = 4
for i in list_model:
    #     print(i)
    plot_auc_curve_model(yte, i.predict_proba(xte)[:, 1])
# plt.savefig('roc_model.tif')
# plt.savefig('roc_model.pdf')
plt.show()




import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
import scipy.stats as stats
import pandas as pd
import seaborn as sns

def calculate_brier_score(y_true, y_pred_prob):
    return brier_score_loss(y_true, y_pred_prob)

def bootstrap_ci(y_true, y_pred_prob, metric_func=brier_score_loss, 
                 n_bootstraps=1000, ci=0.95):

    n_samples = len(y_true)
    scores = []
    for i in range(n_bootstraps):
        # Sample with replacement
        indices = np.random.randint(0, n_samples, n_samples)
        sample_true = np.array(y_true)[indices]
        sample_pred = np.array(y_pred_prob)[indices]
        
        score = metric_func(sample_true, sample_pred)
        scores.append(score)
    alpha = (1 - ci) / 2
    lower_bound = np.percentile(scores, 100 * alpha)
    upper_bound = np.percentile(scores, 100 * (1 - alpha))
    mean_score = np.mean(scores)
    
    return lower_bound, upper_bound, mean_score


def generate_performance_table(models_data, metric_funcs=None):
    if metric_funcs is None:
        from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
        
        # Define custom wrapper for accuracy and f1 that use thresholded predictions
        def accuracy_wrapper(y_true, y_pred_prob, threshold=0.5):
            y_pred = (y_pred_prob >= threshold).astype(int)
            return accuracy_score(y_true, y_pred)
        
        def f1_wrapper(y_true, y_pred_prob, threshold=0.5):
            y_pred = (y_pred_prob >= threshold).astype(int)
            return f1_score(y_true, y_pred)
        
        metric_funcs = {
            'Brier Score': brier_score_loss,
            'AUC-ROC': roc_auc_score,
            'Accuracy': accuracy_wrapper,
            'F1 Score': f1_wrapper
        }
    

    results = []

    for model_name, (y_true, y_pred_prob) in models_data.items():
        model_results = {'Model': model_name}
        
        # Calculate each metric with confidence intervals
        for metric_name, metric_func in metric_funcs.items():
            lower, upper, mean = bootstrap_ci(y_true, y_pred_prob, metric_func)
            model_results[f'{metric_name}'] = f"{mean:.3f} [{lower:.3f}-{upper:.3f}]"
        
        results.append(model_results)
    return pd.DataFrame(results)


bs1 = calculate_brier_score(yte, y_test_pred)
lower1, upper1, mean1 = bootstrap_ci(yte, y_test_pred)
print(f"Model 1 Brier Score: {bs1:.3f} [95% CI: {lower1:.3f}-{upper1:.3f}]")

models_data = {
    "Model 1": (yte, y_test_pred)
}

performance_table = generate_performance_table(models_data)
print("\nPerformance Metrics with 95% Confidence Intervals:")
print(performance_table)
