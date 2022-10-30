# Prediction-of-droughts
Prediction of drought using meterological data like weather and soil data over a particular region
import os
import subprocess
import pandas as pd
import numpy as
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import NearMiss
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as
LDA
from sklearn.decomposition import PCA, KernelPCA
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, c
lassification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import pickle
from google.colab import drive
drive.mount('/content/gdrive')
drive.mount("/content/gdrive", force_remount=True)
!unzip gdrive/My\Drive/train
df=pd.read_csv('train_timeseries.csv')
df.head()
df.shape
df.info()
df.isnull().sum()
df=df.dropna()
df.isnull().sum()
df.dtypes
df['year']=pd.DatetimeIndex(df['date']).year
df['month']=pd.DatetimeIndex(df['date']).month
df['day']=pd.DatetimeIndex(df['date']).day
df['score']=df['score'].round().astype(int)
df.dtypes
df['fips'].nunique()
df['score'].round().value_counts()
df.describe()
column_list=list(df.columns)
plt.figure(figsize=(10,40))
for x in range(1,19):
 plt.subplot(19,1,x)
 sns.boxplot(x =df.columns[x-1], data=df)
 x_name = df.columns[x-1]
 plt.title(f'Distribution of {x_name}') 
plt.tight_layout()
print('Total rows = ',len(df.index))
for i in df.select_dtypes(exclude = ['object']).columns:
 print ('Number of values beyong standard outlier limit in ', i)
 print(len(df[(df[i] >df[i].mean() + 3*df[i].std()) | (df[i] < df[i]
.mean() - 3*df[i].std())]))
df = [(df['PRECTOT'] >=df['PRECTOT'].mean() + 3*df['PRECTOT'].std()) &
 (df['PRECTOT'] >= df['PRECTOT'].mean() - 3*df['PRECTOT'].std())
]
df =[(df['PS'] <= df['PS'].mean() + 3*df['PS'].std()) &
 (df['PS'] >= df['PS'].mean() - 3*df['PS'].std())]
df = df[(df['QV2M'] <= df['QV2M'].mean() + 3*df['QV2M'].std()) &
 (df['QV2M'] >= df['QV2M'].mean() - 3*df['QV2M'].std())]
df = df[(df['T2M'] <= df['T2M'].mean() + 3*df['T2M'].std()) &
 (df['T2M'] >= df['T2M'].mean() - 3*df['T2M'].std())]
df = df[(df['T2MDEW'] <= df['T2MDEW'].mean() + 3*df['T2MDEW'].std()) &
 (df['T2MDEW'] >= df['T2MDEW'].mean() - 3*df['T2MDEW'].std())]
df = df[(df['T2MWET'] <= df['T2MWET'].mean() + 3*df['T2MWET'].std()) &
 (df['T2MWET'] >= df['T2MWET'].mean() - 3*df['T2MWET'].std())]
df = df[(df['T2M_MAX'] <= df['T2M_MAX'].mean() + 3*df['T2M_MAX'].std())
&
 (df['T2M_MAX'] >= df['T2M_MAX'].mean() - 3*df['T2M_MAX'].std())
]
df = df[(df['T2M_MIN'] <= df['T2M_MIN'].mean() + 3*df['T2M_MIN'].std())
&
 ( df['T2M_MIN'] >= df['T2M_MIN'].mean() - 3*df['T2M_MIN'].std()
)]
df = df[(['T2M_RANGE'] <=df['T2M_RANGE'].mean() + 3*df['T2M_RANGE'].std
()) &
 (df['T2M_RANGE'] >= df['T2M_RANGE'].mean() - 3*df['T2M_RANGE'].
std())]
df = df[(df['TS'] <= df['TS'].mean() + 3*df['TS'].std()) &
 (df['TS'] >= df['TS'].mean() - 3* df['TS'].std())
df = df[(df['WS10M'] <= df['WS10M'].mean() + 3*df['WS10M'].std())] &
 (df['WS10M'] >= df['WS10M'].mean() - 3*df['WS10M'].std())
df = df[(df['WS10M_MAX'] <= df['WS10M_MAX'].mean() + 3*df['WS10M_MAX'].
std()) &
 (df['WS10M_MAX'] >= df['WS10M_MAX'].mean() -
3*df['WS10M_MAX'].std())]
df = df[(df['WS10M_MIN'] <= df['WS10M_MIN'].mean() + 3*df['WS10M_MIN'].
std()) &
 (df['WS10M_MIN'] >= df['WS10M_MIN'].mean() - 3*df['WS10M_MIN'].
std())]
df = df[(df['WS10M_RANGE'] <= df['WS10M_RANGE'].mean() + 3*df['WS10M_RA
NGE'].std()) &
 (df['WS10M_RANGE'] >= df['WS10M_RANGE'].mean() - 3*df['WS10M_RA
NGE'].std())]
df = df[(df['WS50M'] <= df['WS50M'].mean() + 3*df['WS50M'].std()) &
 (df['WS50M'] >= df['WS50M'].mean() - 3*df['WS50M'].std())]
df = df[(df['WS50M_MAX'] <= df['WS50M_MAX'].mean() + 3*df['WS50M_MAX'].
std()) &
 (df['WS50M_MAX'] >= df['WS50M_MAX'].mean() - 3*df['WS50M_MAX'].
std())]
df = df[(df['WS50M_MIN'] <= df['WS50M_MIN'].mean() + 3*df['WS50M_MIN'].
std()) &
 (df['WS50M_MIN'] >= df['WS50M_MIN'].mean() - 3*df['WS50M_MIN'].
std())]
df = df[(df['WS50M_RANGE'] <= df['WS50M_RANGE'].mean() + 3*df['WS50M_RA
NGE'].std()) &
 (df['WS50M_RANGE'] >= df['WS50M_RANGE'].mean() - 3*df['WS50M_RA
NGE'].std())]
print('Total rows = ',len(df.index))
df = df[(df['PRECTOT'] <= df['PRECTOT'].mean() + 3*df['PRECTOT'].std())
&
 (df['PRECTOT'] >= df['PRECTOT'].mean() - 3*df['PRECTOT'].std())
]
df = df[(df['PS'] <= df['PS'].mean() + 3*df['PS'].std()) &
 (df['PS'] >= df['PS'].mean() - 3*df['PS'].std())]
df = df[(df['QV2M'] <= df['QV2M'].mean() + 3*df['QV2M'].std()) &
 (df['QV2M'] >= df['QV2M'].mean() - 3*df['QV2M'].std())]
df = df[(df['T2M'] <= df['T2M'].mean() + 3*df['T2M'].std()) &
 (df['T2M'] >= df['T2M'].mean() - 3*df['T2M'].std())]
df = df[(df['T2MDEW'] <= df['T2MDEW'].mean() + 3*df['T2MDEW'].std()) &
 (df['T2MDEW'] >= df['T2MDEW'].mean() - 3*df['T2MDEW'].std())]
df = df[(df['T2MWET'] <= df['T2MWET'].mean() + 3*df['T2MWET'].std()) &
 (df['T2MWET'] >= df['T2MWET'].mean() - 3*df['T2MWET'].std())]
df = df[(df['T2M_MAX'] <= df['T2M_MAX'].mean() + 3*df['T2M_MAX'].std())
&
 (df['T2M_MAX'] >= df['T2M_MAX'].mean() - 3*df['T2M_MAX'].std())
]
df = df[(df['T2M_MIN'] <= df['T2M_MIN'].mean() + 3*df['T2M_MIN'].std())
&
 (df['T2M_MIN'] >= df['T2M_MIN'].mean() - 3*df['T2M_MIN'].std())
]
df = df[(df['T2M_RANGE'] <= df['T2M_RANGE'].mean() + 3*df['T2M_RANGE'].
std()) &
 (df['T2M_RANGE'] >= df['T2M_RANGE'].mean() - 3*df['T2M_RANGE'].
std())]
df = df[(df['TS'] <= df['TS'].mean() + 3*df['TS'].std()) &
 (df['TS'] >= df['TS'].mean() - 3*df['TS'].std())]
df = df[(df['WS10M'] <= df['WS10M'].mean() + 3*df['WS10M'].std()) &
 (df['WS10M'] >= df['WS10M'].mean() - 3*df['WS10M'].std())]
df = df[(df['WS10M_MAX'] <= df['WS10M_MAX'].mean() + 3*df['WS10M_MAX'].
std()) &
 (df['WS10M_MAX'] >= df['WS10M_MAX'].mean() - 3*df['WS10M_MAX'].
std())]
df = df[(df['WS10M_MIN'] <= df['WS10M_MIN'].mean() + 3*df['WS10M_MIN'].
std()) &
(df['WS10M_MIN'] >= df['WS10M_MIN'].mean() - 3*df['WS10M_MIN'].
std())]
df = df[(df['WS10M_RANGE'] <= df['WS10M_RANGE'].mean() + 3*df['WS10M_RA
NGE'].std()) &
 (df['WS10M_RANGE'] >= df['WS10M_RANGE'].mean() - 3*df['WS10M_RA
NGE'].std())]
df = df[(df['WS50M'] <= df['WS50M'].mean() + 3*df['WS50M'].std()) &
 (df['WS50M'] >= df['WS50M'].mean() - 3*df['WS50M'].std())]
df = df[(df['WS50M_MAX'] <= df['WS50M_MAX'].mean() + 3*df['WS50M_MAX'].
std()) &
 (df['WS50M_MAX'] >= df['WS50M_MAX'].mean() - 3*df['WS50M_MAX'].
std())]
df = df[(df['WS50M_MIN'] <= df['WS50M_MIN'].mean() + 3*df['WS50M_MIN'].
std()) &
 (df['WS50M_MIN'] >= df['WS50M_MIN'].mean() - 3*df['WS50M_MIN'].
std())]
df = df[(df['WS50M_RANGE'] <= df['WS50M_RANGE'].mean() + 3*df['WS50M_RA
NGE'].std()) &
 (df['WS50M_RANGE'] >= df['WS50M_RANGE'].mean() - 3*df['WS50M_RA
NGE'].std())]
print('Total rows = ',len(df.index))
categorical_column_list = ['score','year','month','day']
df_categorical = df[['score','year','month','day']]
plt.figure(figsize=(10,40))
for col_name in categorical_column_list:
 plt.figure()
 df_categorical[col_name].value_counts().plot(kind = 'bar')
 x_name = col_name
 y_name = 'Density'
 plt.xlabel(x_name)
 plt.ylabel(y_name)
 plt.title('Distribution of {x_name}'.format(x_name=x_name))
 plt.tight_layout()
plt.scatter(df['year'], df['score'], c ="blue")
plt.show()
plt.scatter(df['QV2M'], df['T2M'], c =df['score'])
plt.xlabel('QV2M')
plt.ylabel('T2M')
plt.title('Variation of T2M vs QV2M')
plt.show()
plt.scatter(df['T2M'], df['T2MDEW'], c =df['score'])
plt.xlabel('T2M')
plt.ylabel('T2MDEW')
plt.title('Variation of T2MDEW vs T2M')
plt.show()
temp_df = df[df['score']==5]
plt.scatter(df['WS10M'], df['WS50M'], c= df['score'])
plt.xlabel('WS10M')
plt.ylabel('WS50M')
plt.title('Variation of WS50M vs WS10M')
plt.show()
independent_variables = df.drop('score', 1)
independent_variables = independent_variables.drop('fips', 1)
independent_variables = independent_variables.drop('date', 1)
independent_variables.head()
target = df['score']
target.head()
target = df['score']
target.head()
print("Train features shape", X_train.shape)
print("Train target shape", y_train.shape)
print("Test features shape", X_test.shape)
print("Test target shape", y_test.shape)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train
model = RandomForestClassifier(n_estimators=10) # n_estimators is the h
yperparameter
rfe = RFE(model, n_features_to_select=15) # n_features_to_select is cho
sen on a trial and error basis
fit = rfe.fit(X_train, y_train)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))
selected_features = independent_variables.columns[(fit.get_support())]
print(selected_features)
independent_variables = independent_variables.drop('PRECTOT', 1)
independent_variables = independent_variables.drop('T2MWET', 1)
independent_variables = independent_variables.drop('WS10M_MAX', 1)
independent_variables = independent_variables.drop('WS10M_MIN', 1)
independent_variables = independent_variables.drop('WS50M_MIN', 1)
independent_variables = independent_variables.drop('month', 1)
independent_variables.head()
X_train, X_test, y_train, y_test = train_test_split(independent_variabl
es, target, test_size=0.2, random_state=0)
print("Train features shape", X_train.shape)
print("Train target shape", y_train.shape)
print("Test features shape", X_test.shape)
print("Test target shape", y_test.shape)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
sm = SMOTE(random_state = 5)
X_train_ures_SMOTE, y_train_ures_SMOTE = sm.fit_resample(X_train, y_tra
in.ravel())
print('Before OverSampling, the shape of train_X: {}'.format(X_train.sh
ape))
print('Before OverSampling, the shape of train_y: {} \n'.format(y_train
.shape))
print('After OverSampling, the shape of train_X: {}'.format(X_train_ure
s_SMOTE.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_
ures_SMOTE.shape))
print("Counts of label '0' - Before Oversampling:{}, After OverSampling
: {}".format(sum(y_train == 0),sum(y_train_ures_SMOTE == 0)))
print("Counts of label '1' - Before Oversampling:{}, After OverSampling
: {}".format(sum(y_train == 1),sum(y_train_ures_SMOTE == 1)))
print("Counts of label '2' - Before Oversampling:{}, After OverSampling
: {}".format(sum(y_train == 2),sum(y_train_ures_SMOTE == 2)))
print("Counts of label '3' - Before Oversampling:{}, After OverSampling
: {}".format(sum(y_train == 3),sum(y_train_ures_SMOTE == 3)))
print("Counts of label '4' - Before Oversampling:{}, After OverSampling
: {}".format(sum(y_train == 4),sum(y_train_ures_SMOTE == 4)))
print("Counts of label '5' - Before Oversampling:{}, After OverSampling
: {}".format(sum(y_train == 5),sum(y_train_ures_SMOTE == 5)))
undersample = NeighbourhoodCleaningRule(n_neighbors=3, threshold_cleani
ng=0.5)
X_train_dres, y_train_dres = undersample.fit_resample(X_train, y_train)
print('Before UnderSampling, the shape of train_X: {}'.format(X_train.s
hape))
print('Before UnderSampling, the shape of train_y: {} \n'.format(y_trai
n.shape))
print('After UnderSampling, the shape of train_X: {}'.format(X_train_dr
es.shape))
print('After UnderSampling, the shape of train_y: {} \n'.format(y_train
_dres.shape))
print("Counts of label '0' - Before UnderSampling:{}, After UnderSampli
ng: {}".format(sum(y_train == 0),sum(y_train_dres == 0)))
print("Counts of label '1' - Before UnderSampling:{}, After UnderSampli
ng: {}".format(sum(y_train == 1),sum(y_train_dres == 1)))
print("Counts of label '2' - Before UnderSampling:{}, After UnderSampli
ng: {}".format(sum(y_train == 2),sum(y_train_dres == 2)))
print("Counts of label '3' - Before UnderSampling:{}, After UnderSampli
ng: {}".format(sum(y_train == 3),sum(y_train_dres == 3)))
print("Counts of label '4' - Before UnderSampling:{}, After UnderSampli
ng: {}".format(sum(y_train == 4),sum(y_train_dres == 4)))
print("Counts of label '5' - Before UnderSampling:{}, After UnderSampli
ng: {}".format(sum(y_train == 5),sum(y_train_dres == 5)))
from imblearn.under_sampling import NearMiss
undersample = NearMiss()
X_train_dres_nm, y_train_dres_nm = undersample.fit_resample(X_train, y_
train)
print('Before UnderSampling, the shape of train_X: {}'.format(X_train.s
hape))
print('Before UnderSampling, the shape of train_y: {} \n'.format(y_trai
n.shape))
print('After UnderSampling, the shape of train_X: {}'.format(X_train_dr
es_nm.shape))
print('After UnderSampling, the shape of train_y: {} \n'.format(y_train
_dres_nm.shape))
print("Counts of label '0' - Before UnderSampling:{}, After UnderSampli
ng: {}".format(sum(y_train == 0),sum(y_train_dres_nm == 0)))
print("Counts of label '1' - Before UnderSampling:{}, After UnderSampli
ng: {}".format(sum(y_train == 1),sum(y_train_dres_nm == 1)))
print("Counts of label '2' - Before UnderSampling:{}, After UnderSampli
ng: {}".format(sum(y_train == 2),sum(y_train_dres_nm == 2)))
print("Counts of label '3' - Before UnderSampling:{}, After UnderSampli
ng: {}".format(sum(y_train == 3),sum(y_train_dres_nm == 3)))
print("Counts of label '4' - Before UnderSampling:{}, After UnderSampli
ng: {}".format(sum(y_train == 4),sum(y_train_dres_nm == 4)))
print("Counts of label '5' - Before UnderSampling:{}, After UnderSampli
ng: {}".format(sum(y_train == 5),sum(y_train_dres_nm == 5)))
pca = PCA()
X_train_dres_nm_PCAreduced = pca.fit_transform(X_train_dres_nm)
X_test_NM_PCA_transformed = pca.transform(X_test)
print(pca.explained_variance_ratio_)
pca = PCA(n_components=5)
X_train_dres_nm_PCAreduced = pca.fit_transform(X_train_dres_nm)
X_test_NM_PCA_transformed = pca.transform(X_test)
pca = PCA()
X_train_ures_SMOTE_PCAreduced = pca.fit_transform(X_train_ures_SMOTE)
X_test_SMOTE_PCA_transformed = pca.transform(X_test)
print(pca.explained_variance_ratio_)
pca = PCA(n_components=5)
X_train_ures_SMOTE_PCAreduced = pca.fit_transform(X_train_ures_SMOTE)
X_test_SMOTE_PCA_transformed = pca.transform(X_test)
print(pca.explained_variance_ratio_)
# poly_kpca = KernelPCA(kernel='poly')
# X_train_dres_nm_polykPCAreduced = poly_kpca.fit_transform(X_train_dre
s_nm)
# X_test_NM_polykPCA_transformed = poly_kpca.transform(X_test)
# print(poly_kpca.explained_variance_ratio_)
# poly_kpca = KernelPCA(kernel='poly')
# X_train_ures_SMOTE_polykPCAreduced = poly_kpca.fit_transform(X_train_
ures_SMOTE)
# X_test_SMOTE_polykPCA_transformed = poly_kpca.transform(X_test)
# print(poly_kpca.explained_variance_ratio)
# poly_kpca = KernelPCA(kernel='poly')
# X_train_polykPCAreduced = poly_kpca.fit_transform(X_train)
# X_test_polykPCA_transformed = poly_kpca.transform(X_test)
# print(poly_kpca.explained_variance_ratio_)
lda=LDA(n_components=5)
X_train_dres_nm_LDAreduced=lda.fit_transform(X_train_dres_nm,y_train_dr
es_nm)
X_test_NM_LDA_transformed=lda.transform(X_test)
print("Train features shape", X_train.shape)
print("LDA Dimensionality reduced features shape on Near Miss downsampl
ed data", X_train_dres_nm_LDAreduced.shape)
print("LDA Dimensionality reduced features shape on test data", X_test_
NM_LDA_transformed.shape)
lda=LDA(n_components=5)
X_train_ures_SMOTE_LDAreduced=lda.fit_transform(X_train_ures_SMOTE,y_tr
ain_ures_SMOTE)
X_test_SMOTE_LDA_transformed=lda.transform(X_test)
print("Train features shape", X_train.shape)
print("LDA Dimensionality reduced features shape on SMOTE Upsampled dat
a", X_train_ures_SMOTE_LDAreduced.shape)
print("LDA Dimensionality reduced features shape on test data", X_test_
NM_LDA_transformed.shape)
DT_classifier_NM = tree.DecisionTreeClassifier(criterion='gini')
DT_classifier_NM.fit(X_train_dres_nm,y_train_dres_nm)
y_pred_NM = DT_classifier_NM.predict(X_test)
pickle.dump(DT_classifier_NM, open('DT_classifier_NM.pkl', 'wb'))
print('Performance of Decision Tree Algorithm with Near Miss Downsampli
ng:\n')
print(confusion_matrix(y_test, y_pred_NM))
print(classification_report(y_test, y_pred_NM))
print('Accuracy:',accuracy_score(y_test, y_pred_NM))
print('Precision:',precision_score(y_test, y_pred_NM, average='weighted
'))
print('Recall:',recall_score(y_test, y_pred_NM, average='weighted'))
print('F1 Score:',f1_score(y_test, y_pred_NM, average='weighted'))
print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_pred_NM))
fpr = dict()
tpr = dict()
thresh = dict()
for i in range(6): 
 fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_pred_NM, pos_label=
i)
plt.plot(fpr[0], tpr[0], linestyle='--
',color='orangered', label='Class 0 vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--
',color='green', label='Class 1 vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--
',color='blue', label='Class 2 vs Rest')
plt.plot(fpr[3], tpr[3], linestyle='--
',color='yellow', label='Class 3 vs Rest')
plt.plot(fpr[4], tpr[4], linestyle='--
',color='purple', label='Class 4 vs Rest')
plt.plot(fpr[5], tpr[5], linestyle='--
',color='magenta', label='Class 5 vs Rest')
plt.title('Multiclass ROC curve for Decision Tree with Near Miss Downsa
mpling')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC curve for Decision Tree with Near Miss Down
sampling',dpi=300)
params = {
 'max_depth': [3, 5, 10, 20],
 'min_samples_leaf': [10, 20, 50, 100],
 'max_features':['log2','sqrt',None]
}
grid_search_DT_NM = GridSearchCV(estimator=DT_classifier_NM,
 param_grid=params,
cv=4, n_jobs=-
1, verbose=1, scoring = "accuracy")
%%time
grid_search_DT_NM.fit(X_train_dres_nm,y_train_dres_nm)
score_df = pd.DataFrame(grid_search_DT_NM.cv_results_)
score_df.nlargest(5,"mean_test_score")
DT_classifier_SMOTE = tree.DecisionTreeClassifier(criterion='gini', max
_depth=70)
DT_classifier_SMOTE.fit(X_train_ures_SMOTE,y_train_ures_SMOTE)
y_pred_SMOTE = DT_classifier_SMOTE.predict(X_test)
pickle.dump(DT_classifier_SMOTE, open('DT_classifier_SMOTE.pkl', 'wb'))
print('Performance of Decision Tree Algorithm with SMOTE Upsampling:\n'
)
print(confusion_matrix(y_test, y_pred_SMOTE))
print(classification_report(y_test, y_pred_SMOTE))
print('Accuracy:',accuracy_score(y_test, y_pred_SMOTE))
print('Precision:',precision_score(y_test, y_pred_SMOTE, average='weigh
ted'))
print('Recall:',recall_score(y_test, y_pred_SMOTE, average='weighted'))
print('F1 Score:',f1_score(y_test, y_pred_SMOTE, average='weighted'))
print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_pred_SMOTE))
fpr = dict()
tpr = dict()
thresh = dict()
for i in range(6): 
 fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_pred_SMOTE, pos_lab
el=i)
plt.plot(fpr[0], tpr[0], linestyle='--
',color='orangered', label='Class 0 vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--
',color='green', label='Class 1 vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--
',color='blue', label='Class 2 vs Rest')
plt.plot(fpr[3], tpr[3], linestyle='--
',color='yellow', label='Class 3 vs Rest')
plt.plot(fpr[4], tpr[4], linestyle='--
',color='purple', label='Class 4 vs Rest')
plt.plot(fpr[5], tpr[5], linestyle='--
',color='magenta', label='Class 5 vs Rest')
plt.title('Multiclass ROC curve for Decision Tree with SMOTE Upsampling
')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC curve for Decision Tree with SMOTE Upsampli
ng',dpi=300)
DT_classifier_NM_PCA = tree.DecisionTreeClassifier(criterion='gini')
DT_classifier_NM_PCA.fit(X_train_dres_nm_PCAreduced,y_train_dres_nm)
y_pred_NM_PCA = DT_classifier_NM_PCA.predict(X_test_NM_PCA_transformed)
pickle.dump(DT_classifier_NM_PCA, open('DT_classifier_NM_PCA.pkl', 'wb'
))
print('Performance of Decision Tree Algorithm with Near Miss Downsampli
ng and PCA:\n')
print(confusion_matrix(y_test, y_pred_NM_PCA))
print(confusion_matrix(y_test, y_pred_NM_PCA))
print(classification_report(y_test, y_pred_NM_PCA))
print('Accuracy:',accuracy_score(y_test, y_pred_NM_PCA))
print('Precision:',precision_score(y_test, y_pred_NM_PCA, average='weig
hted'))
print('Recall:',recall_score(y_test, y_pred_NM_PCA, average='weighted')
)
print('F1 Score:',f1_score(y_test, y_pred_NM_PCA, average='weighted'))
print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_pred_NM_PCA))
fpr = dict()
tpr = dict()
thresh = dict()
for i in range(6): 
 fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_pred_NM_PCA, pos_la
bel=i)
plt.plot(fpr[0], tpr[0], linestyle='--
',color='orangered', label='Class 0 vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--
',color='green', label='Class 1 vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--
',color='blue', label='Class 2 vs Rest')
plt.plot(fpr[3], tpr[3], linestyle='--
',color='yellow', label='Class 3 vs Rest')
plt.plot(fpr[4], tpr[4], linestyle='--
',color='purple', label='Class 4 vs Rest')
plt.plot(fpr[5], tpr[5], linestyle='--
',color='magenta', label='Class 5 vs Rest')
plt.title('Multiclass ROC curve for Decision Tree with Near Miss Downsa
mpling and PCA')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC curve for Decision Tree with Near Miss Down
sampling and PCA',dpi=300)
DT_classifier_SMOTE_PCA = tree.DecisionTreeClassifier(criterion='gini')
DT_classifier_SMOTE_PCA.fit(X_train_ures_SMOTE_PCAreduced,y_train_ures_
SMOTE)
y_pred_SMOTE_PCA = DT_classifier_S
pickle.dump(DT_classifier_SMOTE_PCA, open('DT_classifier_SMOTE_PCA.pkl'
, 'wb'))
print('Performance of Decision Tree Algorithm with SMOTE Upsampling and
PCA:\n')
print(confusion_matrix(y_test, y_pred_SMOTE_PCA))
print(confusion_matrix(y_test, y_pred_SMOTE_PCA))
print(classification_report(y_test, y_pred_SMOTE_PCA))
print('Accuracy:',accuracy_score(y_test, y_pred_SMOTE_PCA))
print('Precision:',precision_score(y_test, y_pred_SMOTE_PCA, average='w
eighted'))
print('Recall:',recall_score(y_test, y_pred_SMOTE_PCA, average='weighte
d'))
print('F1 Score:',f1_score(y_test, y_pred_SMOTE_PCA, average='weighted'
))
print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_pred_SMOTE_PCA))
fpr = dict()
tpr = dict()
thresh = dict()
for i in range(6): 
 fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_pred_SMOTE_PCA, pos
_label=i)
plt.plot(fpr[0], tpr[0], linestyle='--
',color='orangered', label='Class 0 vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--
',color='green', label='Class 1 vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--
',color='blue', label='Class 2 vs Rest')
plt.plot(fpr[3], tpr[3], linestyle='--
',color='yellow', label='Class 3 vs Rest')
plt.plot(fpr[4], tpr[4], linestyle='--
',color='purple', label='Class 4 vs Rest')
plt.plot(fpr[5], tpr[5], linestyle='--
',color='magenta', label='Class 5 vs Rest')
plt.title('Multiclass ROC curve for Decision Tree with SMOTE Upsampling
and PCA')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC curve for Decision Tree with SMOTE Upsampli
ng and PCA',dpi=300)
DT_classifier_NM_LDA = tree.DecisionTreeClassifier(criterion='gini')
DT_classifier_NM_LDA.fit(X_train_dres_nm_LDAreduced,y_train_dres_nm)
y_pred_NM_LDA = DT_classifier_NM_LDA.predict(X_test_NM_LDA_transformed)
print('Performance of Decision Tree Algorithm with Near Miss Downsampli
ng and LDA:\n')
print(confusion_matrix(y_test, y_pred_NM_LDA))
print(confusion_matrix(y_test, y_pred_NM_LDA))
print(classification_report(y_test, y_pred_NM_LDA))
print('Accuracy:',accuracy_score(y_test, y_pred_NM_LDA))
print('Precision:',precision_score(y_test, y_pred_NM_LDA, average='weig
hted'))
print('Recall:',recall_score(y_test, y_pred_NM_LDA, average='weighted')
)
print('F1 Score:',f1_score(y_test, y_pred_NM_LDA, average='weighted'))
print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_pred_NM_LDA))
fpr = dict()
tpr = dict()
thresh = dict()
for i in range(6): 
 fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_pred_NM_LDA, pos_la
bel=i)
plt.plot(fpr[0], tpr[0], linestyle='--
',color='orangered', label='Class 0 vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--
',color='green', label='Class 1 vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--
',color='blue', label='Class 2 vs Rest')
plt.plot(fpr[3], tpr[3], linestyle='--
',color='yellow', label='Class 3 vs Rest')
plt.plot(fpr[4], tpr[4], linestyle='--
',color='purple', label='Class 4 vs Rest')
plt.plot(fpr[5], tpr[5], linestyle='--
',color='magenta', label='Class 5 vs Rest')
plt.title('Multiclass ROC curve for Decision Tree with Near Miss Downsa
mpling and LDA')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC curve for Decision Tree with Near Miss Down
sampling and LDA',dpi=300)
DT_classifier_SMOTE_LDA = tree.DecisionTreeClassifier(criterion='gini')
DT_classifier_SMOTE_LDA.fit(X_train_ures_SMOTE_LDAreduced,y_train_ures_
SMOTE)
y_pred_SMOTE_LDA = DT_classifier_SMOTE_LDA.predict(X_test_SMOTE_LDA_tra
nsformed)
pickle.dump(DT_classifier_SMOTE_LDA, open('DT_classifier_SMOTE_LDA.pkl'
, 'wb'))
print('Performance of Decision Tree Algorithm with SMOTE Upsampling and
LDA:\n')
print(confusion_matrix(y_test, y_pred_SMOTE_LDA))
print(confusion_matrix(y_test, y_pred_SMOTE_LDA))
print(classification_report(y_test, y_pred_SMOTE_LDA))
print('Accuracy:',accuracy_score(y_test, y_pred_SMOTE_LDA))
print('Precision:',precision_score(y_test, y_pred_SMOTE_LDA, average='w
eighted'))
print('Recall:',recall_score(y_test, y_pred_SMOTE_LDA, average='weighte
d'))
print('F1 Score:',f1_score(y_test, y_pred_SMOTE_LDA, average='weighted'
))
print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_pred_SMOTE_LDA))
fpr = dict()
tpr = dict()
thresh = dict()
for i in range(6): 
 fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_pred_SMOTE_LDA, pos
_label=i)
plt.plot(fpr[0], tpr[0], linestyle='--
',color='orangered', label='Class 0 vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--
',color='green', label='Class 1 vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--
',color='blue', label='Class 2 vs Rest')
plt.plot(fpr[3], tpr[3], linestyle='--
',color='yellow', label='Class 3 vs Rest')
plt.plot(fpr[4], tpr[4], linestyle='--
',color='purple', label='Class 4 vs Rest')
plt.plot(fpr[5], tpr[5], linestyle='--
',color='magenta', label='Class 5 vs Rest')
plt.title('Multiclass ROC curve for Decision Tree with SMOTE Upsampling
and LDA')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC curve for Decision Tree with SMOTE Upsampli
ng and LDA',dpi=300)
DT_classifier.get_depth()
params = {
 'max_depth': [40, 50, 60, 70, 80],
# 'max_samples_leaf': [, 20, 50, 100],
 'max_features':['log2','sqrt',None]
}
grid_search = GridSearchCV(estimator=DT_classifier,
 param_grid=params,
cv=4, n_jobs=-
1, verbose=1, scoring = "accuracy")
%%time
grid_search.fit(X_train,y_train)
score_df = pd.DataFrame(grid_search.cv_results_)
score_df.nlargest(5,"mean_test_score")
DT_classifier = tree.DecisionTreeClassifier(criterion='gini', max_depth
=70)
DT_classifier.fit(X_train,y_train)
y_pred_DT = DT_classifier.predict(X_test)
print('Performance of Decision Tree Algorithm without resampling - Afte
r Hyperparameter Tuning:\n')
print(confusion_matrix(y_test, y_pred_DT))
print(classification_report(y_test, y_pred_DT))
print('Accuracy:',accuracy_score(y_test, y_pred_DT))
print('Precision:',precision_score(y_test, y_pred_DT, average='weighted
'))
print('Recall:',recall_score(y_test, y_pred_DT, average='weighted'))
print('F1 Score:',f1_score(y_test, y_pred_DT, average='weighted'))
print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_pred_DT))
fpr = dict()
tpr = dict()
thresh = dict()
for i in range(6): 
fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_pred_DT, pos_label=
i)
plt.plot(fpr[0], tpr[0], linestyle='--
',color='orangered', label='Class 0 vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--
',color='green', label='Class 1 vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--
',color='blue', label='Class 2 vs Rest')
plt.plot(fpr[3], tpr[3], linestyle='--
',color='yellow', label='Class 3 vs Rest')
plt.plot(fpr[4], tpr[4], linestyle='--
',color='purple', label='Class 4 vs Rest')
plt.plot(fpr[5], tpr[5], linestyle='--
',color='magenta', label='Class 5 vs Rest')
plt.title('Multiclass ROC curve for Decision Tree without resampling -
After Hyperparameter Tuning')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC curve for Decision Tree without resampling
- After Hyperparameter Tuning',dpi=300)
knn_classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='minko
wski')
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
print('Performance of KNN Algorithm without resampling:\n')
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))
print('Accuracy:',accuracy_score(y_test, y_pred_knn))
print('Precision:',precision_score(y_test, y_pred_knn, average='weighte
d'))
print('Recall:',recall_score(y_test, y_pred_knn, average='weighted'))
print('F1 Score:',f1_score(y_test, y_pred_knn, average='weighted'))
print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_pred_knn))
fpr = dict()
tpr = dict()
thresh = dict()
for i in range(6): 
 fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_pred_knn, pos_label
=i)
plt.plot(fpr[0], tpr[0], linestyle='--
',color='orangered', label='Class 0 vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--
',color='green', label='Class 1 vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--
',color='blue', label='Class 2 vs Rest')
plt.plot(fpr[3], tpr[3], linestyle='--
',color='yellow', label='Class 3 vs Rest')
plt.plot(fpr[4], tpr[4], linestyle='--
',color='purple', label='Class 4 vs Rest')
plt.plot(fpr[5], tpr[5], linestyle='--
',color='magenta', label='Class 5 vs Rest')
plt.title('Multiclass ROC curve for KNN without resampling')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC curve for KNN without resampling',dpi=300)
k_range = list(range(1, 10))
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn_classifier, param_grid, cv=3, scoring='accuracy
', return_train_score=False,verbose=1)
grid_search=grid.fit(X_train, y_train)
score_df = pd.DataFrame(grid_search.cv_results_)
score_df.nlargest(5,"mean_test_score")
knn_classifier = KNeighborsClassifier(n_neighbors=1, p=2, metric='minko
wski')
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
print('Performance of KNN Algorithm without resampling - After Hyperpar
ameter Tuning:\n')
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))
print('Accuracy:',accuracy_score(y_test, y_pred_knn))
print('Precision:',precision_score(y_test, y_pred_knn, average='weighte
d'))
print('Recall:',recall_score(y_test, y_pred_knn, average='weighted'))
print('F1 Score:',f1_score(y_test, y_pred_knn, average='weighted'))
print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_pred_knn))
fpr = dict()
tpr = dict()
thresh = dict()
for i in range(6): 
 fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_pred_knn, pos_label
=i)
plt.plot(fpr[0], tpr[0], linestyle='--
',color='orangered', label='Class 0 vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--
',color='green', label='Class 1 vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--
',color='blue', label='Class 2 vs Rest')
plt.plot(fpr[3], tpr[3], linestyle='--
',color='yellow', label='Class 3 vs Rest')
plt.plot(fpr[4], tpr[4], linestyle='--
',color='purple', label='Class 4 vs Rest')
plt.plot(fpr[5], tpr[5], linestyle='--
',color='magenta', label='Class 5 vs Rest')
plt.title('Multiclass ROC curve for KNN without resampling - After Hype
rparameter Tuning')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC curve for KNN without resampling - After Hy
perparameter Tuning',dpi=300)
knn_classifier_SMOTE = KNeighborsClassifier(n_neighbors=1, p=2, metric=
'minkowski')
knn_classifier_SMOTE.fit(X_train_ures_SMOTE, y_train_ures_SMOTE)
y_pred_knn_SMOTE = knn_classifier_SMOTE.predict(X_test)
pickle.dump(knn_classifier_SMOTE, open('knn_classifier_SMOTE.pkl', 'wb'
))
print('Performance of KNN Algorithm with SMOTE Upsampling:\n')
print(confusion_matrix(y_test, y_pred_knn_SMOTE))
print(classification_report(y_test, y_pred_knn_SMOTE))
print('Accuracy:',accuracy_score(y_test, y_pred_knn_SMOTE))
print('Precision:',precision_score(y_test, y_pred_knn_SMOTE, average='w
eighted'))
print('Recall:',recall_score(y_test, y_pred_knn_SMOTE, average='weighte
d'))
print('F1 Score:',f1_score(y_test, y_pred_knn_SMOTE, average='weighted'
))
print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_pred_knn_SMOTE))
fpr = dict()
tpr = dict()
thresh = dict()
for i in range(6): 
 fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_pred_knn_SMOTE, pos
_label=i)
plt.plot(fpr[0], tpr[0], linestyle='--
',color='orangered', label='Class 0 vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--
',color='green', label='Class 1 vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--
',color='blue', label='Class 2 vs Rest')
plt.plot(fpr[3], tpr[3], linestyle='--
',color='yellow', label='Class 3 vs Rest')
plt.plot(fpr[4], tpr[4], linestyle='--
',color='purple', label='Class 4 vs Rest')
plt.plot(fpr[5], tpr[5], linestyle='--
',color='magenta', label='Class 5 vs Rest')
plt.title('Multiclass ROC curve for KNN with SMOTE Upsampling')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC curve for KNN with SMOTE Upsampling',dpi=30
0)
knn_classifier_NM = KNeighborsClassifier(n_neighbors=1, p=2, metric='mi
nkowski')
knn_classifier_NM.fit(X_train_dres_nm, y_train_dres_nm)
y_pred_knn_NM = knn_classifier_NM.predict(X_test)
pickle.dump(knn_classifier_NM, open('knn_classifier_NM.pkl', 'wb'))
print('Performance of KNN Algorithm with NM Downsampling:\n')
print(confusion_matrix(y_test, y_pred_knn_NM))
print(classification_report(y_test, y_pred_knn_NM))
print('Accuracy:',accuracy_score(y_test, y_pred_knn_NM))
print('Precision:',precision_score(y_test, y_pred_knn_NM, average='weig
hted'))
print('Recall:',recall_score(y_test, y_pred_knn_NM, average='weighted')
)
print('F1 Score:',f1_score(y_test, y_pred_knn_NM, average='weighted'))
print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_pred_knn_NM))
fpr = dict()
tpr = dict()
thresh = dict()
for i in range(6): 
 fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_pred_knn_NM, pos_la
bel=i)
plt.plot(fpr[0], tpr[0], linestyle='--
',color='orangered', label='Class 0 vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--
',color='green', label='Class 1 vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--
',color='blue', label='Class 2 vs Rest')
plt.plot(fpr[3], tpr[3], linestyle='--
',color='yellow', label='Class 3 vs Rest')
plt.plot(fpr[4], tpr[4], linestyle='--
',color='purple', label='Class 4 vs Rest')
plt.plot(fpr[5], tpr[5], linestyle='--
',color='magenta', label='Class 5 vs Rest')
plt.title('Multiclass ROC curve for KNN with Near Miss Downsampling')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC curve for KNN with Near Miss Downsampling',
dpi=300)
RF_classifier = RandomForestClassifier(n_estimators = 20, max_depth=70,
random_state=0)
RF_classifier.fit(X_train, y_train)
y_pred_RF = RF_classifier.predict(X_test)
print('Performance of RF Algorithm without resampling:\n')
print(confusion_matrix(y_test, y_pred_RF))
print(classification_report(y_test, y_pred_RF))
print('Accuracy:',accuracy_score(y_test, y_pred_RF))
print('Precision:',precision_score(y_test, y_pred_RF, average='weighted
'))
print('Recall:',recall_score(y_test, y_pred_RF, average='weighted'))
print('F1 Score:',f1_score(y_test, y_pred_RF, average='weighted'))
print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_pred_RF))
fpr = dict()
tpr = dict()
thresh = dict()
for i in range(6): 
 fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_pred_RF, pos_label=
i)
plt.plot(fpr[0], tpr[0], linestyle='--
',color='orangered', label='Class 0 vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--
',color='green', label='Class 1 vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--
',color='blue', label='Class 2 vs Rest')
plt.plot(fpr[3], tpr[3], linestyle='--
',color='yellow', label='Class 3 vs Rest')
plt.plot(fpr[4], tpr[4], linestyle='--
',color='purple', label='Class 4 vs Rest')
plt.plot(fpr[5], tpr[5], linestyle='--
',color='magenta', label='Class 5 vs Rest')
plt.title('Multiclass ROC curve for Random Forest without resampling')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC curve for Random Forest without resampling'
,dpi=300)
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 50, num
= 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
 'max_features': max_features,
 'max_depth': max_depth,
 'bootstrap': bootstrap}
RF_random = RandomizedSearchCV(estimator = RF_classifier, param_distrib
utions = random_grid, n_iter = 20, cv = 3, verbose=2, random_state=0, n
_jobs = -1)
RF_random.fit(X_train, y_train)
RF_random.best_params_
RF_classifier = RandomForestClassifier(n_estimators = 50, max_depth=80,
bootstrap=False, max_features='sqrt', random_state=0)
RF_classifier.fit(X_train, y_train)
y_pred_RF = RF_classifier.predict(X_test)
pickle.dump(RF_classifier, open('RF_classifier.pkl', 'wb'))
print('Performance of RF Algorithm without resampling - After Hyperpara
mter Tuning:\n')
print(confusion_matrix(y_test, y_pred_RF))
print(classification_report(y_test, y_pred_RF))
print('Accuracy:',accuracy_score(y_test, y_pred_RF))
print('Precision:',precision_score(y_test, y_pred_RF, average='weighted
'))
print('Recall:',recall_score(y_test, y_pred_RF, average='weighted'))
print('F1 Score:',f1_score(y_test, y_pred_RF, average='weighted'))
print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_pred_RF))
fpr = dict()
tpr = dict()
thresh = dict()
for i in range(6): 
 fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_pred_RF, pos_label=
i)
plt.plot(fpr[0], tpr[0], linestyle='--
',color='orangered', label='Class 0 vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--
',color='green', label='Class 1 vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--
',color='blue', label='Class 2 vs Rest')
plt.plot(fpr[3], tpr[3], linestyle='--
',color='yellow', label='Class 3 vs Rest')
plt.plot(fpr[4], tpr[4], linestyle='--
',color='purple', label='Class 4 vs Rest')
plt.plot(fpr[5], tpr[5], linestyle='--
',color='magenta', label='Class 5 vs Rest')
plt.title('Multiclass ROC curve for Random Forest without resampling -
After Hyperparameter Tuning')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC curve for Random Forest without resampling
- After Hyperparameter Tuning',dpi=300)
# svm_classifier = SVC(kernel='poly', degree=3, C = 1.0)
# svm_classifier.fit(X_train, y_train)
# y_pred_svm = svm_classifier.predict(X_test)
# print('Performance of SVM Algorithm without resampling:\n')
# print(confusion_matrix(y_test, y_pred_svm))
# print(classification_report(y_test, y_pred_svm))
# print('Accuracy:',accuracy_score(y_test, y_pred_svm))
# print('Precision:',precision_score(y_test, y_pred_svm, average='weigh
ted'))
# print('Recall:',recall_score(y_test, y_pred_svm, average='weighted'))
# print('F1 Score:',f1_score(y_test, y_pred_svm, average='weighted'))
# print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_pred_svm))
# svm_classifier = SVC(kernel='rbf', C = 1.0)
# svm_classifier.fit(X_train, y_train)
# y_pred_svm_rbf = svm_classifier.predict(X_test)
# print('Performance of SVM Algorithm with RBF Kernel without resamplin
g:\n')
# print(confusion_matrix(y_test, y_pred_svm_rbf))
# print(classification_report(y_test, y_pred_svm_rbf))
# print('Accuracy:',accuracy_score(y_test, y_pred_svm_rbf))
# print('Precision:',precision_score(y_test, y_pred_svm_rbf, average='w
eighted'))
# print('Recall:',recall_score(y_test, y_pred_svm_rbf, average='weighte
d'))
# print('F1 Score:',f1_score(y_test, y_pred_svm_rbf, average='weighted'
))
# print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_pred_svm_rbf))
svm_classifier_nm = SVC(kernel='poly', degree=3, C = 1.0)
svm_classifier_nm.fit(X_train_dres_nm, y_train_dres_nm)
y_pred_svm_nm = svm_classifier_nm.predict(X_test)
pickle.dump(svm_classifier_nm, open('svm_classifier_nm.pkl', 'wb'))
print('Performance of SVM Algorithm with Near Miss downsampling:\n')
print(confusion_matrix(y_test, y_pred_svm_nm))
print(classification_report(y_test, y_pred_svm_nm))
print('Accuracy:',accuracy_score(y_test, y_pred_svm_nm))
print('Precision:',precision_score(y_test, y_pred_svm_nm, average='weig
hted'))
print('Recall:',recall_score(y_test, y_pred_svm_nm, average='weighted')
)
print('F1 Score:',f1_score(y_test, y_pred_svm_nm, average='weighted'))
print('Cohen Kappa Score:',cohen_kappa_score(y_test, y_pred_svm_nm))
fpr = dict()
tpr = dict()
thresh = dict()
for i in range(6): 
 fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_pred_svm_nm, pos_la
bel=i)
plt.plot(fpr[0], tpr[0], linestyle='--
',color='orangered', label='Class 0 vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--
',color='green', label='Class 1 vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--
',color='blue', label='Class 2 vs Rest')
plt.plot(fpr[3], tpr[3], linestyle='--
',color='yellow', label='Class 3 vs Rest')
plt.plot(fpr[4], tpr[4], linestyle='--
',color='purple', label='Class 4 vs Rest')
plt.plot(fpr[5], tpr[5], linestyle='--
',color='magenta', label='Class 5 vs Rest')
plt.title('Multiclass ROC curve for SVM with Near Miss Downsampling')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC curve for SVM with Near Miss Downsampling',
dpi=300)
all_labels = pd.DataFrame()
all_labels['Actual_label'] = y_test
all_labels['y_pred_DT'] = y_pred_DT
all_labels['y_pred_NM'] = y_pred_NM
all_labels['y_pred_SMOTE'] = y_pred_SMOTE
all_labels['y_pred_NM_PCA'] = y_pred_NM_PCA
all_labels['y_pred_SMOTE_PCA'] = y_pred_SMOTE_PCA
all_labels['y_pred_NM_LDA'] = y_pred_NM_LDA
all_labels['y_pred_SMOTE_LDA'] = y_pred_SMOTE_LDA
all_labels['y_pred_KNN'] = y_pred_knn
all_labels['y_pred_KNN_SMOTE'] = y_pred_knn_SMOTE
all_labels['y_pred_KNN_NM'] = y_pred_knn_NM
all_labels['y_pred_NB'] = y_pred_NB
all_labels['y_pred_RF'] = y_pred_RF
# all_labels['y_pred_svm_nm'] = y_pred_svm_nm
# all_labels[['y_pred_DT','Agglomerative_min_labels','Agglomerative_max
_labels']]=independent_variables[['Agglomerative_labels','Agglomerative
_min_labels','Agglomerative_max_labels']]
data = [
{'Algorithm':'Decision Tree without resampling' ,'Accuracy':0.763336889
83729,'Precision':0.7623049559359242,'Recall':0.76333688983729,'F1 Scor
e':0.7628094905920674,'Cohen Kappa Score':0.596681340983346},
{'Algorithm':'Decision Tree with Near Miss Downsampling','Accuracy':0.2
2480540265282864,'Precision':0.5431846016633978,'Recall':0.224805402652
82864,'F1 Score':0.2626001987113276,'Cohen Kappa Score':0.0787595709141
8129},
{'Algorithm':'Decision Tree with SMOTE Upsampling','Accuracy':0.7642280
365673271,'Precision':0.7725880165635359,'Recall':0.7642280365673271,'F
1 Score':0.7679865604188995,'Cohen Kappa Score':0.6072223355459014},
{'Algorithm':'Decision Tree with Near Miss Downsampling and PCA','Accur
acy':0.18901606084854952,'Precision':0.5208938985991317,'Recall':0.1890
1606084854952,'F1 Score':0.22407231601991243,'Cohen Kappa Score':0.0597
2201042990921},
{'Algorithm':'Decision Tree with SMOTE Upsampling and PCA','Accuracy':0
.6911580461860537,'Precision':0.721815047933934,'Recall':0.691158046186
0537,'F1 Score':0.7032986584993112,'Cohen Kappa Score':0.50450374824989
97},
{'Algorithm':'Decision Tree with Near Miss Downsampling and LDA','Accur
acy':0.20474752863389833,'Precision':0.5142486750813464,'Recall':0.2047
4752863389833,'F1 Score':0.24971268616453107,'Cohen Kappa Score':0.0577
7192884677773},
{'Algorithm':'Decsion Tree with SMOTE Upsampling and LDA','Accuracy':0.
6028092339775455,'Precision':0.6746275672595141,'Recall':0.602809233977
5455,'F1 Score':0.6283270075892382,'Cohen Kappa Score':0.39472278972161
27},
{'Algorithm':'KNN without resampling','Accuracy':0.7986513575337262,'Pr
ecision':0.7982935187700809,'Recall':0.7986513575337262,'F1 Score':0.79
84710410835046,'Cohen Kappa Score':0.6574980649397748},
{'Algorithm':'KNN with SMOTE Upsampling','Accuracy':0.7952666165522927,
'Precision':0.801758151083889,'Recall':0.7952666165522927,'F1 Score':0.
7981975830544615,'Cohen Kappa Score':0.6578269982214404},
{'Algorithm':'KNN with Near Miss Upsampling','Accuracy':0.2325084669043
058,'Precision':0.5664887557511156,'Recall':0.2325084669043058,'F1 Scor
e':0.2688785664243745,'Cohen Kappa Score':0.09355157402001324},
{'Algorithm':'Naive Bayes without resampling','Accuracy':0.585143917165
7896,'Precision':0.4499104487639562,'Recall':0.5851439171657896,'F1 Sco
re':0.4804411924156227,'Cohen Kappa Score':0.08074571293428756},
{'Algorithm':'Random Forest without resampling','Accuracy':0.8089591567
852438,'Precision':0.7969254562173812,'Recall':0.8089591567852438,'F1 S
core':0.7986904178915076,'Cohen Kappa Score':0.6549810654516983},
{'Algorithm':'SVM with Near Miss Downsampling','Accuracy':0.29953442130
02255,'Precision':0.5123237426347645,'Recall':0.2995344213002255,'F1 Sc
ore':0.36286713356946726,'Cohen Kappa Score':0.07811063221491454}]
performance_metrics = pd.DataFrame(data)
performance_metrics.sort_values(by=['Accuracy', 'Cohen Kappa Score'], a
scending=False)
