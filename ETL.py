# import necessary package
# !pip install mysql-connector
# !pip install PyMySQL
# import pymysql
import numpy as np
import pandas as pd
import mysql.connector  # pip install mysql-connector-python
import sklearn
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

USER = 'root'
PWD = '88888888'
HOST = 'localhost'
DATABASE = 'homework'


class DataFrameToMySql:
    def __init__(self, host, database, user, password, schema=None, table_name=None):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.schema = schema or database
        self.table_name = table_name

        # 创建数据库引擎
        self.engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}?charset=utf8mb4")

        # # 获取数据库连接
        # with self.engine.connect() as conn:
        #     conn.execute(f"CREATE DATABASE IF NOT EXISTS {database} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        #     conn.execute(f"USE {self.database};")

    def write_to_mysql(self, df, table_name='', if_exists='replace', index=False, method='multi'):
        """
        将DataFrame写入mysql数据库

        Args:
            df: pandas.DataFrame, 需要写入的DataFrame
            if_exists: str, 如果表格存在时的操作选项, options are {'fail', 'replace', 'append'}, defaults to 'fail'
            index: bool, 是否写入行索引, defaults to False
            method: str, 写入方式, 可选 'multi' 或 'single', defaults to 'multi'

        Returns:
            None
        """
        if table_name is None:
            raise ValueError("No table_name given")

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df should be a pandas DataFrame.")
        try:
            df.to_sql(name=table_name, con=self.engine, if_exists=if_exists, index=index, method=method)
        except Exception as e:
            raise ValueError(f"Cannot write to database. Reason: {e}")


MS = DataFrameToMySql(host=HOST, database=DATABASE, user=USER, password=PWD)


# Data ETL
def etl_data():
    # TODO to mysql
    # load data (extract)
    data = pd.read_csv('bank_churn.csv')
    # transform
    necessary_col = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
                     'IsActiveMember', 'EstimatedSalary', 'Exited']
    data = data[necessary_col]
    # Null value processing
    null_count = data.isnull().sum().sum()
    if null_count:
        data.replace(' ', np.nan, inplace=True)
    # Delete duplicate data
    data.drop_duplicates(inplace=True)
    # Specify data type
    data[['CreditScore', 'Age', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Exited']] = data[
        ['CreditScore', 'Age', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Exited']].astype(int)
    data[['Balance', 'EstimatedSalary']] = data[['Balance', 'EstimatedSalary']].astype(np.float64)
    data.to_csv('bank_churn_etl.csv', index=False)
    MS.write_to_mysql(df=data, table_name='etl_data', if_exists='replace')
    # return data


etl_data()
# Part 1: Data Exploration
df = pd.read_csv('bank_churn_etl.csv')

y = df['Exited']
X = df

desc = df[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']].describe()
print(desc)

# Part 2: Feature Preprocessing
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1207)

encoder1 = OneHotEncoder().fit(xtrain[['Geography']])
ohe_column1 = pd.DataFrame(encoder1.transform(xtrain[['Geography']]).toarray(),
                           columns=encoder1.get_feature_names_out())
ohe_column2 = pd.DataFrame(encoder1.transform(xtest[['Geography']]).toarray(), columns=encoder1.get_feature_names_out())

xtrain_data = xtrain.drop(['Geography'], axis=1).reset_index(drop=True)
xtest_data = xtest.drop(['Geography'], axis=1).reset_index(drop=True)

xtrain_data = pd.concat([xtrain_data, ohe_column1], axis=1)
xtest_data = pd.concat([xtest_data, ohe_column2], axis=1)

encoder2 = OneHotEncoder().fit(xtrain[['Gender']])
ohe_column11 = pd.DataFrame(encoder2.transform(xtrain[['Gender']]).toarray(),
                            columns=encoder2.get_feature_names_out()).iloc[:, :1]
ohe_column22 = pd.DataFrame(encoder2.transform(xtest[['Gender']]).toarray(),
                            columns=encoder2.get_feature_names_out()).iloc[:, :1]

xtrain_data = xtrain_data.drop(['Gender'], axis=1).reset_index(drop=True)
xtest_data = xtest_data.drop(['Gender'], axis=1).reset_index(drop=True)

xtrain_data = pd.concat([xtrain_data, ohe_column11], axis=1)
xtest_data = pd.concat([xtest_data, ohe_column22], axis=1)

num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
scaler = StandardScaler()
scaler.fit(xtrain_data[num_cols])
xtrain_data[num_cols] = scaler.transform(xtrain_data[num_cols])
xtest_data[num_cols] = scaler.transform(xtest_data[num_cols])

# Part 3: Model training:
# Part 3.1: Random forest:
rfc = RandomForestClassifier()
frc = rfc.fit(xtrain_data, ytrain)
result = rfc.score(xtest_data, ytest)

aa = []
for i in ['entropy', 'gini']:
    rfc = RandomForestClassifier(criterion=i, random_state=1)
    rf_cv = cross_val_score(rfc, xtrain_data, ytrain, cv=5).mean()
    aa.append(rf_cv)
print(max(aa), aa.index(max(aa)))
aa_ = rfc.n_estimators
print(aa_)

parameters = {
    'n_estimators': [96, 100, 103],
    'max_depth': [8, 10, 13],
    'max_features': [4, 5, 6],
    'min_samples_split': [3, 5, 6, 7]
}
Grid_RF = GridSearchCV(RandomForestClassifier(criterion='gini'), parameters, cv=5)
Grid_RF.fit(xtrain_data, ytrain)
print(Grid_RF.best_params_, Grid_RF.best_score_)
best_RF_model = Grid_RF.best_estimator_

# Part 3.2: Logistic Regression:
classifier_logistic = LogisticRegression()
classifier_logistic.fit(xtrain_data, ytrain)
classifier_logistic.score(xtest_data, ytest)

parameters = {
    'penalty': ('l1', 'l2'),
    'C': (0.01, 0.05, 0.1, 0.5)
}
Grid_LR = GridSearchCV(LogisticRegression(solver='liblinear'), parameters, cv=5)
Grid_LR.fit(xtrain_data, ytrain)
print(Grid_LR.best_params_, Grid_LR.best_score_)
best_LR_model = Grid_LR.best_estimator_
# LR_models = pd.DataFrame(Grid_LR.cv_results_)
# res = (LR_models.pivot(index='param_penalty', columns='param_C', values='mean_test_score'))
# _ = sns.heatmap(res, cmap='viridis')

# Part 3.3: KNN
classifier_KNN = KNeighborsClassifier()
parameters = {
    'n_neighbors': [1, 3, 5, 7, 9, 10]
}
Grid_KNN = GridSearchCV(KNeighborsClassifier(), parameters, cv=5)
Grid_KNN.fit(xtrain_data, ytrain)
print(Grid_KNN.best_params_, Grid_KNN.best_score_)
best_KNN_model = Grid_KNN.best_estimator_

# Part 4: Model Evaluation
# Part 4.1: Model Evaluation - Confusion Matrix (Precision, Recall, Accuracy)
confusion_matrices = [
    ("Random Forest", confusion_matrix(ytest, best_RF_model.predict(xtest_data))),
    ("Logistic Regression", confusion_matrix(ytest, best_LR_model.predict(xtest_data))),
    ("K nearest neighbor", confusion_matrix(ytest, best_KNN_model.predict(xtest_data)))
]


# calculate accuracy, precision and recall, [[tn, fp],[]]
def cal_evaluation(classifier, cm):
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    accuracy = (tp + tn) / (tp + fp + fn + tn + 0.0)
    precision = tp / (tp + fp + 0.0)
    recall = tp / (tp + fn + 0.0)
    print(classifier)
    print("Accuracy is: " + str(accuracy))
    print("precision is: " + str(precision))
    print("recall is: " + str(recall))
    print('*' * 50)
    df = pd.DataFrame({'Accuracy': [accuracy], 'precision': [precision], 'recall': [recall]})
    df['classifier'] = classifier
    return df


# print out confusion matrices
def draw_confusion_matrices(confusion_matricies):
    class_names = ['Not', 'Churn']
    dd = pd.DataFrame()
    for cm in confusion_matrices:
        classifier, cm = cm[0], cm[1]
        c_df = cal_evaluation(classifier, cm)
        dd = dd._append(c_df)
    MS.write_to_mysql(df=dd, table_name='model_result', if_exists='replace')
    print(dd)


print('*' * 50)
draw_confusion_matrices(confusion_matrices)
print('*' * 50)

print(classification_report(ytest, best_RF_model.predict(xtest_data)))
print(classification_report(ytest, best_KNN_model.predict(xtest_data)))
print(classification_report(ytest, best_LR_model.predict(xtest_data)))

# Part 4.2: Model Evaluation - Confusion Matrix (ROC,AUC)

y_pred_rf = best_RF_model.predict_proba(xtest_data)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(ytest, y_pred_rf)

y_pred_lr = best_LR_model.predict_proba(xtest_data)[:, 1]
fpr_lr, tpr_lr, thresh = roc_curve(ytest, y_pred_lr)

y_pred_KNN = best_KNN_model.predict_proba(xtest_data)[:, 1]
fpr_KNN, tpr_KNN, thresh = roc_curve(ytest, y_pred_KNN)
metrics.auc(fpr_rf, tpr_rf)

# Part 5: Feature Importance Discussion
X_RF = X.copy()

encoder3 = OneHotEncoder().fit(X_RF[['Geography']])
ohe_column3 = pd.DataFrame(encoder3.transform(X_RF[['Geography']]).toarray(), columns=encoder3.get_feature_names_out())

encoder4 = OneHotEncoder().fit(X_RF[['Gender']])
ohe_column4 = pd.DataFrame(encoder4.transform(X_RF[['Gender']]).toarray(),
                           columns=encoder4.get_feature_names_out()).iloc[:, :1]

X_RF = X_RF.drop(['Geography', 'Gender'], axis=1).reset_index(drop=True)
X_RF = pd.concat([X_RF, ohe_column3, ohe_column4], axis=1)

# check feature importance of random forest for feature selection
forest = RandomForestClassifier()
forest.fit(X_RF, y)

importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature importance ranking by Random Forest Model:")
for ind in range(X.shape[1]):
    print("{0} : {1}".format(X_RF.columns[indices[ind]], round(importances[indices[ind]], 4)))
