import numpy as np
import pandas as pd
df = pd.read_csv('dataset_traffic_accident.csv')
df.head()
df.shape
df.columns
df.info()
df.describe()
df.describe(include='O')
df.isnull().sum()
df.isnull()
import matplotlib.pyplot as plt
col = 'Traffic_Density'

print(f"{col} - Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}")

df[col].plot(kind='hist', bins=30, title=col, color='lightblue', edgecolor='black')
plt.axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
plt.axvline(df[col].median(), color='blue', linestyle='--', linewidth=2, label='Median')
plt.legend()
plt.show()
df['Traffic_Density'].fillna(df['Traffic_Density'].median(), inplace=True)
col = 'Speed_Limit'

print(f"{col} - Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}")

df[col].plot(kind='hist', bins=30, title=col, color='lightblue', edgecolor='black')
plt.axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
plt.axvline(df[col].median(), color='blue', linestyle='--', linewidth=2, label='Median')
plt.legend()
plt.show()

df['Speed_Limit'].fillna(df['Speed_Limit'].median(), inplace=True)
col = 'Number_of_Vehicles'

print(f"{col} - Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}")

df[col].plot(kind='hist', bins=30, title=col, color='lightblue', edgecolor='black')
plt.axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
plt.axvline(df[col].median(), color='blue', linestyle='--', linewidth=2, label='Median')
plt.legend()
plt.show()
df['Number_of_Vehicles'].fillna(df['Number_of_Vehicles'].median(), inplace=True)
col = 'Driver_Alcohol'

print(f"{col} - Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}")

df[col].plot(kind='hist', bins=30, title=col, color='lightblue', edgecolor='black')
plt.axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
plt.axvline(df[col].median(), color='blue', linestyle='--', linewidth=2, label='Median')
plt.legend()
plt.show()
df['Driver_Alcohol'].fillna(df['Driver_Alcohol'].median(), inplace=True)
col = 'Driver_Age'

print(f"{col} - Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}")

df[col].plot(kind='hist', bins=30, title=col, color='lightblue', edgecolor='black')
plt.axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
plt.axvline(df[col].median(), color='blue', linestyle='--', linewidth=2, label='Median')
plt.legend()
plt.show()
df['Driver_Age'].fillna(df['Driver_Age'].median(), inplace=True)
col = 'Driver_Experience'

print(f"{col} - Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}")

df[col].plot(kind='hist', bins=30, title=col, color='lightblue', edgecolor='black')
plt.axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
plt.axvline(df[col].median(), color='blue', linestyle='--', linewidth=2, label='Median')
plt.legend()
plt.show()
df['Driver_Experience'].fillna(df['Driver_Experience'].median(), inplace=True)
col = 'Accident'

print(f"{col} - Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}")

df[col].plot(kind='hist', bins=30, title=col, color='lightblue', edgecolor='black')
plt.axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
plt.axvline(df[col].median(), color='blue', linestyle='--', linewidth=2, label='Median')
plt.legend()
plt.show()
df['Accident'].fillna(df['Accident'].median(), inplace=True)

df.isnull().sum()
cols = ['Weather', 'Road_Type', 'Time_of_Day', 'Accident_Severity', 'Road_Condition', 'Vehicle_Type', 'Road_Light_Condition']

for i in cols:
    print(f"{i} - Mode: {df[i].mode()}")
for i in cols:
    df[i] = df[i].fillna(df[i].mode()[0])

df.isnull().sum()
df.head()
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()
df.shape
import seaborn as sns
sns.boxplot(x=df['Speed_Limit'], color='lightblue')
plt.title('Speed_Limit')
plt.show()
sns.boxplot(x=df['Number_of_Vehicles'], color='lightgreen')
plt.title('Number_of_Vehicles')
plt.show()
sns.boxplot(x=df['Driver_Age'], color='lightcoral')
plt.title('Driver_Age')
plt.show()
sns.boxplot(x=df['Driver_Experience'], color='lightyellow')
plt.title('Driver_Experience')
plt.show()

sns.boxplot(x=df['Traffic_Density'], color='skyblue')
plt.title('Traffic_Density')
plt.show()
sns.countplot(x='Accident', data=df)
plt.title('Traffic Accidents Distribution')
plt.xticks([0, 1], ['No Accident', 'Accident'])
plt.show()

daccident_rate = df['Accident'].mean() * 100
print(f'Accident Rate: {daccident_rate:.2f}%')
# Accident probability by road type
plt.figure(figsize=(10, 5))
sns.barplot(x='Road_Type', y='Accident', data=df)
plt.title('Accident Probability by Road Type')
plt.show()
# Average severity by weather condition
severity_map = {'Low': 1, 'Moderate': 2, 'High': 3}
df['Accident_Severity_Num'] = df['Accident_Severity'].map(severity_map)

plt.figure(figsize=(10, 5))
sns.barplot(x='Weather', y='Accident_Severity_Num', data=df)
plt.title('Average Accident Severity by Weather Condition')
plt.ylabel('Average Severity Level (1=Low, 3=High)')
plt.show()
# Accident frequency by time of day
plt.figure(figsize=(10, 5))
sns.barplot(x='Time_of_Day', y='Accident', data=df)
plt.title('Accident Probability by Time of Day')
plt.show()
# Accident frequency by traffic density
plt.figure(figsize=(10, 5))
sns.barplot(x='Traffic_Density', y='Accident', data=df)
plt.title('Accident Probability by Traffic Density')
plt.show()
# Accident frequency by alcoholic driving
plt.figure(figsize=(10, 5))
sns.barplot(x='Driver_Alcohol', y='Accident', data=df)
plt.title('Accident Probability by Driver Alcohol')
plt.show()
sns.lineplot(data=df, x="Speed_Limit", y="Accident_Severity_Num")
plt.title("Trend of Accident Severity with Speed Limit")
plt.show()
sns.lineplot(data=df, x="Driver_Experience", y="Accident_Severity_Num")
plt.title("Trend of Accident Severity by Driver Experience")
plt.show()
from pandas.plotting import scatter_matrix
cols = ['Traffic_Density', 'Speed_Limit', 'Number_of_Vehicles', 'Driver_Age', 'Driver_Experience', 'Accident']

plt.figure(figsize=(12,12))
scatter_matrix(df[cols], alpha=0.7, diagonal='kde', figsize=(12,12))
plt.suptitle("Pair Plot (Scatter Matrix) of Key Features", fontsize=16)
plt.show()
df = df.drop(['Accident_Severity_Num'], axis = 1)
df.columns
df = pd.get_dummies(df, columns=['Weather', 'Road_Type', 'Time_of_Day', 'Accident_Severity', 
                                  'Road_Condition', 'Vehicle_Type', 'Road_Light_Condition'], drop_first=True)
df.info()
numerical_columns_df = df.select_dtypes(include=['int64', 'float64'])
numerical_columns_df
numerical_columns_df.corr()
sns.heatmap(numerical_columns_df.corr(),annot=True)
df['Age_vs_Experience'] = df['Driver_Age'] - df['Driver_Experience']
df = df.drop(['Driver_Age', 'Driver_Experience'], axis = 1)
df['Age_vs_Experience'].isnull().sum()
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)
numeric_columns = ['Speed_Limit', 'Number_of_Vehicles', 'Age_vs_Experience']
from sklearn.preprocessing import RobustScaler
rs = RobustScaler()
scaled_data = rs.fit_transform(df[numeric_columns])
scaled_df = pd.DataFrame(scaled_data, columns=numeric_columns)
print(scaled_df.head())
scaled_df = pd.DataFrame(scaled_data, columns=numeric_columns)
print(scaled_df.head())
scaled_df.head()
df_merged = df.copy()
df_merged[numeric_columns] = scaled_df[numeric_columns].values
x = df_merged.drop(['Accident'], axis = 1)
y = df_merged['Accident']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state=100)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train,y_train)
knn_pred = knn.predict(x_test)
knn_pred
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score,f1_score, classification_report

print("Confusion Matrix:\n", confusion_matrix(y_test,knn_pred))
print("\nAccuracy Score:\n", accuracy_score(y_test,knn_pred))
print("\nPrecision Score:\n", precision_score(y_test,knn_pred))
print("\nRecall Score:\n", recall_score(y_test,knn_pred))
print("\nF1 Score:\n", f1_score(y_test,knn_pred))
print("\nClassification Report:\n", classification_report(y_test,knn_pred))
knn_train_pred = knn.predict(x_train)
acc_train = accuracy_score(y_train, knn_train_pred)
print("Training Accuracy:", acc_train)
print("Testing accuracy:", accuracy_score(y_test,knn_pred))
from sklearn.metrics import roc_curve, auc

knn_prob = knn.predict_proba(x_test)[:, 1]

fpr1, tpr1, _ = roc_curve(y_test, knn_prob)
roc_auc1 = auc(fpr1, tpr1)
roc_auc1

plt.figure(figsize=(6,5))
plt.plot(fpr1, tpr1, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc1:.3f})")
plt.plot([0,1], [0,1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - KNN")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
# plotting the ROC curve for Hypertuned KNN
from sklearn.model_selection import GridSearchCV
knn_param = {'n_neighbors': range(1, 30),
             'weights': ['uniform', 'distance'],
             'algorithm': ['auto', 'ball_tree', 'kd_tree'],
             'leaf_size': range(10, 50, 5)}

grid_search_knn = GridSearchCV(knn, knn_param, cv=5, scoring='f1', n_jobs=-1, verbose=1)
# n_jobs=-1 uses all CPU cores for parallel processing.
# verbose=1 gives progress updates.
grid_search_knn.fit(x_train, y_train)
grid_search_knn.best_params_
tuned_knn = KNeighborsClassifier(algorithm = 'auto', leaf_size = 10, n_neighbors = 1, weights = 'uniform')
tuned_knn.fit(x_train, y_train)
tuned_knn_pred = tuned_knn.predict(x_test)
tuned_knn_pred
tuned_knn_train_pred = tuned_knn.predict(x_train)
acc_train = accuracy_score(y_train, tuned_knn_train_pred)
print("Training Accuracy:", acc_train)
print("Testing accuracy:", accuracy_score(y_test,tuned_knn_pred))
f1_score(y_test,tuned_knn_pred)
knn_prob = tuned_knn.predict_proba(x_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, knn_prob)
roc_auc = auc(fpr, tpr)
roc_auc

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0,1], [0,1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Hypertuned KNN")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
# plotting the ROC curve for Hypertuned KNN

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train,y_train)
rfc_pred = rfc.predict(x_test)
rfc_pred
print("Confusion Matrix:\n", confusion_matrix(y_test,rfc_pred))
print("\nAccuracy Score:\n", accuracy_score(y_test,rfc_pred))
print("\nPrecision Score:\n", precision_score(y_test,rfc_pred))
print("\nRecall Score:\n", recall_score(y_test,rfc_pred))
print("\nF1 Score:\n", f1_score(y_test,rfc_pred))
print("\nClassification Report:\n", classification_report(y_test,rfc_pred))
rfc_train_pred = rfc.predict(x_train)
acc_train = accuracy_score(y_train, rfc_train_pred)
print("Training Accuracy:", acc_train)
print("Testing accuracy:", accuracy_score(y_test, rfc_pred))
rfc_prob = rfc.predict_proba(x_test)[:, 1]

fpr2, tpr2, _ = roc_curve(y_test, rfc_prob)
roc_auc2 = auc(fpr2, tpr2)
roc_auc2

plt.figure(figsize=(6,5))
plt.plot(fpr2, tpr2, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc2:.3f})")
plt.plot([0,1], [0,1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
# plotting the ROC curve for Random Forest Classifier
rfc_param = {'max_depth':[3, 5, 10],
             'n_estimators':[150, 200, 300],
             'min_samples_split': [2, 5, 10],  
             'min_samples_leaf': [1, 2, 4]}

grid_search_rfc = GridSearchCV(rfc, rfc_param, cv = 5, scoring='f1', n_jobs=-1, verbose=1)
grid_search_rfc.fit(x_train, y_train)
grid_search_rfc.best_params_
tuned_rfc = RandomForestClassifier(max_depth = 10, min_samples_leaf = 1, min_samples_split = 5, n_estimators = 300)
tuned_rfc.fit(x_train, y_train)
tuned_rfc_pred = tuned_rfc.predict(x_test)
tuned_rfc_pred
tuned_rfc_train_pred = tuned_rfc.predict(x_train)
acc_train3 = accuracy_score(y_train, tuned_rfc_train_pred)
print("Training Accuracy:", acc_train3)
print("Testing accuracy:", accuracy_score(y_test, tuned_rfc_pred))
f1_score(y_test, tuned_rfc_pred)
tuned_rfc_prob = tuned_rfc.predict_proba(x_test)[:, 1]

fpr3, tpr3, _ = roc_curve(y_test, tuned_rfc_prob)
roc_auc3 = auc(fpr3, tpr3)
roc_auc3

plt.figure(figsize=(6,5))
plt.plot(fpr3, tpr3, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc3:.3f})")
plt.plot([0,1], [0,1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Hypertuned Random Forest")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
# plotting the ROC curve for Hypertuned Random Forest Classifier

from xgboost import XGBClassifier
xgb = XGBClassifier(eval_metric='mlogloss')
xgb.fit(x_train,y_train)
xgb_pred = xgb.predict(x_test)
xgb_pred
print("Confusion Matrix:\n", confusion_matrix(y_test,xgb_pred))
print("\nAccuracy Score:\n", accuracy_score(y_test,xgb_pred))
print("\nPrecision Score:\n", precision_score(y_test,xgb_pred))
print("\nRecall Score:\n", recall_score(y_test,xgb_pred))
print("\nF1 Score:\n", f1_score(y_test,xgb_pred))
print("\nClassification Report:\n", classification_report(y_test,xgb_pred))
xgb_train_pred = xgb.predict(x_train)
acc_train = accuracy_score(y_train, xgb_train_pred)
print("Training Accuracy:", acc_train)
print("Testing accuracy:", accuracy_score(y_test, xgb_pred))
xgb_param = {'n_estimators': [100, 150, 200],
                   'learning_rate': [0.05, 0.1, 0.2],
                   'max_depth': [3, 4, 5],
                   'subsample': [0.7, 0.8, 0.9],
                   'colsample_bytree': [0.7, 0.8, 0.9],
                   'min_child_weight': [1, 3, 5],
                   'reg_alpha': [0, 0.01, 0.1],
                   'reg_lambda': [1, 1.5, 2]}
grid_search_xgb = GridSearchCV(estimator = xgb,
                               param_grid = xgb_param,
                               cv = 5,
                               scoring = 'f1_weighted',
                               verbose = 1,
                               n_jobs = -1)
grid_search_xgb.fit(x_train, y_train)
grid_search_xgb.best_params_
best_params = grid_search_xgb.best_params_
tuned_xgb = XGBClassifier(**best_params, eval_metric='mlogloss', n_jobs=-1)
tuned_xgb.fit(x_train, y_train)

tuned_xgb_pred = tuned_xgb.predict(x_test)
tuned_xgb_pred
tuned_xgb_train_pred = tuned_xgb.predict(x_train)
acc_train4 = accuracy_score(y_train, tuned_xgb_train_pred)
print("Training Accuracy:", acc_train4)
print("Testing accuracy:", accuracy_score(y_test, tuned_xgb_pred))
f1_score(y_test, tuned_xgb_pred)