#Hybrid SVR model
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error,mean_squared_error
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# 读取数据集
df=pd.read_csv('PRODIGY dataset.csv')
#PRODIGY特征集合列表
selected_features_list=['Ics_charg-charg','Ics_charg-apolar', 'Ics_polar-polar', 'Ics_polar-apolar', '%NISapol','%NISchar']
selected_features_list_y=['Binding_affinity','Ics_charg-charg','Ics_charg-apolar', 'Ics_polar-polar', 'Ics_polar-apolar', '%NISapol','%NISchar']
data=df[selected_features_list_y]
print(data)

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

svr_average_pred=pd.DataFrame({"y":y})
cycle_times=3
n_repeats = 10

kernel_avgpredict=['rbf1_y_pred','rbf2_y_pred','rbf3_y_pred']
u=0
for i in range(cycle_times):
    svr_pred_data=[]
    random_state_number=[a for a in range(i*10+1,i*10+11)]
    for j in range(n_repeats):
        parameters = {
            'C': [0.1, 1, 2,3,4,5,6,7,8,9,10, 100],
            'epsilon': [0.01, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1, 2,3,4,5, 10]
        }
        kf = KFold(n_splits=4, shuffle=True, random_state=random_state_number[j])
        grid_search = GridSearchCV(SVR(kernel='rbf'), parameters, cv=kf, scoring='r2')
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        pipeline = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=best_params['C'], epsilon=best_params['epsilon']))
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)
        svr_pred_data.append(y_pred)
        R, _ = pearsonr(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
    svr_average_pred[kernel_avgpredict[u]]=np.mean(svr_pred_data, axis=0)
    u=u+1

x = svr_average_pred.iloc[:, 1:].values
y = svr_average_pred.iloc[:, 0].values

n_splits = 4
n_repeats = 10
coefficients = []

random_state_number=[1,2,3,4,5,6,7,8,9,10]
for _ in range(n_repeats):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state_number[_])
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = sm.OLS(y_train, sm.add_constant(x_train)).fit()
        coefficients.append(model.params)

# 计算所有回归系数的平均值
average_coefficients = np.mean(coefficients, axis=0)
variable_names=['SVR_1','SVR_2','SVR_3']
intercept = round(average_coefficients[0],5)
coef_str = ' + '.join([f'{coef:.5f} * {variable_names[i]}' for i, coef in enumerate(average_coefficients[1:])])
regression_equation = f'ΔG = {coef_str} + {intercept}'
print('Hybrid SVR model equation:', regression_equation)

#计算SVR-based mixed model的R和RMSE
average_coefficients1 = np.mean(coefficients, axis=0)[1:]
intercept = average_coefficients[0]
y_pred = np.dot(x, average_coefficients1) + intercept
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
R,a= pearsonr(y, y_pred)
print(f'混合SVR相关系数 (R): {R:.2f}')
print(f'混合SVR均方根误差(RMSE): {rmse:.2f}')