#Hybrid DNN model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,PReLU,Input
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
import random as python_random
from sklearn.metrics import mean_absolute_error,mean_squared_error
from scipy.stats import pearsonr
from tensorflow.keras.layers import Layer
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 读取数据集
df=pd.read_csv('UPR dataset.csv')
#定义新界面特征
df['NIBa'] = df['Ics_charg-polar'] * df['BSApolar']
df['NIBp'] = df['Ics_charg-apolar'] * df['BSAapolar']
#输入我们的特征集合F
feature_set_F = ['NIBa','NIBp','Ics_polar-apolar','Ics_charg-charg','Ics_apolar-apolar','%NISpol','%NISapol']
feature_set_F_y=['Binding_affinity','Ics_charg-charg','NIBa','NIBp', 'Ics_polar-apolar','Ics_apolar-apolar','%NISapol','%NISpol']
data=df[feature_set_F_y]
print(data)

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

cycle_times=3
columnnames=['PReLU1','PReLU2','PReLU3']
n_repeats = 10

DNNs_average_pred=pd.DataFrame({"y":y})
activation_pred_data=[]
u=0
for j in range(cycle_times):
    activation_pred_data=[]
    random_state_number=[a for a in range(j*10+1,j*10+11)]
    for i in range(n_repeats):
        python_random.seed(random_state_number[i])
        tf.random.set_seed(random_state_number[i])
        # 自定义一输出层加权激活函数
        class WeightedSumActivation(Layer):
            def __init__(self, alpha=0.49, **kwargs):
                super(WeightedSumActivation, self).__init__(**kwargs)
                self.alpha = tf.Variable(initial_value=alpha, trainable=True, constraint=lambda x: tf.clip_by_value(x, 0, 1))
            def call(self, inputs):
                linear = tf.keras.activations.linear(inputs)
                relu = tf.keras.activations.relu(inputs)
                return self.alpha * linear + (1 - self.alpha) * relu
        def create_model(units=64):
            model = Sequential([
                Input(shape=(X.shape[1],)),
                Dense(units),
                PReLU(),
                Dense(units),
                PReLU(),
                Dense(units),
                PReLU(),
                Dense(1),
                WeightedSumActivation(alpha=0.49)  # 使用自定义激活函数
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            return model
        model=KerasRegressor(model=create_model,model__units=64,epochs=100,batch_size=10,verbose=0)
        param_grid = {
            'epochs': [50, 100],
            'batch_size': [3, 9],
            'model__units': [64,128]
        }
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=4, scoring='neg_mean_absolute_error',verbose=0)
        grid_result = grid.fit(X, y)
        best_model = create_model(units=grid_result.best_params_['model__units'])
        best_model.fit(X, y, epochs=grid_result.best_params_['epochs'], batch_size=grid_result.best_params_['batch_size'], verbose=0)
        predictions = best_model.predict(X, verbose=0).flatten()
        activation_pred_data.append(predictions)
        mae = mean_absolute_error(y, predictions)
        r, _ = pearsonr(y, predictions)
    DNNs_average_pred[columnnames[u]]=np.mean(activation_pred_data, axis=0)
    u=u+1

# 建立混合模型
x = DNNs_average_pred.iloc[:, 1:].values
y = DNNs_average_pred.iloc[:, 0].values
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
variable_names=['DNNs_1','DNNs_2','DNNs_3']
intercept = round(average_coefficients[0],5)
coef_str = ' + '.join([f'{coef:.5f} * {variable_names[i]}' for i, coef in enumerate(average_coefficients[1:])])
regression_equation = f'ΔG = {coef_str} + {intercept}'
print('Hybrid DNN model equation:', regression_equation)

# 计算DNN-based mixed model的R和RMSE
average_coefficients1 = np.mean(coefficients, axis=0)[1:]
intercept = average_coefficients[0]
y_pred = np.dot(x, average_coefficients1) + intercept
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
R,a= pearsonr(y, y_pred)
print(f'Hybrid DNN model相关系数 (R): {R:.2f}')
print(f'Hybrid DNN model均方根误差(RMSE): {rmse:.2f}')