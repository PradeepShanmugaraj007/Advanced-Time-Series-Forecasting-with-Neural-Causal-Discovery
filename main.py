import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.tsa.stattools as st

n = 1000
vars = 5
np.random.seed(42)
data = np.zeros((n, vars))
data[:,0] = np.random.normal(size=n)
data[:,1] = 0.7 * np.roll(data[:,0], 1) + np.random.normal(size=n)
data[:,2] = 0.5 * np.roll(data[:,1], 1) + 0.3 * np.roll(data[:,0], 2) + np.random.normal(size=n)
data[:,3] = 0.4 * np.roll(data[:,2], 1) + 0.3 * np.roll(data[:,1], 2) + np.random.normal(size=n)
data[:,4] = 0.8 * np.roll(data[:,3], 1) + np.random.normal(size=n)
data = np.nan_to_num(data)
df = pd.DataFrame(data, columns=[f'var{i}' for i in range(vars)])
df.to_csv('dataset.csv', index=False)

def granger_causality(df, maxlag=3):
    results = {}
    for col_y in df.columns:
        for col_x in df.columns:
            if col_x != col_y:
                test_result = st.grangercausalitytests(df[[col_y, col_x]], maxlag=maxlag, verbose=False)
                p_values = [test_result[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag+1)]
                min_p = min(p_values)
                results[(col_x, col_y)] = min_p
    return results

causality_results = granger_causality(df)

def make_supervised(df, lag=3):
    X, y = [], []
    for i in range(lag, len(df)):
        X.append(df.values[i-lag:i].flatten())
        y.append(df.values[i, 0])
    return np.array(X), np.array(y)

lag = 3
X_all, y_all = make_supervised(df, lag)
scaler = StandardScaler().fit(X_all)
X_all_scaled = scaler.transform(X_all)
model_base = LinearRegression().fit(X_all_scaled, y_all)
y_pred_base = model_base.predict(X_all_scaled)
rmse_base = np.sqrt(mean_squared_error(y_all, y_pred_base))
mae_base = mean_absolute_error(y_all, y_pred_base)

cause_vars = [k[0] for k,v in causality_results.items() if k[1]=='var0' and v < 0.05]
if cause_vars:
    idxs = [df.columns.get_loc(c) for c in cause_vars]
    selected = []
    for l in range(lag):
        for i in idxs:
            selected.append(i + l*vars)
    X_causal = X_all[:,selected]
else:
    X_causal = X_all[:,[0, vars, 2*vars]]
scaler_causal = StandardScaler().fit(X_causal)
X_causal_scaled = scaler_causal.transform(X_causal)
model_causal = LinearRegression().fit(X_causal_scaled, y_all)
y_pred_causal = model_causal.predict(X_causal_scaled)
rmse_causal = np.sqrt(mean_squared_error(y_all, y_pred_causal))
mae_causal = mean_absolute_error(y_all, y_pred_causal)

with open('output.txt', 'w') as f:
    f.write('Baseline Model Results
')
    f.write(f'RMSE: {rmse_base:.4f}
MAE: {mae_base:.4f}
')
    f.write('Causal-Aware Model Results
')
    f.write(f'RMSE: {rmse_causal:.4f}
MAE: {mae_causal:.4f}
')
    f.write('
Causal Links Discovered (p < 0.05):
')
    for k,v in causality_results.items():
        if v < 0.05:
            f.write(f'{k}: {v:.4e}
')
