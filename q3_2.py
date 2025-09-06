# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import os

# -------------------------------
# 1. 数据加载与预处理
# -------------------------------
file_path = r"D:\RUC\国赛\附件.xlsx"
male_df = pd.read_excel(file_path, sheet_name="男胎检测数据")

# 孕周解析函数
def parse_weeks(x):
    try:
        if isinstance(x, str) and "w" in x:
            parts = x.replace("w", "").split("+")
            w = float(parts[0])
            d = float(parts[1]) if len(parts) > 1 and parts[1] != "" else 0
            return w + d / 7
        elif isinstance(x, (int, float)):
            return float(x)
        else:
            return np.nan
    except:
        return np.nan

male_df["孕周数值"] = male_df["检测孕周"].apply(parse_weeks)
male_df['Y达标'] = (male_df['Y染色体浓度'] >= 0.04).astype(int)

# 计算每位孕妇达标时间 Ti 和事件指示 delta_i
Ti_list, delta_list = [], []
for code, group in male_df.groupby('孕妇代码'):
    idx = group[group['Y达标'] == 1]
    if not idx.empty:
        Ti_list.append(idx['孕周数值'].min())
        delta_list.append(1)
    else:
        Ti_list.append(np.nan)
        delta_list.append(0)

pregnant_info = male_df.groupby('孕妇代码').first().reset_index()
pregnant_info['T'] = Ti_list
pregnant_info['delta'] = delta_list

# 特征选择
features = ['年龄', '身高', '体重', '孕妇BMI', '原始读段数', '在参考基因组上比对的比例',
            '重复读段的比例', '唯一比对的读段数  ', 'GC含量', 'X染色体浓度']

X = pregnant_info[features].fillna(pregnant_info[features].median())
y_event = pregnant_info['delta']
y_time = pregnant_info['T']

# 只对达标孕妇建模
mask = y_event == 1
X_reg = X[mask]
y_reg = np.log(y_time[mask])  # 对数变换

# 标准化
scaler = StandardScaler()
X_reg_scaled = scaler.fit_transform(X_reg)

# -------------------------------
# 2. LASSO特征选择
# -------------------------------
lasso_reg = LassoCV(cv=5, max_iter=20000, tol=1e-4)
lasso_reg.fit(X_reg_scaled, y_reg)
selected_features_reg = np.array(features)[lasso_reg.coef_ != 0]
print("LASSO选择的特征（回归）:", selected_features_reg)

# 筛选特征
X_reg_selected = X_reg_scaled[:, lasso_reg.coef_ != 0]

# -------------------------------
# 3. XGBoost回归建模
# -------------------------------
xgb_reg = xgb.XGBRegressor(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_reg.fit(X_reg_selected, y_reg)

# 预测
y_pred_logT = xgb_reg.predict(X_reg_selected)
y_pred_T = np.exp(y_pred_logT)  # 对数反变换

# 评估
mae = mean_absolute_error(y_time[mask], y_pred_T)
rmse = np.sqrt(mean_squared_error(y_time[mask], y_pred_T))
r2 = r2_score(y_time[mask], y_pred_T)
print(f"回归 MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}")

# -------------------------------
# 4. 可视化
# -------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_time[mask], y=y_pred_T)
plt.plot([y_time[mask].min(), y_time[mask].max()],
         [y_time[mask].min(), y_time[mask].max()], 'r--')
plt.xlabel("实际达标孕周")
plt.ylabel("预测达标孕周")
plt.title("XGBoost回归预测 vs 实际")
os.makedirs("./figure", exist_ok=True)
plt.savefig("./figure/XGBoost回归预测 vs 实际.jpg", dpi=500)

# -------------------------------
# 5. BMI分组与最佳NIPT时点（基于XGBoost回归）
# -------------------------------
# 定义BMI分组
bmi_bins = [0, 28, 32, 36, 40, 100]
bmi_labels = ['<28', '28-32', '32-36', '36-40', '>=40']
pregnant_info['BMI组'] = pd.cut(pregnant_info['孕妇BMI'], bins=bmi_bins, labels=bmi_labels, right=False)

weeks_grid = np.arange(8, 26, 0.5)
best_weeks = {}

for grp in bmi_labels:
    grp_idx = pregnant_info['BMI组'] == grp
    if grp_idx.sum() == 0:
        best_weeks[grp] = np.nan
        continue
    X_grp = scaler.transform(pregnant_info.loc[grp_idx, features])[..., lasso_reg.coef_ != 0]
    y_pred_logT_grp = xgb_reg.predict(X_grp)
    y_pred_T_grp = np.exp(y_pred_logT_grp)
    # 假设达标概率随孕周增加，选择平均预测孕周作为参考最佳NIPT时点
    best_weeks[grp] = y_pred_T_grp.mean()

print("根据男胎孕妇的BMI，各组最佳NIPT时点（孕周）：", best_weeks)

# -------------------------------
# 6. 测量误差蒙特卡洛模拟
# -------------------------------
B = 5  # 模拟次数
sigma_eta = 0.005
mc_best_weeks = {grp: [] for grp in bmi_labels}

for b in range(B):
    noise = np.random.normal(0, sigma_eta, size=male_df.shape[0])
    male_df['Y浓度模拟'] = male_df['Y染色体浓度'] + noise
    male_df['Y达标模拟'] = (male_df['Y浓度模拟'] >= 0.04).astype(int)

    Ti_list_mc, delta_list_mc = [], []
    for code, group in male_df.groupby('孕妇代码'):
        idx = group[group['Y达标模拟'] == 1]
        Ti_list_mc.append(idx['孕周数值'].min() if not idx.empty else np.nan)
        delta_list_mc.append(1 if not idx.empty else 0)
    pregnant_info['T_mc'] = Ti_list_mc
    pregnant_info['delta_mc'] = delta_list_mc

    mask_mc = pregnant_info['delta_mc'] == 1
    if mask_mc.sum() < 2:
        continue

    X_reg_mc = scaler.transform(pregnant_info.loc[mask_mc, features])[..., lasso_reg.coef_ != 0]
    y_reg_mc = np.log(pregnant_info.loc[mask_mc, 'T_mc'])
    xgb_reg.fit(X_reg_mc, y_reg_mc)

    for grp in bmi_labels:
        grp_idx = pregnant_info['BMI组'] == grp
        if grp_idx.sum() == 0:
            mc_best_weeks[grp].append(np.nan)
            continue
        X_grp_mc = scaler.transform(pregnant_info.loc[grp_idx, features])[..., lasso_reg.coef_ != 0]
        y_pred_T_grp_mc = np.exp(xgb_reg.predict(X_grp_mc))
        mc_best_weeks[grp].append(y_pred_T_grp_mc.mean())

# 输出95%置信区间
for grp in bmi_labels:
    vals = np.array(mc_best_weeks[grp])
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        continue
    ci_lower, ci_upper = np.percentile(vals, [2.5, 97.5])
    print(f"{grp} BMI组最佳孕周95%CI: [{ci_lower:.1f}, {ci_upper:.1f}]")

# -------------------------------
# 7. 可视化
# -------------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(x='BMI组', y='T', data=pregnant_info)
plt.ylabel("达标时间（孕周）")
plt.title("不同BMI组达标时间分布")
plt.savefig("./figure/不同BMI组达标时间分布_XGB.jpg", dpi=500)