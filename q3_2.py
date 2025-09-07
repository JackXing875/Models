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
from clean_data import CleanData  

# 数据加载与预处理
file_path = r"D:\RUC\国赛\附件.xlsx"
cleaner = CleanData(file_path, sheet="男胎检测数据")
processed_df = cleaner.process_all()  
male_df = processed_df.copy()

# 构造达标事件时间数据
male_df['y_reach'] = (male_df['y_conc'] >= 0.04).astype(int)

Ti_list, delta_list = [], []
for pid, group in male_df.groupby('pid'):
    idx = group[group['y_reach'] == 1]
    if not idx.empty:
        Ti_list.append(idx['gest_week'].min())
        delta_list.append(1)
    else:
        Ti_list.append(np.nan)
        delta_list.append(0)

pregnant_info = male_df.groupby('pid').first().reset_index()
pregnant_info['T'] = Ti_list
pregnant_info['delta'] = delta_list

# 特征选择
features = ['age', 'height', 'weight', 'bmi', 'reads_total', 'map_ratio',
            'dup_ratio', 'reads_unique', 'gc_ratio', 'x_conc']

X = pregnant_info[features].fillna(pregnant_info[features].median())
y_event = pregnant_info['delta']
y_time = pregnant_info['T']

mask = y_event == 1
X_reg = X[mask]
y_reg = np.log(y_time[mask])  # 对数变换

# 标准化
scaler = StandardScaler()
X_reg_scaled = scaler.fit_transform(X_reg)

# LASSO特征选择
lasso_reg = LassoCV(cv=5, max_iter=20000, tol=1e-4)
lasso_reg.fit(X_reg_scaled, y_reg)
selected_features_reg = np.array(features)[lasso_reg.coef_ != 0]
print("LASSO选择的特征:", selected_features_reg)

# 筛选特征
X_reg_selected = X_reg_scaled[:, lasso_reg.coef_ != 0]

# XGBoost回归建模
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

# 可视化
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_time[mask], y=y_pred_T)
plt.plot([y_time[mask].min(), y_time[mask].max()],
         [y_time[mask].min(), y_time[mask].max()], 'r--')
plt.xlabel("Actual gestational week")
plt.ylabel("Predicted gestational week")
plt.title("XGBoost prediction vs Actual")
os.makedirs("./figure", exist_ok=True)
plt.savefig("./figure/XGBoost_prediction_vs_actual.jpg", dpi=500)
plt.close()

# BMI分组与最佳NIPT时点
bmi_bins = [0, 28, 32, 36, 40, 100]
bmi_labels = ['<28', '28-32', '32-36', '36-40', '>=40']
pregnant_info['bmi_group'] = pd.cut(pregnant_info['bmi'], bins=bmi_bins, labels=bmi_labels, right=False)

best_weeks = {}
for grp in bmi_labels:
    grp_idx = pregnant_info['bmi_group'] == grp
    if grp_idx.sum() == 0:
        best_weeks[grp] = np.nan
        continue
    X_grp = scaler.transform(pregnant_info.loc[grp_idx, features])[..., lasso_reg.coef_ != 0]
    y_pred_logT_grp = xgb_reg.predict(X_grp)
    y_pred_T_grp = np.exp(y_pred_logT_grp)
    best_weeks[grp] = y_pred_T_grp.mean()

print("Best NIPT week by BMI group:", best_weeks)

# 测量误差蒙特卡洛模拟
B = 5
sigma_eta = 0.005
mc_best_weeks = {grp: [] for grp in bmi_labels}

for b in range(B):
    noise = np.random.normal(0, sigma_eta, size=male_df.shape[0])
    male_df['y_conc_sim'] = male_df['y_conc'] + noise
    male_df['y_reach_sim'] = (male_df['y_conc_sim'] >= 0.04).astype(int)

    Ti_list_mc, delta_list_mc = [], []
    for pid, group in male_df.groupby('pid'):
        idx = group[group['y_reach_sim'] == 1]
        Ti_list_mc.append(idx['gest_week'].min() if not idx.empty else np.nan)
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
        grp_idx = pregnant_info['bmi_group'] == grp
        if grp_idx.sum() == 0:
            mc_best_weeks[grp].append(np.nan)
            continue
        X_grp_mc = scaler.transform(pregnant_info.loc[grp_idx, features])[..., lasso_reg.coef_ != 0]
        y_pred_T_grp_mc = np.exp(xgb_reg.predict(X_grp_mc))
        mc_best_weeks[grp].append(y_pred_T_grp_mc.mean())

# 95%置信区间
for grp in bmi_labels:
    vals = np.array(mc_best_weeks[grp])
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        continue
    ci_lower, ci_upper = np.percentile(vals, [2.5, 97.5])
    print(f"{grp} group best week 95% CI: [{ci_lower:.1f}, {ci_upper:.1f}]")

# 可视化
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False  # 避免负号显示成方块

# BMI组孕周分布箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(
    x='bmi_group', y='T', data=pregnant_info,
    showmeans=True,  # 显示均值
    boxprops=dict(alpha=0.7, facecolor="skyblue"),
    medianprops=dict(color="red", linewidth=2),
    meanprops=dict(marker="o", markerfacecolor="black", markersize=6)
)
plt.ylabel("达到阈值的孕周", fontsize=12)
plt.xlabel("BMI分组", fontsize=12)
plt.title("不同BMI组达到阈值孕周分布", fontsize=14, fontweight='bold')
plt.savefig("./figure/BMI组孕周分布.jpg", dpi=500, bbox_inches="tight")
plt.show()
plt.close()

# 残差分布直方图
residuals = y_time[mask] - y_pred_T
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30, color="teal", alpha=0.7)
plt.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
plt.xlabel("残差 (真实值 - 预测值)", fontsize=12)
plt.ylabel("频数", fontsize=12)
plt.title("残差分布", fontsize=14, fontweight='bold')
plt.savefig("./figure/残差分布.jpg", dpi=500, bbox_inches="tight")
plt.show()
plt.close()

# 特征重要性
fig, ax = plt.subplots(figsize=(8, 6))
xgb.plot_importance(
    xgb_reg, importance_type='weight',
    ax=ax, color="cornflowerblue"
)
plt.title("特征重要性", fontsize=14, fontweight='bold')
plt.savefig("./figure/特征重要性.jpg", dpi=500, bbox_inches="tight")
plt.show()
plt.close()

# 残差与预测值
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=y_pred_T, y=residuals, alpha=0.6,
    color="purple", edgecolor="w", s=70
)
sns.regplot(
    x=y_pred_T, y=residuals, scatter=False,
    lowess=True, color="red", line_kws={"linewidth": 2}, ci=None  # 平滑趋势线
)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel("预测孕周", fontsize=12)
plt.ylabel("残差", fontsize=12)
plt.title("残差与预测孕周关系", fontsize=14, fontweight='bold')
plt.savefig("./figure/残差与预测孕周关系.jpg", dpi=500, bbox_inches="tight")
plt.show()
plt.close()
