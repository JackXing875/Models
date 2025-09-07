import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pygam import LinearGAM, s, te
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from clean_data import CleanData
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline

# Spearman相关系数
def spearman(df, cols):
    corr, pval = spearmanr(df[cols])
    print("Spearman相关系数矩阵：\n", corr)
    return corr, pval

# 二维平滑热力图
def heatmap(df, x='gest_week', y='bmi', target='y_conc', h1=1, h2=1, grid_size=50):
    x_min, x_max = df[x].min(), df[x].max()
    y_min, y_max = df[y].min(), df[y].max()
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    mu = np.zeros((grid_size, grid_size))
    sigma = np.zeros((grid_size, grid_size))
    
    for i, u in enumerate(x_grid):
        for j, v in enumerate(y_grid):
            w = np.exp(-((df[x]-u)**2)/(2*h1**2) - ((df[y]-v)**2)/(2*h2**2))
            mu[i,j] = np.sum(w*df[target])/np.sum(w)
            sigma[i,j] = np.sqrt(np.sum(w*(df[target]-mu[i,j])**2)/np.sum(w))
    
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.contourf(x_grid, y_grid, mu.T, 20, cmap='viridis')
    plt.colorbar(label='局部均值')
    plt.xlabel(x); plt.ylabel(y); plt.title('y局部均值热力图')
    
    plt.subplot(1,2,2)
    plt.contourf(x_grid, y_grid, sigma.T, 20, cmap='magma')
    plt.colorbar(label='局部标准差')
    plt.xlabel(x); plt.ylabel(y); plt.title('y局部标准差热力图')
    
    plt.show()

def regression_tree(df, target='y_conc', features=None, max_depth=3, min_samples_leaf=10, plot=True):
    """
    回归树
    """
    X = df[features].values
    y = df[target].values
    tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    tree.fit(X, y)
    
    if plot:
        plt.figure(figsize=(15,8))
        plot_tree(tree, feature_names=features, filled=True, fontsize=10)
        plt.show()
    
    importances = tree.feature_importances_
    feature_importance_dict = {f: imp for f, imp in zip(features, importances)}
    print("特征重要性：")
    for f, imp in sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"{f:20s} -> {imp:.4f}")
    
    return tree, feature_importance_dict


def Lasso(df, target='y_conc', features=None, degree=2, cv=5, random_state=2025):
    """
    多项式扩展 + Lasso 特征选择
    target: 目标变量
    features: 自变量列表
    degree: 多项式阶数
    cv: 交叉验证折数
    """
    X = df[features].values
    y = df[target].values
    
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(cv=cv, random_state=random_state, n_jobs=-1))
    ])
    
    pipe.fit(X, y)
    lasso = pipe.named_steps['lasso']
    poly = pipe.named_steps['poly']
    
    feature_names = poly.get_feature_names_out(features)
    coef = lasso.coef_
    
    selected = [(f, c) for f, c in zip(feature_names, coef) if abs(c) > 1e-6]
    
    print("LassoCV 最优 α:", lasso.alpha_)
    print("保留下来的特征：")
    for f, c in selected:
        print(f"{f:20s} -> {c:.4f}")
    
    return selected, pipe

def get_features(selected, features_all):
    """
    从 Lasso 结果里提取原始自变量名
    """
    raw_selected = []
    for f, c in selected:
        for orig in features_all:
            if orig in f:  
                raw_selected.append(orig)
    return list(set(raw_selected))

# GAMM 拟合
def fit_gamm(df, target='y_conc', covariates=None):
    """
    target: 目标变量列名
    covariates: 额外协变量列表
    """
    smooth_vars = ['gest_week', 'bmi']
    if covariates:
        covariates = [v for v in covariates if v not in smooth_vars]

    # 自变量矩阵
    X_smooth = df[smooth_vars].values
    y = df[target].values

    terms = te(0,1)  # gest_week × bmi 二维平滑
    if covariates:
        X_cov = df[covariates].values
        X = np.hstack([X_smooth, X_cov])
        for i in range(X_smooth.shape[1], X.shape[1]):
            terms = terms + s(i) 
    else:
        X = X_smooth

    gam = LinearGAM(terms).fit(X, y)
    return gam, X, y

# 置换检验
def permutation_test(df, target='y_conc', covariates=None, n_perm=200):
    gam, X, y = fit_gamm(df, target, covariates)
    original_coefs = gam.coef_

    unique_pids = df['pid'].unique()
    perm_coefs = []

    for _ in range(n_perm):
        y_perm = y.copy()
        for pid in unique_pids:
            mask = df['pid'] == pid
            y_perm[mask] = np.random.permutation(y[mask])
        gam_perm, _, _ = fit_gamm(df.assign(**{target: y_perm}), target, covariates)
        perm_coefs.append(gam_perm.coef_)

    perm_coefs = np.array(perm_coefs)
    p_values = np.mean(np.abs(perm_coefs) >= np.abs(original_coefs), axis=0)
    return p_values

# Bootstrap
def bootstrap(df, target='y_conc', covariates=None, n_boot=200):
    unique_pids = df['pid'].unique()
    boot_coefs = []

    for _ in range(n_boot):
        sampled_pids = np.random.choice(unique_pids, size=len(unique_pids), replace=True)
        df_boot = pd.concat([df[df['pid'] == pid] for pid in sampled_pids], ignore_index=True)
        gam, _, _ = fit_gamm(df_boot, target, covariates)
        boot_coefs.append(gam.coef_)

    boot_coefs = np.array(boot_coefs)
    ci_lower = np.percentile(boot_coefs, 2.5, axis=0)
    ci_upper = np.percentile(boot_coefs, 97.5, axis=0)
    return ci_lower, ci_upper

def draw(df, x='gest_week', y='bmi', target='y_conc', h1=1, h2=1, grid_size=50):
    """
    绘制Y浓度随孕周和BMI变化的热力图和等高线
    - x, y: 自变量列名
    - target: Y浓度
    - h1, h2: 高斯核带宽
    - grid_size: 网格分辨率
    """
    x_min, x_max = df[x].min(), df[x].max()
    y_min, y_max = df[y].min(), df[y].max()
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)

    mu = np.zeros((grid_size, grid_size))

    # 二维高斯平滑
    for i, u in enumerate(x_grid):
        for j, v in enumerate(y_grid):
            w = np.exp(-((df[x]-u)**2)/(2*h1**2) - ((df[y]-v)**2)/(2*h2**2))
            mu[i,j] = np.sum(w*df[target])/np.sum(w)

    X, Y = np.meshgrid(x_grid, y_grid)
    plt.figure(figsize=(10,7))

    # 热力图
    plt.contourf(X, Y, mu.T, 20, cmap='viridis')
    plt.colorbar(label='Y浓度')
    
    # 等高线图
    cs = plt.contour(X, Y, mu.T, colors='white', linewidths=1.2)
    plt.clabel(cs, fmt="%.2f", colors='white')
    
    plt.xlabel('孕周')
    plt.ylabel('BMI')
    plt.title('Y浓度随孕周和BMI的变化（局部均值 + 等高线）')
    plt.show()


if __name__ == "__main__":
    path = r"D:\RUC\国赛\附件.xlsx"
    cleaner = CleanData(path)
    df = cleaner.process_all()

    print(df.columns)

    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    # Spearman
    cols_corr = ['gest_week','bmi','y_conc','reads_total','gc_ratio']
    spearman(df, cols_corr)
    
    # 热力图
    heatmap(df)
    
    # 决策树
    tree_features = ['gest_week','bmi','age','reads_total','gc_ratio']
    tree, feature_importance = regression_tree(df, target='y_conc', features=tree_features)

    # Lasso 特征选择
    features_all = ['gest_week','bmi','age','reads_total','gc_ratio']
    selected, lasso_pipe = Lasso(df, target='y_conc', features=features_all, degree=2)

    # 自动传递给 GAMM
    covariates = get_features(selected, features_all)
    print("传递给 GAMM 的自变量：", covariates)

    gam_model, X, y = fit_gamm(df, covariates=covariates)
    print("GAMM 拟合完成")

    # 置换检验
    p_values_perm = permutation_test(df, covariates=covariates, n_perm=200)  # 可调大
    print("置换检验p值：", p_values_perm)
    
    # Bootstrap
    ci_lower, ci_upper = bootstrap(df, covariates=covariates, n_boot=200)  # 可调大
    print("Bootstrap 95% CI下限：", ci_lower)
    print("Bootstrap 95% CI上限：", ci_upper)

    draw(df)

    X, Y = np.meshgrid(np.linspace(df['gest_week'].min(), df['gest_week'].max(), 50),
                    np.linspace(df['bmi'].min(), df['bmi'].max(), 50))
    Z = np.zeros_like(X)
    # 用你的高斯平滑结果填充 Z
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            w = np.exp(-((df['gest_week']-X[i,j])**2)/(2*1**2) - ((df['bmi']-Y[i,j])**2)/(2*1**2))
            Z[i,j] = np.sum(w*df['y_conc']) / np.sum(w)

    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k')
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Y浓度')
    ax.set_xlabel('孕周'); ax.set_ylabel('BMI'); ax.set_zlabel('Y浓度')
    plt.show()
