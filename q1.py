import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pygam import LinearGAM, s, l, te
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from clean_data import CleanData
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline


# =====================
# 1. Spearman相关系数
# =====================
def compute_spearman(df, cols):
    corr, pval = spearmanr(df[cols])
    print("Spearman相关系数矩阵：\n", corr)
    return corr, pval

# =====================
# 2. 二维平滑热力图（局部均值与标准差）
# =====================
def plot_heatmap(df, x='gest_week', y='bmi', target='y_conc', h1=1, h2=1, grid_size=50):
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

# =====================
# 3. 回归树
# =====================
def regression_tree(df, target='y_conc', features=None, max_depth=3, min_samples_leaf=10, plot=True):
    """
    回归树 + 输出特征重要性
    """
    X = df[features].values
    y = df[target].values
    tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    tree.fit(X, y)
    
    # 可视化决策树
    if plot:
        plt.figure(figsize=(15,8))
        plot_tree(tree, feature_names=features, filled=True, fontsize=10)
        plt.show()
    
    # 输出特征重要性
    importances = tree.feature_importances_
    feature_importance_dict = {f: imp for f, imp in zip(features, importances)}
    print("特征重要性：")
    for f, imp in sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"{f:20s} -> {imp:.4f}")
    
    return tree, feature_importance_dict


def lasso_poly_selection(df, target='y_conc', features=None, degree=2, cv=5, random_state=2025):
    """
    多项式扩展 + LassoCV 特征选择
    df: DataFrame
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
    
    # 提取特征名
    feature_names = poly.get_feature_names_out(features)
    coef = lasso.coef_
    
    selected = [(f, c) for f, c in zip(feature_names, coef) if abs(c) > 1e-6]
    
    print("LassoCV 最优 α:", lasso.alpha_)
    print("保留下来的特征：")
    for f, c in selected:
        print(f"{f:20s} -> {c:.4f}")
    
    return selected, pipe

def get_selected_features_from_lasso(selected, features_all):
    """
    从 Lasso 结果里提取原始自变量名（去掉多项式展开的高阶符号）
    """
    raw_selected = []
    for f, c in selected:
        for orig in features_all:
            if orig in f:   # 只要多项式里包含这个原始特征，就认为它有用
                raw_selected.append(orig)
    # 去重
    return list(set(raw_selected))


# =====================
# 4. GAMM 拟合
# =====================

def fit_gamm(df, target='y_conc', covariates=None):
    """
    df: DataFrame, 已处理好的数据
    target: 目标变量列名
    covariates: 额外协变量列表（Lasso筛选后）
    """
    # 保证 covariates 不重复
    smooth_vars = ['gest_week', 'bmi']
    if covariates:
        covariates = [v for v in covariates if v not in smooth_vars]

    # 自变量矩阵
    X_smooth = df[smooth_vars].values
    y = df[target].values

    # 构建 term
    terms = te(0,1)  # gest_week × bmi 二维平滑
    if covariates:
        X_cov = df[covariates].values
        X = np.hstack([X_smooth, X_cov])
        # 对额外协变量添加平滑 s(i) 或线性项，这里用线性
        for i in range(X_smooth.shape[1], X.shape[1]):
            terms = terms + s(i)  # 如果想做线性，可以换成 l(i)
    else:
        X = X_smooth

    gam = LinearGAM(terms).fit(X, y)
    return gam, X, y


# =====================
# 5. 置换检验
# =====================
def permutation_test_gamm(df, target='y_conc', covariates=None, n_perm=200):
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

# =====================
# 6.Bootstrap
# =====================

def bootstrap_gamm(df, target='y_conc', covariates=None, n_boot=200):
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

def plot_y_conc_contour(df, x='gest_week', y='bmi', target='y_conc', h1=1, h2=1, grid_size=50):
    """
    绘制Y浓度随孕周和BMI变化的热力图和等高线
    - df: 已处理数据
    - x, y: 自变量列名
    - target: Y浓度
    - h1, h2: 高斯核带宽
    - grid_size: 网格分辨率
    """
    # 网格
    x_min, x_max = df[x].min(), df[x].max()
    y_min, y_max = df[y].min(), df[y].max()
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)

    mu = np.zeros((grid_size, grid_size))  # 局部均值

    # 二维高斯平滑
    for i, u in enumerate(x_grid):
        for j, v in enumerate(y_grid):
            w = np.exp(-((df[x]-u)**2)/(2*h1**2) - ((df[y]-v)**2)/(2*h2**2))
            mu[i,j] = np.sum(w*df[target])/np.sum(w)

    # 绘图
    X, Y = np.meshgrid(x_grid, y_grid)
    plt.figure(figsize=(10,7))
    # 热力图
    plt.contourf(X, Y, mu.T, 20, cmap='viridis')
    plt.colorbar(label='Y浓度')
    # 等高线
    cs = plt.contour(X, Y, mu.T, colors='white', linewidths=1.2)
    plt.clabel(cs, fmt="%.2f", colors='white')
    
    plt.xlabel('孕周')
    plt.ylabel('BMI')
    plt.title('Y浓度随孕周和BMI的变化（局部均值 + 等高线）')
    plt.show()

# =====================
# 7. 主流程示例
# =====================
if __name__ == "__main__":
    path = r"D:\RUC\国赛\附件.xlsx"
    cleaner = CleanData(path)
    df = cleaner.process_all()

    print(df.columns)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
    plt.rcParams['axes.unicode_minus'] = False
    # Spearman
    cols_corr = ['gest_week','bmi','y_conc','reads_total','gc_ratio']
    compute_spearman(df, cols_corr)
    
    # 热力图
    plot_heatmap(df)
    
    # 决策树
    tree_features = ['gest_week','bmi','age','reads_total','gc_ratio']
    tree, feature_importance = regression_tree(df, target='y_conc', features=tree_features)

    # Lasso 特征选择
    features_all = ['gest_week','bmi','age','reads_total','gc_ratio']
    selected, lasso_pipe = lasso_poly_selection(df, target='y_conc', features=features_all, degree=2)

    # 自动传递给 GAMM
    covariates = get_selected_features_from_lasso(selected, features_all)
    print("传递给 GAMM 的自变量：", covariates)

    gam_model, X, y = fit_gamm(df, covariates=covariates)
    print("GAMM 拟合完成")

    # 置换检验
    p_values_perm = permutation_test_gamm(df, covariates=covariates, n_perm=200)  # 可调大
    print("置换检验p值：", p_values_perm)
    
    # Bootstrap
    ci_lower, ci_upper = bootstrap_gamm(df, covariates=covariates, n_boot=200)  # 可调大
    print("Bootstrap 95% CI下限：", ci_lower)
    print("Bootstrap 95% CI上限：", ci_upper)

    plot_y_conc_contour(df)
