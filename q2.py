import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from clean_data import CleanData

# 从重复测量数据构建生存记录
def construct_intervals(df, pid_col='pid', week_col='gest_week', y_col='y_conc', threshold=0.04):
    """
    对每个孕妇按孕周排序，如果出现从 <threshold 到 >=threshold 的第一次跃升，
    则定义达标区间 (L, R] = (week_{i-1}, week_i].
    若首次检测就 >= threshold，则 L = 0, R = week_1.
    若一直未达标，则视为右删失
    """
    rows = []
    for pid, g in df.groupby(pid_col):
        gg = g.sort_values(week_col)
        weeks = gg[week_col].values
        ys = gg[y_col].values

        found = False
        for i in range(len(ys)):
            if ys[i] >= threshold:
                if i == 0:
                    L = 0.0
                    R = float(weeks[i])
                else:
                    L = float(weeks[i-1])
                    R = float(weeks[i])
                event = 1
                found = True
                break
        if not found:
            L = float(weeks[-1])
            R = np.inf
            event = 0

        rows.append({'pid': pid, 'L': L, 'R': R, 'event': event, 'last_week': float(weeks[-1])})
    surv_df = pd.DataFrame(rows).set_index('pid')
    return surv_df


def intervals(surv_df, mode='right'):
    """
    模式: 'right' 或 'midpoint'
    """
    out = surv_df.copy()
    durations = []
    events = []
    for pid, row in out.iterrows():
        if row['event'] == 1:
            if mode == 'right':
                duration = row['R']
            elif mode == 'midpoint':
                duration = (row['L'] + row['R']) / 2.0
            else:
                raise ValueError("mode must be 'right' or 'midpoint'")
            event = 1
        else:
            duration = row['last_week']
            event = 0
        durations.append(duration)
        events.append(event)
    out['duration'] = durations
    out['event_observed'] = events
    return out


def fit_cox(df_surv, cov_df, covariates, duration_col='duration', event_col='event_observed', scale_covariates=False):
    """
    df_surv：每位患者的DataFrame，索引为 pid，包含的列是 duration（生存时间）和 event（结局事件）。  
    cov_df：每位患者的协变量数据框（DataFrame），索引为 pid，包含建模所需的协变量。  
    covariates：协变量名称的列表，这些名称对应于 cov_df 中的列。
    """
    # 只保留需要的协变量
    cov_df_for_cox = cov_df[covariates].copy()
    for col in covariates:
        cov_df_for_cox[col] = pd.to_numeric(cov_df_for_cox[col], errors='coerce')
    cov_df_for_cox = cov_df_for_cox.dropna(subset=covariates)

    df_fit = df_surv[[duration_col, event_col]].join(cov_df_for_cox, how='inner')  # inner join 丢掉没有协变量的行

    scaler = None
    if scale_covariates:
        scaler = StandardScaler()
        df_fit[covariates] = scaler.fit_transform(df_fit[covariates])

    cph = CoxPHFitter()
    cph.fit(df_fit, duration_col=duration_col, event_col=event_col, show_progress=False)

    cph._scaler = scaler
    cph._covariates = covariates
    cph._df_fit = df_fit
    return cph, df_fit

# 计算每个个体的生存函数矩阵 S(t)
def survival_matrix(cph, cov_df_for_predict, times):
    covs = cph._covariates
    Xpredict = cov_df_for_predict[covs].copy().reset_index(drop=True)
    if getattr(cph, '_scaler', None) is not None:
        Xpredict = pd.DataFrame(cph._scaler.transform(Xpredict), columns=covs)

    sf = cph.predict_survival_function(Xpredict, times=times)
    S = sf.values 
    pid_list = cov_df_for_predict.index.to_list()
    return np.array(times), S, pid_list


# 基于特征 [bmi, LP] 的层次聚类分析
def cluster(bmi_arr, lp_arr, k=3, plot_dendrogram=False, plot_fname=None):
    X = np.vstack([bmi_arr, lp_arr]).T
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Z = linkage(Xs, method='ward', metric='euclidean')
    labels = fcluster(Z, k, criterion='maxclust')  # 1..k
    centers = []
    for c in range(1, k+1):
        mask = labels == c
        if np.sum(mask) == 0:
            centers.append(np.array([np.nan]*X.shape[1]))
        else:
            centers.append(Xs[mask].mean(axis=0))
    centers = np.vstack(centers)
    if plot_dendrogram:
        plt.figure(figsize=(10, 4))
        dendrogram(Z, no_labels=True, color_threshold=None)
        plt.title("Hierarchical clustering dendrogram (ward)")
        if plot_fname:
            plt.savefig(plot_fname, bbox_inches='tight', dpi=150)
        plt.show()
    return labels, scaler, centers


# 根据分组生存函数（S_group）通过解析公式求 t*：
def get_tstar(times, S_group, alpha=5.0, beta=1.0):
    q = beta / (alpha + beta)

    if S_group[0] <= q: return float(times[0])
    if S_group[-1] >= q: return float(times[-1])
    idx = np.where(S_group <= q)[0]
    if len(idx) == 0: return float(times[-1])
    i = idx[0]
    if i == 0:
        return float(times[0])
    t1, t2 = times[i-1], times[i]
    S1, S2 = S_group[i-1], S_group[i]
    if S2 == S1:
        return float(t1)
    t_star = t1 + (q - S1) * (t2 - t1) / (S2 - S1)
    return float(t_star)

# Pipeline: 半参数 Cox + 聚类
def pipeline_cox_clustering(
    df_measurements,
    pid_col='pid',
    week_col='gest_week',
    y_col='y_conc',
    covariate_cols=['bmi'],
    cox_covariates=['bmi'],
    threshold=0.04,
    time_mode='right',
    k_clusters=3,
    alpha=5.0,
    beta=1.0,
    times_step=0.2,
    scale_covariates=False,
    plot_dendrogram=False,
    verbose=True       # 新增参数
):
    df_measurements = df_measurements.copy()
    
    # 构建生存区间
    surv_intervals = construct_intervals(df_measurements, pid_col=pid_col,
                                                 week_col=week_col, y_col=y_col, threshold=threshold)
    
    # 构建协变量
    base = df_measurements.sort_values([pid_col, week_col]).groupby(pid_col).first()
    cov_df = base[covariate_cols].copy()
    cov_df = cov_df.reindex(surv_intervals.index)
    cov_df_for_cox = cov_df.copy()
    for col in cox_covariates:
        cov_df_for_cox[col] = pd.to_numeric(cov_df_for_cox[col], errors='coerce')
    cov_df_for_cox = cov_df_for_cox.dropna(subset=cox_covariates).astype(float)
    
    # 构建生存数据
    surv_for_cox = intervals(surv_intervals.loc[cov_df_for_cox.index], mode=time_mode)
    
    # 拟合 Cox 模型
    cph, df_fit = None, None
    try:
        cph, df_fit = fit_cox(surv_for_cox, cov_df_for_cox, cox_covariates, scale_covariates=scale_covariates)
        if verbose:
            print("Cox summary :")
            print(cph.summary)
    except Exception as e:
        if verbose:
            print(f"Cox model fitting failed: {e}")

    # 生存矩阵和风险得分
    max_time = float(np.ceil(df_measurements[week_col].max()))
    times = np.arange(0.0, max_time + times_step, times_step)
    S_matrix, pid_array = None, []
    LP = None
    if cph is not None:
        times, S_matrix, pid_list = survival_matrix(cph, cov_df_for_cox, times)
        LP = np.log(cph.predict_partial_hazard(cov_df_for_cox).values.ravel())
        pid_array = cov_df_for_cox.index.to_list()
    
    # 层次聚类
    cluster_results = {}
    if LP is not None:
        bmi_arr = cov_df_for_cox['bmi'].astype(float).reindex(pid_array).values
        labels, scaler_cluster, centers = cluster(bmi_arr, LP, k=k_clusters, plot_dendrogram=plot_dendrogram)
        cluster_assignments = pd.Series(labels, index=pid_array, name='cluster')
        
        for c in range(1, k_clusters + 1):
            members = [pid for pid, lab in cluster_assignments.items() if lab == c]
            if len(members) == 0:
                cluster_results[c] = {'n': 0, 't_star': np.nan, 'S_group': None, 'members': []}
                continue
            idxs = [pid_array.index(pid) for pid in members]
            S_group = S_matrix[:, idxs].mean(axis=1)
            t_star = get_tstar(times, S_group, alpha=alpha, beta=beta)
            cluster_results[c] = {'n': len(members), 't_star': t_star, 'S_group': S_group, 'members': members}
        
        if verbose:
            print("Original clustering results (per-cluster t*):")
            for c in cluster_results:
                print(f"Cluster {c}: n={cluster_results[c]['n']}, t*={cluster_results[c]['t_star']:.3f} weeks")
    else:
        # 如果 Cox 拟合失败，返回空 cluster
        for c in range(1, k_clusters + 1):
            cluster_results[c] = {'n': 0, 't_star': np.nan, 'S_group': None, 'members': []}

    out = {
        'cph': cph,
        'surv_intervals': surv_intervals,
        'cov_df': cov_df_for_cox,
        'surv_for_cox': surv_for_cox,
        'times': times,
        'S_matrix': S_matrix,
        'pid_list': pid_array,
        'LP': LP,
        'cluster_results': cluster_results
    }
    return out

# Bootstrap 求置信区间 
def bootstrap(df_measurements, cluster_results, pid_col='pid', week_col='gest_week', y_col='y_conc',
                    covariate_cols=['bmi'], cox_covariates=['bmi'], alpha=5.0, beta=1.0,
                    times_step=0.2, n_boot=100, scale_covariates=False):
    boot_tstars_per_cluster = {c: [] for c in cluster_results.keys()}
    rng = np.random.RandomState(12345)
    unique_pids_all = df_measurements[pid_col].unique()
    n_pids = len(unique_pids_all)

    for _ in range(n_boot):
        sampled_pids = rng.choice(unique_pids_all, size=n_pids, replace=True)
        df_boot = pd.concat([df_measurements[df_measurements[pid_col] == pid] for pid in sampled_pids], ignore_index=True)
        
        surv_intervals_b = construct_intervals(df_boot, pid_col=pid_col,
                                                       week_col=week_col, y_col=y_col, threshold=0.04)
        base_b = df_boot.sort_values([pid_col, week_col]).groupby(pid_col).first()
        cov_df_b = base_b[covariate_cols].copy()
        cov_df_b_for_cox = cov_df_b.copy()
        for col in cox_covariates:
            cov_df_b_for_cox[col] = pd.to_numeric(cov_df_b_for_cox[col], errors='coerce')
        cov_df_b_for_cox = cov_df_b_for_cox.dropna(subset=cox_covariates)
        if cov_df_b_for_cox.empty:
            continue
        
        surv_for_cox_b = intervals(surv_intervals_b.loc[cov_df_b_for_cox.index], mode='right')
        try:
            cph_b, _ = fit_cox(surv_for_cox_b, cov_df_b_for_cox, cox_covariates, scale_covariates=scale_covariates)
        except Exception:
            continue
        
        times_b = np.arange(0.0, float(np.ceil(df_measurements[week_col].max())) + times_step, times_step)
        times_b, S_matrix_b, pid_list_b = survival_matrix(cph_b, cov_df_b_for_cox, times_b)
        LP_b = np.log(cph_b.predict_partial_hazard(cov_df_b_for_cox).values.ravel())
        
        # 聚类
        bmi_arr = cov_df_b_for_cox['bmi'].values
        labels_b, _, _ = cluster(bmi_arr, LP_b, k=len(cluster_results), plot_dendrogram=False)
        
        # 每簇 t*
        for c in cluster_results.keys():
            members_boot = [pid_list_b[idx] for idx, lab in enumerate(labels_b) if lab == c]
            if len(members_boot) == 0:
                boot_tstars_per_cluster[c].append(np.nan)
                continue
            idxs_boot = [pid_list_b.index(pid) for pid in members_boot]
            S_group_b = S_matrix_b[:, idxs_boot].mean(axis=1)
            t_star_b = get_tstar(times_b, S_group_b, alpha=alpha, beta=beta)
            boot_tstars_per_cluster[c].append(t_star_b)
    
    summary = {}
    for c in cluster_results.keys():
        arr = np.array([v for v in boot_tstars_per_cluster[c] if (v is not None) and (not np.isnan(v))])
        if arr.size == 0:
            ci_low, ci_high, mean_t = np.nan, np.nan, np.nan
        else:
            ci_low = np.percentile(arr, 2.5)
            ci_high = np.percentile(arr, 97.5)
            mean_t = np.nanmean(arr)
        summary[c] = {
            'boot_mean_tstar': mean_t,
            'boot_ci_2.5': ci_low,
            'boot_ci_97.5': ci_high,
            'boot_samples_collected': len(boot_tstars_per_cluster[c])
        }

    return {c: list(map(float, vals)) for c, vals in boot_tstars_per_cluster.items()}

# 灵敏度分析
def measurement_error_simulation(df_measurements, pipeline_func, n_sim=100,
                                 noise_sds=[0.005, 0.01, 0.02], random_seed=2025, **pipeline_kwargs):
    rng = np.random.RandomState(random_seed)
    out = {}
    base_df = df_measurements.copy().reset_index(drop=True)
    for sd in noise_sds:
        tstar_mat = []
        for s in range(n_sim):
            df_sim = base_df.copy()
            noise = rng.normal(loc=0.0, scale=sd, size=len(df_sim))
            df_sim['y_conc'] = df_sim['y_conc'] + noise
            df_sim['y_conc'] = df_sim['y_conc'].clip(lower=0.0)

            sim_res = pipeline_func(df_sim, **pipeline_kwargs)
            tstars = [sim_res['cluster_results'][c].get('t_star', np.nan)
                      for c in sorted(sim_res['cluster_results'].keys())]
            tstar_mat.append(tstars)
        
        tstar_mat = np.array(tstar_mat)
        out[sd] = pd.DataFrame(tstar_mat, columns=[f"cluster_{c}" for c in sorted(sim_res['cluster_results'].keys())])
    return out


def summarize(df, results):
    cov_df = results['cov_df']
    cluster_results = results['cluster_results']
    
    bootstrap_summary = results.get('bootstrap_summary', 
                                    {c: {'boot_mean_tstar': np.nan, 
                                         'boot_ci_2.5': np.nan, 
                                         'boot_ci_97.5': np.nan} 
                                     for c in cluster_results.keys()})

    df_orig = df.sort_values(['pid', 'gest_week']).groupby('pid').first()
    gest_orig = df_orig['gest_week']

    rows = []
    for c in sorted(cluster_results.keys()):
        members = cluster_results[c].get('members', [])
        if len(members) == 0:
            rows.append({
                'cluster': c,
                'n': 0,
                'gest_min': np.nan, 'gest_median': np.nan, 'gest_max': np.nan,
                'tstar_orig': np.nan,
                'tstar_boot_mean': np.nan,
                'tstar_ci_2.5': np.nan,
                'tstar_ci_97.5': np.nan
            })
            continue
        gest_vals = gest_orig.loc[members].values
        rows.append({
            'cluster': c,
            'n': len(members),
            'gest_min': float(np.nanmin(gest_vals)),
            'gest_median': float(np.nanmedian(gest_vals)),
            'gest_max': float(np.nanmax(gest_vals)),
            'tstar_orig': float(cluster_results[c]['t_star']),
            'tstar_boot_mean': float(bootstrap_summary[c]['boot_mean_tstar']),
            'tstar_ci_2.5': bootstrap_summary[c]['boot_ci_2.5'],
            'tstar_ci_97.5': bootstrap_summary[c]['boot_ci_97.5']
        })
    return pd.DataFrame(rows).set_index('cluster')


def pipeline_wrapper_for_sim(df_sim, **kwargs):
    kwargs_sim = kwargs.copy()
    kwargs_sim.pop('n_boot', None)
    kwargs_sim['verbose'] = False  
    return pipeline_cox_clustering(df_sim, **kwargs_sim)

def ci_from_bootstrap(values, ci=0.95):
    """由 bootstrap 样本计算置信区间"""
    values = np.array(values)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    lower = np.percentile(values, (1-ci)/2*100)
    upper = np.percentile(values, (1+ci)/2*100)
    return lower, upper


if __name__ == "__main__":
    path = r"D:\RUC\国赛\附件.xlsx"
    cleaner = CleanData(path)
    df = cleaner.process_all()

    results = pipeline_cox_clustering(df)
    df_summary = summarize(df, results)
    print("\n每簇 BMI 区间与推荐 NIPT 时点：")
    print(df_summary)
    df_summary.to_csv("cluster_bmi_tstar_summary.csv", float_format="%.3f")
    print("已保存 cluster_bmi_tstar_summary.csv")

    boot_res = bootstrap(
        df_measurements=df,
        cluster_results=results['cluster_results'],
        pid_col='pid', week_col='gest_week', y_col='y_conc',
        covariate_cols=['bmi'], cox_covariates=['bmi'],
        alpha=5.0, beta=1.0, times_step=0.2,
        n_boot=200, scale_covariates=False
    )

    ci_rows = []
    for c, tstars in boot_res.items():
        mean_t = np.nanmean(tstars)
        std_t = np.nanstd(tstars)
        ci_low, ci_high = ci_from_bootstrap(tstars, ci=0.95)
        ci_rows.append({
            "cluster": c,
            "mean_tstar": mean_t,
            "std_tstar": std_t,
            "ci_low": ci_low,
            "ci_high": ci_high
        })
    df_ci = pd.DataFrame(ci_rows)
    print("\nBootstrap 置信区间结果：")
    print(df_ci)
    df_ci.to_csv("bootstrap_ci_results.csv", index=False)
    print("已保存 bootstrap_ci_results.csv")

    # ===================== 测量误差敏感性分析 =====================
    noise_sds = [0.005, 0.01, 0.02]
    n_sim = 100

    print("\n开始误差敏感性分析")
    sim_results = measurement_error_simulation(
        df_measurements=df,
        pipeline_func=pipeline_wrapper_for_sim,
        n_sim=n_sim,
        noise_sds=noise_sds,
        random_seed=2025,
        pid_col='pid', week_col='gest_week', y_col='y_conc',
        covariate_cols=['bmi'], cox_covariates=['bmi'],
        threshold=0.04, time_mode='right', k_clusters=3,
        alpha=5.0, beta=1.0, times_step=0.2,
        scale_covariates=False, plot_dendrogram=False
    )

    me_err_summary_rows = []
    for sd, df_t in sim_results.items():
        for col in df_t.columns:
            arr = df_t[col].dropna().values
            cluster_num = int(col.split('_')[-1])
            members = results['cluster_results'][cluster_num]['members']
            gest_weeks = df[df['pid'].isin(members)]['gest_week'].values

            me_err_summary_rows.append({
                'noise_sd': sd,
                'cluster': col,
                'mean_tstar': np.nanmean(arr) if arr.size > 0 else np.nan,
                'std_tstar': np.nanstd(arr) if arr.size > 0 else np.nan,
                'gest_min': float(np.nanmin(gest_weeks)) if len(gest_weeks) > 0 else np.nan,
                'gest_q1': float(np.nanpercentile(gest_weeks, 25)) if len(gest_weeks) > 0 else np.nan,
                'gest_median': float(np.nanmedian(gest_weeks)) if len(gest_weeks) > 0 else np.nan,
                'gest_q3': float(np.nanpercentile(gest_weeks, 75)) if len(gest_weeks) > 0 else np.nan,
                'gest_max': float(np.nanmax(gest_weeks)) if len(gest_weeks) > 0 else np.nan
            })

    me_err_summary = pd.DataFrame(me_err_summary_rows)
    print("\n测量误差敏感性结果（每簇 t* 的均值与标准差 原始孕周）：")
    print(me_err_summary)
    me_err_summary.to_csv("measurement_error_sensitivity.csv", index=False)
    print("已保存 measurement_error_sensitivity.csv")
