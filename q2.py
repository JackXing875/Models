# pipeline_cox_clustering.py
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, WeibullAFTFitter
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings
from clean_data import CleanData
warnings.filterwarnings("ignore")

path = r"D:\RUC\国赛\附件.xlsx"
cleaner = CleanData(path)
df = cleaner.process_all()
# -----------------------------
# 1) 从重复测量数据构建“生存”记录（按孕妇）
# -----------------------------
def construct_patient_intervals(df, pid_col='pid', week_col='gest_week', y_col='y_conc', threshold=0.04):
    """
    对每个孕妇按孕周排序，如果出现从 <threshold 到 >=threshold 的第一次跃升，
    则定义达标区间 (L, R] = (week_{i-1}, week_i].
    若首次检测就 >= threshold，则 L = 0, R = week_1.
    若一直未达标，则视为右删失（L = last_week, R = np.inf, event=0）。
    返回 DataFrame per-patient：
      pid, L, R, event_observed (1 if interval observed), last_week,
      plus baseline covariates chosen (bmi) - we'll attach them later.
    """
    rows = []
    for pid, g in df.groupby(pid_col):
        gg = g.sort_values(week_col)
        weeks = gg[week_col].values
        ys = gg[y_col].values

        found = False
        for i in range(len(ys)):
            if ys[i] >= threshold:
                # first time at or above threshold
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


# -----------------------------
# 2) 将 L,R 转为 Cox 可用的 duration/event（两种近似）
# -----------------------------
def intervals_to_right_or_midpoint(surv_df, mode='right'):
    """
    mode: 'right' or 'midpoint'
    'right'    -> duration = R (if observed), else last_week; event flag as is.
    'midpoint' -> duration = (L+R)/2 for observed; else last_week.
    返回 DataFrame 包含 duration 与 event 列（index 为 pid）
    """
    out = surv_df.copy()
    durations = []
    events = []
    for pid, row in out.iterrows():
        if row['event'] == 1:
            if mode == 'right':
                duration = row['R']
            elif mode == 'midpoint':
                # if L == 0 and R finite, midpoint is R/2
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


# -----------------------------
# 3) Fit Cox model (semi-parametric), return fitted model and covariate DataFrame
# -----------------------------
def fit_cox(df_surv, cov_df, covariates, duration_col='duration', event_col='event_observed', scale_covariates=False):
    """
    df_surv: per-patient DataFrame with index=pid and columns duration/event
    cov_df: per-patient covariate DataFrame (index=pid) containing covariates used
    covariates: list of covariate column names in cov_df
    Returns fitted CoxPHFitter and the combined df used for fitting.
    """
    # 只保留需要的协变量，并强制转为数值
    cov_df_for_cox = cov_df[covariates].copy()
    for col in covariates:
        cov_df_for_cox[col] = pd.to_numeric(cov_df_for_cox[col], errors='coerce')
    cov_df_for_cox = cov_df_for_cox.dropna(subset=covariates)

    # Merge survival data with covariates
    df_fit = df_surv[[duration_col, event_col]].join(cov_df_for_cox, how='inner')  # inner join 丢掉没有协变量的行

    # 不 reset index，避免 pid 被当作协变量
    # optional scaling
    scaler = None
    if scale_covariates:
        scaler = StandardScaler()
        df_fit[covariates] = scaler.fit_transform(df_fit[covariates])

    cph = CoxPHFitter()
    cph.fit(df_fit, duration_col=duration_col, event_col=event_col, show_progress=False)

    # attach scaler for later prediction if needed
    cph._scaler = scaler
    cph._covariates = covariates
    cph._df_fit = df_fit
    return cph, df_fit


# -----------------------------
# 4) 计算每个个体的生存函数矩阵 S(t)（times x individuals）
# -----------------------------
def predict_survival_matrix(cph, cov_df_for_predict, times):
    """
    cph: fitted CoxPHFitter
    cov_df_for_predict: DataFrame index=pid with covariates (columns same as used in fit)
    times: 1D array of times at which to evaluate survival
    Returns: times (np.array), S (2D np.array shape (n_times, n_individuals)) and pid list
    """
    covs = cph._covariates
    Xpredict = cov_df_for_predict[covs].copy().reset_index(drop=True)
    # if scaler used when fitting, apply same transform
    if getattr(cph, '_scaler', None) is not None:
        Xpredict = pd.DataFrame(cph._scaler.transform(Xpredict), columns=covs)

    # lifelines returns DataFrame: index times, columns one per observation (names 0..n-1)
    sf = cph.predict_survival_function(Xpredict, times=times)
    # sf: DataFrame index=times, columns=0..n-1
    S = sf.values  # shape (len(times), n_obs)
    pid_list = cov_df_for_predict.index.to_list()
    return np.array(times), S, pid_list


# -----------------------------
# 5) Hierarchical clustering on features [bmi, LP]
# -----------------------------
def cluster_patients(bmi_arr, lp_arr, k=3, plot_dendrogram=False, plot_fname=None):
    """
    bmi_arr, lp_arr: 1D arrays with same order (index alignment)
    returns labels (1..k), scaler object, centers (k x features)
    """
    X = np.vstack([bmi_arr, lp_arr]).T
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # hierarchical linkage
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


# -----------------------------
# 6) 根据分组生存函数（S_group）通过解析公式求 t*：
#    E[L(t)] = alpha * ∫_{t}^{∞} S(u) du + beta * ∫_{0}^{t} (1 - S(u)) du
#    推导得到一阶导 = 0 => S(t*) = beta / (alpha + beta)
#    所以 t* 为使得 S_group(t*) = q 的反函数（插值求解）
# -----------------------------
def find_tstar_from_S(times, S_group, alpha=2.0, beta=1.0):
    """
    times: monotone increasing array
    S_group: array of same length (survival prob at each times element)
    returns t_star (float) by linear interpolation where S_group crosses q = beta/(alpha+beta).
    If crossing not found within times, returns times[-1] (or 0 if at start).
    """
    q = beta / (alpha + beta)
    # ensure monotone decreasing S_group
    # handle edgecases
    if S_group[0] <= q:
        return float(times[0])
    if S_group[-1] >= q:
        # survival still above q at last time -> event extremely rare, choose last time
        return float(times[-1])
    # find i such that S[i] >= q >= S[i+1]
    idx = np.where(S_group <= q)[0]
    if len(idx) == 0:
        return float(times[-1])
    i = idx[0]
    if i == 0:
        return float(times[0])
    t1, t2 = times[i-1], times[i]
    S1, S2 = S_group[i-1], S_group[i]
    if S2 == S1:
        return float(t1)
    # linear interpolation
    t_star = t1 + (q - S1) * (t2 - t1) / (S2 - S1)
    return float(t_star)


# -----------------------------
# 7) 主 pipeline：一条龙运行 + Bootstrap
# -----------------------------
def pipeline_cox_clustering(
    df_measurements,
    pid_col='pid',
    week_col='gest_week',
    y_col='y_conc',
    covariate_cols=['bmi'],
    cox_covariates=['bmi'],  # covariates to use in Cox
    threshold=0.04,
    time_mode='right',   # 'right' or 'midpoint'
    k_clusters=3,
    alpha=2.0,
    beta=1.0,
    times_step=0.2,
    n_boot=50,
    scale_covariates=False,
    plot_dendrogram=False
):
    """
    Robust pipeline: construct intervals, fit Cox, cluster using [bmi, LP],
    compute t* per cluster, and bootstrap for t* CI. Includes defensive checks
    and numeric coercion for covariates to avoid string->float errors (e.g. 'A082').
    """
    # Defensive copy
    df_measurements = df_measurements.copy()

    # --- 0. Basic checks ---
    if pid_col not in df_measurements.columns:
        raise KeyError(f"{pid_col} not in df_measurements")
    if week_col not in df_measurements.columns:
        raise KeyError(f"{week_col} not in df_measurements")
    if y_col not in df_measurements.columns:
        raise KeyError(f"{y_col} not in df_measurements")

    # 1) construct per-patient intervals
    surv_intervals = construct_patient_intervals(df_measurements, pid_col=pid_col, week_col=week_col, y_col=y_col, threshold=threshold)

    # 2) build baseline covariates per pid - choose first measurement per patient
    base = df_measurements.sort_values([pid_col, week_col]).groupby(pid_col).first()
    cov_df = base[covariate_cols].copy()

    # ensure cov_df index aligns with surv_intervals (index = pid)
    cov_df = cov_df.reindex(surv_intervals.index)

    # 2b) COERCION: make sure cox_covariates are numeric (avoid strings like 'A082')
    # Work on a copy to avoid SettingWithCopyWarning
    cov_df_for_cox = cov_df.copy()

# 2. 强制每个协变量为数字，非数字转 NaN
    for col in cox_covariates:
        cov_df_for_cox[col] = pd.to_numeric(cov_df_for_cox[col], errors='coerce')

    # 3. 丢掉所有 NaN 行
    cov_df_for_cox = cov_df_for_cox.dropna(subset=cox_covariates)

    # 4. 只保留 Cox 模型需要的列，并确保类型是 float
    cov_df_for_cox = cov_df_for_cox[cox_covariates].astype(float)

    # 5. 如果你有生存数据表 surv_intervals，需要对齐 index
    surv_for_cox = surv_intervals.loc[cov_df_for_cox.index]

    # Report and drop rows with NaN in any cox covariate
    n_before = len(cov_df_for_cox)
    cov_df_for_cox = cov_df_for_cox.dropna(subset=cox_covariates)
    dropped = n_before - len(cov_df_for_cox)
    if dropped > 0:
        print(f"Warning: {dropped} patients dropped because cox covariates contained non-numeric or missing values after coercion.")

    # Ensure surv_intervals and cov_df_for_cox have consistent index for cox fitting:
    surv_intervals_for_cox = surv_intervals.loc[cov_df_for_cox.index]

    # 3) convert to duration/event for Cox
    surv_for_cox = intervals_to_right_or_midpoint(surv_intervals_for_cox, mode=time_mode)

    # 4) fit Cox on original data
    try:
        cph, df_fit = fit_cox(surv_for_cox, cov_df_for_cox, cox_covariates, scale_covariates=scale_covariates)
    except Exception as e:
        raise RuntimeError(f"Cox fitting failed on original data: {e}")

    print("Cox summary (original fit):")
    print(cph.summary)

    # 5) predict LP and survival functions for each patient on a common time grid
    max_time = float(np.ceil(df_measurements[week_col].max()))
    times = np.arange(0.0, max_time + times_step, times_step)

    # predict_survival_matrix expects cov_df indexed by pid with covariates used in fit
    times, S_matrix, pid_list = predict_survival_matrix(cph, cov_df_for_cox, times)
    # S_matrix: shape (n_times, n_patients); pid_list matches cov_df_for_cox.index order

    # compute LP (log hazard ratio) for each patient:
    # Use cov_df_for_cox as provided (indexed by pid)
    part_haz = cph.predict_partial_hazard(cov_df_for_cox)
    LP = np.log(part_haz.values.ravel())

    # pid order we used for clustering / S_matrix
    pid_array = cov_df_for_cox.index.to_list()

    # 6) clustering using bmi, LP
    # For clustering, use covariates from cov_df_for_cox (not the original cov_df which may contain non-numeric)
    # prepare arrays aligned with pid_array
    bmi_arr = cov_df_for_cox['bmi'].astype(float).reindex(pid_array).values
    lp_arr = LP  # already aligned with pid_array

    labels, scaler_cluster, centers = cluster_patients(bmi_arr, lp_arr, k=k_clusters, plot_dendrogram=plot_dendrogram)
    # labels aligned with cov_df_for_cox.index == pid_array

    # create cluster membership mapping
    cluster_assignments = pd.Series(labels, index=pid_array, name='cluster')

    # 7) for each cluster compute S_group (mean S over members) and compute t*
    cluster_results = {}
    for c in range(1, k_clusters + 1):
        members = [pid for pid, lab in cluster_assignments.items() if lab == c]
        if len(members) == 0:
            cluster_results[c] = {'n': 0, 't_star': np.nan, 'S_group': None, 'members': []}
            continue
        # indices of members in pid_array
        idxs = [pid_array.index(pid) for pid in members]
        S_group = S_matrix[:, idxs].mean(axis=1)
        t_star = find_tstar_from_S(times, S_group, alpha=alpha, beta=beta)
        cluster_results[c] = {'n': len(members), 't_star': t_star, 'S_group': S_group, 'members': members}

    print("Original clustering results (per-cluster t*):")
    for c in cluster_results:
        print(f"Cluster {c}: n={cluster_results[c]['n']}, t*={cluster_results[c]['t_star']:.3f} weeks")

    # 8) Bootstrap: 全过程重估计（按孕妇为单位）
    boot_tstars_per_cluster = {c: [] for c in range(1, k_clusters + 1)}
    rng = np.random.RandomState(12345)
    unique_pids_all = df_measurements[pid_col].unique()
    n_pids = len(unique_pids_all)
    print("\nBootstrap (this may take a while) ...")

    # Precompute raw centers of original clusters in raw (unscaled) space for matching
    raw_centers_orig = []   # ✅ 用 list 存放每个簇的中心

    for c in range(1, k_clusters+1):
        members = cluster_results[c].get('members', [])
        if len(members) == 0:
            raw_centers_orig.append(np.array([np.nan, np.nan]))
        else:
            arr = np.vstack([
                cov_df_for_cox.loc[members, 'bmi'].values,
                LP[[pid_array.index(pid) for pid in members]]
            ]).T
            raw_centers_orig.append(np.nanmean(arr, axis=0))

    raw_centers_orig = np.vstack(raw_centers_orig)   # ✅ 最后一次性转为 ndarray



    for b in range(n_boot):
        try:
            sampled_pids = rng.choice(unique_pids_all, size=n_pids, replace=True)
            df_boot = pd.concat([df_measurements[df_measurements[pid_col] == pid] for pid in sampled_pids], ignore_index=True)

            # bootstrap intervals and baseline covariates
            surv_intervals_b = construct_patient_intervals(df_boot, pid_col=pid_col, week_col=week_col, y_col=y_col, threshold=threshold)
            base_b = df_boot.sort_values([pid_col, week_col]).groupby(pid_col).first()
            cov_df_b = base_b[covariate_cols].copy()

            # coerce cox covariates to numeric and drop NaN rows
            cov_df_b_for_cox = cov_df_b.copy()
            for col in cox_covariates:
                cov_df_b_for_cox[col] = pd.to_numeric(cov_df_b_for_cox[col], errors='coerce')
            cov_df_b_for_cox = cov_df_b_for_cox.dropna(subset=cox_covariates)
            if cov_df_b_for_cox.empty:
                # bad bootstrap sample
                continue

            surv_for_cox_b = intervals_to_right_or_midpoint(surv_intervals_b.loc[cov_df_b_for_cox.index], mode=time_mode)

            # fit cox on bootstrap
            try:
                cph_b, _ = fit_cox(surv_for_cox_b, cov_df_b_for_cox, cox_covariates, scale_covariates=scale_covariates)
            except Exception:
                continue

            # predict survival for ORIGINAL cov_df_for_cox patients using bootstrap-fitted model
            try:
                times_b, S_matrix_b, pid_list_b = predict_survival_matrix(cph_b, cov_df_for_cox, times)
            except Exception:
                continue

            part_haz_b = cph_b.predict_partial_hazard(cov_df_for_cox)
            LP_b = np.log(part_haz_b.values.ravel())

            # re-cluster original patients using [bmi, LP_b]
            labels_b, _, _ = cluster_patients(cov_df_for_cox['bmi'].values, LP_b, k=k_clusters, plot_dendrogram=False)

            # compute bootstrap raw centers (unscaled)
            raw_centers_b = []
            for kk in range(1, k_clusters+1):
                maskk = (labels_b == kk)
                if maskk.sum() == 0:
                    raw_centers_b.append(np.array([np.nan, np.nan, np.nan]))
                else:
                    arr = np.vstack([
                        cov_df_for_cox['bmi'].values[maskk],
                        LP_b[maskk]
                    ]).T
                    raw_centers_b.append(np.nanmean(arr, axis=0))
            raw_centers_b = np.vstack(raw_centers_b)

            # if NaNs in centers, skip this bootstrap
            if np.isnan(raw_centers_b).any() or np.isnan(raw_centers_orig).any():
                continue

            # Hungarian matching between orig centers and bootstrap centers
            cost = cdist(raw_centers_orig, raw_centers_b, metric='euclidean')
            try:
                row_ind, col_ind = linear_sum_assignment(cost)
            except Exception:
                continue

            # for each matched pair compute t* (use S_matrix_b and membership in labels_b)
            for i_orig, i_boot in zip(row_ind, col_ind):
                boot_label = i_boot + 1
                members_boot = [pid_array[idx] for idx, lab in enumerate(labels_b) if lab == boot_label]
                if len(members_boot) == 0:
                    boot_tstars_per_cluster[i_orig + 1].append(np.nan)
                    continue
                idxs_boot = [pid_array.index(pid) for pid in members_boot]
                S_group_b = S_matrix_b[:, idxs_boot].mean(axis=1)
                t_star_b = find_tstar_from_S(times_b, S_group_b, alpha=alpha, beta=beta)
                boot_tstars_per_cluster[i_orig + 1].append(t_star_b)

        except Exception:
            # general safe guard: skip this bootstrap iteration on unexpected error
            continue

    # 9) summarize bootstrap results into CI per original cluster
    summary = {}
    for c in range(1, k_clusters + 1):
        arr = np.array([v for v in boot_tstars_per_cluster[c] if (v is not None) and (not np.isnan(v))])
        if arr.size == 0:
            ci_low, ci_high, mean_t = np.nan, np.nan, np.nan
        else:
            ci_low = np.percentile(arr, 2.5)
            ci_high = np.percentile(arr, 97.5)
            mean_t = np.nanmean(arr)
        summary[c] = {
            'orig_tstar': cluster_results[c]['t_star'],
            'n_members': cluster_results[c]['n'],
            'boot_mean_tstar': mean_t,
            'boot_ci_2.5': ci_low,
            'boot_ci_97.5': ci_high,
            'boot_samples_collected': len(boot_tstars_per_cluster[c])
        }

    out = {
        'cph': cph,
        'surv_intervals': surv_intervals,
        'cov_df': cov_df_for_cox,           # numeric-only covariate df used for cox/prediction
        'surv_for_cox': surv_for_cox,
        'times': times,
        'S_matrix': S_matrix,
        'pid_list': pid_array,
        'LP': LP,
        'cluster_assignments': cluster_assignments,
        'cluster_results': cluster_results,
        'bootstrap_summary': summary,
        'boot_tstars_per_cluster': boot_tstars_per_cluster
    }
    return out


def summarize_clusters_BMI_tstar_and_gest(df, results):
    cov_df = results['cov_df']  # index=pid, numeric-only covariates
    cluster_assignments = results['cluster_assignments']
    cluster_results = results['cluster_results']
    bootstrap_summary = results['bootstrap_summary']

    # 原始孕周：从 df_measurements 的第一条记录取 gest_week
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


# ===== 输出 summary 表（显示并保存） =====
results = pipeline_cox_clustering(df)
df_summary = summarize_clusters_BMI_tstar_and_gest(df, results)
print("\n每簇 BMI 区间与推荐 NIPT 时点：")
print(df_summary)
df_summary.to_csv("cluster_bmi_tstar_summary.csv", float_format="%.3f")
print("已保存 cluster_bmi_tstar_summary.csv")

# === 测量误差敏感性分析（对 y_conc 添加高斯噪声） ===
def measurement_error_simulation(df_measurements, pipeline_func, n_sim=100, noise_sds=[0.005, 0.01, 0.02], random_seed=2025, **pipeline_kwargs):
    """
    对每个 noise_sd 做 n_sim 次重复：
      - 在每条记录的 y_conc 上加 N(0, noise_sd^2)
      - 截断 y_conc 到 [0, 1]（或 0 上限）
      - 运行 pipeline_func（应该是 pipeline_cox_clustering 的 wrapper）得到每簇的 t*
    返回 dict: noise_sd -> DataFrame (n_sim x n_clusters) of t* values
    """
    rng = np.random.RandomState(random_seed)
    out = {}
    base_df = df_measurements.copy().reset_index(drop=True)
    for sd in noise_sds:
        tstar_mat = []
        for s in range(n_sim):
            df_sim = base_df.copy()
            noise = rng.normal(loc=0.0, scale=sd, size=len(df_sim))
            df_sim['y_conc'] = df_sim['y_conc'] + noise
            df_sim['y_conc'] = df_sim['y_conc'].clip(lower=0.0)  # 保证非负
            # 运行 pipeline，注意这里我们只需要最终的 t* per cluster
            sim_res = pipeline_func(df_sim, **pipeline_kwargs)
            # sim_res['cluster_results'][c]['t_star'] 可用，但pipeline可能在拟合失败时跳过
            tstars = []
            for c in sorted(sim_res['cluster_results'].keys()):
                t = sim_res['cluster_results'][c].get('t_star', np.nan)
                tstars.append(t if t is not None else np.nan)
            tstar_mat.append(tstars)
        tstar_mat = np.array(tstar_mat)  # shape (n_sim, n_clusters)
        out[sd] = pd.DataFrame(tstar_mat, columns=[f"cluster_{c}" for c in sorted(sim_res['cluster_results'].keys())])
    return out

# === wrapper 用已有 pipeline_cox_clustering 函数 ===
def pipeline_wrapper_for_sim(df_sim, **kwargs):
    kwargs_sim = kwargs.copy()
    # keep n_boot small for simulation speed; but ideally use full n_boot
    kwargs_sim['n_boot'] = kwargs_sim.get('n_boot', 50)  # 不改变用户指定
    return pipeline_cox_clustering(df_sim, **kwargs_sim)


# === 运行模拟（示例参数，视时间可调） ===
noise_sds = [0.005, 0.01, 0.02]   # 例如 0.5%, 1%, 2% 绝对浓度单位（对应 y_conc 为 0.04）
n_sim = 100   # 建议越多越稳健；100为演示
print("\n开始测量误差敏感性分析（这将运行很多次 pipeline，可能耗时）...")
sim_results = measurement_error_simulation(
    df_measurements=df,                          # 原始测序记录 DataFrame（未 aggregate）
    pipeline_func=pipeline_wrapper_for_sim,
    n_sim=n_sim,
    noise_sds=noise_sds,
    random_seed=2025,
    pid_col='pid', week_col='gest_week', y_col='y_conc',
    covariate_cols=['bmi'], cox_covariates=['bmi'],
    threshold=0.04, time_mode='right', k_clusters=3,
    alpha=2.0, beta=1.0, times_step=0.2, n_boot=50,
    scale_covariates=False, plot_dendrogram=False
)

# === 汇总并展示每个 noise level 下每簇 t* 的均值和95%区间 ===
me_err_summary_rows = []
for sd, df_t in sim_results.items():
    for col in df_t.columns:
        arr = df_t[col].dropna().values
        # 找到对应 cluster 的原始孕周
        cluster_num = int(col.split('_')[-1])
        members = results['cluster_results'][cluster_num]['members']
        if len(members) == 0:
            gest_weeks = []
        else:
            # 用原始孕周（未标准化）
            gest_weeks = df.loc[members, 'gest_week'].values

        if arr.size == 0:
            me_err_summary_rows.append({
                'noise_sd': sd,
                'cluster': col,
                'mean_tstar': np.nan,
                'ci2.5': np.nan,
                'ci97.5': np.nan,
                'std_tstar': np.nan,
                'gest_min': np.nan,
                'gest_q1': np.nan,
                'gest_median': np.nan,
                'gest_q3': np.nan,
                'gest_max': np.nan
            })
        else:
            me_err_summary_rows.append({
                'noise_sd': sd,
                'cluster': col,
                'mean_tstar': np.nanmean(arr),
                'ci2.5': np.percentile(arr, 2.5),
                'ci97.5': np.percentile(arr, 97.5),
                'std_tstar': np.nanstd(arr),
                'gest_min': float(np.nanmin(gest_weeks)) if len(gest_weeks) > 0 else np.nan,
                'gest_q1': float(np.nanpercentile(gest_weeks, 25)) if len(gest_weeks) > 0 else np.nan,
                'gest_median': float(np.nanmedian(gest_weeks)) if len(gest_weeks) > 0 else np.nan,
                'gest_q3': float(np.nanpercentile(gest_weeks, 75)) if len(gest_weeks) > 0 else np.nan,
                'gest_max': float(np.nanmax(gest_weeks)) if len(gest_weeks) > 0 else np.nan
            })

me_err_summary = pd.DataFrame(me_err_summary_rows)
print("\n测量误差敏感性结果（每簇 t* 的均值与 95% 区间 + 原始孕周）：")
print(me_err_summary)
me_err_summary.to_csv("measurement_error_sensitivity_with_gestweek.csv", index=False)
print("已保存 measurement_error_sensitivity_with_gestweek.csv")
