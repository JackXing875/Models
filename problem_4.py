import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report
import xgboost as xgb
import shap
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score

# 数据加载与预处理
def load_and_preprocess_data(file_path):

    data = pd.read_excel(file_path, sheet_name="女胎检测数据")
    
    def convert_gestational_weeks(week_str):
        try:
            if isinstance(week_str, str) and "w" in week_str:
                parts = week_str.replace("w", "").split("+")
                weeks = float(parts[0])
                days = float(parts[1]) if len(parts) > 1 and parts[1] != "" else 0
                return weeks + days / 7
            elif isinstance(week_str, (int, float)):
                return float(week_str)
            else:
                return np.nan
        except:
            return np.nan
    
    data["gestational_weeks_numeric"] = data["检测孕周"].apply(convert_gestational_weeks)
    data['is_abnormal'] = data['染色体的非整倍体'].apply(
        lambda x: 1 if str(x).startswith('T') else 0
    )
    
    return data

# 特征工程
def create_features(data, features):

    data_eng = data.copy()
    
    if '原始读段数' in features:
        for chr in [13, 18, 21, 'X']:
            z_col = f'z_chr{chr}'
            if z_col in features:
                data_eng[f'{z_col}_squared'] = data_eng[z_col] ** 2
                features.append(f'{z_col}_squared')
    
    if 'GC含量' in features and '原始读段数' in features:
        data_eng['gc_reads_interaction'] = data_eng['GC含量'] * data_eng['原始读段数']
        features.append('gc_reads_interaction')
        
    if all(col in features for col in ['21号染色体的z值', '18号染色体的z值', '13号染色体的z值']):
        data_eng['z_score_ratio'] = data_eng['21号染色体的z值'] / (data_eng['18号染色体的z值'] + data_eng['13号染色体的z值'] + 1e-6)
        features.append('z_score_ratio')
        
    return data_eng, features

# 模型训练与评估
def train_and_evaluate_model(X, y):

    # 数据标准化
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    # 处理类别不平衡
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_normalized, y)
    
    # 基础分类器
    base_classifiers = [
        ('random_forest', RandomForestClassifier(
            n_estimators=90, max_depth=2, class_weight='balanced', random_state=42
        )),
        ('gradient_boosting', GradientBoostingClassifier(
            n_estimators=90, max_depth=2, random_state=42
        )),
        ('support_vector', SVC(
            kernel='rbf', probability=True, class_weight='balanced', random_state=42
        )),
        ('xgboost', xgb.XGBClassifier(
            n_estimators=90, max_depth=2, learning_rate=0.01, random_state=42
        ))
    ]
    
    # 生成元特征
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    meta_features = np.zeros((X_balanced.shape[0], len(base_classifiers)))
    
    for idx, (name, clf) in enumerate(base_classifiers):
        fold_preds = np.zeros(X_balanced.shape[0])
        for train_idx, val_idx in cv.split(X_balanced, y_balanced):
            clf.fit(X_balanced[train_idx], y_balanced[train_idx])
            fold_preds[val_idx] = clf.predict_proba(X_balanced[val_idx])[:, 1]
        meta_features[:, idx] = fold_preds
    
    # 元分类器
    meta_clf = LogisticRegression(
        penalty='l2', class_weight='balanced', solver='liblinear', max_iter=1000
    )
    meta_clf.fit(meta_features, y_balanced)
    
    # 测试集评估
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=0.2, stratify=y, random_state=42
    )
    
    test_meta_features = np.zeros((X_test.shape[0], len(base_classifiers)))
    for idx, (name, clf) in enumerate(base_classifiers):
        clf.fit(X_balanced, y_balanced)
        test_meta_features[:, idx] = clf.predict_proba(X_test)[:, 1]
    
    test_probs = meta_clf.predict_proba(test_meta_features)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)
    
    # 计算指标
    auc_score = roc_auc_score(y_test, test_probs)
    pr_auc = average_precision_score(y_test, test_probs)
    precision, recall, thresholds = precision_recall_curve(y_test, test_probs)
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    cm = confusion_matrix(y_test, test_preds)
    cr = classification_report(y_test, test_preds)
    
    return {
        'auc_score': auc_score,
        'pr_auc': pr_auc,
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'fpr': fpr,
        'tpr': tpr,
        'cm': cm,
        'cr': cr,
        'test_probs': test_probs,
        'test_preds': test_preds,
        'y_test': y_test,
        'base_classifiers': base_classifiers,
        'X_test': X_test
    }

# 可视化函数
def setup_plot_style():

    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    plt.rcParams['font.size'] = 12
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

def plot_performance_curves(results, save_path='performance_curves.png'):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # ROC曲线
    ax1.plot(results['fpr'], results['tpr'], 
             label=f'AUC = {results["auc_score"]:.3f}', 
             color='#FF6B6B', linewidth=2.5)
    ax1.fill_between(results['fpr'], results['tpr'], alpha=0.2, color='#FF6B6B')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1.5)
    ax1.set_xlabel('假阳性率', fontweight='bold')
    ax1.set_ylabel('真阳性率', fontweight='bold')
    ax1.set_title('ROC曲线', fontweight='bold', pad=20)
    ax1.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # PR曲线
    ax2.plot(results['recall'], results['precision'], 
             label=f'AUC = {results["pr_auc"]:.3f}', 
             color='#4ECDC4', linewidth=2.5)
    ax2.fill_between(results['recall'], results['precision'], alpha=0.2, color='#4ECDC4')
    ax2.set_xlabel('召回率', fontweight='bold')
    ax2.set_ylabel('精确率', fontweight='bold')
    ax2.set_title('精确率-召回率曲线', fontweight='bold', pad=20)
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['正常', '异常'], yticklabels=['正常', '异常'],
                annot_kws={"size": 14, "weight": "bold"},
                cbar_kws={"shrink": 0.8})
    plt.title('混淆矩阵', fontweight='bold', pad=20)
    plt.ylabel('真实标签', fontweight='bold')
    plt.xlabel('预测标签', fontweight='bold')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_threshold_analysis(precision, recall, thresholds, save_path='threshold_analysis.png'):

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_f1_idx]
    
    target_recall = 0.8
    target_recall_idx = np.where(recall >= target_recall)[0][0]
    target_threshold = thresholds[target_recall_idx]
    
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, precision[:-1], "b-", label="精确率", linewidth=3, alpha=0.8)
    plt.plot(thresholds, recall[:-1], "g-", label="召回率", linewidth=3, alpha=0.8)
    plt.plot(thresholds, f1_scores[:-1], "r-", label="F1分数", linewidth=3, alpha=0.8)

    plt.xlabel("分类阈值", fontweight='bold')
    plt.ylabel("评分", fontweight='bold')
    plt.title("不同阈值下的模型性能", fontweight='bold', pad=20)
    plt.legend(loc="center right", frameon=True, fancybox=True, shadow=True)

    plt.axvline(x=optimal_threshold, color='r', linestyle='--', 
                label=f'最佳F1阈值 ({optimal_threshold:.2f})', linewidth=2)
    plt.axvline(x=target_threshold, color='g', linestyle='--', 
                label=f'80%召回率阈值 ({target_threshold:.2f})', linewidth=2)

    plt.axvspan(optimal_threshold-0.05, optimal_threshold+0.05, alpha=0.1, color='red')
    plt.axvspan(target_threshold-0.05, target_threshold+0.05, alpha=0.1, color='green')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return optimal_threshold, target_threshold

def plot_probability_distribution(probs, y_test, save_path='probability_distribution.png'):

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.hist(probs[y_test == 0], bins=30, alpha=0.7, color='green', label='正常样本')
    plt.xlabel('预测概率')
    plt.ylabel('频数')
    plt.title('正常样本的预测概率分布')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(probs[y_test == 1], bins=30, alpha=0.7, color='red', label='异常样本')
    plt.xlabel('预测概率')
    plt.ylabel('频数')
    plt.title('异常样本的预测概率分布')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# SHAP分析
def perform_shap_analysis(model, X_test, feature_names, save_prefix='shap'):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # 特征重要性
    plt.figure(figsize=(14, 10))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, 
                      plot_type="bar", show=False, color='#45B7D1')
    plt.title("基于SHAP值的特征重要性排序", fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 特征影响
    plt.figure(figsize=(14, 10))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False, 
                      cmap=plt.get_cmap("coolwarm"), alpha=0.7)
    plt.title("特征值与SHAP值的关系", fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 依赖图
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_features_indices = np.argsort(mean_abs_shap)[-5:]
    
    for idx in top_features_indices:
        # 不再手动创建figure
        shap.dependence_plot(idx, shap_values, X_test, 
                             feature_names=feature_names, show=False,
                             alpha=0.7, dot_size=16)
        plt.title(f"SHAP依赖图 - {feature_names[idx]}", fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_dependence_{feature_names[idx]}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return explainer, shap_values

#瀑布图
def plot_waterfall_chart(shap_values, selected_features, explainer, X_test, sample_idx=0, abnormal_indices=[]):

    if len(abnormal_indices) > 0:
        shap.waterfall_plot(    
        shap.Explanation(
            values=shap_values[sample_idx],
            base_values=explainer.expected_value,  # 使用传入的explainer
            data=X_test[sample_idx],  # 使用传入的X_test
            feature_names=selected_features
        ),
        show=False,
        max_display=12  # 只显示最重要的12个特征
    )
    plt.title(f"异常样本SHAP瀑布图 (索引: {sample_idx})", fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('shap_waterfall_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

# 主函数
def main():

    # 加载数据
    data_file = "D:\desktop\建模\附件.xlsx"
    female_data = load_and_preprocess_data(data_file)
    
    # 特征选择
    selected_features = [
        '年龄', '身高', '体重', '孕妇BMI', '原始读段数',
        '在参考基因组上比对的比例', '重复读段的比例', '唯一比对的读段数',
        'GC含量',
        '13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值',
        'X染色体浓度',
        '13号染色体的GC含量', '18号染色体的GC含量', '21号染色体的GC含量'
    ]
    
    # 特征工程
    female_data, selected_features = create_features(female_data, selected_features)
    
    # 准备数据
    X_data = female_data[selected_features].fillna(female_data[selected_features].median())
    y_labels = female_data['is_abnormal']
    
    # 训练和评估模型
    results = train_and_evaluate_model(X_data, y_labels)
    
    # 打印结果
    print(f"ROC AUC评分: {results['auc_score']:.3f}")
    print("混淆矩阵:\n", results['cm'])
    print("分类报告:\n", results['cr'])
    
    # 设置绘图样式
    setup_plot_style()
    
    # 绘制性能曲线
    plot_performance_curves(results)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(results['cm'])
    
    # 阈值分析
    optimal_threshold, target_threshold = plot_threshold_analysis(
        results['precision'], results['recall'], results['thresholds']
    )
    
    print(f"\n基于F1分数的最佳阈值: {optimal_threshold:.3f}")
    print(f"达到80%召回率所需的阈值: {target_threshold:.3f}")
    
    # 使用新阈值进行预测
    adjusted_preds = (results['test_probs'] >= target_threshold).astype(int)
    print("\n使用新阈值(80%召回率)后的性能:")
    print(classification_report(results['y_test'], adjusted_preds))
    
    adjusted_preds = (results['test_probs'] >= optimal_threshold).astype(int)
    print("\n使用新阈值(F1分数最佳)后的性能:")
    print(classification_report(results['y_test'], adjusted_preds))
    
    # 绘制概率分布
    plot_probability_distribution(results['test_probs'], results['y_test'])
    
    # SHAP分析
    xgb_model = results['base_classifiers'][3][1]
    explainer, shap_values = perform_shap_analysis(xgb_model, results['X_test'], selected_features)
    
    # 决策图
    plt.figure(figsize=(14, 10))
    shap.decision_plot(
        explainer.expected_value, 
        shap_values[:20], 
        selected_features,
        show=False,
        highlight=0,
        feature_order='importance',
        feature_display_range=slice(-10, None)
    )
    plt.title("SHAP决策图 (前20个样本)", fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('shap_decision_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 瀑布图
    plot_waterfall_chart(shap_values, selected_features, explainer, results['X_test'], sample_idx=3, abnormal_indices=[3])
    
    print("\n=== 分析完成 ===")

if __name__ == "__main__":
    main()