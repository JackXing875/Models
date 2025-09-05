import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import re

class CleanData:
    def __init__(self, path, sheet="男胎检测数据"):
        self.df = pd.read_excel(path, sheet_name=sheet)
        self.rename_columns() 

    def rename_columns(self):
        """变量映射（统一列名）"""
        rename_dict = {
            "序号": "idx",
            "孕妇代码": "pid",
            "年龄": "age",
            "身高": "height",
            "体重": "weight",
            "末次月经": "lmp",
            "IVF妊娠": "ivf",
            "检测日期": "test_date",
            "检测抽血次数": "visit",
            "检测孕周": "gest_week_raw",
            "孕妇BMI": "bmi",
            "原始读段数": "reads_total",
            "在参考基因组上比对的比例": "map_ratio",
            "重复读段的比例": "dup_ratio",
            "唯一比对的读段数  ": "reads_unique",
            "GC含量": "gc_ratio",
            "13号染色体的Z值": "z_chr13",
            "18号染色体的Z值": "z_chr18",
            "21号染色体的Z值": "z_chr21",
            "X染色体的Z值": "z_chrX",
            "Y染色体的Z值": "z_chrY",
            "Y染色体浓度": "y_conc",
            "X染色体浓度": "x_conc",
            "13号染色体的GC含量": "gc_chr13",
            "18号染色体的GC含量": "gc_chr18",
            "21号染色体的GC含量": "gc_chr21",
            "被过滤掉读段数的比例": "filter_ratio",
            "染色体的非整倍体": "aneuploidy",
            "怀孕次数": "gravidity",
            "生产次数": "parity",
            "胎儿是否健康": "fetus_health",
        }
        self.df = self.df.rename(columns=rename_dict)

    def parse_gestational_age(self, s: str):
        """孕周字符串转为小数"""
        try:
            if isinstance(s, str) and "w+" in s:
                week, day = s.split("w+")
                return float(int(week) + int(day) / 7)
            elif isinstance(s, str) and "w" in s:
                return int(s.replace("w", ""))
            else:
                return float(s) if pd.notna(s) else np.nan
        except:
            return np.nan

    def gestational_age(self):
        """处理孕周数据"""
        self.df["gest_week"] = self.df["gest_week_raw"].apply(self.parse_gestational_age)
        
    def process_categorical_vars(self):
        """处理分类变量"""
        # IVF妊娠转为二进制
        self.df["ivf_binary"] = (self.df["ivf"] == "IVF妊娠").astype(int)
        
        # 胎儿是否健康转为二进制
        self.df["healthy_binary"] = (self.df["fetus_health"] == "是").astype(int)
        
        # 处理怀孕次数中的特殊值
        self.df["gravidity"] = self.df["gravidity"].replace("≥3", 3)
        self.df["gravidity"] = pd.to_numeric(self.df["gravidity"], errors='coerce')
        
    def create_time_features(self):
        """创建时间相关特征"""
        # 检测日期转为datetime
        self.df["test_date"] = pd.to_datetime(self.df["test_date"], errors='coerce')
        self.df["lmp"] = pd.to_datetime(self.df["lmp"], errors='coerce')
        
        # 计算实际检测孕周（从末次月经开始）
        if pd.notna(self.df["lmp"]).any() and pd.notna(self.df["test_date"]).any():
            self.df["actual_gest_week"] = (self.df["test_date"] - self.df["lmp"]).dt.days / 7
        
        # 创建孕妇检测次数特征
        visit_count = self.df.groupby("pid")["visit"].transform('count')
        self.df["total_visits"] = visit_count
        
    def create_technical_features(self):
        """创建技术指标特征"""
        # 有效读段比例
        self.df["effective_read_ratio"] = self.df["reads_unique"] / self.df["reads_total"]
        
        # GC偏差（样本GC与参考GC的差异）
        self.df["gc_bias"] = self.df["gc_ratio"] - 0.41  # 参考基因组GC含量约41%
        
        # 染色体Z值绝对值（异常检测用）
        for chr in [13, 18, 21, 'X', 'Y']:
            self.df[f"abs_z_chr{chr}"] = np.abs(self.df[f"z_chr{chr}"])
        
        # Y/X染色体浓度比
        self.df["y_x_ratio"] = self.df["y_conc"] / (self.df["x_conc"] + 1e-6)
        
    def missing_data(self, method="mice"):
        """缺失值处理：MICE 或 median, 不直接插补reads_total/read_unique"""
        # 可插补的数值列（排除 reads_total 和 reads_unique）
        num_cols = [
            "age", "height", "weight", "bmi", "gc_ratio",
            "map_ratio", "dup_ratio", "filter_ratio",
            "y_conc", "x_conc", "z_chrX", "z_chrY", "z_chr21", "z_chr18", "z_chr13",
            "gc_chr13", "gc_chr18", "gc_chr21", "gest_week", "gravidity", "parity"
        ]

        if method == "mice":
            try:
                imp = IterativeImputer(random_state=42, max_iter=10)
                self.df[num_cols] = imp.fit_transform(self.df[num_cols])
            except Exception as e:
                print("MICE 失败，退化为中位数填补:", e)
                for col in num_cols:
                    self.df[col] = self.df[col].fillna(self.df[col].median())
        else:
            for col in num_cols:
                self.df[col] = self.df[col].fillna(self.df[col].median())

        # 分类变量用众数填补
        cat_cols = ["ivf", "fetus_health"]
        for col in cat_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else "未知")



    def outlier_detection(self):
        """异常值检测并标记低质量读段"""
        # 技术指标异常值
        tech_cols = ["reads_total", "map_ratio", "dup_ratio", "filter_ratio"]
        for col in tech_cols:
            if col == "reads_total":
                # 低质量读段标记
                self.df["low_quality_reads"] = (self.df[col] < self.df[col].quantile(0.01)).astype(int)
            Q1, Q3 = self.df[col].quantile([0.01, 0.99])
            IQR = Q3 - Q1
            low, high = Q1 - 3 * IQR, Q3 + 3 * IQR
            self.df[f"{col}_tech_outlier"] = ((self.df[col] < low) | (self.df[col] > high)).astype(int)

        # 生物学指标异常值
        bio_cols = ["y_conc", "x_conc", "z_chr13", "z_chr18", "z_chr21", "z_chrX", "z_chrY"]
        for col in bio_cols:
            Q1, Q3 = self.df[col].quantile([0.05, 0.95])
            IQR = Q3 - Q1
            low, high = Q1 - 2.5 * IQR, Q3 + 2.5 * IQR
            self.df[f"{col}_bio_outlier"] = ((self.df[col] < low) | (self.df[col] > high)).astype(int)

        # 标记极端Z值（可能指示非整倍体）
        for chr in [13, 18, 21]:
            self.df[f"z_chr{chr}_extreme"] = (np.abs(self.df[f"z_chr{chr}"]) > 3).astype(int)

        # GC异常
        self.df["gc_outlier"] = ~self.df["gc_ratio"].between(0.39, 0.61)

        # 最终的异常列集合
        outlier_cols = [col for col in self.df.columns if "_outlier" in col] + ["low_quality_reads"]

        # ⚠️ 不直接 drop，保留标记，敏感性分析时可用：
        # self.df_clean = self.df[~self.df[outlier_cols].any(axis=1)].reset_index(drop=True)

   

    def create_interaction_features(self):
        """创建交互特征"""
        # 孕周与读段数的交互
        self.df["gest_week_reads_interaction"] = self.df["gest_week"] * self.df["reads_total"]
        
        # BMI与年龄的交互
        self.df["bmi_age_interaction"] = self.df["bmi"] * self.df["age"]
        
        # Y浓度与孕周的交互
        self.df["y_conc_gest_interaction"] = self.df["y_conc"] * self.df["gest_week"]

    def split_data(self, test_size=0.2, val_size=0.2, random_state=42):
        """按孕妇分层的训练/验证/测试集划分"""
        unique_pids = self.df["pid"].unique()
        
        # 先分测试集
        train_val_pids, test_pids = train_test_split(
            unique_pids, test_size=test_size, random_state=random_state
        )
        
        # 再从训练集中分验证集
        train_pids, val_pids = train_test_split(
            train_val_pids, test_size=val_size/(1-test_size), random_state=random_state
        )
        
        train_df = self.df[self.df["pid"].isin(train_pids)]
        val_df = self.df[self.df["pid"].isin(val_pids)]
        test_df = self.df[self.df["pid"].isin(test_pids)]
        
        return train_df, val_df, test_df

    def get_feature_sets(self):
        """返回不同类型的特征集合，便于模型选择"""
        basic_features = ["age", "height", "weight", "bmi", "gravidity", "parity", "ivf_binary"]
        
        technical_features = [
            "reads_total", "reads_unique", "map_ratio", "dup_ratio", 
            "filter_ratio", "gc_ratio", "effective_read_ratio", "gc_bias"
        ]
        
        chromosomal_features = [
            "z_chr13", "z_chr18", "z_chr21", "z_chrX", "z_chrY",
            "abs_z_chr13", "abs_z_chr18", "abs_z_chr21", "abs_z_chrX", "abs_z_chrY",
            "y_conc", "x_conc", "y_x_ratio"
        ]
        
        time_features = ["gest_week", "visit", "total_visits"]
        
        interaction_features = [
            "gest_week_reads_interaction", "bmi_age_interaction", "y_conc_gest_interaction"
        ]
        
        outlier_features = [col for col in self.df.columns if '_outlier' in col or '_extreme' in col]
        
        return {
            'basic': basic_features,
            'technical': technical_features,
            'chromosomal': chromosomal_features,
            'time': time_features,
            'interaction': interaction_features,
            'outlier': outlier_features,
            'all': basic_features + technical_features + chromosomal_features + 
                  time_features + interaction_features + outlier_features
        }

    def process_all(self):
        """执行完整的预处理流程"""
        print("开始数据预处理...")
        
        # 1. 处理孕周
        self.gestational_age()
        print("✓ 孕周处理完成")
        
        # 2. 处理分类变量
        self.process_categorical_vars()
        print("✓ 分类变量处理完成")
        
        # 3. 创建时间特征
        self.create_time_features()
        print("✓ 时间特征创建完成")
        
        # 4. 创建技术特征
        self.create_technical_features()
        print("✓ 技术特征创建完成")
        
        # 5. 缺失值处理
        self.missing_data(method="mice")
        print("✓ 缺失值处理完成")
        
        # 6. 异常值检测
        self.outlier_detection()
        print("✓ 异常值检测完成")
        

        
        # 8. 创建交互特征
        self.create_interaction_features()
        print("✓ 交互特征创建完成")
        
        # 9. 按孕妇排序
        self.df = self.df.sort_values(by=["pid", "gest_week"])
        print("✓ 数据排序完成")
        
        print(f"预处理完成，最终数据形状: {self.df.shape}")
        
        return self.df

if __name__ == "__main__":
    # 文件路径
    path = r"D:\RUC\国赛\附件.xlsx"

    # 1. 实例化
    cleaner = CleanData(path, sheet="男胎检测数据")
    
    # 2. 执行完整预处理
    processed_df = cleaner.process_all()

    processed_df.to_csv('清理后的数据.csv', index=False, encoding='utf-8-sig')
    
    # 3. 获取特征集合
    feature_sets = cleaner.get_feature_sets()
    print("\n可用特征集合:")
    for key, features in feature_sets.items():
        print(f"{key}: {len(features)}个特征")
    
    # 4. 数据划分
    train_df, val_df, test_df = cleaner.split_data(test_size=0.2, val_size=0.2)
    
    print(f"\n数据划分结果:")
    print(f"训练集: {train_df.shape} ({(len(train_df)/len(processed_df)*100):.1f}%)")
    print(f"验证集: {val_df.shape} ({(len(val_df)/len(processed_df)*100):.1f}%)")
    print(f"测试集: {test_df.shape} ({(len(test_df)/len(processed_df)*100):.1f}%)")
    
    print(processed_df.columns)
