import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import string

# ==========================================
# STEP 0: SETUP
# Rishi Pawani 5551
# ==========================================
print("Loading Data...")
# Make sure 'Global Health Insights.csv' is in the same folder as this script
df = pd.read_csv("Global Health Insights.csv")
pd.set_option('display.max_columns', None)
print("Data Loaded Successfully!\n")

# ==========================================
# 1. RENAMING COLUMN
# ==========================================
print("--- Op 1: Renaming Column ---")
print("Before:", [col for col in df.columns if 'vitamin_d' in col])
df.rename(columns={'condition_vitamin_d_deficiency_pct': 'Vit_D_Deficiency'}, inplace=True)
print("After:", [col for col in df.columns if 'Vit_D' in col])
print("-" * 30)

# ==========================================
# 2. DATATYPE CONVERSION
# ==========================================
print("--- Op 2: Datatype Conversion ---")
print("Before Dtype:", df['period'].dtype)
df['period'] = pd.to_datetime(df['period'])
print("After Dtype:", df['period'].dtype)
print("-" * 30)

# ==========================================
# 3. DETECTING OUTLIERS
# ==========================================
print("--- Op 3: Detecting Outliers ---")
Q1 = df['monthly_analyses'].quantile(0.25)
Q3 = df['monthly_analyses'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['monthly_analyses'] < (Q1 - 1.5 * IQR)) | (df['monthly_analyses'] > (Q3 + 1.5 * IQR))]
print(f"Number of outliers detected in 'monthly_analyses': {len(outliers)}")
print("-" * 30)

# ==========================================
# 4. SCALING DATA (MinMax)
# Rishi Pawani 5551
# ==========================================
print("--- Op 4: Scaling Data ---")
scaler_minmax = MinMaxScaler()
df['unique_users_scaled'] = scaler_minmax.fit_transform(df[['unique_users']])
print(df[['unique_users', 'unique_users_scaled']].head(3))
print("-" * 30)

# ==========================================
# 5. NORMALIZATION (StandardScaler)
# ==========================================
print("--- Op 5: Normalization ---")
scaler_std = StandardScaler()
df['avg_health_score_norm'] = scaler_std.fit_transform(df[['avg_health_score']])
print(df[['avg_health_score', 'avg_health_score_norm']].head(3))
print("-" * 30)

# ==========================================
# 6. BINNING
# ==========================================
print("--- Op 6: Binning ---")
bins = [0, 70, 73, 100]
labels = ['Low', 'Medium', 'High']
df['health_score_category'] = pd.cut(df['avg_health_score'], bins=bins, labels=labels)
print(df[['avg_health_score', 'health_score_category']].head(3))
print("-" * 30)

# ==========================================
# 7. AGGREGATION
# ==========================================
print("--- Op 7: Aggregation ---")
agg_data = df.groupby('region')['monthly_analyses'].mean()
print("Mean Monthly Analyses by Region:")
print(agg_data)
print("-" * 30)

# ==========================================
# 8. MEAN
# Rishi Pawani 5551
# ==========================================
print("--- Op 8: Mean ---")
mean_val = df['avg_biomarkers_per_test'].mean()
print(f"Mean Biomarkers per Test: {mean_val:.2f}")
print("-" * 30)

# ==========================================
# 9. MEDIAN
# ==========================================
print("--- Op 9: Median ---")
median_val = df['monthly_analyses'].median()
print(f"Median Monthly Analyses: {median_val}")
print("-" * 30)

# ==========================================
# 10. MODE
# ==========================================
print("--- Op 10: Mode ---")
mode_val = df['top_abnormal_biomarker_1'].mode()[0]
print(f"Most Frequent Abnormal Biomarker: {mode_val}")
print("-" * 30)

# ==========================================
# 11, 12, 13. PLOTS
# Rishi Pawani 5551
# ==========================================
print("--- Ops 11, 12, 13: Generating Plots... ---")
plt.figure(figsize=(15, 5))

# Op 11: Histogram
plt.subplot(1, 3, 1)
plt.hist(df['avg_health_score'], bins=10, color='skyblue', edgecolor='black')
plt.title('Op 11: Histogram (Health Score)')
plt.xlabel('Score')

# Op 12: Boxplot
plt.subplot(1, 3, 2)
sns.boxplot(x=df['monthly_analyses'])
plt.title('Op 12: Boxplot (Monthly Analyses)')

# Op 13: Scatter Plot
plt.subplot(1, 3, 3)
plt.scatter(df['avg_health_score'], df['avg_nutrient_score'], alpha=0.5)
plt.title('Op 13: Scatter (Health vs Nutrient)')
plt.xlabel('Health Score')
plt.ylabel('Nutrient Score')

plt.tight_layout()
print("Displaying plots now (Close window to continue)...")
plt.show() 
print("-" * 30)

# ==========================================
# 14. ONE HOT ENCODING
# ==========================================
print("--- Op 14: One Hot Encoding ---")
# Using prefix to easily identify new columns
df = pd.get_dummies(df, columns=['region'], prefix='region')
print("New columns created:", [col for col in df.columns if 'region_' in col])
print("-" * 30)

# ==========================================
# 15. LABEL ENCODING
# Rishi Pawani 5551
# ==========================================
print("--- Op 15: Label Encoding ---")
le = LabelEncoder()
df['sub_region_encoded'] = le.fit_transform(df['sub_region'])
print(df[['sub_region', 'sub_region_encoded']].head(3))
print("-" * 30)

# ==========================================
# 16. FEATURE EXTRACTION
# ==========================================
print("--- Op 16: Feature Extraction ---")
df['Total_High_Risk'] = df['risk_elevated_pct'] + df['risk_critical_pct']
print(df[['risk_elevated_pct', 'risk_critical_pct', 'Total_High_Risk']].head(3))
print("-" * 30)

# ==========================================
# 17. DIMENSIONALITY REDUCTION (PCA)
# ==========================================
print("--- Op 17: PCA ---")
# Selecting condition columns and the renamed vitamin D column
cols = [c for c in df.columns if 'condition_' in c] + ['Vit_D_Deficiency']
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df[cols])
df['pca_1'] = pca_result[:, 0]
df['pca_2'] = pca_result[:, 1]
print(df[['pca_1', 'pca_2']].head(3))
print("-" * 30)

# ==========================================
# 18. FEATURE CONSTRUCTION (CONCATENATION)
# Rishi Pawani 5551
# ==========================================
print("--- Op 18: Feature Construction ---")
df['all_biomarkers_text'] = (
    df['top_abnormal_biomarker_1'] + " " +
    df['top_abnormal_biomarker_2'] + " " +
    df['top_abnormal_biomarker_3']
)
print("Combined Text Sample:", df['all_biomarkers_text'].iloc[0])
print("-" * 30)

# ==========================================
# 19. LOWERCASING
# ==========================================
print("--- Op 19: Lowercasing ---")
df['all_biomarkers_lower'] = df['all_biomarkers_text'].str.lower()
print("Lowercased Sample:", df['all_biomarkers_lower'].iloc[0])
print("-" * 30)

# ==========================================
# 20. TOKENIZATION
# ==========================================
print("--- Op 20: Tokenization ---")
df['biomarker_tokens'] = df['all_biomarkers_lower'].apply(lambda x: x.split())
print("Tokens Sample:", df['biomarker_tokens'].iloc[0])
print("-" * 30)

print("\nAll 20 Operations Completed Successfully!")
