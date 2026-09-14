import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# 1. VERİ YÜKLEME
# ─────────────────────────────────────────
incidents  = pd.read_csv("incidents_master.csv")
financial  = pd.read_csv("financial_impact (1).csv")# düzeltildi
market     = pd.read_csv("market_impact.csv")

print("Incidents shape :", incidents.shape)
print("Financial shape :", financial.shape)
print("Market shape    :", market.shape)

# ─────────────────────────────────────────
# 2. BİRLEŞTİRME & HEDEF DEĞİŞKEN
# ─────────────────────────────────────────
merged_df = incidents.merge(financial, on="incident_id", how="left")
print("\nMerged shape:", merged_df.shape)

df = merged_df[merged_df["total_loss_usd"].notnull()].copy()
print("Hedefi olan satır sayısı:", df.shape[0])

median_loss = df["total_loss_usd"].median()
print("Median total_loss_usd:", median_loss)

df["severity"] = (df["total_loss_usd"] > median_loss).map({True: "High", False: "Low"})
print("\nSeverity dağılımı:\n", df["severity"].value_counts())

# ─────────────────────────────────────────
# 3. ÖZELLİK SEÇİMİ
# ─────────────────────────────────────────
selected_features = [
    "company_revenue_usd", "employee_count",
    "country_hq", "industry_primary", "industry_secondary",
    "is_public_company",
    "attack_vector_primary", "attack_vector_secondary", "attack_chain",
    "attributed_group", "attribution_confidence",
    "data_compromised_records", "data_type", "systems_affected",
    "downtime_hours", "data_source_type", "confidence_tier",
    "quality_score", "quality_grade", "review_flag"
]

# Sadece var olan sütunları al
selected_features = [c for c in selected_features if c in df.columns]

X = df[selected_features].copy()
y = df["severity"].copy()
print("\nFeature matrix shape:", X.shape)

# ─────────────────────────────────────────
# 4. TEMİZLEME & ENCODİNG
# ─────────────────────────────────────────
num_cols = X.select_dtypes(include=["int64","float64"]).columns
cat_cols = X.select_dtypes(include=["object","bool"]).columns

X[num_cols] = X[num_cols].fillna(X[num_cols].median())
X[cat_cols] = X[cat_cols].fillna("Unknown")

X = pd.get_dummies(X, drop_first=True)
print("Encoding sonrası shape:", X.shape)

# ─────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ─────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")

# ─────────────────────────────────────────
# 6. MODELLER
# ─────────────────────────────────────────
from sklearn.linear_model        import LogisticRegression
from sklearn.ensemble            import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors           import KNeighborsClassifier
from sklearn.naive_bayes         import GaussianNB
from sklearn.svm                 import SVC
from sklearn.neural_network      import MLPClassifier

def evaluate(name, y_true, y_pred):
    return {
        "Model"    : name,
        "Accuracy" : round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, pos_label="High", zero_division=0), 4),
        "Recall"   : round(recall_score(y_true, y_pred,    pos_label="High", zero_division=0), 4),
        "F1 Score" : round(f1_score(y_true, y_pred,        pos_label="High", zero_division=0), 4),
    }

results = []

# 1. Logistic Regression
lr = LogisticRegression(max_iter=2000, random_state=42)
lr.fit(X_train_s, y_train)
results.append(evaluate("Logistic Regression", y_test, lr.predict(X_test_s)))

# 2. KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_s, y_train)
results.append(evaluate("kNN", y_test, knn.predict(X_test_s)))

# 3. Naive Bayes
nb = GaussianNB()
nb.fit(X_train_s, y_train)
results.append(evaluate("Naive Bayes", y_test, nb.predict(X_test_s)))

# 4. Linear SVM
svm_l = SVC(kernel="linear", random_state=42)
svm_l.fit(X_train_s, y_train)
results.append(evaluate("Linear SVM", y_test, svm_l.predict(X_test_s)))

# 5. RBF SVM
svm_r = SVC(kernel="rbf", gamma="scale", random_state=42)
svm_r.fit(X_train_s, y_train)
results.append(evaluate("RBF SVM", y_test, svm_r.predict(X_test_s)))

# 6. Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
results.append(evaluate("Random Forest", y_test, rf.predict(X_test)))

# 7. MLP
mlp = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, early_stopping=False, random_state=42)
mlp.fit(X_train_s, y_train.to_numpy())
results.append(evaluate("MLP", y_test, mlp.predict(X_test_s)))

# 8. Gradient Boosting (XGBoost yerine — network erişimi yok)
gb = GradientBoostingClassifier(n_estimators=200, random_state=42)
gb.fit(X_train, y_train)
results.append(evaluate("Gradient Boosting", y_test, gb.predict(X_test)))

# ─────────────────────────────────────────
# 7. SONUÇ TABLOSU
# ─────────────────────────────────────────
results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False)
print("\n" + "="*65)
print("MODEL KARŞILAŞTIRMA TABLOSU")
print("="*65)
print(results_df.to_string(index=False))

best = results_df.iloc[0]
print(f"\n✅ EN İYİ MODEL: {best['Model']}")
print(f"   Accuracy={best['Accuracy']}  Precision={best['Precision']}  Recall={best['Recall']}  F1={best['F1 Score']}")

# CSV olarak kaydet
results_df.to_csv("model_results.csv", index=False)
print("\nSonuçlar 'model_results.csv' olarak kaydedildi.")

# ─────────────────────────────────────────
# 8. FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────
importances = rf.feature_importances_
top10_idx   = np.argsort(importances)[::-1][:10]

print("\n--- TOP 10 ÖZELLİK (Random Forest) ---")
for rank, i in enumerate(top10_idx, 1):
    print(f"  {rank:2}. {X.columns[i]:45s} {importances[i]:.4f}")

feat_imp_df = pd.DataFrame({
    "Feature"   : X.columns[top10_idx],
    "Importance": importances[top10_idx]
})
feat_imp_df.to_csv("feature_importance.csv", index=False)
print("Feature importance 'feature_importance.csv' olarak kaydedildi.")