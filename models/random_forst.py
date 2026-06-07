# ── Rico Bullock — Random Forest ──────────────────────────────────────────────
df_rf = df.copy()
# Feature Engineering
df_rf["Return"]      = df_rf["Close"].pct_change()
df_rf["HL_Range"]    = (df_rf["High"] - df_rf["Low"]) / df_rf["Close"]
df_rf["OC_Move"]     = (df_rf["Close"] - df_rf["Open"]) / df_rf["Open"]

df_rf["MA5"]         = df_rf["Close"].rolling(5).mean()
df_rf["MA20"]        = df_rf["Close"].rolling(20).mean()
df_rf["Close_MA5"]   = df_rf["Close"] / df_rf["MA5"] - 1
df_rf["Close_MA20"]  = df_rf["Close"] / df_rf["MA20"] - 1
df_rf["MA5_MA20"]    = df_rf["MA5"] / df_rf["MA20"] - 1

df_rf["Vol5"]        = df_rf["Return"].rolling(5).std()
df_rf["Vol10"]       = df_rf["Return"].rolling(10).std()

df_rf["Vol_MA5"]     = df_rf["Volume"].rolling(5).mean()
df_rf["Vol_Ratio"]   = df_rf["Volume"] / df_rf["Vol_MA5"]
df_rf["Up_day"]      = (df_rf["Close"] > df_rf["Open"]).astype(int)
df_rf["Buy_Vol"]     = df_rf["Volume"] * df_rf["Up_day"]
df_rf["Sell_Vol"]    = df_rf["Volume"] * (1 - df_rf["Up_day"])
df_rf["Vol_Imbalance"] = (df_rf["Buy_Vol"] - df_rf["Sell_Vol"]) / (df_rf["Buy_Vol"] + df_rf["Sell_Vol"])

df_rf["Ret_lag1"]    = df_rf["Return"].shift(1)
df_rf["Ret_lag2"]    = df_rf["Return"].shift(2)
df_rf["Ret_lag3"]    = df_rf["Return"].shift(3)
df_rf["Ret_lag5"]    = df_rf["Return"].shift(5)

df_rf["Target"]      = (df_rf["Close"].shift(-1) > df_rf["Close"]).astype(int)
df_rf.dropna(inplace=True)
df_rf.reset_index(drop=True, inplace=True)

# Features, X, y
features = ["Return", "HL_Range", "OC_Move",
            "Close_MA5", "Close_MA20", "MA5_MA20",
            "Vol5", "Vol10", "Vol_Ratio",
            "Ret_lag1", "Ret_lag2", "Ret_lag3", "Ret_lag5",
            "Vol_Imbalance"]

X = df_rf[features].values
y = df_rf["Target"].values

# Train/Validation/Test Split
train_mask = df_rf["Date"] < "2024-01-01"
valid_mask = (df_rf["Date"] >= "2024-01-01") & (df_rf["Date"] < "2025-01-01")
test_mask  = df_rf["Date"] >= "2025-01-01"

X_train, y_train = X[train_mask], y[train_mask]
X_valid, y_valid = X[valid_mask], y[valid_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

# Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay,
                              roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Baseline Random Forest
rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
rf_base.fit(X_train, y_train)
print(f"Baseline train accuracy:      {accuracy_score(y_train, rf_base.predict(X_train)):.4f}")
print(f"Baseline validation accuracy: {accuracy_score(y_valid, rf_base.predict(X_valid)):.4f}")

# Hyperparameter Tuning
param_grid = {
    "n_estimators":     [100, 200],
    "max_depth":        [3, 5, 10, None],
    "min_samples_leaf": [5, 10, 20],
    "max_features":     ["sqrt", 0.5]
}
tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=tscv, scoring="accuracy", n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
print(f"Best parameters:          {grid_search.best_params_}")
print(f"Best CV accuracy:         {grid_search.best_score_:.4f}")
print(f"Tuned validation accuracy:{accuracy_score(y_valid, best_rf.predict(X_valid)):.4f}")

# Test Set Evaluation
y_pred       = best_rf.predict(X_test)
y_pred_proba = best_rf.predict_proba(X_test)[:, 1]
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Test ROC-AUC:  {roc_auc_score(y_test, y_pred_proba):.4f}")
print(classification_report(y_test, y_pred, target_names=["Down (0)", "Up (1)"]))