from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Ubaidillah Mohammad Razali — SVM ──────────────────────────────────────────
df_svm = df.copy()

# feature engineering
df_svm["Return"]     = df_svm["Close"].pct_change()
df_svm["MA5"]        = df_svm["Close"].rolling(5).mean()
df_svm["MA10"]       = df_svm["Close"].rolling(10).mean()
df_svm["MA20"]       = df_svm["Close"].rolling(20).mean()
df_svm["Volatility"] = df_svm["Return"].rolling(5).std()
df_svm["Momentum5"]  = df_svm["Close"] - df_svm["Close"].shift(5)
df_svm["Vol_Ratio"]  = df_svm["Volume"] / df_svm["Volume"].rolling(10).mean()

delta    = df_svm["Close"].diff()
gain     = delta.clip(lower=0)
loss     = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs       = avg_gain / avg_loss
df_svm["RSI"] = 100 - (100 / (1 + rs))

ema12 = df_svm["Close"].ewm(span=12).mean()
ema26 = df_svm["Close"].ewm(span=26).mean()
df_svm["MACD"]        = ema12 - ema26
df_svm["MACD_signal"] = df_svm["MACD"].ewm(span=9).mean()

df_svm["BB_mid"]   = df_svm["Close"].rolling(20).mean()
bb_std             = df_svm["Close"].rolling(20).std()
df_svm["BB_upper"] = df_svm["BB_mid"] + 2 * bb_std
df_svm["BB_lower"] = df_svm["BB_mid"] - 2 * bb_std

df_svm["Ret1"] = df_svm["Return"].shift(1)
df_svm["Ret2"] = df_svm["Return"].shift(2)
df_svm["Ret3"] = df_svm["Return"].shift(3)

df_svm["Target"] = (df_svm["Close"].shift(-1) > df_svm["Close"]).astype(int)
df_svm = df_svm.dropna()

# feature set
features_svm = [
    "Open", "High", "Low", "Close", "Volume",
    "Return", "MA5", "MA10", "MA20",
    "Volatility", "Momentum5", "Vol_Ratio",
    "RSI", "MACD", "MACD_signal",
    "BB_mid", "BB_upper", "BB_lower",
    "Ret1", "Ret2", "Ret3"
]

X_svm = df_svm[features_svm]
y_svm = df_svm["Target"]

# train/test split
split_svm = int(len(df_svm) * 0.8)
X_train_svm, X_test_svm = X_svm.iloc[:split_svm], X_svm.iloc[split_svm:]
y_train_svm, y_test_svm = y_svm.iloc[:split_svm], y_svm.iloc[split_svm:]

# model
pipeline_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svc",    SVC(class_weight="balanced"))
])

# hyperparameters
tscv = TimeSeriesSplit(n_splits=5)
param_grid_svm = {
    "svc__kernel": ["linear", "rbf"],
    "svc__C":      [0.01, 0.1, 1, 10, 100],
    "svc__gamma":  ["scale", "auto"]
}

grid_svm = GridSearchCV(
    pipeline_svm,
    param_grid=param_grid_svm,
    cv=tscv,
    scoring="accuracy",
    n_jobs=-1
)

grid_svm.fit(X_train_svm, y_train_svm)
best_model_svm = grid_svm.best_estimator_

# prediction
y_pred_svm = best_model_svm.predict(X_test_svm)

# evaluation
print("\nbest parameters:")
print(grid_svm.best_params_)
print("\ncross validation score:")
print(grid_svm.best_score_)
print("\ntest accuracy:")
print(accuracy_score(y_test_svm, y_pred_svm))
print("\nclassification report:")
print(classification_report(y_test_svm, y_pred_svm))
print("\nconfusion matrix:")
print(confusion_matrix(y_test_svm, y_pred_svm))

# plot
plt.figure(figsize=(12, 5))
plt.plot(y_test_svm.values, label="Actual")
plt.plot(y_pred_svm, label="Predicted", alpha=0.7)
plt.title("SVM Stock Direction Prediction")
plt.legend()
plt.grid()
plt.show()
