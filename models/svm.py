
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt

df.columns = [c.strip() for c in df.columns]

df["Volume"] = df["Volume"].astype(str).str.replace(",", "")
df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

for col in ["Open", "High", "Low", "Close"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
df = df.iloc[::-1].reset_index(drop=True)

# -- feature engineering

# returns
df["Return"] = df["Close"].pct_change()
# moving averages
df["MA5"] = df["Close"].rolling(5).mean()
df["MA10"] = df["Close"].rolling(10).mean()
df["MA20"] = df["Close"].rolling(20).mean()
# volatility
df["Volatility"] = df["Return"].rolling(5).std()
# momentum
df["Momentum5"] = df["Close"] - df["Close"].shift(5)
# relative volume
df["Vol_Ratio"] = df["Volume"] / df["Volume"].rolling(10).mean()
# RSI
delta = df["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))
# MACD
ema12 = df["Close"].ewm(span=12).mean()
ema26 = df["Close"].ewm(span=26).mean()
df["MACD"] = ema12 - ema26
df["MACD_signal"] = df["MACD"].ewm(span=9).mean()
# bollinger bands
df["BB_mid"] = df["Close"].rolling(20).mean()
bb_std = df["Close"].rolling(20).std()
df["BB_upper"] = df["BB_mid"] + 2 * bb_std
df["BB_lower"] = df["BB_mid"] - 2 * bb_std
# lag features
df["Ret1"] = df["Return"].shift(1)
df["Ret2"] = df["Return"].shift(2)
df["Ret3"] = df["Return"].shift(3)

# -- target
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
df = df.dropna()

# -- feature set
features = [
    "Open", "High", "Low", "Close", "Volume",
    "Return", "MA5", "MA10", "MA20",
    "Volatility", "Momentum5", "Vol_Ratio",
    "RSI", "MACD", "MACD_signal",
    "BB_mid", "BB_upper", "BB_lower",
    "Ret1", "Ret2", "Ret3"
]

X = df[features]
y = df["Target"]

# -- train/test split
split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# -- model
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(class_weight="balanced"))
])

# -- hyperparameters
tscv = TimeSeriesSplit(n_splits=5)

param_grid = {
    "svc__kernel": ["linear", "rbf"],
    "svc__C": [0.01, 0.1, 1, 10, 100],
    "svc__gamma": ["scale", "auto"]
}

grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=tscv,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# -- prediction
y_pred = best_model.predict(X_test)

# -- evaluation
print("\nbest parameters:")
print(grid.best_params_)

print("\ncross validation score:")
print(grid.best_score_)

print("\ntest accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nclassification report :")
print(classification_report(y_test, y_pred))

print("\nconfusion matrix")
print(confusion_matrix(y_test, y_pred))

# plot
plt.figure(figsize=(12,5))
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted", alpha=0.7)
plt.title(" SVM Stock Direction Prediction")
plt.legend()
plt.grid()
plt.show()
