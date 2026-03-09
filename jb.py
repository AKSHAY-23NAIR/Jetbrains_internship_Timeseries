
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix


def create_sliding_windows(df, P, F):
    X = []
    y = []
    meta = []

    for (store, product), group in df.groupby(["store", "product"]):
        group = group.sort_values("Date").reset_index(drop=True)

        sales = group["number_sold"].values
        incidents = group["incident"].values
        dates = group["Date"].values

        
        for t in range(P, len(group) - F + 1):
            window = sales[t -P:t]
            future_incident = incidents[t:t + F]

            label = 1 if np.max(future_incident) == 1 else 0

            X.append(window)
            y.append(label)

            meta.append({
                "store": store,
                "product": product,
                "window_end_date": dates[t - 1],
                "future_start_date": dates[t],
                "future_end_date": dates[t + F - 1]
            })

    return np.array(X), np.array(y), pd.DataFrame(meta)

df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

#converting to datetime format
df["Date"] = pd.to_datetime(df["Date"],dayfirst=True)
test_df["Date"] = pd.to_datetime(test_df["Date"],dayfirst=True)

# Sort data so that each store-product time series is in correct time order
df = df.sort_values(["store", "product", "Date"]).reset_index(drop=True)
test_df = test_df.sort_values(["store", "product", "Date"]).reset_index(drop=True)

print("Dataset shape:", df.shape)
print(df.head())

#HerePe create an incident where we give the label 1 if the number of sales exceed 1000
INCIDENT_THRESHOLD = 1000
df["incident"] = (df["number_sold"] > INCIDENT_THRESHOLD).astype(int)
test_df["incident"] = (test_df["number_sold"] > INCIDENT_THRESHOLD).astype(int)

#Now we will create sliding window parameters

P = 5
F = 3

# X = []
# y = []
# meta = []

# for (store, product), group in df.groupby(["store", "product"]):

#     group = group.sort_values("Date").reset_index(drop=True)

#     sales = group["number_sold"].values
#     incidents = group["incident"].values
#     dates = group["Date"].values

#     for t in range(P, len(group) - F):

#         # input window: previous W sales values
#         window = sales[t-P:t]

#         # future horizon: next H incident values
#         future_incident = incidents[t:t+F]

#         # label = 1 if any incident occurs in the next H steps
#         label = 1 if np.max(future_incident) == 1 else 0

#         X.append(window)
#         y.append(label)

#         # metadata for understanding predictions later
#         meta.append({
#             "store": store,
#             "product": product,
#             "window_end_date": dates[t-1],
#             "future_start_date": dates[t],
#             "future_end_date": dates[t+F-1]
#         })

# X = np.array(X)
# y = np.array(y)

# print(X)
# print(y)
# print("Window dataset shape:", X.shape)

X_train, y_train, train_meta = create_sliding_windows(df, P, F)
X_test, y_test, test_meta = create_sliding_windows(test_df, P, F)

print("\nSliding-window train shape:", X_train.shape)
print("Sliding-window test shape:", X_test.shape)
print("Train positive labels:", y_train.sum())
print("Test positive labels:", y_test.sum())

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

print("\nModel training complete.")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n=== Classification Report on Test Set ===")
print(classification_report(y_test, y_pred, digits=4))

print("=== Confusion Matrix on Test Set ===")
print(confusion_matrix(y_test, y_pred))

results = test_meta.copy()
results["true_label"] = y_test
results["predicted_label_default"] = y_pred
results["predicted_probability"] = y_prob
#results["predicted_label_thresholded"] = y_alert

print("\nSample test results:")
print(results.head(10))
