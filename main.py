import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

# Step 1: Simulate INR price dataset
np.random.seed(42)
dates = pd.date_range(start='2021-01-01', periods=500)
data = pd.DataFrame({
    'Date': dates,
    'Open': np.random.uniform(70, 75, size=500),
    'High': np.random.uniform(75, 77, size=500),
    'Low': np.random.uniform(68, 70, size=500),
    'Close': np.random.uniform(70, 75, size=500)
})
data['Next_Close'] = data['Close'].shift(-1)
data.dropna(inplace=True)

# Step 2: Feature engineering
data['Change'] = data['Next_Close'] - data['Close']
data['Target'] = (data['Change'] > 0).astype(int)

# Step 3: Prepare features and targets
features = ['Open', 'High', 'Low', 'Close']
X = data[features]
y_reg = data['Next_Close']
y_clf = data['Target']

# Step 4: Train/test split
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

# Step 5: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Linear Regression (Regression)
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train_reg)
y_pred_reg = lin_reg.predict(X_test_scaled)
rmse = mean_squared_error(y_test_reg, y_pred_reg, squared=False)

# Step 7: Logistic Regression (Classification)
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train_clf)
y_pred_clf_log = log_reg.predict(X_test_scaled)
log_acc = accuracy_score(y_test_clf, y_pred_clf_log)

# Step 8: Random Forest (Ensemble Classification)
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train_scaled, y_train_clf)
y_pred_clf_rf = rf_clf.predict(X_test_scaled)
rf_acc = accuracy_score(y_test_clf, y_pred_clf_rf)

# Step 9: Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, None],
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
grid_search.fit(X_train_scaled, y_train_clf)
best_rf = grid_search.best_estimator_
best_rf_acc = accuracy_score(y_test_clf, best_rf.predict(X_test_scaled))

# Step 10: Output Results
print("Linear Regression RMSE:", rmse)
print("Logistic Regression Accuracy:", log_acc)
print("Random Forest Accuracy:", rf_acc)
print("Best Tuned RF Accuracy:", best_rf_acc)
print("Best Params:", grid_search.best_params_)
