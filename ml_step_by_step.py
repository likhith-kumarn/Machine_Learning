#  STUDENT PERFORMANCE — STEP BY STEP ML PROJECT

# STEP 0 ▸ INSTALL & IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("All libraries imported successfully!")

# STEP 1: LOAD THE DATASET
# Load the dataset
df = pd.read_csv('Student_Performance.csv')

print("Shape:", df.shape)          # (rows, columns)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn data types:")
print(df.dtypes)


# STEP 2 : EXPLORE THE DATA (EDA)

#2a. Basic statistics 
print("Basic Statistics:")
print(df.describe())

#2b. Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

#2c. Check unique values in categorical column
print("\n Unique values in 'Extracurricular Activities':")
print(df['Extracurricular Activities'].value_counts())

#2d. Visualize target distribution
plt.figure(figsize=(8, 4))
plt.hist(df['Performance Index'], bins=30, color='#7c6af7', edgecolor='white')
plt.title('Distribution of Performance Index (Target)')
plt.xlabel('Performance Index')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

#2e. Correlation heatmap
import matplotlib.colors as mcolors

df_temp = df.copy()
df_temp['Extracurricular Activities'] = (df_temp['Extracurricular Activities'] == 'Yes').astype(int)
corr = df_temp.corr()

fig, ax = plt.subplots(figsize=(7, 6))
cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(cax)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha='left', fontsize=9)
ax.set_yticklabels(corr.columns, fontsize=9)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=8)
plt.title('Correlation Matrix', pad=20)
plt.tight_layout()
plt.show()


# STEP 3: PREPROCESSING — ENCODE + SPLIT
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#3a. Encode categorical column
le = LabelEncoder()
df['Extracurricular Activities'] = le.fit_transform(df['Extracurricular Activities'])
# Yes → 1,  No → 0
print("After encoding:")
print(df['Extracurricular Activities'].value_counts())

#3b. Separate features (X) and target (y)
X = df.drop('Performance Index', axis=1)   # All columns except target
y = df['Performance Index']                # Target column

print("\n Features (X):", X.columns.tolist())
print(" Target (y):", y.name)
print(" X shape:", X.shape)

#3c. Train / Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)  # 20% for testing   
# Fixed seed for reproducibility

print(f"\n Training samples : {X_train.shape[0]}")
print(f" Testing samples  : {X_test.shape[0]}")


# STEP 4 : EVALUATION HELPER FUNCTION
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# We'll call this function after training each model
def evaluate(model_name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n {model_name}")
    print(f"   RMSE : {rmse:.4f}  - lower is better")
    print(f"   MAE  : {mae:.4f}  - lower is better")
    print(f"   R²   : {r2:.4f}  - closer to 1 is better")
    return {'Model': model_name, 'RMSE': round(rmse,4),
            'MAE': round(mae,4), 'R²': round(r2,4)}

all_results = []   # We'll collect results from every model here
print(" evaluate() function ready!")


# STEP 5 ▸ MODEL 1 — LINEAR REGRESSION
# WHAT IT DOES:
#    Finds the best-fit line (or hyperplane) through the data.
#    Equation:  y = b0 + b1*x1 + b2*x2 + ..
#    It minimises the sum of squared errors (OLS method).
#     Great baseline |  Very interpretable

from sklearn.linear_model import LinearRegression

#5a. Create & train the model
lr = LinearRegression()
lr.fit(X_train, y_train)          # Train on training data

#5b. Make predictions 
y_pred_lr = lr.predict(X_test)    # Predict on test data

# 5c. Evaluate
res = evaluate("Linear Regression", y_test, y_pred_lr)
all_results.append(res)

#5d. Inspect coefficients (which features matter most?)
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\n Coefficients (how much each feature moves the score):")
print(coef_df.to_string(index=False))
print(f"\nIntercept (b0): {lr.intercept_:.4f}")

#5e. Plot: Actual vs Predicted
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred_lr, alpha=0.3, color='#7c6af7', s=10)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression — Actual vs Predicted')
plt.legend()
plt.tight_layout()
plt.show()


# STEP 6 ▸ MODEL 2 — DECISION TREE
#  WHAT IT DOES:
#    Splits the dataset into smaller groups using IF/THEN rules.
#    Each leaf gives a prediction = mean of samples in that leaf.
#     Visualisable |  Handles non-linearity
#     Overfits easily if not limited (use max_depth!)

from sklearn.tree import DecisionTreeRegressor, export_text

#6a. Create & train
dt = DecisionTreeRegressor(
    max_depth=5,          # Limit depth to prevent overfitting
    min_samples_leaf=10,  # Leaf must have at least 10 samples
    random_state=42
)
dt.fit(X_train, y_train)

#6b. Predict & evaluate
y_pred_dt = dt.predict(X_test)
res = evaluate("Decision Tree", y_test, y_pred_dt)
all_results.append(res)

#6c. Print tree rules (first 3 levels)
print("\n Decision Tree Rules (depth ≤ 3):")
rules = export_text(dt, feature_names=list(X.columns), max_depth=3)
print(rules[:1000], "\n...")

#6d. Feature importance
fi = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt.feature_importances_
}).sort_values('Importance', ascending=False)
print("\n Feature Importances:")
print(fi.to_string(index=False))

#6e. Plot feature importance
plt.figure(figsize=(7, 4))
plt.barh(fi['Feature'][::-1], fi['Importance'][::-1], color='#56d364')
plt.xlabel('Importance Score')
plt.title('Decision Tree — Feature Importance')
plt.tight_layout()
plt.show()

# OVERFITTING CHECK: compare train vs test R²
train_r2 = r2_score(y_train, dt.predict(X_train))
test_r2  = r2_score(y_test,  y_pred_dt)
print(f"\n Overfitting check → Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
print("   (Large gap = overfitting. Reduce max_depth to fix.)")


# ──────────────────────────────────────────────────────────────
# STEP 7 ▸ MODEL 3 — RANDOM FOREST
# ──────────────────────────────────────────────────────────────
# 📖 WHAT IT DOES:
#    Trains MANY decision trees, each on a random subset of data
#    and random subset of features (this is called "bagging").
#    Final prediction = AVERAGE of all trees.
#    ✅ Much less overfitting than single tree
#    ✅ Built-in OOB (Out-of-Bag) score = free validation!

from sklearn.ensemble import RandomForestRegressor

# --- 7a. Create & train ---
rf = RandomForestRegressor(
    n_estimators=100,   # Number of trees
    max_depth=10,       # Max depth per tree
    random_state=42,
    oob_score=True,     # Free cross-validation estimate!
    n_jobs=-1           # Use all CPU cores
)
rf.fit(X_train, y_train)

# --- 7b. Predict & evaluate ---
y_pred_rf = rf.predict(X_test)
res = evaluate("Random Forest", y_test, y_pred_rf)
all_results.append(res)

# --- 7c. OOB Score (bonus: free R² estimate without test set) ---
print(f"\n🎯 OOB Score (Out-of-Bag R²): {rf.oob_score_:.4f}")
print("   (This is a free validation score — no test data used!)")

# --- 7d. Feature importance ---
fi_rf = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)
print("\n📊 Feature Importances:")
print(fi_rf.to_string(index=False))

# --- 7e. Plot: number of trees vs error ---
errors = []
for n in range(1, 101, 5):
    rf_temp = RandomForestRegressor(n_estimators=n, random_state=42, n_jobs=-1)
    rf_temp.fit(X_train, y_train)
    e = np.sqrt(mean_squared_error(y_test, rf_temp.predict(X_test)))
    errors.append(e)

plt.figure(figsize=(7, 4))
plt.plot(range(1, 101, 5), errors, color='#ffa657', lw=2, marker='o', markersize=4)
plt.xlabel('Number of Trees')
plt.ylabel('RMSE')
plt.title('Random Forest — RMSE vs Number of Trees')
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig('step7_random_forest.png', dpi=120)
plt.show()
print("✅ Plot saved: step7_random_forest.png")


# ──────────────────────────────────────────────────────────────
# STEP 8 ▸ MODEL 4 — GRADIENT BOOSTING
# ──────────────────────────────────────────────────────────────
# 📖 WHAT IT DOES:
#    Builds trees SEQUENTIALLY. Each new tree fixes the ERRORS
#    (residuals) made by the previous trees.
#    Uses gradient descent in function space.
#    ✅ Very powerful | ✅ Often best on tabular data
#    ❌ Slower to train | ❌ More hyperparams to tune

from sklearn.ensemble import GradientBoostingRegressor

# --- 8a. Create & train ---
gb = GradientBoostingRegressor(
    n_estimators=200,    # Number of boosting rounds
    learning_rate=0.1,   # Step size (lower = more rounds needed)
    max_depth=4,         # Keep trees shallow in boosting
    subsample=0.8,       # Use 80% of data per tree (reduces overfitting)
    random_state=42
)
gb.fit(X_train, y_train)

# --- 8b. Predict & evaluate ---
y_pred_gb = gb.predict(X_test)
res = evaluate("Gradient Boosting", y_test, y_pred_gb)
all_results.append(res)

# --- 8c. Training loss curve ---
train_loss = gb.train_score_
print(f"\n📉 Loss: {train_loss[0]:.2f} (start) → {train_loss[-1]:.4f} (end)")

plt.figure(figsize=(7, 4))
plt.plot(train_loss, color='#f97583', lw=2)
plt.xlabel('Boosting Round')
plt.ylabel('Training Loss (MSE)')
plt.title('Gradient Boosting — Training Loss Curve')
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig('step8_gradient_boosting.png', dpi=120)
plt.show()
print("✅ Plot saved: step8_gradient_boosting.png")

# --- 8d. Learning rate experiment ---
print("\n📊 Effect of learning_rate (n_estimators fixed at 100):")
for lr_val in [0.01, 0.05, 0.1, 0.2, 0.5]:
    gb_temp = GradientBoostingRegressor(n_estimators=100,
                                         learning_rate=lr_val, random_state=42)
    gb_temp.fit(X_train, y_train)
    r2_temp = r2_score(y_test, gb_temp.predict(X_test))
    print(f"   learning_rate={lr_val:.2f}  → R²: {r2_temp:.4f}")


# ──────────────────────────────────────────────────────────────
# STEP 9 ▸ MODEL 5 — XGBOOST
# ──────────────────────────────────────────────────────────────
# 📖 WHAT IT DOES:
#    XGBoost = eXtreme Gradient Boosting.
#    Same idea as GradientBoosting but:
#    ✦ Faster (histogram binning, parallel trees)
#    ✦ Built-in L1 + L2 regularisation
#    ✦ Handles missing values natively
#    ✅ Wins most Kaggle tabular competitions!

# Option A: If you have XGBoost installed (pip install xgboost)
try:
    from xgboost import XGBRegressor
    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,    # Random subset of features per tree
        reg_alpha=0.1,           # L1 regularisation
        reg_lambda=1.0,          # L2 regularisation
        random_state=42,
        verbosity=0
    )
    xgb.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False)
    y_pred_xgb = xgb.predict(X_test)
    res = evaluate("XGBoost", y_test, y_pred_xgb)
    all_results.append(res)
    print("\n✅ Used: xgboost.XGBRegressor")

# Option B: XGBoost not installed → use sklearn's equivalent
except ImportError:
    print("ℹ️  xgboost not installed. Using sklearn HistGradientBoostingRegressor")
    print("   (Same idea — histogram-based boosting with regularisation)")
    from sklearn.ensemble import HistGradientBoostingRegressor
    xgb = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=5,
        l2_regularization=1.0,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    res = evaluate("XGBoost-style (HistGB)", y_test, y_pred_xgb)
    all_results.append(res)

# --- 9b. Plot Actual vs Predicted ---
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred_xgb, alpha=0.3, color='#79c0ff', s=10)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('XGBoost — Actual vs Predicted')
plt.legend()
plt.tight_layout()
plt.savefig('step9_xgboost.png', dpi=120)
plt.show()
print("✅ Plot saved: step9_xgboost.png")


# ──────────────────────────────────────────────────────────────
# STEP 10 ▸ HYPERPARAMETER TUNING — GridSearchCV
# ──────────────────────────────────────────────────────────────
# 📖 WHAT IT DOES:
#    Automatically tries every combination of parameters you specify.
#    Uses cross-validation to pick the best combination.
#    ✅ Removes guesswork from tuning
#    ❌ Can be slow for large grids (use RandomizedSearchCV then)

from sklearn.model_selection import GridSearchCV

# --- 10a. Define the parameter grid ---
# Each key = parameter name, each list = values to try
param_grid = {
    'n_estimators':    [50, 100, 150],    # 3 values
    'max_depth':       [5, 8, 10],        # 3 values
    'min_samples_leaf': [1, 2, 4]         # 3 values
}
# Total combinations: 3 × 3 × 3 = 27
# With cv=3:  27 × 3 = 81 model fits

print(f"🔍 Searching {3*3*3} combinations × 3-fold = 81 fits...")
print("⏳ Please wait...\n")

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=3,                                  # 3-fold cross validation
    scoring='neg_root_mean_squared_error', # Optimise for RMSE
    n_jobs=-1,                             # Parallel
    verbose=1
)
grid_search.fit(X_train, y_train)

# --- 10b. Best results ---
print(f"\n✅ Best Parameters : {grid_search.best_params_}")
print(f"✅ Best CV RMSE    : {-grid_search.best_score_:.4f}")

# --- 10c. Evaluate best model on test set ---
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)
res = evaluate("RF (GridSearch Tuned)", y_test, y_pred_tuned)
all_results.append(res)

# --- 10d. View top 10 combinations ---
cv_results_df = pd.DataFrame(grid_search.cv_results_)
top10 = cv_results_df.nlargest(10, 'mean_test_score')[
    ['param_n_estimators', 'param_max_depth',
     'param_min_samples_leaf', 'mean_test_score']
].copy()
top10['mean_test_score'] = -top10['mean_test_score']   # convert to positive RMSE
top10.columns = ['n_estimators', 'max_depth', 'min_samples_leaf', 'mean_RMSE']
print("\n📊 Top 10 Parameter Combinations:")
print(top10.round(4).to_string(index=False))

# --- 10e. Heatmap of n_estimators vs max_depth ---
pivot = cv_results_df.pivot_table(
    index='param_max_depth',
    columns='param_n_estimators',
    values='mean_test_score',
    aggfunc='max'
)
pivot = -pivot   # make positive RMSE

plt.figure(figsize=(7, 4))
plt.imshow(pivot.values, cmap='YlOrRd_r', aspect='auto')
plt.colorbar(label='RMSE')
plt.xticks(range(len(pivot.columns)), [f'n={v}' for v in pivot.columns])
plt.yticks(range(len(pivot.index)),   [f'd={v}' for v in pivot.index])
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        plt.text(j, i, f'{pivot.values[i,j]:.2f}',
                 ha='center', va='center', fontsize=9, color='black')
plt.title('GridSearchCV Heatmap — RMSE (lower=better)')
plt.xlabel('n_estimators')
plt.ylabel('max_depth')
plt.tight_layout()
plt.savefig('step10_gridsearch.png', dpi=120)
plt.show()
print("✅ Plot saved: step10_gridsearch.png")


# ──────────────────────────────────────────────────────────────
# STEP 11 ▸ PIPELINE + CROSS VALIDATION
# ──────────────────────────────────────────────────────────────
# 📖 PIPELINE:
#    Chains steps together: [Scaler → Model] as one object.
#    WHY: Without Pipeline, if you scale X before splitting,
#         the scaler "sees" test data → DATA LEAKAGE!
#    Pipeline fits the scaler ONLY on training fold each time.
#
# 📖 CROSS VALIDATION:
#    Splits data into k folds. Trains on k-1, validates on 1.
#    Rotates k times → k scores. Report mean ± std.
#    ✅ Better performance estimate than a single train/test split.

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold

# --- 11a. Build pipelines for each model ---
pipelines = {
    'Linear Regression': Pipeline([
        ('scaler', StandardScaler()),           # Step 1: scale features
        ('model',  LinearRegression())          # Step 2: fit model
    ]),
    'Decision Tree': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  DecisionTreeRegressor(max_depth=5, random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ]),
    'Gradient Boosting': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  GradientBoostingRegressor(n_estimators=100,
                                              learning_rate=0.1,
                                              max_depth=4, random_state=42))
    ]),
    'XGBoost-style': Pipeline([
        ('scaler', StandardScaler()),
        ('model',  HistGradientBoostingRegressor(max_iter=200,
                                                  learning_rate=0.05,
                                                  max_depth=5, random_state=42))
    ]),
}

# --- 11b. Define K-Fold strategy ---
kf = KFold(
    n_splits=5,       # 5 folds
    shuffle=True,     # Shuffle data before splitting
    random_state=42
)

# --- 11c. Run cross validation on every pipeline ---
print(f"\n{'Model':<25} {'Mean R²':>9} {'±':>3} {'Std':>7}   {'Mean RMSE':>10} {'±':>3} {'Std':>7}")
print("-" * 70)

cv_records = []
for name, pipe in pipelines.items():
    r2_scores   = cross_val_score(pipe, X, y, cv=kf, scoring='r2', n_jobs=-1)
    rmse_scores = np.sqrt(-cross_val_score(pipe, X, y, cv=kf,
                           scoring='neg_mean_squared_error', n_jobs=-1))

    print(f"{name:<25} {r2_scores.mean():>9.4f} {'±':>3} {r2_scores.std():>7.4f}"
          f"   {rmse_scores.mean():>10.4f} {'±':>3} {rmse_scores.std():>7.4f}")

    cv_records.append({
        'Model': name,
        'Mean R²': round(r2_scores.mean(), 4),
        'Std R²':  round(r2_scores.std(),  4),
        'Mean RMSE': round(rmse_scores.mean(), 4),
        'Std RMSE':  round(rmse_scores.std(),  4),
        'R2_scores': r2_scores
    })

# --- 11d. Plot CV score distributions ---
plt.figure(figsize=(9, 5))
labels = [r['Model'] for r in cv_records]
data   = [r['R2_scores'] for r in cv_records]
bp = plt.boxplot(data, patch_artist=True, labels=labels,
                  medianprops=dict(color='white', lw=2))
colors = ['#7c6af7','#56d364','#ffa657','#f97583','#79c0ff']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
plt.ylabel('R² Score')
plt.title('5-Fold Cross Validation — R² Distribution per Model')
plt.xticks(rotation=15, ha='right')
plt.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig('step11_cross_validation.png', dpi=120)
plt.show()
print("✅ Plot saved: step11_cross_validation.png")


# ──────────────────────────────────────────────────────────────
# STEP 12 ▸ FINAL MODEL COMPARISON
# ──────────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("  🏆 FINAL MODEL COMPARISON (Test Set)")
print("=" * 55)

results_df = pd.DataFrame(all_results).sort_values('R²', ascending=False)
print(results_df.to_string(index=False))

best = results_df.iloc[0]
print(f"\n🥇 Best Model : {best['Model']}")
print(f"   R²   = {best['R²']}")
print(f"   RMSE = {best['RMSE']}")

# --- Final comparison bar chart ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = ['#7c6af7','#56d364','#ffa657','#f97583','#79c0ff','#f778ba'][:len(results_df)]

axes[0].barh(results_df['Model'], results_df['R²'],
             color=colors, edgecolor='none')
axes[0].set_xlabel('R² Score')
axes[0].set_title('Model Comparison — R²')
axes[0].set_xlim(0.9, 1.0)
for i, v in enumerate(results_df['R²']):
    axes[0].text(v + 0.0002, i, str(v), va='center', fontsize=9)

axes[1].barh(results_df['Model'], results_df['RMSE'],
             color=colors, edgecolor='none')
axes[1].set_xlabel('RMSE')
axes[1].set_title('Model Comparison — RMSE (lower = better)')
for i, v in enumerate(results_df['RMSE']):
    axes[1].text(v + 0.01, i, str(v), va='center', fontsize=9)

plt.suptitle('Final Model Comparison — Student Performance Dataset',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('step12_final_comparison.png', dpi=120)
plt.show()
print("✅ Plot saved: step12_final_comparison.png")

print("\n🎉 PROJECT COMPLETE! All models trained and compared.")
