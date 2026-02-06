import os
import warnings
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from optuna.trial import Trial
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xarray as xr
import time
import joblib
warnings.filterwarnings('ignore')

# ------------------ 路径配置 ------------------
climate_path = r''
delta_SOS_path = r''
IGBP_path = r''
Soil_texture_path = r''

DYNAMIC_FEATURE_NAMES = [
    'TEMP', 'SR', 'VPD', 'Rain_fre', 'SM', 'Frost_days'
]
STATIC_FEATURE_NAMES = ['Soil_clay', 'Soil_soc', 'IGBP']
ALL_FEATURE_NAMES = DYNAMIC_FEATURE_NAMES + STATIC_FEATURE_NAMES

MODEL_FILENAME = r''
PLOT_FILENAME = r''
CSV_FILENAME = r''

def load_data():
    print("正在加载训练数据...")

    Climate_data = xr.open_dataset(climate_path + 'name').sel(lat=slice(30, 90))
    TEMP = Climate_data['variable'].values
    SR = Climate_data['variable'].values
    VPD = Climate_data['variable'].values
    Rain_fre = Climate_data['variable'].values
    SM = Climate_data['variable'].values
    Frost_days = Climate_data['variable'].values

    Delta_SOS = xr.open_dataset(delta_SOS_path).sel(lat=slice(30, 90))['Delta_SOS'].values
    IGBP = xr.open_dataset(IGBP_path).sel(lat=slice(30, 90))['IGBP'].values
    Soil_clay = xr.open_dataset(Soil_texture_path + 'T_CLAY_0.25deg.nc').sel(lat=slice(30, 90))['T_CLAY'].values
    Soil_soc = xr.open_dataset(Soil_texture_path + 'T_OC_0.25deg.nc').sel(lat=slice(30, 90))['T_OC'].values

    return (TEMP, SR, VPD, Rain_fre, SM, Frost_days,
            Soil_clay, Soil_soc, IGBP, Delta_SOS)

def preprocess_data(TEMP, SR, VPD, Rain_fre, SM, Frost_days,
            Soil_clay, Soil_soc, IGBP, Delta_SOS):
    igbp_mask = np.isin(IGBP, [1, 2, 3, 4])
    dynamic_features = np.stack([TEMP, SR, VPD, Rain_fre, SM, Frost_days], axis=-1)
    static_features = np.stack([Soil_clay, Soil_soc, IGBP], axis=-1)

    combined_valid_mask = (
            (~np.isnan(static_features).any(axis=-1)) &
            (~np.isnan(dynamic_features).any(axis=(0, -1))) &
            (~np.isnan(Delta_SOS).any(axis=0)) &
            (igbp_mask)
    )
    valid_points = np.where(combined_valid_mask)
    X_dynamic = dynamic_features[:, valid_points[0], valid_points[1], :]
    X_static = static_features[valid_points[0], valid_points[1], :]
    y = Delta_SOS[:, valid_points[0], valid_points[1]]
    return X_dynamic, X_static, y


# ------------------ XGBoost 优化与训练 ------------------
def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',  # 推荐使用 hist 以提高速度
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-6, 1.0, log=True),
        'enable_categorical': True  # 必须开启以处理 IGBP
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, preds))


def train_xgb_model_optimized(X_dynamic, X_static, y, n_trials=20):
    print("正在构建特征矩阵...")
    n_time_steps = X_dynamic.shape[0]
    n_dynamic_features = X_dynamic.shape[2]

    # 1. 展平与扩展
    X_dynamic_reshaped = X_dynamic.reshape(-1, n_dynamic_features)
    X_static_tiled = np.tile(X_static, (n_time_steps, 1))
    y_reshaped = y.reshape(-1)

    # 2. 合并为 DataFrame
    X = pd.DataFrame(np.concatenate([X_dynamic_reshaped, X_static_tiled], axis=1),
                     columns=ALL_FEATURE_NAMES)


    # 释放内存
    del X_dynamic_reshaped, X_static_tiled

    # 3. 强制转换分类变量
    X['IGBP'] = X['IGBP'].astype('category')

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y_reshaped, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=42)

    # 5. Optuna 调参
    print(f"开始 Optuna 搜索 (共 {n_trials} 次)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)

    # 6. 使用最佳参数训练最终模型
    print(f"\n最佳 RMSE: {study.best_trial.value:.4f}")
    best_params = study.best_params
    best_params['enable_categorical'] = True

    best_model = xgb.XGBRegressor(**best_params)
    best_model.fit(X_train_full, y_train_full)

    # 7. 保存
    best_model.save_model(MODEL_FILENAME)
    print(f"模型已保存至: {MODEL_FILENAME}")

    y_pred = best_model.predict(X_test)
    return y_test, y_pred, X_test


# ------------------ 绘图与输出 (修正版) ------------------
def plot_performance(y_test, y_pred, output_path, csv_path=None):
    print("\n绘制模型性能图...")
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 8))


    if len(y_test) > 50000:
        idx = np.random.choice(len(y_test), 50000, replace=False)

        ax.scatter(y_test[idx], y_pred[idx], alpha=0.3, s=15, edgecolors='k', c='skyblue')
    else:
        ax.scatter(y_test, y_pred, alpha=0.3, s=15, edgecolors='k', c='skyblue')

    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=2, label='1:1 线')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    stats_text = f'$R^2$={r2:.3f}\nRMSE={rmse:.3f}\nMAE={mae:.3f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f"性能图已保存：{output_path}")

    if csv_path:
        pd.DataFrame({'y_test': y_test, 'y_pred': y_pred}).to_csv(csv_path, index=False)
        print(f"结果已保存：{csv_path}")


# ------------------ 主程序 ------------------
if __name__ == '__main__':
    start_time = time.time()

    # 1. 加载数据
    data_tuple = load_data()

    # 2. 预处理 
    X_dynamic, X_static, y = preprocess_data(*data_tuple)

    # 3. 训练模型
    y_test, y_pred, X_test = train_xgb_model_optimized(X_dynamic, X_static, y, n_trials=50)

    # 4. 绘图
    plot_performance(y_test, y_pred, PLOT_FILENAME, csv_path=CSV_FILENAME)

    end_time = time.time()
    print(f"\n全过程耗时 {(end_time - start_time) / 60:.2f} 分钟。")