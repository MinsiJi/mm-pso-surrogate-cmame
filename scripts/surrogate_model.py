import numpy as np
from scipy.integrate import solve_ivp
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import qmc
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import os
import pandas as pd
# 配置支持数学符号的字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC','SimHei','DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False


def ballistic_model(params, constants):
    P1, T1, A1, V1, L = params
    m, Cd, gamma, R, Cv, Cp, T0, A = constants
    M0 = P1 * V1 / (R * T1)
    x0 = 0.0
    v0 = 0.0
    T0_system = T1
    def ode_system(t, y):
        x, v, M, T = y
        V = V1 + A * x
        P = M * R * T / V
        # 计算质量泄漏率
        leak = Cd * A1 * P * np.sqrt(gamma / (R * T)) * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
        dvdt = (P * A) / m if x <= L else 0.0
        dMdt = -leak
        dTdt = (-R * T * leak - P * A * v) / (Cv * M) if M > 1e-6 else 0.0
        dxdt = v
        return [dxdt, dvdt, dMdt, dTdt]
    t_span = (0, 0.1)
    y0 = [x0, v0, M0, T0_system]
    
    sol = solve_ivp(
        ode_system, 
        t_span, 
        y0, 
        method='LSODA', 
        events=lambda t, y: y[0] - L,
        dense_output=True
    )
    if sol.t_events[0].size > 0:
        t_end = sol.t_events[0][0]
        x_end, v_end, M_end, T_end = sol.sol(t_end)
    else:
        t_end = sol.t[-1]
        x_end, v_end, M_end, T_end = sol.y[:, -1]
    
    V_end = V1 + A * x_end
    P_end = M_end * R * T_end / V_end
    
    # 计算累积泄漏能量
    t_points = np.linspace(0, t_end, 1000) 
    sol_vals = sol.sol(t_points)
    leak_energy = 0.0
    
    for i in range(1, len(t_points)):
        dt = t_points[i] - t_points[i-1]
        t = t_points[i]
        x, v, M, T = sol_vals[0][i], sol_vals[1][i], sol_vals[2][i], sol_vals[3][i]
        V = V1 + A * x
        P = M * R * T / V
        
        # 计算当前时刻的质量泄漏率
        leak = Cd * A1 * P * np.sqrt(gamma / (R * T)) * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
        # 计算泄漏能量
        h = Cp * T 
        leak_energy += leak * h * dt
    
    U0 = Cv * M0 * T1
    Uf = Cv * M_end * T_end
    Patm = 101325
    

    W = (U0 + P1 * V1) - (Uf + P_end * (V1 + A * x_end)) - leak_energy
    yita = 100 * (m * v_end ** 2 / 2) / W if W > 0 else 0.0
    
    U0_total = Cp * M0 * T1
    yita_zong = 100*0.5 * m * v_end ** 2 / U0_total if U0_total > 0 else 0.0
    

    pressure_ok = P_end >= Patm
    feasible = (v_end >= 50) and (x_end >= L) and pressure_ok
    
    return v_end, yita, yita_zong, feasible 



# 数据生成：拉丁超立方采样
def generate_training_data(constants, num_samples=10000):
    sampler = qmc.LatinHypercube(d=5)
    params = sampler.random(num_samples)
    low = np.array([5e5, 300, 1e-5, 0.001, 0.6])
    high = np.array([10e6, 800, 5e-5, 0.005, 1.5])
    params = qmc.scale(params, low, high)
    X, y = [], []
    for p in params:
        v_end, yita, yita_zong, feasible = ballistic_model(p, constants)
        if feasible and yita > 1e-8:
            X.append(p)
            y.append([v_end, yita, yita_zong])  # 保存三个输出
    return np.array(X), np.array(y)


# 模型定义
class SurrogateModel:
    def __init__(self):
        # 多输出代理模型（v, yita, yita_zong）
        self.model = make_pipeline(
            StandardScaler(),  # 输入标准化
            MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                max_iter=5000,
                alpha=0.0005,
                learning_rate='adaptive',
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=50,
                tol=1e-6
            )
        )
        self.scaler = StandardScaler()  # 输出标准化器
    
    def train_with_validation(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # 标准化输出
        y_train_scaled = self.scaler.fit_transform(y_train)
        y_test_scaled = self.scaler.transform(y_test)
        
        # 训练模型
        self.model.fit(X_train, y_train_scaled)
        
        # 预测并逆标准化
        y_train_pred_scaled = self.model.predict(X_train)
        self.y_train_pred = self.scaler.inverse_transform(y_train_pred_scaled)
        
        y_test_pred_scaled = self.model.predict(X_test)
        self.y_test_pred = self.scaler.inverse_transform(y_test_pred_scaled)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # 输出训练信息
        print(f"速度模型最终损失: {self.model.steps[1][1].loss_:.6f}")
        print(f"效率模型最终损失: {self.model.steps[1][1].loss_:.6f}")
        print(f"总体效率模型最终损失: {self.model.steps[1][1].loss_:.6f}")
        print(f"模型训练轮数: {self.model.steps[1][1].n_iter_}")
    
    def calculate_validation_metrics(self):
        metrics = {}
        
        # 速度指标
        v_train_true, v_train_pred = self.y_train[:, 0], self.y_train_pred[:, 0]
        v_test_true, v_test_pred = self.y_test[:, 0], self.y_test_pred[:, 0]
        metrics['v'] = {
            'train': {'MAE': mean_absolute_error(v_train_true, v_train_pred),
                      'RMSE': np.sqrt(mean_squared_error(v_train_true, v_train_pred)),
                      'R²': r2_score(v_train_true, v_train_pred)},
            'test': {'MAE': mean_absolute_error(v_test_true, v_test_pred),
                     'RMSE': np.sqrt(mean_squared_error(v_test_true, v_test_pred)),
                     'R²': r2_score(v_test_true, v_test_pred)}
        }
        
        # 效率指标
        yita_train_true, yita_train_pred = self.y_train[:, 1], self.y_train_pred[:, 1]
        yita_test_true, yita_test_pred = self.y_test[:, 1], self.y_test_pred[:, 1]
        metrics['yita'] = {
            'train': {'MAE': mean_absolute_error(yita_train_true, yita_train_pred),
                      'RMSE': np.sqrt(mean_squared_error(yita_train_true, yita_train_pred)),
                      'R²': r2_score(yita_train_true, yita_train_pred)},
            'test': {'MAE': mean_absolute_error(yita_test_true, yita_test_pred),
                     'RMSE': np.sqrt(mean_squared_error(yita_test_true, yita_test_pred)),
                     'R²': r2_score(yita_test_true, yita_test_pred)}
        }
        
        # 总体效率指标
        yita_zong_train_true, yita_zong_train_pred = self.y_train[:, 2], self.y_train_pred[:, 2]
        yita_zong_test_true, yita_zong_test_pred = self.y_test[:, 2], self.y_test_pred[:, 2]
        metrics['yita_zong'] = {
            'train': {'MAE': mean_absolute_error(yita_zong_train_true, yita_zong_train_pred),
                      'RMSE': np.sqrt(mean_squared_error(yita_zong_train_true, yita_zong_train_pred)),
                      'R²': r2_score(yita_zong_train_true, yita_zong_train_pred)},
            'test': {'MAE': mean_absolute_error(yita_zong_test_true, yita_zong_test_pred),
                     'RMSE': np.sqrt(mean_squared_error(yita_zong_test_true, yita_zong_test_pred)),
                     'R²': r2_score(yita_zong_test_true, yita_zong_test_pred)}
        }
        
        return metrics

    def export_prediction_comparisons(self, save_dir='comparison_data'):
        os.makedirs(save_dir, exist_ok=True)

        labels = ['Speed', 'Utilize_Efficiency', 'Total_Efficiency']
        for i, label in enumerate(labels):
            df = pd.DataFrame({
                f'{label}_True': self.y_test[:, i],
                f'{label}_Pred': self.y_test_pred[:, i]
            })
            file_path = os.path.join(save_dir, f'{label.lower()}_comparison.csv')
            df.to_csv(file_path, index=False)
    def loss_curves(self,save_dir=None, show=True):
        loss = self.model.steps[1][1].loss_curve_[:135]
        with open(save_dir, 'w') as f:
            for i, val in enumerate(loss):
                f.write(f"{i}\t{val:.8f}\n")


# 超参数调优：随机搜索
def hyperparameter_tuning(X, y_yita_zong, n_iter=25):
    """优化后的超参数调优（移除n_outputs_参数）"""
    param_dist = {
        'mlpregressor__hidden_layer_sizes': [
            (128, 64), (256, 128), (128, 64, 32), (256, 128, 64)
        ],
        'mlpregressor__alpha': np.logspace(-5, -3, 5),
        'mlpregressor__learning_rate': ['constant', 'adaptive'],
        'mlpregressor__learning_rate_init': np.logspace(-4, -2, 10),
        'mlpregressor__early_stopping': [True],
        'mlpregressor__n_iter_no_change': [20, 30, 50], 
        'mlpregressor__max_iter': [1000, 2000, 5000],
        'mlpregressor__solver': ['adam']
    }
    pipeline = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            random_state=42,
            max_iter=5000,
            early_stopping=True,
            validation_fraction=0.1
            # 关键修改：移除n_outputs_参数
        )
    )
    # 设置 n_jobs=1 禁用并行计算
    random_search = RandomizedSearchCV(
        pipeline, param_dist, n_iter=n_iter, cv=3, scoring='neg_mean_squared_error', n_jobs=1, verbose=1
    )
    random_search.fit(X, y_yita_zong)
    return random_search.best_estimator_

# 集成学习：Stacking融合
def ensemble_model(X, y_yita_zong):
    base_models = [
        ('mlp', MLPRegressor(hidden_layer_sizes=(128, 64), random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=200, random_state=42)),
        ('gbm', GradientBoostingRegressor(random_state=42))
    ]
    meta_model = LinearRegression()
    
    stacking = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        n_jobs=1
    )
    stacking.fit(X, y_yita_zong)
    return stacking


# 主程序调用
if __name__ == "__main__":
    constants = (1, 0.62, 1.4, 287, 718, 1005, 300, 0.001256)
    Patm = 101325  # 环境压力，用于约束条件
    
    print("开始生成训练数据...")
    X, y = generate_training_data(constants, num_samples=10000)
    print(f"数据生成完成，有效样本数量: {X.shape[0]}")
    
    print("开始训练代理模型...")
    surrogate = SurrogateModel()
    surrogate.train_with_validation(X, y)
    print("代理模型训练完成")
    
    print("开始超参数调优...")
    # 超参数调优
    best_estimator = hyperparameter_tuning(X, y[:, 2])
    print("超参数调优完成")
    
    print("开始集成学习...")
    # 集成学习
    stacking_model = ensemble_model(X, y[:, 2])
    print("集成学习完成")
    
    # 计算验证指标
    metrics = surrogate.calculate_validation_metrics()
    print("\n=== 代理模型验证指标 ===")
    
    # 速度指标输出
    print("---------------------- 速度 (v_end) ----------------------")
    print(f"训练集 MAE: {metrics['v']['train']['MAE']:.4f} m/s")
    print(f"训练集 RMSE: {metrics['v']['train']['RMSE']:.4f} m/s")
    print(f"训练集 R²: {metrics['v']['train']['R²']:.4f}")
    print(f"测试集 MAE: {metrics['v']['test']['MAE']:.4f} m/s")
    print(f"测试集 RMSE: {metrics['v']['test']['RMSE']:.4f} m/s")
    print(f"测试集 R²: {metrics['v']['test']['R²']:.4f}")
    
    # 效率指标输出
    print("\n---------------------- 效率 (yita) ----------------------")
    print(f"训练集 MAE: {metrics['yita']['train']['MAE']:.4f} %")
    print(f"训练集 RMSE: {metrics['yita']['train']['RMSE']:.4f} %")
    print(f"训练集 R²: {metrics['yita']['train']['R²']:.4f}")
    print(f"测试集 MAE: {metrics['yita']['test']['MAE']:.4f} %")
    print(f"测试集 RMSE: {metrics['yita']['test']['RMSE']:.4f} %")
    print(f"测试集 R²: {metrics['yita']['test']['R²']:.4f}")
    
    # 总体效率指标输出
    print("\n------------------ 总体效率 (yita_zong) ------------------")
    print(f"训练集 MAE: {metrics['yita_zong']['train']['MAE']:.6f}")
    print(f"训练集 RMSE: {metrics['yita_zong']['train']['RMSE']:.6f}")
    print(f"训练集 R²: {metrics['yita_zong']['train']['R²']:.4f}")
    print(f"测试集 MAE: {metrics['yita_zong']['test']['MAE']:.6f}")
    print(f"测试集 RMSE: {metrics['yita_zong']['test']['RMSE']:.6f}")
    print(f"测试集 R²: {metrics['yita_zong']['test']['R²']:.4f}")
    
    # 绘制对比图
    print("生成预测对比数据")
    surrogate.export_prediction_comparisons('figures/data')
    

    surrogate.loss_curves('loss_curve.txt')



