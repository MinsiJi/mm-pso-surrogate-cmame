import numpy as np
from scipy.integrate import solve_ivp
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from scipy.stats import qmc
import os
os.chdir(os.path.dirname(__file__))  # 切换到当前脚本所在目录
# 配置支持数学符号的字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']  # 优先使用DejaVu
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学符号使用STIX字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.rcParams['text.usetex'] = False  # 禁用LaTeX引擎

# 已知参常数
m = 1.0  # 弹射物体质量
Rd=0.02
A = np.pi*Rd**2  # 面积
gamma = 1.4  # 比热比
R = 287.0  # 气体常数

Cv = R / (gamma - 1)  # 定容比热
Cp = gamma * Cv  # 定压比热
T0 = 288  # 环境温度
Patm = 101325  # 环境压力

constants = (m, gamma, R, Cv, Cp, T0, A, Patm)

def cd_model( P1, T1, A1, V1):
    """
    可变流量系数 Cd:
    """

    if T1 <= 0: T1 = 1e-8

    p0_MPa = P1 / 1e6        
    T0_K   = T1               
    A1_m2 = A1         
    V1_m3  = V1               

    Cd = (1.90 + 0.0012 * V1_m3**2 - 0.017 * p0_MPa**2 - 3.02 * V1_m3 * p0_MPa
              + 0.27 * V1_m3 * T0_K + 0.099 * A1_m2 * p0_MPa + 10.28 * A1_m2 * T0_K
              + 0.0098 * V1_m3 + 0.015 * A1_m2 - 0.0086 * p0_MPa - 0.0054 * T0_K)

    Cd = float(np.clip(Cd, 0.01, 1.0))
    return Cd
def unified_equations(t, state, P1, T1, A1, V1, L, Patm):
    v, M, T, x = state
    
    # 确保物理量的合理性
    if T <= 0:
        T = 1e-10
    if M <= 0:
        M = 1e-10
    
    # 计算体积和压力
    V = V1 + A * x
    P = M * R * T / V
    
    # 方程二：质量泄漏
    Cd_n=cd_model(P, T, A1, V)
    leak = Cd_n * A1 * P * np.sqrt(gamma / (R * T)) * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
    dMdt = -leak
    
    # 方程一：加速度
    dvdt = P * A / m if P > Patm else 0  # 压力低于环境时不再加速
    
    # 方程五：温度变化
    dTdt = (-R * T * leak - P * A * v) / (Cv * M) if M > 1e-6 else 0
    
    # 位移变化
    dxdt = v
    
    return [dvdt, dMdt, dTdt, dxdt]

def calculate_efficiency(P1, T1, A1, V1, L, Patm=101325, method='Radau'):
    # 初始条件
    M0 = P1 * V1 / (R * T1)
    v0 = 0
    T0 = T1
    x0 = 0
    initial_state = [v0, M0, T0, x0]
    
    def event(t, state, *args):
        return state[3] - L
    event.terminal = True
    
    # 求解方程组
    solution = solve_ivp(
        unified_equations, 
        t_span=(0, 0.1),
        y0=initial_state, 
        args=(P1, T1, A1, V1, L, Patm), 
        method=method,
        events=event,
        dense_output=True
    )
    
    # 提取最终状态
    if solution.t_events[0].size > 0:
        t_end = solution.t_events[0][0]
        v_final, M_final, T_final, x_final = solution.sol(t_end)
    else:
        v_final, M_final, T_final, x_final = solution.y[:, -1]
    
    # 确保物理量的合理性
    if T_final <= 0:
        T_final = 1e-10
    
    # 计算最终体积和压力
    V_final = V1 + A * x_final
    P_final = M_final * R * T_final / V_final
    h = Cp*T0
    
    # 检查是否满足压力约束
    pressure_constraint = P_final >= Patm
    
    # 计算效率
    if pressure_constraint and x_final >= L - 1e-3:
        leak_energy = 0
        for i in range(1, len(solution.t)):
            dt = solution.t[i] - solution.t[i - 1]  # 时间间隔
            
            # 获取当前时间步的状态变量
            t_current = solution.t[i]
            v_current = solution.y[0][i]       # 当前速度
            M_current = solution.y[1][i]       # 当前质量
            T_current = solution.y[2][i]       # 当前温度
            x_current = solution.y[3][i]       # 当前位移
            
                # 当前体积
            V_current = V1 + A * max(x_current, 0.0)
            
            # 当前压力
            P_current = M_current * R * T_current / V_current if V_current > 1e-12 else 1.0
            
            # 动态计算当前Cd
            Cd_current = cd_model(P_current, T_current, A1, V_current) 
            # 调用cd_model计算当前时间步的Cd
            Cd_current = cd_model(
                P1=P_current,
                T1=T_current,
                A1=A1,
                V1=V_current,
            )
            
            V_current = V1 + A * max(x_current, 0.0)
            
            # 计算当前压力
            if V_current > 1e-12:
                P_current = M_current * R * T_current / V_current
            else:
                P_current = 1.0 
            
            # 计算当前泄漏率
            leak = Cd_current * A1 * P_current * np.sqrt(gamma / (R * T_current)) * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))

            h = Cp * T_current
            leak_energy += leak * h * dt  
        
        # 计算能量
        U0 = Cp * M0 * T1
        Uf = Cp * M_final * T_final
        W_total = U0 - Uf - leak_energy
        yita = 0.5 * m * v_final ** 2 / W_total if W_total > 0 else 0
        yita_zong = 0.5 * m * v_final ** 2 / U0
    else:
        yita = 1e-6
        yita_zong = 1e-6
        pressure_constraint = False
    
    return yita, yita_zong, v_final, x_final, pressure_constraint, P_final

def dynamic_penalty(iteration, max_iterations):
    """动态惩罚系数，随迭代次数增加而增强"""
    return 1 + 10 * (iteration / max_iterations) ** 2

def _normalize_params(params, bounds):
    """把参数按边界归一化到[0,1]，便于计算距离"""
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    return (np.array(params) - lb) / (ub - lb + 1e-12)

def _pairwise_min_distance(points):
    """点集中每个点到最近其它点的距离（欧氏）"""
    if len(points) <= 1:
        return np.array([0.0]*len(points))
    P = np.array(points, dtype=float)
    D = np.sqrt(((P[:, None, :] - P[None, :, :])**2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    return D.min(axis=1)

def _dedup_by_threshold(points, thresh):
    """按最近邻距离阈值去重，返回代表解索引"""
    if len(points) == 0:
        return []
    keep = []
    used = np.zeros(len(points), dtype=bool)
    for i in np.argsort([-p[0] for p in points]):  # 按性能从高到低选代表
        if used[i]: 
            continue
        keep.append(i)
        # 标记与代表点太近的为已用
        for j in range(len(points)):
            if used[j] or j == i:
                continue
            dist = np.linalg.norm(points[i][1] - points[j][1])
            if dist <= thresh:
                used[j] = True
        used[i] = True
    return keep

def compute_metrics_and_plots(all_runs_results, weight_labels, bounds,
                              delta=0.02, d0=0.15, tau_mode='p90',
                              outdir='results'):

    import csv
    os.makedirs(outdir, exist_ok=True)
    all_scores = []

    # 汇总所有 run 的得分用于确定全局阈值/最优
    for runs in all_runs_results:
        for r in runs:
            obj = float(r['objective'])
            if np.isfinite(obj):
                all_scores.append(-obj)

    if len(all_scores) == 0:
        raise RuntimeError("没有可用的 objective 值。")

    f_star = np.max(all_scores)
    if tau_mode == 'p90':
        tau = float(np.percentile(all_scores, 90))
        tau_desc = 'p90(final)'
    elif tau_mode == 'best*98%':
        tau = 0.98 * f_star
        tau_desc = '0.98*best(final)'
    else:
        # 也允许你传固定数值
        tau = float(tau_mode)
        tau_desc = f'fixed={tau:.6g}'

    csv_path = os.path.join(outdir, 'metrics_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            '权重组合', 'N', '终值中位数(Score)', '终值IQR(Score)', 'SR_tau(%)',
            'Topδ阈值(Score)', 'C_delta', 'D_delta_median'
        ])
        writer.writeheader()

        box_data_eff = []
        box_data_total_eff = []
        box_data_vel = []
        box_labels = []

        for w_idx, runs in enumerate(all_runs_results):
            # —— 统一在一次循环里收集 —— 
            pairs = []                  # [(run, score=−objective), ...]
            effs, teffs, vels = [], [], []

            for r in runs:
                obj = float(r['objective'])
                if np.isfinite(obj):
                    score = -obj        # 统一成“得分”（越大越好）
                    pairs.append((r, score))
                    effs.append(r['efficiency'] * 100.0)
                    teffs.append(r['total_efficiency'] * 100.0)
                    vels.append(r['velocity'])

            # 本组若无有效样本，写空并继续
            if not pairs:
                writer.writerow({
                    '权重组合': weight_labels[w_idx],
                    'N': len(runs),
                    '终值中位数(Score)': '',
                    '终值IQR(Score)': '',
                    'SR_tau(%)': '0.00',
                    'Topδ阈值(Score)': '',
                    'C_delta': 0,
                    'D_delta_median': ''
                })
                box_data_eff.append([])
                box_data_total_eff.append([])
                box_data_vel.append([])
                box_labels.append(weight_labels[w_idx])
                continue

            # ===== 统计（全部基于 pairs，确保一一对应）=====
            scores = np.asarray([s for _, s in pairs], dtype=float)

            # FBQ（得分中位数/IQR）
            med = float(np.median(scores))
            iqr = float(np.percentile(scores, 75) - np.percentile(scores, 25))

            # 成功率（相对 τ）
            sr = 100.0 * float(np.mean(scores >= tau))

            # Top-δ 阈值（“组内”Top-δ，更公平）
            thr = (1.0 - delta) * float(np.max(scores))

            # 候选与去重
            cand = []
            for r, s in pairs:
                if (s >= thr) and r.get('pressure_ok', True):   # 如需先不强制可行，去掉第二个条件即可
                    x_norm = _normalize_params(r['parameters'], bounds)
                    cand.append((float(s), x_norm))

            C_delta, D_delta_med = 0, 0.0
            if len(cand) > 0:
                keep_idx = _dedup_by_threshold(cand, d0)
                reps = [cand[i] for i in keep_idx]
                C_delta = len(reps)
                if C_delta >= 2:
                    rep_points = np.array([p[1] for p in reps])
                    dmins = _pairwise_min_distance(rep_points)
                    D_delta_med = float(np.median(dmins))
                # 画带宽图
                _plot_topdelta_bandwidth(reps, weight_labels[w_idx], outdir)

            # 写表
            writer.writerow({
                '权重组合': weight_labels[w_idx],
                'N': len(runs),
                '终值中位数(Score)': f'{med:.6f}',
                '终值IQR(Score)': f'{iqr:.6f}',
                'SR_tau(%)': f'{sr:.2f}',
                'Topδ阈值(Score)': f'{thr:.6f}',
                'C_delta': C_delta,
                'D_delta_median': f'{D_delta_med:.4f}',
            })

            # 箱线图数据
            box_data_eff.append(effs)
            box_data_total_eff.append(teffs)
            box_data_vel.append(vels)
            box_labels.append(weight_labels[w_idx])


    print(f"[OK] 指标已保存：{csv_path}")
    print(f"[OK] 箱线图与Top-δ带宽图已输出到：{outdir}")


def _plot_box(data_list, labels, ylabel, save_path):
    plt.figure(figsize=(8, 4.8))
    plt.boxplot(data_list, labels=labels, showfliers=False)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig(save_path, dpi=1200)
    plt.close()

def _plot_topdelta_bandwidth(reps, weight_label, outdir):
    reps_sorted = sorted(reps, key=lambda z: z[0], reverse=True)
    objs = np.array([z[0] for z in reps_sorted], dtype=float)
    X = np.array([z[1] for z in reps_sorted], dtype=float)

    # 最近邻距离
    dmins = _pairwise_min_distance(X)
    idx = np.arange(1, len(reps_sorted)+1)

    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    ax1.plot(idx, objs, marker='o', linewidth=2)
    ax1.set_xlabel('Representative solution index (Top-δ)')
    ax1.set_ylabel('Objective (higher is better)')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(idx, dmins, alpha=0.35)
    ax2.set_ylabel('Nearest-neighbor distance (normalized space)')

    plt.title(f"Top-δ Portfolio — {weight_label}")
    plt.tight_layout()
    fname = f"topdelta_band_{weight_label.replace(' ', '').replace('=', '').replace(',', '_')}.png"
    plt.savefig(os.path.join(outdir, fname), dpi=1200)
    plt.close()

def _first_hit_indices(histories, tau):
    first_hits = []
    for h in histories:
        h = np.asarray(h, dtype=float)
        idx = np.where(h >= tau)[0]
        first_hits.append(idx[0] if idx.size > 0 else np.inf)
    return np.array(first_hits)

def _ecdf_curve(first_hits, T):
    N = len(first_hits)
    y = np.zeros(T, dtype=float)
    # 对每个 t 统计命中率
    for t in range(T):
        y[t] = np.mean(first_hits <= t)
    return y

def compute_and_plot_ecdf_ert(all_weight_histories, weight_labels,
                              tau_mode='p90', outdir='results',
                              per_weight_tau=False):
    import csv
    os.makedirs(outdir, exist_ok=True)

    # 统一长度
    max_T = max(max(len(h) for h in runs) for runs in all_weight_histories)

    # 如果是全局 τ：先算所有 finals
    if not per_weight_tau:
        finals_all = np.asarray([np.asarray(h)[-1]
                                 for runs in all_weight_histories for h in runs], float)
        tau_global = (float(np.percentile(finals_all, 90)) if tau_mode=='p90'
                      else 0.98*float(np.max(finals_all)) if tau_mode=='best*98%' 
                      else float(tau_mode))
        tau_desc_global = f"{'p90(final)' if tau_mode=='p90' else ('0.98*best(final)' if tau_mode=='best*98%' else f'fixed={tau_global:.6g}')}"
    # 叠加对比图
    plt.figure(figsize=(8,4.8))

    rows = []
    for w_idx, runs in enumerate(all_weight_histories):
        # 组内 finals
        finals = np.asarray([np.asarray(h, float)[-1] for h in runs], float)
        if per_weight_tau:
            tau = (float(np.percentile(finals, 90)) if tau_mode=='p90'
                   else 0.98*float(np.max(finals)) if tau_mode=='best*98%'
                   else float(tau_mode))
            tau_desc = f"{'p90(final)' if tau_mode=='p90' else ('0.98*best(final)' if tau_mode=='best*98%' else f'fixed={tau:.6g}')}"
        else:
            tau, tau_desc = tau_global, tau_desc_global

        # 标准化长度
        runs_fixed = []
        for h in runs:
            h = np.asarray(h, float)
            if len(h)<max_T:
                h = np.concatenate([h, np.full(max_T-len(h), h[-1])])
            runs_fixed.append(h)
        runs_fixed = np.asarray(runs_fixed, float)

        # 首次命中
        first_hits = _first_hit_indices(runs_fixed, tau)
        y = _ecdf_curve(first_hits, max_T)

        # 单图
        plt.figure(figsize=(8,4.8))
        plt.plot(np.arange(max_T), y, linewidth=2)
        plt.xlabel('Iteration (budget proxy)')
        plt.ylabel('Proportion of runs ≥ τ')
        plt.title(f"ECDF — {weight_labels[w_idx]}  (τ: {tau_desc}, {tau:.6g})")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        fname = f"ecdf_{weight_labels[w_idx].replace(' ', '').replace('=', '').replace(',', '_')}.png"
        plt.savefig(os.path.join(outdir, fname), dpi=600); plt.close()

        # 叠加
        plt.plot(np.arange(max_T), y, linewidth=2, label=weight_labels[w_idx])

        # 统计
        success = np.isfinite(first_hits)
        sr = 100.0*np.mean(success)
        ert = float(np.mean(first_hits[success])) if np.any(success) else float('inf')
        med_hit = float(np.median(first_hits[success])) if np.any(success) else float('inf')
        rows.append([weight_labels[w_idx], len(runs), tau_desc, f'{tau:.6f}',
                     f'{sr:.2f}', 'inf' if np.isinf(ert) else f'{ert:.2f}',
                     'inf' if np.isinf(med_hit) else f'{med_hit:.2f}'])

    # 汇总对比
    plt.xlabel('Iteration (budget proxy)'); plt.ylabel('Proportion of runs ≥ τ')
    ttl = "ECDF (All weights) — per-weight τ" if per_weight_tau else f"ECDF (All weights) — τ: {tau_desc_global}, {tau_global:.6g}"
    plt.title(ttl); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'ecdf_all_weights.png'), dpi=800); plt.close()

    # CSV
    import csv
    with open(os.path.join(outdir,'ecdf_ert_summary.csv'),'w',newline='') as f:
        w = csv.writer(f)
        w.writerow(['权重组合','N_runs','Tau描述','Tau值','SR_tau(%)','ERT(迭代步)','中位首次命中步'])
        w.writerows(rows)


def objective_function(params, constants, iteration, max_iterations, alpha=1.0, beta=1.0, gamma_weight=0.0, v_ref=60.0):
    P1, T1, A1, V1, L = params
    m, gamma, R, Cv, Cp, T0, A, Patm = constants

    # 精确计算物理量
    yita, yita_zong, v_final, x_final, pressure_ok, P_final = calculate_efficiency(P1, T1, A1, V1, L, Patm)

    # 复合目标：加权和
    obj_value = alpha * yita + beta * yita_zong + gamma_weight * (v_final / v_ref)

    # 惩罚项
    penalty = 0.0
    if v_final < v_ref:
        penalty += dynamic_penalty(iteration, max_iterations) * (v_ref - v_final)
    if not pressure_ok:
        penalty += dynamic_penalty(iteration, max_iterations) * (Patm - P_final) * 100
    if x_final < L - 1e-3:
        penalty += dynamic_penalty(iteration, max_iterations) * (L - x_final) * 100

    return -obj_value + penalty  # 最小化目标

class SurrogateModel:
    def __init__(self):
        # 多输出代理模型（v, η, η_total）
        self.model = make_pipeline(
            StandardScaler(),  # 输入标准化
            MLPRegressor(
                hidden_layer_sizes=(256, 128, 64),
                activation='relu',
                solver='adam',
                max_iter=5000,
                alpha=0.0005,
                learning_rate='adaptive',
                random_state=42
            )
        )
        self.scaler_y = StandardScaler()  # 输出标准化器

    def train(self, X, y):
        """训练：输入X，输出[y_v, y_η, y_η_total]"""
        y_scaled = self.scaler_y.fit_transform(y)
        self.model.fit(X, y_scaled)

    def predict(self, X):
        """预测并逆标准化"""
        y_pred_scaled = self.model.predict(X)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred 

# 生成训练数据
def generate_training_data(constants, num_samples=10000):
    """使用拉丁超立方采样生成训练数据"""
    sampler = qmc.LatinHypercube(d=5)  # 5维参数空间
    params = sampler.random(num_samples)
    low = np.array([5e5, 300, 1e-5, 0.001, 0.6])
    high = np.array([10e6, 800, 5e-5, 0.005, 1.5])
    params = qmc.scale(params, low, high)  # 映射到参数范围
    X, y = [], []
    m, gamma, R, Cv, Cp, T0, A, Patm = constants
    for p in params:
        P1, T1, A1, V1, L = p
        # 解包6个返回值
        yita, yita_zong, v_end, x_end, pressure_ok, P_final = calculate_efficiency(P1, T1, A1, V1, L, Patm)
        if pressure_ok and x_end >= L - 1e-3:  # 保留有效样本
            X.append(p)
            # 保存 v、效率η、总效率η_total
            y.append([v_end, yita, yita_zong])
    return np.array(X), np.array(y)

# 粒子群优化器
class HybridPSO:
    def __init__(self, n_particles, dimensions, bounds, constants, max_iter, alpha=1.0, beta=1.0, gamma_weight=0.0, v_ref=60.0):
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.bounds = bounds  # 各参数上下限
        self.constants = constants
        self.max_iter = max_iter
        self.global_best_position = None
        self.global_best_value = float('inf')
        self.particles = []
        self.convergence_history = []  # 保存每次迭代的最优适应度
        # 权重参数
        self.alpha = alpha
        self.beta = beta
        self.gamma_weight = gamma_weight
        self.v_ref = v_ref
        
    def optimize(self, surrogate_model):
        for i in range(self.max_iter):
            for particle in self.particles:
                # 70%概率用代理模型，30%用精确计算
                if np.random.rand() < 0.7:
                    # 代理模型预测 v, η, η_total
                    pred = surrogate_model.predict([particle.position])[0]
                    pred_v, pred_yita, pred_yita_zong = pred

                    # 简化压力约束判断（仅用于代理模型阶段）
                    P1, T1, A1, V1, L = particle.position
                    V_final = V1 + A * L
                    M_final = P1 * V1/(R*T1) -0.62*A1*P1*np.sqrt(gamma/(R*T1)) * (2/(gamma+1))**((gamma+1)/(2*(gamma-1))) * 0.05
                    T_final = T1 * (V1/V_final)**(gamma-1)
                    P_final = M_final*R*T_final/V_final
                    pressure_ok = P_final >= self.constants[-1]

                    if pressure_ok:
                        penalty = 0.0
                        if pred_v < self.v_ref:
                            penalty += dynamic_penalty(i, self.max_iter) * (self.v_ref - pred_v)
                        # 代理模型预测的目标值
                        obj_value = self.alpha * pred_yita + self.beta * pred_yita_zong + self.gamma_weight * (pred_v / self.v_ref)
                        particle.fitness = -obj_value + penalty
                    else:
                        particle.fitness = float('inf')
                else:
                    # 精确计算目标函数
                    particle.fitness = objective_function(
                        particle.position, self.constants, i, self.max_iter,
                        alpha=self.alpha, beta=self.beta, gamma_weight=self.gamma_weight, v_ref=self.v_ref
                    )

                # 更新全局最优
                if particle.fitness < self.global_best_value:
                    self.global_best_value = particle.fitness
                    self.global_best_position = particle.position.copy()

            # 记录收敛历史
            self.convergence_history.append(self.global_best_value)

            # 动态惯性权重
            w = 0.9 - 0.5 * (i / self.max_iter)
            
            # 更新粒子速度和位置
            for particle in self.particles:
                particle.update_velocity(self.global_best_position, w)
                particle.apply_bounds()
            
            # 引入遗传算法操作
            if i % 10 == 0:  # 每10代进行一次遗传操作
                self.genetic_operation()
        
        # 最终评估（使用精确模型）
        yita, yita_zong, v_final, x_final, pressure_ok, P_final = calculate_efficiency(
            *self.global_best_position, Patm=self.constants[-1]
        )
        final_obj = self.alpha * yita + self.beta * yita_zong + self.gamma_weight * (v_final / self.v_ref)
        
        # 输出最终优化结果
        print(f"Final Optimization Results (α={self.alpha}, β={self.beta}, γ={self.gamma_weight}):")
        print(f"  Optimized Efficiency: {yita*100:.2f}%")
        print(f"  Total Efficiency: {yita_zong*100:.2f}%")
        print(f"  Final Velocity: {v_final:.2f} m/s")
        print(f"  Final Position: {x_final:.2f} m")
        print(f"  Final Pressure: {P_final/1e5:.2f} bar (Constraint satisfied: {pressure_ok})")
        optimized_params_formatted = [f"{param:.6f}" for param in self.global_best_position]
        print(f"  Optimized Parameters: {optimized_params_formatted}")
        
        return self.global_best_position, final_obj, self.convergence_history

    def genetic_operation(self):
        # 选择操作
        fitness_values = np.array([p.fitness for p in self.particles])
        probabilities = np.exp(-fitness_values) / np.sum(np.exp(-fitness_values))
        selected_indices = np.random.choice(self.n_particles, size=self.n_particles, p=probabilities)
        selected_particles = [self.particles[i] for i in selected_indices]
        
        # 交叉操作
        new_particles = []
        for j in range(0, self.n_particles, 2):
            parent1 = selected_particles[j]
            parent2 = selected_particles[j + 1]
            child1, child2 = self.crossover(parent1, parent2)
            new_particles.extend([child1, child2])
        
        # 变异操作
        for particle in new_particles:
            self.mutate(particle)
        
        self.particles = new_particles

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, len(parent1.position))
        child1_position = np.concatenate(
            (parent1.position[:crossover_point], parent2.position[crossover_point:]))
        child2_position = np.concatenate(
            (parent2.position[:crossover_point], parent1.position[crossover_point:]))
        child1 = Particle(child1_position, np.zeros_like(child1_position), self.bounds)
        child2 = Particle(child2_position, np.zeros_like(child2_position), self.bounds)
        return child1, child2

    def mutate(self, particle):
        mutation_rate = 0.1
        for k in range(len(particle.position)):
            if np.random.rand() < mutation_rate:
                particle.position[k] = np.random.uniform(self.bounds[k][0], self.bounds[k][1])
        particle.apply_bounds()

# 粒子类
class Particle:
    def __init__(self, position, velocity, bounds):
        self.position = position
        self.velocity = velocity
        self.bounds = bounds
        self.fitness = float('inf')

    def update_velocity(self, global_best_position, w):
        c1 = 1.5
        c2 = 1.5
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))
        cognitive = c1 * r1 * (self.position - global_best_position)
        social = c2 * r2 * (self.position - global_best_position)
        self.velocity = w * self.velocity + cognitive + social

    def apply_bounds(self):
        for i in range(len(self.position)):
            self.position[i] = np.clip(self.position[i], self.bounds[i][0], self.bounds[i][1])
# 收敛曲线
def plot_convergence(all_histories, title, save_path):
    plt.figure(figsize=(8, 4.8))
    for i, history in enumerate(all_histories):
        plt.plot(history, alpha=0.3, label=f'Run {i+1}' if i < 5 else None)
    plt.plot(np.mean(all_histories, axis=0), 'k-', linewidth=2, label='Mean')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness (Negative Objective)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path, dpi=1200)
    plt.close()

# 保存所有结果到CSV
def save_all_results(all_runs_results, weight_labels):
    import csv
    os.makedirs('results', exist_ok=True)
    
    with open('results/all_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['权重组合', '运行编号', '效率(%)', '总效率(%)', '速度(m/s)', '最终压力(bar)', 
                      'P1', 'T1', 'A1', 'V1', 'L', '满足约束']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for w_idx, weight_label in enumerate(weight_labels):
            for run_idx, run in enumerate(all_runs_results[w_idx]):
                writer.writerow({
                    '权重组合': weight_label,
                    '运行编号': run_idx + 1,
                    '效率(%)': run['efficiency'] * 100,
                    '总效率(%)': run['total_efficiency'] * 100,
                    '速度(m/s)': run['velocity'],
                    '最终压力(bar)': run['final_pressure'] / 1e5,
                    'P1': run['parameters'][0],
                    'T1': run['parameters'][1],
                    'A1': run['parameters'][2],
                    'V1': run['parameters'][3],
                    'L': run['parameters'][4],
                    '满足约束': run['pressure_ok']
                })

# 主程序
if __name__ == "__main__":
    BASE_SEED = 42
    num_runs_per_weight = 20  # 每个权重组合运行20次
    
    # 权重组合
    weight_combinations = [
        (0.5, 0.5, 0.0),  # 侧重效率
        (0.33, 0.33, 0.33),  # 平衡
        (0.25, 0.25, 0.5),  # 侧重速度
        (0.2, 0.8, 0.0),  # 侧重效率，总效率次之
        (0.8, 0.2, 0.0)   # 侧重总效率，效率次之
    ]
    weight_labels = [f"α={w[0]},β={w[1]},γ={w[2]}" for w in weight_combinations]
    v_ref = 60.0
    
    # 生成训练数据
    print("生成训练数据...")
    X, y = generate_training_data(constants, num_samples=5000)
    
    # 训练代理模型
    print("训练代理模型...")
    model = SurrogateModel()
    model.train(X, y)
    
    # 存储所有运行结果
    all_runs_results = []  
    all_weight_histories = []
    # 为每个权重组合执行多次运行
    for w_idx, weights in enumerate(weight_combinations):
        alpha, beta, gamma_weight = weights
        print(f"\n=== 权重组合 {w_idx+1}/{len(weight_combinations)}: (α={alpha}, β={beta}, γ={gamma_weight}) ===")
        
        weight_runs_results = []  # 存储当前权重的所有运行结果
        weight_convergence_histories = []  # 存储收敛历史
        
        for run_idx in range(num_runs_per_weight):
            print(f"  运行 {run_idx+1}/{num_runs_per_weight}...")
            
            # 设置随机种子
            run_seed = BASE_SEED + w_idx * num_runs_per_weight + run_idx
            np.random.seed(run_seed)
            
            # 初始化PSO
            pso = HybridPSO(
                n_particles=50, 
                dimensions=5, 
                bounds=np.array([[5e5, 10e6], [300, 800], [1e-5, 5e-5], [0.001, 0.005], [0.6, 1.5]]),
                constants=constants, 
                max_iter=100,
                alpha=alpha, 
                beta=beta, 
                gamma_weight=gamma_weight, 
                v_ref=v_ref
            )
            
            # 初始化粒子
            for j in range(pso.n_particles):
                np.random.seed(run_seed + j)
                position = np.random.uniform(pso.bounds[:, 0], pso.bounds[:, 1])
                velocity = np.random.uniform(-1, 1, pso.dimensions)
                pso.particles.append(Particle(position, velocity, pso.bounds))
            
            # 运行优化
            best_params, best_obj, convergence_history = pso.optimize(model)
            weight_convergence_histories.append([-x for x in convergence_history])  # 转换回正值
            
            # 使用精确模型评估最终结果
            yita, yita_zong, v_final, x_final, pressure_ok, P_final = calculate_efficiency(
                *best_params, Patm
            )
            
            # 保存结果
            weight_runs_results.append({
                'weights': weights,
                'efficiency': yita,
                'total_efficiency': yita_zong,
                'velocity': v_final,
                'position': x_final,
                'pressure_ok': pressure_ok,
                'final_pressure': P_final,
                'parameters': best_params,
                'objective': pso.global_best_value
            })
        
        # 保存当前权重的所有运行结果
        all_runs_results.append(weight_runs_results)
        
        # 绘制当前权重的收敛曲线对比图
        plot_convergence(
            weight_convergence_histories,
            f"Comparison of Convergence Curves from Multiple Runs (α={alpha}, β={beta}, γ={gamma_weight})",
            f"convergence_weight_{w_idx+1}.png"
        )
        all_weight_histories.append(weight_convergence_histories)
        # 使用最佳参数进行敏感性分析
        best_run = max(weight_runs_results, key=lambda x: x['efficiency'])

    
    # 保存所有结果到CSV
    save_all_results(all_runs_results, weight_labels)
    
    # 输出统计摘要
    print("\n\n------------------- 统计摘要 -------------------")
    for w_idx, weight_label in enumerate(weight_labels):
        runs = all_runs_results[w_idx]
        
        # 提取指标
        efficiencies = [run['efficiency'] * 100 for run in runs]
        total_efficiencies = [run['total_efficiency'] * 100 for run in runs]
        velocities = [run['velocity'] for run in runs]
        pressures = [run['final_pressure'] / 1e5 for run in runs]
        constraint_satisfied = [run['pressure_ok'] for run in runs]
        
        # 计算统计量
        stats = {
            'efficiency': {
                'mean': np.mean(efficiencies),
                'std': np.std(efficiencies),
                'max': max(efficiencies),
                'min': min(efficiencies)
            },
            'total_efficiency': {
                'mean': np.mean(total_efficiencies),
                'std': np.std(total_efficiencies),
                'max': max(total_efficiencies),
                'min': min(total_efficiencies)
            },
            'velocity': {
                'mean': np.mean(velocities),
                'std': np.std(velocities),
                'max': max(velocities),
                'min': min(velocities)
            },
            'pressure': {
                'mean': np.mean(pressures),
                'std': np.std(pressures),
                'max': max(pressures),
                'min': min(pressures)
            },
            'constraint_rate': sum(constraint_satisfied) / len(constraint_satisfied)
        }
        
        # 输出结果
        print(f"\n权重组合 {w_idx+1}: {weight_label}")
        print(f"  效率: {stats['efficiency']['mean']:.2f}% ± {stats['efficiency']['std']:.2f}% (范围: {stats['efficiency']['min']:.2f}% - {stats['efficiency']['max']:.2f}%)")
        print(f"  总效率: {stats['total_efficiency']['mean']:.2f}% ± {stats['total_efficiency']['std']:.2f}% (范围: {stats['total_efficiency']['min']:.2f}% - {stats['total_efficiency']['max']:.2f}%)")
        print(f"  速度: {stats['velocity']['mean']:.2f} m/s ± {stats['velocity']['std']:.2f} m/s (范围: {stats['velocity']['min']:.2f} - {stats['velocity']['max']:.2f} m/s)")
        print(f"  最终压力: {stats['pressure']['mean']:.2f} bar ± {stats['pressure']['std']:.2f} bar (范围: {stats['pressure']['min']:.2f} - {stats['pressure']['max']:.2f} bar)")
        print(f"  约束满足率: {stats['constraint_rate']*100:.2f}%")
        # 找出最佳运行
        best_run_idx = np.argmax(efficiencies)
        best_run = runs[best_run_idx]
        print(f"  最佳运行 (运行 {best_run_idx+1}):")
        print(f"    效率: {best_run['efficiency']*100:.2f}%")
        print(f"    总效率: {best_run['total_efficiency']*100:.2f}%")
        print(f"    速度: {best_run['velocity']:.2f} m/s")
        print(f"    参数: {[f'{p:.6f}' for p in best_run['parameters']]}")

    compute_metrics_and_plots(
        all_runs_results=all_runs_results,
        weight_labels=weight_labels,
        bounds=np.array([[5e5, 10e6], [300, 800], [1e-5, 5e-5], [0.001, 0.005], [0.6, 1.5]]),
        delta=0.02,
        d0=0.15,
        tau_mode='p90',
        outdir='results'
    )

    compute_and_plot_ecdf_ert(
        all_weight_histories, weight_labels,
        tau_mode='p90', outdir='results',
        per_weight_tau=True
    )

