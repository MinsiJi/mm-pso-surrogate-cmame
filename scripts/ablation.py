import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import qmc
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ------------------------------
# 全局绘图设置
# ------------------------------
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# ==============================
# 物理常数 & 基本模型
# ==============================
m = 1.0               # 弹丸质量 (kg)
Rd = 0.02             # 发射管半径 (m)
A  = np.pi * Rd**2    # 截面积 (m^2)
gamma = 1.4
R  = 287.0
Cv = R / (gamma - 1)
Cp = gamma * Cv
T_env = 288.0
Patm = 101325.0
CONSTS = (m, gamma, R, Cv, Cp, T_env, A, Patm)

def cd_model(P1, T1, A1, V1):
    """Cd 拟合式"""
    if T1 <= 0: T1 = 1e-8
    p0_MPa = P1 / 1e6
    T0_K   = T1
    A1_m2  = A1
    V1_m3  = V1
    Cd = (1.90 + 0.0012*V1_m3**2 - 0.017*p0_MPa**2 - 3.02*V1_m3*p0_MPa
          + 0.27*V1_m3*T0_K + 0.099*A1_m2*p0_MPa + 10.28*A1_m2*T0_K
          + 0.0098*V1_m3 + 0.015*A1_m2 - 0.0086*p0_MPa - 0.0054*T0_K)
    return float(np.clip(Cd, 0.01, 1.0))

def unified_equations(t, state, P1, T1, A1, V1, L, Patm):
    """状态: [v, M, T, x]"""
    v, M, T, x = state
    if T <= 0: T = 1e-10
    if M <= 0: M = 1e-10
    V = V1 + A * x
    P = M * R * T / V
    Cd_n = cd_model(P, T, A1, V)

    # 泄漏质量流率
    leak = Cd_n * A1 * P * np.sqrt(gamma/(R*T)) * (2/(gamma+1))**((gamma+1)/(2*(gamma-1)))

    dMdt = -leak
    dvdt = max(P - Patm, 0.0) * A / m
    dTdt = (-R*T*leak - P*A*v)/(Cv*M) if M > 1e-6 else 0.0
    dxdt = v
    return [dvdt, dMdt, dTdt, dxdt]

def calculate_efficiency(P1, T1, A1, V1, L, Patm=Patm, method='Radau'):
    """返回: eta, eta_total, v_end, x_end, pressure_ok, P_end"""
    M0 = P1*V1/(R*T1)
    state0 = [0.0, M0, T1, 0.0]

    def event_xL(t, state, *args):
        return state[3] - L
    event_xL.terminal = True

    sol = solve_ivp(unified_equations, (0, 0.1), state0,
                    args=(P1, T1, A1, V1, L, Patm),
                    method=method, events=event_xL, dense_output=True)

    if sol.t_events[0].size > 0:
        t_end = sol.t_events[0][0]
        v_end, M_end, T_end, x_end = sol.sol(t_end)
    else:
        v_end, M_end, T_end, x_end = sol.y[:, -1]

    if T_end <= 0: T_end = 1e-10
    V_end = V1 + A * x_end
    P_end = M_end * R * T_end / V_end
    pressure_ok = (P_end >= Patm)

    # 计算泄漏焓和效率
    if pressure_ok and x_end >= L - 1e-3:
        leak_energy = 0.0
        for i in range(1, len(sol.t)):
            dt = sol.t[i] - sol.t[i-1]
            M_i  = sol.y[1][i]
            T_i  = sol.y[2][i]
            x_i  = sol.y[3][i]
            V_i  = V1 + A * max(x_i, 0.0)
            P_i  = M_i * R * T_i / V_i if V_i > 1e-12 else 1.0
            Cd_i = cd_model(P_i, T_i, A1, V_i)
            leak_i = Cd_i * A1 * P_i * np.sqrt(gamma/(R*T_i)) * (2/(gamma+1))**((gamma+1)/(2*(gamma-1)))
            h_i = Cp * T_i
            leak_energy += leak_i * h_i * dt

        U0 = Cp * M0 * T1
        Uf = Cp * M_end * T_end
        W  = U0 - Uf - leak_energy
        eta = 0.5*m*v_end**2 / W if W > 0 else 1e-6
        eta_total = 0.5*m*v_end**2 / U0 if U0 > 0 else 1e-6
    else:
        eta = 1e-6
        eta_total = 1e-6
        pressure_ok = False

    return eta, eta_total, v_end, x_end, pressure_ok, P_end

# ==============================
# 消融开关（默认：Full）
# ==============================
USE_SURROGATE   = True    # Surrogate-off -> False
USE_GP          = True    # GP-off        -> False
USE_DYNAMIC_PEN = True    # DP-off        -> False
SURROGATE_PROB  = 0.70    # 仅在 USE_SURROGATE=True 时生效
GP_PERIOD       = 10

def dynamic_penalty(iteration, max_iterations):
    """动态惩罚系数；DP-off 时恒为 1.0"""
    if not USE_DYNAMIC_PEN:
        return 1.0
    return 1.0 + 10.0 * (iteration / max_iterations) ** 2

def objective_true(params, iteration, max_iterations,
                   alpha=1.0, beta=1.0, gamma_weight=0.0, v_ref=60.0):
    """返回：(fitness, feasible_flag, constraint_violation_scalar)"""
    P1, T1, A1, V1, L = params
    eta, eta_total, v_end, x_end, ok, P_end = calculate_efficiency(P1, T1, A1, V1, L, Patm)
    J = alpha*eta + beta*eta_total + gamma_weight*(v_end / v_ref)

    # 约束惩罚
    pen = 0.0
    if v_end < v_ref:
        pen += dynamic_penalty(iteration, max_iterations) * (v_ref - v_end)
    if not ok:
        pen += dynamic_penalty(iteration, max_iterations) * (Patm - P_end) * 100.0
    if x_end < L - 1e-3:
        pen += dynamic_penalty(iteration, max_iterations) * (L - x_end) * 100.0

    fit = -(J) + pen

    # 违反度（用于 MCV）
    viol = (v_ref - v_end if v_end < v_ref else 0.0) \
           + (Patm - P_end if not ok else 0.0) \
           + (L - x_end if x_end < L - 1e-3 else 0.0)

    return fit, ok, viol

# ==============================
# 代理模型
# ==============================
class SurrogateModel:
    def __init__(self):
        self.model = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(256,128,64),
                         activation='relu', solver='adam',
                         max_iter=5000, alpha=5e-4,
                         learning_rate='adaptive', random_state=42)
        )
        self.scaler_y = StandardScaler()

    def train(self, X, y):
        y_sc = self.scaler_y.fit_transform(y)
        self.model.fit(X, y_sc)

    def predict(self, X):
        y_sc = self.model.predict(X)
        y = self.scaler_y.inverse_transform(y_sc)
        # 返回顺序：v_end, eta, eta_total
        return np.array(y)

# ==============================
# PSO
# ==============================
class Particle:
    def __init__(self, position, velocity, bounds):
        self.position = position
        self.velocity = velocity
        self.bounds   = bounds
        self.fitness  = np.inf
        self.pbest     = position.copy()
        self.pbest_fit = np.inf

    def apply_bounds(self):
        self.position = np.clip(self.position, self.bounds[:,0], self.bounds[:,1])

    def update_velocity(self, gbest, w):
        c1 = 1.5; c2 = 1.5
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))
        self.velocity = (w * self.velocity
                         + c1 * r1 * (self.pbest - self.position)
                         + c2 * r2 * (gbest - self.position))

class HybridPSO:
    def __init__(self, n_particles, bounds, max_iter,
                 alpha=1.0, beta=1.0, gamma_weight=0.0, v_ref=60.0, surrogate=None):
        self.n = n_particles
        self.bounds = bounds
        self.dim = bounds.shape[0]
        self.T = max_iter
        self.alpha, self.beta, self.gamma = alpha, beta, gamma_weight
        self.v_ref = v_ref
        self.surrogate = surrogate
        self.pop = []
        self.gbest_pos = None
        self.gbest_fit = np.inf
        self.history = []
        self.real_evals_per_iter = []  # 每代真实评估次数

    def _evaluate_one(self, x, it, rng):
        """统一评估入口：按开关/概率选择代理或真实"""
        use_sur = (USE_SURROGATE and (rng.random() < SURROGATE_PROB))
        if use_sur and (self.surrogate is not None):
            v, eta, eta_total = self.surrogate.predict([x])[0]
            P1, T1, A1, V1, L = x
            V_final = V1 + A * L
            M_final = P1*V1/(R*T1) - 0.62*A1*P1*np.sqrt(gamma/(R*T1))*(2/(gamma+1))**((gamma+1)/(2*(gamma-1))) * 0.05
            T_final = T1 * (V1 / max(V_final, 1e-9))**(gamma-1)
            P_final = M_final * R * T_final / max(V_final, 1e-9)
            pressure_ok = (P_final >= Patm)

            pen = 0.0
            if v < self.v_ref:
                pen += dynamic_penalty(it, self.T) * (self.v_ref - v)
            if not pressure_ok:
                pen += dynamic_penalty(it, self.T) * (Patm - P_final) * 100.0

            J = self.alpha*eta + self.beta*eta_total + self.gamma*(v / self.v_ref)
            fit = -(J) + pen
            real_eval = 0  # 代理评估不计入真实评估数
            return fit, real_eval
        else:
            fit, ok, _ = objective_true(x, it, self.T, self.alpha, self.beta, self.gamma, self.v_ref)
            return fit, 1  # 真实评估 +1

    def _genetic_perturb(self, rng):
        """
        遗传扰动
        """
        n_cur = len(self.pop)
        f = np.array([p.fitness for p in self.pop], dtype=float)
        finite_mask = np.isfinite(f)
        if not finite_mask.any():
            probs = np.ones(n_cur, dtype=float) / n_cur
        else:
            worst = np.nanmax(f[finite_mask])
            f = np.where(np.isfinite(f), f, worst + 1.0)
            z = f - f.min()
            probs = np.exp(-z)
            s = probs.sum()
            if not np.isfinite(s) or s <= 0:
                probs = np.ones(n_cur, dtype=float) / n_cur
            else:
                probs /= s
        idx = rng.choice(n_cur, size=n_cur, p=probs)

        new_pop = []
        for i in range(0, n_cur, 2):
            a = self.pop[idx[i]]
            b = self.pop[idx[(i + 1) % n_cur]]
            mid = 0.5 * (a.position + b.position)
            sigma = 0.1 * (self.bounds[:, 1] - self.bounds[:, 0])
            pos1 = mid + rng.normal(0.0, sigma)
            pos2 = mid + rng.normal(0.0, sigma)
            vel1 = rng.normal(0.0, 0.5, size=self.dim)
            vel2 = rng.normal(0.0, 0.5, size=self.dim)

            c1 = Particle(pos1, vel1, self.bounds)
            c2 = Particle(pos2, vel2, self.bounds)
            c1.apply_bounds(); c2.apply_bounds()

            for c in (c1, c2):
                for k in range(self.dim):
                    if rng.random() < 0.15:
                        c.position[k] += rng.normal(0.0, 0.2 * (self.bounds[k, 1] - self.bounds[k, 0]))
                c.apply_bounds()

            new_pop.extend([c1, c2])


        while len(new_pop) < self.n:
            k = rng.integers(0, len(new_pop))
            dup_pos = new_pop[k].position + rng.normal(0.0, 0.05 * (self.bounds[:, 1] - self.bounds[:, 0]))
            dup_vel = new_pop[k].velocity + rng.normal(0.0, 0.5, size=self.dim)
            d = Particle(dup_pos, dup_vel, self.bounds)
            d.apply_bounds()
            new_pop.append(d)

        if len(new_pop) > self.n:
            new_pop = new_pop[:self.n]

        self.pop = new_pop

    def optimize(self, rng):
        # 初始化群体
        if not self.pop:
            init_pos = rng.uniform(self.bounds[:,0], self.bounds[:,1], size=(self.n, self.dim))
            init_vel = rng.uniform(-1, 1, size=(self.n, self.dim))
            self.pop = [Particle(init_pos[i], init_vel[i], self.bounds) for i in range(self.n)]

        real_eval_counter = 0
        for p in self.pop:
            fit, cnt = self._evaluate_one(p.position, it=0, rng=rng)
            real_eval_counter += cnt
            p.fitness = fit
            p.pbest = p.position.copy()
            p.pbest_fit = fit
        best_idx = int(np.argmin([pp.fitness for pp in self.pop]))
        self.gbest_pos = self.pop[best_idx].position.copy()
        self.gbest_fit = self.pop[best_idx].fitness

        # 记录
        self.history = [self.gbest_fit]
        self.real_evals_per_iter = [real_eval_counter]

        # 迭代
        for it in range(1, self.T):
            real_eval_counter = 0

            # 评估 & 更新 pbest/gbest
            for p in self.pop:
                fit, cnt = self._evaluate_one(p.position, it=it, rng=rng)
                real_eval_counter += cnt
                p.fitness = fit
                if fit < p.pbest_fit:
                    p.pbest_fit = fit
                    p.pbest = p.position.copy()
                if fit < self.gbest_fit:
                    self.gbest_fit = fit
                    self.gbest_pos = p.position.copy()

            # 速度/位置更新
            w = 0.9 - 0.5*(it/self.T)  # 惯性权重线性递减
            for p in self.pop:
                p.update_velocity(self.gbest_pos, w)
                p.position += p.velocity
                p.apply_bounds()

            # 遗传扰动
            if USE_GP and (it > 0) and (it % GP_PERIOD == 0):
                self._genetic_perturb(rng)

            # 记录
            self.history.append(self.gbest_fit)
            self.real_evals_per_iter.append(real_eval_counter)

        # 结束时计算 FR/MCV：对最终种群做一次真实评估
        feas_flags = []
        viol_vals  = []
        for p in self.pop:
            fit, ok, viol = objective_true(p.position, self.T-1, self.T,
                                           self.alpha, self.beta, self.gamma, self.v_ref)
            feas_flags.append(1 if ok and fit < 1e9 else 0)
            viol_vals.append(viol)
        FR  = float(np.mean(feas_flags))
        MCV = float(np.mean(viol_vals))
        return self.gbest_pos, self.gbest_fit, self.history, self.real_evals_per_iter, FR, MCV

# ==============================
# 数据生成（训练代理）
# ==============================
def generate_training_data(consts, num_samples=5000, seed=2024):
    rng = np.random.default_rng(seed)
    bounds = np.array([[5e5, 10e6], [300, 800], [1e-5, 5e-5], [0.001, 0.005], [0.6, 1.5]])
    sampler = qmc.LatinHypercube(d=bounds.shape[0], seed=seed)
    U = sampler.random(n=num_samples)
    X_raw = qmc.scale(U, bounds[:,0], bounds[:,1])

    X, Y = [], []
    for p in X_raw:
        P1, T1, A1, V1, L = p
        eta, eta_total, v_end, x_end, ok, P_end = calculate_efficiency(P1, T1, A1, V1, L, Patm)
        # 过滤明显异常/无效样本
        if np.isfinite(eta) and np.isfinite(eta_total) and np.isfinite(v_end):
            if eta > 0 and eta_total > 0:
                X.append(p)
                Y.append([v_end, eta, eta_total])
    return np.asarray(X, float), np.asarray(Y, float)

# ==============================
# 统计与可视化
# ==============================
def median_iqr(arr):
    """对 NaN/Inf 安全的中位数与 IQR 统计"""
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    med = np.nanmedian(arr)
    q75 = np.nanpercentile(arr, 75)
    q25 = np.nanpercentile(arr, 25)
    return med, (q75 - q25)

def plot_ablation_convergence(hist_dict, title, save_path):
    plt.figure(figsize=(8, 6))
    for label, runs in hist_dict.items():
        H = np.array(runs, dtype=float)
        mean = np.nanmean(H, axis=0)
        std  = np.nanstd(H, axis=0)
        x = np.arange(len(mean))
        plt.plot(x, mean, label=label, linewidth=2)
        plt.fill_between(x, mean-std, mean+std, alpha=0.15)
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness (Negative Objective, lower is better)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 20)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=1200)
    plt.close()

def evals_to_target(history, target, real_evals_per_iter):
    """累积真实评估数何时首次达到目标（负目标：越低越好）；未达则返回 inf"""
    cum = 0
    for t, val in enumerate(history):
        cum += real_evals_per_iter[t]
        if val <= target:
            return cum
    return np.inf

# ==============================
# 实验主流程
# ==============================
def run_suite_for_weight(weight, bounds, runs=20, seed_base=42, T=100, pop=50, v_ref=60.0,
                         surrogate_model=None, save_tag="W"):
    alpha, beta, gamma_w = weight
    settings = [
        ("Full",           True,  True,  True),
        ("Surrogate-off",  False, True,  True),
        ("GP-off",         True,  False, True),
        ("DP-off",         True,  True,  False),
    ]

    all_hist_full  = []
    all_real_full  = []
    all_final_full = []
    FRs_full = []; MCVs_full = []; Times_full = []

    for r in range(runs):
        rng = np.random.default_rng(seed_base + r)
        # 统一初始种群
        init_pos = rng.uniform(bounds[:,0], bounds[:,1], size=(pop, bounds.shape[0]))
        init_vel = rng.uniform(-1, 1, size=(pop, bounds.shape[0]))

        # 开关：Full
        global USE_SURROGATE, USE_GP, USE_DYNAMIC_PEN
        USE_SURROGATE, USE_GP, USE_DYNAMIC_PEN = True, True, True

        opt = HybridPSO(pop, bounds, T, alpha, beta, gamma_w, v_ref, surrogate=surrogate_model)
        opt.pop = [Particle(init_pos[i].copy(), init_vel[i].copy(), bounds) for i in range(pop)]
        t0 = time.time()
        _, best_fit, hist, real_per_iter, FR, MCV = opt.optimize(rng)
        dt = time.time() - t0

        all_hist_full.append(hist)
        all_real_full.append(real_per_iter)
        all_final_full.append(hist[-1])
        FRs_full.append(FR)
        MCVs_full.append(MCV)
        Times_full.append(dt)

    # 阈值
    target = np.percentile([x for x in all_final_full if np.isfinite(x)], 25)

    # Full 的 E2T
    e2t_full = [evals_to_target(all_hist_full[i], target, all_real_full[i]) for i in range(runs)]

    # ---------- 其它三种设置 ----------
    hist_by_setting = {"Full": all_hist_full}
    summary_rows = []

    mJ, iJ = median_iqr([h[-1] for h in all_hist_full])
    mFR, iFR = median_iqr(FRs_full)
    mMCV, iMCV = median_iqr(MCVs_full)
    mT, iT = median_iqr(Times_full)
    mE, iE = median_iqr(e2t_full)

    summary_rows.append(("Full", mJ, iJ, mFR, mMCV, mT, iT, mE, iE))

    for name, US, UG, UDP in settings:
        if name == "Full":
            continue
        USE_SURROGATE, USE_GP, USE_DYNAMIC_PEN = US, UG, UDP

        histories = []
        E2Ts = []
        FRs  = []
        MCVs = []
        Times = []

        for r in range(runs):
            rng = np.random.default_rng(seed_base + r)
            # 与 Full 相同初始种群
            rng_replay = np.random.default_rng(seed_base + r)
            init_pos = rng_replay.uniform(bounds[:,0], bounds[:,1], size=(pop, bounds.shape[0]))
            init_vel = rng_replay.uniform(-1, 1, size=(pop, bounds.shape[0]))

            opt = HybridPSO(pop, bounds, T, alpha, beta, gamma_w, v_ref, surrogate=surrogate_model)
            opt.pop = [Particle(init_pos[i].copy(), init_vel[i].copy(), bounds) for i in range(pop)]
            t0 = time.time()
            _, best_fit, hist, real_per_iter, FR, MCV = opt.optimize(rng)
            dt = time.time() - t0

            histories.append(hist)
            E2Ts.append(evals_to_target(hist, target, real_per_iter))
            FRs.append(FR)
            MCVs.append(MCV)
            Times.append(dt)

        hist_by_setting[name] = histories

        mJ, iJ = median_iqr([h[-1] for h in histories])
        mFR, iFR = median_iqr(FRs)
        mMCV, iMCV = median_iqr(MCVs)
        mT, iT = median_iqr(Times)
        mE, iE = median_iqr(E2Ts)
        summary_rows.append((name, mJ, iJ, mFR, mMCV, mT, iT, mE, iE))

    # ---------- 绘图 ----------
    title = f"Ablation Convergence (α={alpha}, β={beta}, γ={gamma_w})"
    os.makedirs("ablation_plots", exist_ok=True)
    plot_ablation_convergence(hist_by_setting, title,
                              f"ablation_plots/ablation_convergence_{save_tag}.png")

    # ---------- 控制台摘要 ----------
    print(f"\n=== 消融摘要（{save_tag}）: α={alpha}, β={beta}, γ={gamma_w} ===")
    print("设定            J_end med[IQR]   FR med[IQR]   MCV med[IQR]     Time(s) med[IQR]    E2T med[IQR]")
    for row in summary_rows:
        name, mJ, iJ, mFR, mMCV, mT, iT, mE, iE = row
        print(f"{name:14s}  {mJ:.6f}[{iJ:.6f}]   "
              f"{(mFR if np.isfinite(mFR) else float('nan')):.2f}[{(iFR if np.isfinite(iFR) else float('nan')):.2f}]   "
              f"{(mMCV if np.isfinite(mMCV) else float('nan')):.4f}[{(iMCV if np.isfinite(iMCV) else float('nan')):.4f}]   "
              f"{(mT if np.isfinite(mT) else float('nan')):.2f}[{(iT if np.isfinite(iT) else float('nan')):.2f}]      "
              f"{(mE if np.isfinite(mE) else float('inf')):.0f}[{(iE if np.isfinite(iE) else float('nan')):.0f}]")

# ==============================
# 主程序
# ==============================
if __name__ == "__main__":
    # 权重组合
    weight_combinations = [
        (1.0, 1.0, 0.0),  # W1 均衡
        (0.8, 0.8, 0.4),  # W2 略偏速度
        (0.5, 0.5, 1.0),  # W3 强偏速度
        (1.2, 0.8, 0.0),  # W4 偏效率
        (0.8, 1.2, 0.0),  # W5 偏总效率
    ]

    # 参数空间边界
    bounds = np.array([
        [5e5, 10e6],   # P1 (Pa)
        [300, 800],    # T1 (K)
        [1e-5, 5e-5],  # A1 (m^2)
        [0.001, 0.005],# V1 (m^3)
        [0.6, 1.5],    # L (m)
    ])

    POP  = 50
    T    = 100
    RUNS = 20
    VREF = 60.0
    SEED = 42

    # 1) 训练代理
    print(">> 生成代理训练数据...")
    X, Y = generate_training_data(CONSTS, num_samples=5000, seed=SEED)
    print(f"   有效样本: {len(X)}")
    print(">> 训练 MLP 代理...")
    surrogate = SurrogateModel()
    surrogate.train(X, Y)

    # 2) 逐权重做消融
    for wi, w in enumerate(weight_combinations, 1):
        run_suite_for_weight(w, bounds, runs=RUNS, seed_base=SEED,
                             T=T, pop=POP, v_ref=VREF,
                             surrogate_model=surrogate, save_tag=f"W{wi}")


    
    