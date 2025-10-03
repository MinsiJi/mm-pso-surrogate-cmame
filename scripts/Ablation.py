

import os
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import qmc
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ---- 画图中文/数学设置 ----
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# ==============================
# 物理常数 & 基本模型
# ==============================
m = 1.0
Rd = 0.02
A  = np.pi * Rd**2
gamma = 1.4
R  = 287.0
Cv = R / (gamma - 1)
Cp = gamma * Cv
T_env = 288
Patm = 101325
CONSTS = (m, gamma, R, Cv, Cp, T_env, A, Patm)

def cd_model(P1, T1, A1, V1):
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
    v, M, T, x = state
    if T <= 0: T = 1e-10
    if M <= 0: M = 1e-10
    V = V1 + A * x
    P = M * R * T / V
    Cd_n = cd_model(P, T, A1, V)
    leak = Cd_n * A1 * P * np.sqrt(gamma/(R*T)) * (2/(gamma+1))**((gamma+1)/(2*(gamma-1)))
    dMdt = -leak
    dvdt = P*A/m if P > Patm else 0.0
    dTdt = (-R*T*leak - P*A*v)/(Cv*M) if M > 1e-6 else 0.0
    dxdt = v
    return [dvdt, dMdt, dTdt, dxdt]

def calculate_efficiency(P1, T1, A1, V1, L, Patm=Patm, method='Radau'):
    # 初值
    M0 = P1*V1/(R*T1)
    state0 = [0.0, M0, T1, 0.0]

    def event(t, state, *args):  # 终止：x=L
        return state[3] - L
    event.terminal = True

    sol = solve_ivp(unified_equations, (0, 0.1), state0,
                    args=(P1, T1, A1, V1, L, Patm),
                    method=method, events=event, dense_output=True)

    if sol.t_events[0].size > 0:
        t_end = sol.t_events[0][0]
        v_end, M_end, T_end, x_end = sol.sol(t_end)
    else:
        v_end, M_end, T_end, x_end = sol.y[:, -1]

    if T_end <= 0: T_end = 1e-10
    V_end = V1 + A * x_end
    P_end = M_end * R * T_end / V_end
    pressure_ok = (P_end >= Patm)

    # 若满足基本可行性，积分泄漏焓
    if pressure_ok and x_end >= L - 1e-3:
        leak_energy = 0.0
        for i in range(1, len(sol.t)):
            dt = sol.t[i] - sol.t[i-1]
            v_i  = sol.y[0][i]
            M_i  = sol.y[1][i]
            T_i  = sol.y[2][i]
            x_i  = sol.y[3][i]
            V_i  = V1 + A * max(x_i, 0.0)
            P_i  = M_i * R * T_i / V_i if V_i > 1e-12 else 1.0
            Cd_i = cd_model(P_i, T_i, A1, V_i)
            leak = Cd_i * A1 * P_i * np.sqrt(gamma/(R*T_i)) * (2/(gamma+1))**((gamma+1)/(2*(gamma-1)))
            h_i  = Cp * T_i
            leak_energy += leak * h_i * dt

        U0 = Cp * M0 * T1
        Uf = Cp * M_end * T_end
        W  = U0 - Uf - leak_energy
        eta = 0.5*m*v_end**2 / W if W > 0 else 1e-6
        eta_total = 0.5*m*v_end**2 / U0
    else:
        eta = 1e-6
        eta_total = 1e-6
        pressure_ok = False

    return eta, eta_total, v_end, x_end, pressure_ok, P_end

# ==============================
# 消融开关
# ==============================
USE_SURROGATE   = True   # Surrogate-off -> False
USE_GP          = True   # GP-off -> False
USE_DYNAMIC_PEN = True   # DP-off -> False
SURROGATE_PROB  = 0.70   # 仅在 USE_SURROGATE=True 时生效
GP_PERIOD       = 10

def dynamic_penalty(iteration, max_iterations):
    """动态惩罚系数；DP-off 时恒为 1.0"""
    if not USE_DYNAMIC_PEN:
        return 1.0
    return 1.0 + 10.0 * (iteration / max_iterations) ** 2

def objective_true(params, iteration, max_iterations,
                   alpha=1.0, beta=1.0, gamma_weight=0.0, v_ref=60.0):
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
    return -(J) + pen, ok, (v_ref - v_end if v_end < v_ref else 0.0) + \
           (Patm - P_end if not ok else 0.0) + \
           (L - x_end if x_end < L - 1e-3 else 0.0)

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
        yp_sc = self.model.predict(X)
        return self.scaler_y.inverse_transform(yp_sc)  # [v, eta, eta_total]

def generate_training_data(constants, num_samples=5000):
    sampler = qmc.LatinHypercube(d=5)
    Xr = sampler.random(num_samples)
    low  = np.array([5e5, 300, 1e-5, 0.001, 0.6])
    high = np.array([10e6, 800, 5e-5, 0.005, 1.5])
    params = qmc.scale(Xr, low, high)
    X, Y = [], []
    for p in params:
        P1, T1, A1, V1, L = p
        eta, eta_total, v_end, x_end, ok, _ = calculate_efficiency(P1, T1, A1, V1, L, Patm)
        if ok and x_end >= L - 1e-3:
            X.append(p)
            Y.append([v_end, eta, eta_total])
    return np.asarray(X), np.asarray(Y)

# ==============================
# PSO + 遗传扰动
# ==============================
class Particle:
    def __init__(self, position, velocity, bounds):
        self.position = position
        self.velocity = velocity
        self.bounds   = bounds
        self.fitness  = np.inf

    def apply_bounds(self):
        self.position = np.clip(self.position, self.bounds[:,0], self.bounds[:,1])

    def update_velocity(self, gbest, w):
        # 简化 PSO
        c1 = 1.5; c2 = 1.5
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))
        cog = c1 * r1 * (self.position - gbest)
        soc = c2 * r2 * (self.position - gbest)
        self.velocity = w * self.velocity + cog + soc

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
        # Surrogate path?
        use_sur = (USE_SURROGATE and (rng.random() < SURROGATE_PROB))
        if use_sur and (self.surrogate is not None):
            v, eta, eta_total = self.surrogate.predict([x])[0]
            # 简化压力可行性判断
            P1, T1, A1, V1, L = x
            V_final = V1 + A * L
            M_final = P1*V1/(R*T1) - 0.62*A1*P1*np.sqrt(gamma/(R*T1)) * (2/(gamma+1))**((gamma+1)/(2*(gamma-1))) * 0.05
            T_final = T1 * (V1/V_final)**(gamma-1)
            P_final = M_final * R * T_final / V_final
            pressure_ok = (P_final >= Patm)

            pen = 0.0
            if v < self.v_ref:
                pen += dynamic_penalty(it, self.T) * (self.v_ref - v)
            if not pressure_ok:
                # 代理分支仅作轻量惩罚（不计入真实评估）
                pen += dynamic_penalty(it, self.T) * (Patm - P_final) * 100.0
            J = self.alpha*eta + self.beta*eta_total + self.gamma*(v / self.v_ref)
            fit = -(J) + pen
            real_eval = 0  # 代理不计入真实评估
            return fit, real_eval
        else:
            fit, ok, _ = objective_true(x, it, self.T, self.alpha, self.beta, self.gamma, self.v_ref)
            return fit, 1  # 真实评估+1

    def optimize(self, rng):
        for it in range(self.T):
            real_count_this_iter = 0
            # 评估
            for p in self.pop:
                fit, real_inc = self._evaluate_one(p.position, it, rng)
                p.fitness = fit
                real_count_this_iter += real_inc
                if fit < self.gbest_fit:
                    self.gbest_fit = fit
                    self.gbest_pos = p.position.copy()

            self.history.append(self.gbest_fit)
            self.real_evals_per_iter.append(real_count_this_iter)

            # 速度/位置更新
            w = 0.9 - 0.5*(it/self.T)
            for p in self.pop:
                p.update_velocity(self.gbest_pos, w)
                p.position += p.velocity
                p.apply_bounds()

            # 遗传扰动
            if USE_GP and (it>0) and (it % GP_PERIOD == 0):
                self._genetic_perturb(rng)

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

    def _genetic_perturb(self, rng):
        # 轮盘赌
        f = np.array([p.fitness for p in self.pop])
        probs = np.exp(-f - f.min())  # 稳定化
        probs /= probs.sum()
        idx = rng.choice(self.n, size=self.n, p=probs)
        sel = [self.pop[i] for i in idx]

        # 单点交叉
        childs = []
        for j in range(0, self.n, 2):
            a, b = sel[j], sel[(j+1) % self.n]
            cp = rng.integers(1, self.dim)
            c1 = np.concatenate([a.position[:cp], b.position[cp:]])
            c2 = np.concatenate([b.position[:cp], a.position[cp:]])
            childs += [Particle(c1, np.zeros(self.dim), self.bounds),
                       Particle(c2, np.zeros(self.dim), self.bounds)]
        # 高斯变异
        mut_rate = 0.10
        for c in childs:
            for k in range(self.dim):
                if rng.random() < mut_rate:
                    lo, hi = self.bounds[k]
                    sigma = 0.1 * (hi - lo)
                    c.position[k] += rng.normal(0.0, sigma)
            c.apply_bounds()
        self.pop = childs[:self.n]

# ==============================
# 运行与统计
# ==============================
def median_iqr(arr):
    import numpy as np
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    med = np.median(arr)
    q75 = np.percentile(arr, 75)
    q25 = np.percentile(arr, 25)
    return med, (q75 - q25)

def fmt_num(x, nd=2):
    import numpy as np
    return "–" if not np.isfinite(x) else f"{x:.{nd}f}"

def plot_ablation_convergence(hist_dict, title, save_path):
    """
    hist_dict: {label: [runs of history]}
    """
    plt.figure(figsize=(9,5.2))
    for label, runs in hist_dict.items():
        H = np.array(runs)
        mean = H.mean(axis=0)
        std  = H.std(axis=0)
        x = np.arange(len(mean))
        plt.plot(x, mean, label=label, linewidth=2)
        plt.fill_between(x, mean-std, mean+std, alpha=0.15)
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness (Negative Objective, lower is better)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=1200)
    plt.close()

def evals_to_target(history, target, real_evals_per_iter):
    cum = 0
    for t, val in enumerate(history):
        cum += real_evals_per_iter[t]
        if val <= target: 
            return cum
    return np.inf

def run_suite_for_weight(weight, bounds, runs=20, seed_base=42, T=100, pop=50, v_ref=60.0,
                         surrogate_model=None, save_tag="W"):
    alpha, beta, gamma_w = weight
    settings = [
        ("Full",           True,  True,  True),
        ("Surrogate-off",  False, True,  True),
        ("GP-off",         True,  False, True),
        ("DP-off",         True,  True,  False),
    ]

    # —— 确定 target（5%分位）并记录统计 —— #
    all_final_full = []
    all_hist_full  = []
    all_real_full  = []
    FRs_full = []
    MCVs_full = []
    Times_full = []

    for r in range(runs):
        rng = np.random.default_rng(seed_base + r)
        init_pos = rng.uniform(bounds[:,0], bounds[:,1], size=(pop, bounds.shape[0]))
        init_vel = rng.uniform(-1, 1, size=(pop, bounds.shape[0]))

        global USE_SURROGATE, USE_GP, USE_DYNAMIC_PEN
        USE_SURROGATE, USE_GP, USE_DYNAMIC_PEN = True, True, True

        opt = HybridPSO(pop, bounds, T, alpha, beta, gamma_w, v_ref, surrogate=surrogate_model)
        opt.pop = [Particle(init_pos[i].copy(), init_vel[i].copy(), bounds) for i in range(pop)]

        t0 = time.time()
        _, best_fit, hist, real_per_iter, FR, MCV = opt.optimize(rng)
        dt = time.time() - t0

        all_final_full.append(hist[-1])
        all_hist_full.append(hist)
        all_real_full.append(real_per_iter)
        FRs_full.append(FR)
        MCVs_full.append(MCV)
        Times_full.append(dt)

    # target（5%分位）
    vals_full = np.asarray(all_final_full, dtype=float)
    vals_full = vals_full[np.isfinite(vals_full)]
    target = np.percentile(vals_full, 5) if vals_full.size > 0 else np.inf

    # Full 的 E2T 与统计
    e2t_full = [evals_to_target(all_hist_full[i], target, all_real_full[i]) for i in range(runs)]
    mJ,  iJ  = median_iqr([h[-1] for h in all_hist_full])
    mFR, iFR = median_iqr(FRs_full)
    mMCV,iMCV= median_iqr(MCVs_full)
    mT,  iT  = median_iqr(Times_full)
    mE,  iE  = median_iqr(e2t_full)

    # —— 汇总容器 —— #
    hist_by_setting = {"Full": all_hist_full}
    summary_rows = []
    summary_rows.append(("Full", mJ, iJ, mFR, mMCV, mT, iT, mE, iE))

    # —— 其它三种设置—— #
    for name, us, ug, up in settings:
        if name == "Full":
            continue
        USE_SURROGATE, USE_GP, USE_DYNAMIC_PEN = us, ug, up

        histories = []
        E2Ts = []
        FRs  = []
        MCVs = []
        Times = []

        for r in range(runs):
            rng = np.random.default_rng(seed_base + r)
            # 与 Full 使用相同的初始种群（保证公平）
            rng_replay = np.random.default_rng(seed_base + r)
            init_pos = rng_replay.uniform(bounds[:,0], bounds[:,1], size=(pop, bounds.shape[0]))
            init_vel = rng_replay.uniform(-1, 1, size=(pop, bounds.shape[0]))

            opt = HybridPSO(pop, bounds, T, alpha, beta, gamma_w, v_ref,
                            surrogate=surrogate_model if us else None)
            opt.pop = [Particle(init_pos[i].copy(), init_vel[i].copy(), bounds) for i in range(pop)]

            t0 = time.time()
            _, best_fit, hist, real_per_iter, FR, MCV = opt.optimize(rng)
            Times.append(time.time() - t0)

            histories.append(hist)
            FRs.append(FR); MCVs.append(MCV)
            E2Ts.append(evals_to_target(hist, target, real_per_iter))

        hist_by_setting[name] = histories
        mJ,  iJ  = median_iqr([h[-1] for h in histories])
        mFR, iFR = median_iqr(FRs)
        mMCV,iMCV= median_iqr(MCVs)
        mT,  iT  = median_iqr(Times)
        mE,  iE  = median_iqr(E2Ts)
        summary_rows.append((name, mJ, iJ, mFR, mMCV, mT, iT, mE, iE))

    # —— 汇总图 —— #
    title = f"Ablation Convergence (α={alpha}, β={beta}, γ={gamma_w})"
    os.makedirs("ablation_plots", exist_ok=True)
    plot_ablation_convergence(hist_by_setting, title,
                              f"ablation_plots/ablation_convergence_{save_tag}.png")

    # —— 控制台摘要（安全格式化）—— #
    print(f"\n=== 消融摘要（{save_tag}）: α={alpha}, β={beta}, γ={gamma_w} ===")
    print("设定\t\tJ_end med[IQR]\tFR med\tMCV med\tTime(s) med[IQR]\tE2T med[IQR]")
    for row in summary_rows:
        name, mJ, iJ, mFR, mMCV, mT, iT, mE, iE = row
        print(f"{name:14s}\t{fmt_num(mJ,4)}[{fmt_num(iJ,4)}]\t"
              f"{fmt_num(mFR,2)}\t{fmt_num(mMCV,4)}\t"
              f"{fmt_num(mT,2)}[{fmt_num(iT,2)}]\t\t"
              f"{fmt_num(mE,0)}[{fmt_num(iE,0)}]")


# ==============================
# 主程序
# ==============================
if __name__ == "__main__":
    # 权重组合（与你原文一致，可按需调整）
    weight_combinations = [
        (0.5, 0.5, 0.0),  # 侧重效率
        (0.33, 0.33, 0.33),  # 平衡
        (0.25, 0.25, 0.5),  # 侧重速度
        (0.2, 0.8, 0.0),  # 侧重效率，总效率次之
        (0.8, 0.2, 0.0)   # 侧重总效率，效率次之
    ]
    bounds = np.array([[5e5, 10e6], [300, 800], [1e-5, 5e-5], [0.001, 0.005], [0.6, 1.5]])
    POP  = 50
    T    = 100
    RUNS = 20
    VREF = 60.0
    SEED = 42

    # 1) 训练代理（一次即可）
    print(">> 生成代理训练数据(可能需数分钟，可按机器性能在函数内调小样本量)...")
    X, Y = generate_training_data(CONSTS, num_samples=5000)
    print(f"   有效样本: {len(X)}")
    print(">> 训练 MLP 代理...")
    surrogate = SurrogateModel()
    surrogate.train(X, Y)

    # 2) 逐权重做消融
    for wi, w in enumerate(weight_combinations, 1):
        run_suite_for_weight(w, bounds, runs=RUNS, seed_base=SEED,
                             T=T, pop=POP, v_ref=VREF,
                             surrogate_model=surrogate, save_tag=f"W{wi}")


    
    