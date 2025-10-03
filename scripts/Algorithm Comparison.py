import numpy as np
from scipy.integrate import solve_ivp
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from scipy.stats import qmc
import pandas as pd
from pandas.plotting import parallel_coordinates
import os
import time

# === 基本设置 ===
os.chdir(os.path.dirname(__file__))
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# === 物理常量 ===
m = 1.0                    # 质量
A = np.pi * 0.02**2        # 截面积
gamma = 1.4
R = 287.0
Cv = R / (gamma - 1)
Cp = gamma * Cv
T0 = 288
Patm = 101325
constants = (m, gamma, R, Cv, Cp, T0, A, Patm)

# === 可变 Cd 模型 ===
def cd_model(P1, T1, A1, V1):
    if T1 <= 0: T1 = 1e-8
    p0_MPa = P1 / 1e6
    T0_K   = T1
    A1_m2  = A1
    V1_m3  = V1
    Cd = (1.90 + 0.0012 * V1_m3**2 - 0.017 * p0_MPa**2 - 3.02 * V1_m3 * p0_MPa
          + 0.27 * V1_m3 * T0_K + 0.099 * A1_m2 * p0_MPa + 10.28 * A1_m2 * T0_K
          + 0.0098 * V1_m3 + 0.015 * A1_m2 - 0.0086 * p0_MPa - 0.0054 * T0_K)
    return float(np.clip(Cd, 0.01, 1.0))

# === ODE ===
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

# === 仿真 + 效率 ===
def calculate_efficiency(P1, T1, A1, V1, L, Patm=101325, method='Radau'):
    M0 = P1 * V1 / (R * T1)
    v0, Tinit, x0 = 0.0, T1, 0.0
    initial_state = [v0, M0, Tinit, x0]

    def event(t, state, *args): return state[3] - L
    event.terminal = True

    sol = solve_ivp(
        unified_equations, (0, 0.1), initial_state,
        args=(P1, T1, A1, V1, L, Patm),
        method=method, events=event, dense_output=True
    )

    if sol.t_events[0].size > 0:
        t_end = sol.t_events[0][0]
        v_final, M_final, T_final, x_final = sol.sol(t_end)
    else:
        v_final, M_final, T_final, x_final = sol.y[:, -1]

    if T_final <= 0: T_final = 1e-10
    V_final = V1 + A * x_final
    P_final = M_final * R * T_final / V_final
    pressure_ok = (P_final >= Patm)

    if pressure_ok and x_final >= L - 1e-3:
        leak_energy = 0.0
        for i in range(1, len(sol.t)):
            dt = sol.t[i] - sol.t[i-1]
            v_i = sol.y[0][i]
            M_i = sol.y[1][i]
            T_i = sol.y[2][i]
            x_i = sol.y[3][i]
            V_i = V1 + A * max(x_i, 0.0)
            P_i = M_i * R * T_i / V_i if V_i > 1e-12 else 1.0
            Cd_i = cd_model(P_i, T_i, A1, V_i)
            leak = Cd_i * A1 * P_i * np.sqrt(gamma/(R*T_i)) * (2/(gamma+1))**((gamma+1)/(2*(gamma-1)))
            h = Cp * T_i
            leak_energy += leak * h * dt

        U0 = Cp * M0 * T1
        Uf = Cp * M_final * T_final
        W_total = U0 - Uf - leak_energy
        yita = 0.5 * m * v_final**2 / W_total if W_total > 0 else 0.0
        yita_zong = 0.5 * m * v_final**2 / U0 if U0 > 0 else 0.0
    else:
        yita = 1e-6
        yita_zong = 1e-6
        pressure_ok = False

    return yita, yita_zong, v_final, x_final, pressure_ok, P_final

# === 目标与代理 ===
def dynamic_penalty(iteration, max_iterations):
    return 1 + 10 * (iteration / max_iterations) ** 2

def objective_function(params, constants, iteration, max_iterations, alpha=0.5, beta=0.5, gamma_weight=0.0, v_ref=60.0):
    P1, T1, A1, V1, L = params
    m_, gamma_, R_, Cv_, Cp_, T0_, A_, Patm_ = constants
    yita, yita_zong, v_final, x_final, pressure_ok, P_final = calculate_efficiency(P1, T1, A1, V1, L, Patm_)
    obj_value = alpha*yita + beta*yita_zong + gamma_weight*(v_final / v_ref)
    penalty = 0.0
    if v_final < v_ref:        penalty += dynamic_penalty(iteration, max_iterations) * (v_ref - v_final)
    if not pressure_ok:        penalty += dynamic_penalty(iteration, max_iterations) * (Patm_ - P_final) * 100
    if x_final < L - 1e-3:     penalty += dynamic_penalty(iteration, max_iterations) * (L - x_final) * 100
    return -obj_value + penalty           # 最小化

class SurrogateModel:
    def __init__(self):
        self.model = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(256,128,64), activation='relu',
                         solver='adam', max_iter=5000, alpha=0.0005,
                         learning_rate='adaptive', random_state=42)
        )
        self.scaler_y = StandardScaler()
    def train(self, X, y):
        y_scaled = self.scaler_y.fit_transform(y)
        self.model.fit(X, y_scaled)
    def predict(self, X):
        y_pred_scaled = self.model.predict(X)
        return self.scaler_y.inverse_transform(y_pred_scaled)  # [v, yita, yita_zong]

def generate_training_data(constants, num_samples=10000):
    sampler = qmc.LatinHypercube(d=5)
    params = qmc.scale(sampler.random(num_samples),
                       np.array([5e5, 300, 1e-5, 0.001, 0.6]),
                       np.array([1e7, 800, 5e-5, 0.005, 1.5]))
    X, y = [], []
    for p in params:
        P1, T1, A1, V1, L = p
        yita, yita_zong, v_end, x_end, ok, _ = calculate_efficiency(P1, T1, A1, V1, L, Patm)
        if ok and x_end >= L - 1e-3:
            X.append(p)
            y.append([v_end, yita, yita_zong])
    return np.array(X), np.array(y)

# === 多样性工具 ===
def _normalize_params(params, bounds):
    lb, ub = bounds[:,0], bounds[:,1]
    return (np.array(params) - lb) / (ub - lb + 1e-12)

def _pairwise_min_distance(points):
    if len(points) <= 1: return np.array([0.0]*len(points))
    P = np.array(points, float)
    D = np.sqrt(((P[:,None,:]-P[None,:,:])**2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    return D.min(axis=1)

def _dedup_by_threshold(cand, d0):
    # cand: [(score, x_norm, x_raw), ...]
    if not cand: return []
    keep, used = [], np.zeros(len(cand), bool)
    order = np.argsort([-c[0] for c in cand])
    for i in order:
        if used[i]: continue
        keep.append(i); used[i] = True
        for j in range(len(cand)):
            if used[j] or j==i: continue
            if np.linalg.norm(cand[i][1] - cand[j][1]) <= d0:
                used[j] = True
    return [cand[i] for i in keep]

def _plot_topdelta_bandwidth(reps, title, save_path):
    if not reps: return
    reps_sorted = sorted(reps, key=lambda z: z[0], reverse=True)
    scores = np.array([z[0] for z in reps_sorted], float)
    Xn = np.array([z[1] for z in reps_sorted], float)
    dmins = _pairwise_min_distance(Xn)
    idx = np.arange(1, len(reps_sorted)+1)
    fig, ax1 = plt.subplots(figsize=(8,4.8))
    ax1.plot(idx, scores, marker='o', linewidth=2)
    ax1.set_xlabel('Representative idx (Top-δ)')
    ax1.set_ylabel('Score (higher is better)')
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.bar(idx, dmins, alpha=0.35)
    ax2.set_ylabel('Nearest-neighbor dist (normed)')
    plt.title(title); plt.tight_layout()
    plt.savefig(save_path, dpi=1200); plt.close()

def calc_score_from_true(x, constants, alpha, beta, gamma_weight, v_ref):
    yita, yita_zong, v, x_end, ok, _ = calculate_efficiency(*x, Patm=constants[-1])
    if (not ok) or (x_end < x[-1]-1e-3):
        return None, (yita, yita_zong, v, ok)
    score = alpha*yita + beta*yita_zong + gamma_weight*(v/v_ref)
    return float(score), (yita, yita_zong, v, ok)

def build_elite_archive_from_positions(positions, bounds, constants,
                                       alpha, beta, gamma_weight, v_ref,
                                       delta=0.05, d0=0.15, require_feasible=True):
    scored = []
    for x in positions:
        s, _ = calc_score_from_true(x, constants, alpha, beta, gamma_weight, v_ref)
        if s is None:
            if require_feasible: 
                continue
            else:
                continue
        scored.append((s, x.copy()))
    if not scored: return []
    scores = np.array([s for s,_ in scored], float)
    thr = (1.0 - delta) * float(np.max(scores))
    cand = []
    for s, x in scored:
        if s >= thr:
            xn = _normalize_params(x, bounds)
            cand.append((s, xn, x))
    reps = _dedup_by_threshold(cand, d0)
    reps.sort(key=lambda z: z[0], reverse=True)
    return reps

# === 粒子与三种算法 ===
class Particle:
    def __init__(self, position, velocity, bounds):
        self.position = position
        self.velocity = velocity
        self.bounds = bounds
        self.fitness = float('inf')
        self.best_fitness = float('inf')
        self.best_position = position.copy()
    def update_velocity(self, global_best_position, w):
        c1 = 1.5; c2 = 1.5
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))
        cognitive = c1 * r1 * (self.best_position - self.position)
        social    = c2 * r2 * (global_best_position - self.position)
        self.velocity = w*self.velocity + cognitive + social
    def apply_bounds(self):
        for i in range(len(self.position)):
            self.position[i] = np.clip(self.position[i], self.bounds[i][0], self.bounds[i][1])
    def update_best(self):
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()

class MMPSO:
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
        self.time_elapsed = 0.0   # ← 新增
        
    def optimize(self, surrogate_model):
        start = time.time()  
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
        fin = {
            'efficiency': yita,
            'total_efficiency': yita_zong,
            'velocity': v_final,
            'pressure_ok': pressure_ok,
            'final_pressure': P_final,
            'particles': [p.position.copy() for p in self.particles]  # 末代粒子位置
        }

        # 计时
        self.time_elapsed = time.time() - start

        # 返回与其他算法统一的四元组
        return self.global_best_position, final_obj, self.convergence_history, fin

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

class StandardPSO:
    def __init__(self, n_particles, dimensions, bounds, constants, max_iter):
        self.n_particles = n_particles; self.dimensions = dimensions
        self.bounds = bounds; self.constants = constants
        self.max_iter = max_iter
        self.global_best_position = None; self.global_best_value = float('inf')
        self.particles = []; self.convergence_history = []; self.time_elapsed = 0.0
    def optimize(self):
        start = time.time()
        for i in range(self.max_iter):
            for p in self.particles:
                p.fitness = objective_function(p.position, self.constants, i, self.max_iter)
                if p.fitness < p.best_fitness:
                    p.best_fitness = p.fitness; p.best_position = p.position.copy()
                if p.fitness < self.global_best_value:
                    self.global_best_value = p.fitness; self.global_best_position = p.position.copy()
            self.convergence_history.append(self.global_best_value)
            w = 0.9 - 0.5 * (i / self.max_iter)
            for p in self.particles:
                c1, c2 = 1.5, 1.5
                r1, r2 = np.random.rand(self.dimensions), np.random.rand(self.dimensions)
                cognitive = c1*r1*(p.best_position - p.position)
                social    = c2*r2*(self.global_best_position - p.position)
                p.velocity = w*p.velocity + cognitive + social
                p.position += p.velocity
                p.apply_bounds()
        self.time_elapsed = time.time() - start
        final_positions = [p.position.copy() for p in self.particles]
        return self.global_best_position, -self.global_best_value, self.convergence_history, final_positions

class GeneticAlgorithm:
    def __init__(self, population_size, dimensions, bounds, constants, max_iter):
        self.pop_size = population_size; self.dim = dimensions
        self.bounds = bounds; self.constants = constants; self.max_iter = max_iter
        self.best_position = None; self.global_best_value = float('inf')
        self.convergence_history = []; self.time_elapsed = 0.0
        self.elitism_rate = 0.1
        self.min_bounds = bounds[:,0]; self.max_bounds = bounds[:,1]
        self.current_iter = 0
    def initialize_population(self):
        return np.array([np.random.uniform(self.min_bounds, self.max_bounds, self.dim)
                         for _ in range(self.pop_size)]).clip(self.min_bounds, self.max_bounds)
    def check_bounds(self, individual):
        return np.all(individual >= self.min_bounds) and np.all(individual <= self.max_bounds)
    def selection(self, population, fitness):
        pop_size = len(population)
        elite_size = min(int(pop_size*self.elitism_rate), pop_size-1)
        elite_idx = np.argsort(fitness)[:elite_size]
        elite = population[elite_idx]
        fitness = np.array(fitness, float)
        min_f, max_f = np.min(fitness), np.max(fitness)
        if max_f == min_f:
            probs = np.ones(pop_size)/pop_size
        else:
            norm = fitness - min_f
            probs = np.ones(pop_size)/pop_size if norm.sum()==0 else norm / norm.sum()
        sel_idx = list(elite_idx)
        while len(sel_idx) < pop_size:
            new_idx = np.random.choice(pop_size, size=pop_size-len(sel_idx), replace=True, p=probs)
            sel_idx.extend(new_idx)
        sel_idx = np.unique(sel_idx)[:pop_size]
        return population[sel_idx]
    def crossover(self, parents):
        pop_size = self.pop_size
        elite_size = min(int(pop_size*self.elitism_rate), pop_size-1)
        children = []
        if len(parents) >= elite_size:
            children.extend(parents[:elite_size])
        else:
            children.extend(parents)
        n_par = len(parents)
        for i in range(0, n_par, 2):
            if len(children) >= pop_size: break
            if i+1 < n_par:
                p1, p2 = parents[i], parents[i+1]
                alpha = np.random.rand()
                c1 = alpha*p1 + (1-alpha)*p2
                c2 = (1-alpha)*p1 + alpha*p2
                c1 = np.clip(c1, self.min_bounds, self.max_bounds)
                c2 = np.clip(c2, self.min_bounds, self.max_bounds)
                children.extend([c1, c2])
        while len(children) < pop_size:
            children.append(np.random.uniform(self.bounds[:,0], self.bounds[:,1], self.dim))
        return np.array(children[:pop_size])
    def mutation(self, population, rate=0.1):
        for i in range(len(population)):
            if np.random.rand() < rate:
                idx = np.random.randint(0, self.dim)
                step = 0.1 * (1 - self.current_iter/self.max_iter)
                population[i, idx] += np.random.normal(0, step)*(self.max_bounds[idx]-self.min_bounds[idx])
                population[i, idx] = np.clip(population[i, idx], self.min_bounds[idx], self.max_bounds[idx])
        return population
    def optimize(self):
        start = time.time()
        population = self.initialize_population()
        for i in range(self.max_iter):
            self.current_iter = i
            fitness = []
            for p in population:
                if not self.check_bounds(p):
                    fitness.append(1e10); continue
                fitness.append(objective_function(p, self.constants, i, self.max_iter))
            fitness = np.array(fitness)
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.global_best_value:
                self.global_best_value = fitness[best_idx]
                self.best_position = population[best_idx].copy()
            self.convergence_history.append(self.global_best_value)
            parents = self.selection(population, fitness)
            children = self.crossover(parents)
            population = self.mutation(children)
        self.time_elapsed = time.time() - start
        final_yita, final_yita_zong, final_v, _, ok, final_P = calculate_efficiency(*self.best_position, Patm)
        return self.best_position, -self.global_best_value, self.convergence_history, {
            'efficiency': final_yita, 'total_efficiency': final_yita_zong, 'velocity': final_v,
            'pressure_ok': ok, 'final_pressure': final_P,
            'population': population.copy()   # 末代种群
        }

# === 主程序：三算法质量+多样性对比 ===
if __name__ == "__main__":
    BASE_SEED = 42
    num_runs = 20
    algorithms = ['MMPSO','GeneticAlgorithm','StandardPSO']
    results = {alg: [] for alg in algorithms}
    run_times = {alg: [] for alg in algorithms}
    elite_by_alg = {alg: [] for alg in algorithms}

    # 统一权重（score口径）
    alpha, beta, gamma_weight = 0.5, 0.5, 0.0
    v_ref = 60.0

    for alg in algorithms:
        print(f"\n=================== 运行 {alg} ===================")
        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}")
            run_seed = BASE_SEED + run
            np.random.seed(run_seed)

            # 代理训练
            X, y = generate_training_data(constants, num_samples=5000)
            model = SurrogateModel(); model.train(X, y)

            # 统一搜索空间与迭代
            n_particles = 50
            dimensions  = 5
            bounds = np.array([[5e5, 1e7], [300, 800], [1e-5, 5e-5], [0.001, 0.005], [0.6, 1.5]])
            max_iter = 100

            if alg == 'StandardPSO':
                pso = StandardPSO(n_particles, dimensions, bounds, constants, max_iter)
                for i in range(n_particles):
                    np.random.seed(run_seed + i)
                    pos = np.random.uniform(bounds[:,0], bounds[:,1])
                    vel = np.random.uniform(-1, 1, dimensions)
                    pso.particles.append(Particle(pos, vel, bounds))
                best_pos, best_score, _, final_positions = pso.optimize()
                yita, yita_zong, v, _, ok, Pfin = calculate_efficiency(*best_pos, Patm)
                reps = build_elite_archive_from_positions(
                    positions=final_positions, bounds=bounds, constants=constants,
                    alpha=alpha, beta=beta, gamma_weight=gamma_weight, v_ref=v_ref,
                    delta=0.05, d0=0.15, require_feasible=True
                )
                elite_by_alg[alg].extend(reps)
                run_time = pso.time_elapsed

            elif alg == 'GeneticAlgorithm':
                ga = GeneticAlgorithm(n_particles, dimensions, bounds, constants, max_iter)
                best_pos, best_score, _, fin = ga.optimize()
                yita, yita_zong, v, ok, Pfin = fin['efficiency'], fin['total_efficiency'], fin['velocity'], fin['pressure_ok'], fin['final_pressure']
                reps = build_elite_archive_from_positions(
                    positions=fin['population'], bounds=bounds, constants=constants,
                    alpha=alpha, beta=beta, gamma_weight=gamma_weight, v_ref=v_ref,
                    delta=0.05, d0=0.15, require_feasible=True
                )
                elite_by_alg[alg].extend(reps)
                run_time = ga.time_elapsed

            elif alg == 'MMPSO':
                pso = MMPSO(n_particles, dimensions, bounds, constants, max_iter,
                            alpha=alpha, beta=beta, gamma_weight=gamma_weight, v_ref=v_ref)
                for i in range(n_particles):
                    np.random.seed(run_seed + i)
                    pos = np.random.uniform(bounds[:,0], bounds[:,1])
                    vel = np.random.uniform(-1, 1, dimensions)
                    pso.particles.append(Particle(pos, vel, bounds))
                best_pos, best_score, _, fin = pso.optimize(model)
                yita, yita_zong, v, ok, Pfin = fin['efficiency'], fin['total_efficiency'], fin['velocity'], fin['pressure_ok'], fin['final_pressure']
                reps = build_elite_archive_from_positions(
                    positions=fin.get('particles', []), bounds=bounds, constants=constants,
                    alpha=alpha, beta=beta, gamma_weight=gamma_weight, v_ref=v_ref,
                    delta=0.05, d0=0.15, require_feasible=True
                )
                elite_by_alg[alg].extend(reps)
                run_time = pso.time_elapsed

            P1, T1, A1, V1, L = best_pos
            print(f"    最优参数 | P1={P1/1e6:.2f} MPa, T1={T1:.0f} K, A1={A1*1e6:.2f} mm², V1={V1*1e3:.2f} mL, L={L:.2f} m")
            print(f"    指标     | η={yita*100:.2f}%, η_total={yita_zong*100:.2f}%, v={v:.2f} m/s, 约束={'满足' if ok else '不满足'}, 用时={run_time:.2f}s")

            results[alg].append({'efficiency': yita, 'total_efficiency': yita_zong,
                                 'velocity': v, 'pressure_ok': ok, 'final_pressure': Pfin})
            run_times[alg].append(run_time)

        # 小结
        eff  = [r['efficiency']*100 for r in results[alg]]
        teff = [r['total_efficiency']*100 for r in results[alg]]
        vel  = [r['velocity'] for r in results[alg]]
        okrt = np.mean([r['pressure_ok'] for r in results[alg]])*100
        print(f"\n{alg} 汇总：η={np.mean(eff):.2f}%±{np.std(eff):.2f}%，η_total={np.mean(teff):.2f}%±{np.std(teff):.2f}%")
        print(f"            v={np.mean(vel):.2f}±{np.std(vel):.2f} m/s，成功率={okrt:.2f}% ，平均用时={np.mean(run_times[alg]):.2f}s\n")

    # === 质量对比图 ===
    efficiencies      = {alg: [r['efficiency']*100 for r in results[alg]] for alg in algorithms}
    total_efficiencies= {alg: [r['total_efficiency']*100 for r in results[alg]] for alg in algorithms}
    velocities        = {alg: [r['velocity'] for r in results[alg]] for alg in algorithms}

    plt.figure(figsize=(8,6))
    plt.boxplot(efficiencies.values(), labels=algorithms)
    plt.xlabel('Algorithm'); plt.ylabel('Utilize Efficiency (%)')
    plt.title('Utilize efficiency distribution across algorithms')
    plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
    plt.savefig('utilize_efficiency_comparison.png', dpi=1200); plt.close()

    plt.figure(figsize=(8,6))
    plt.boxplot(total_efficiencies.values(), labels=algorithms)
    plt.xlabel('Algorithm'); plt.ylabel('Total Efficiency (%)')
    plt.title('Total efficiency distribution across algorithms')
    plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
    plt.savefig('total_efficiency_comparison.png', dpi=1200); plt.close()

    plt.figure(figsize=(8,6))
    plt.boxplot(velocities.values(), labels=algorithms)
    plt.xlabel('Algorithm'); plt.ylabel('Final Velocity (m/s)')
    plt.title('Final velocity distribution across algorithms')
    plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
    plt.savefig('velocity_comparison.png', dpi=1200); plt.close()

    # 可行率 + 平均时间
    x = np.arange(len(algorithms))
    success = [np.mean([r['pressure_ok'] for r in results[alg]])*100 for alg in algorithms]
    avg_time = [np.mean(run_times[alg]) for alg in algorithms]
    plt.figure(figsize=(8,5))
    plt.bar(x-0.2, success, width=0.4)
    plt.bar(x+0.2, avg_time, width=0.4)
    plt.xticks(x, algorithms); plt.ylabel('Value')
    plt.title('Success rate (%) and Avg runtime (s)')
    plt.legend(['Success rate (%)','Avg runtime (s)'])
    plt.grid(True, axis='y', alpha=0.3); plt.tight_layout()
    plt.savefig('success_time_comparison.png', dpi=1200); plt.close()

    # === 多样性（Top-δ）对比 ===
    C_delta, D_delta_med = {}, {}
    for alg in algorithms:
        reps = elite_by_alg[alg]
        if len(reps)==0:
            C_delta[alg] = 0; D_delta_med[alg] = 0.0
            print(f"[{alg}] 无 Top-δ 代表解（可能 δ 太小或可行率低）")
            continue
        Xn = np.array([r[1] for r in reps], float)
        dmins = _pairwise_min_distance(Xn) if len(Xn) else np.array([])
        C_delta[alg] = len(reps)
        D_delta_med[alg] = float(np.median(dmins)) if len(dmins) else 0.0
        # 带宽图
        _plot_topdelta_bandwidth(reps, f"{alg} Top-δ Portfolio", f"{alg.lower()}_topdelta_bandwidth.png")
        # 平行坐标图
        cols = ["P1","T1","A1","V1","L"]
        df = pd.DataFrame([dict(zip(cols, r[2])) for r in reps])
        df["score"] = [r[0] for r in reps]
        if len(df) >= 3:
            qh, qm = np.percentile(df["score"], [67, 33])
            df["tier"] = np.where(df["score"]>=qh, "high", np.where(df["score"]<=qm, "low", "mid"))
        else:
            df["tier"] = "all"
        plt.figure(figsize=(10,6))
        parallel_coordinates(df.assign(tier=df["tier"]), "tier", cols, linewidth=1.2, alpha=0.6)
        plt.title(f"{alg} Top-δ Representatives — Parallel Coordinates")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(f"{alg.lower()}_parallel_coords.png", dpi=1200); plt.close()

    # 多样性条形图
    plt.figure(figsize=(8,5))
    plt.bar(x-0.2, [C_delta.get(a,0) for a in algorithms], width=0.4)
    plt.bar(x+0.2, [D_delta_med.get(a,0.0) for a in algorithms], width=0.4)
    plt.xticks(x, algorithms); plt.ylabel('Value')
    plt.title('Top-δ diversity: C_delta (count) & D_delta_median')
    plt.legend(['C_delta','D_delta_median'])
    plt.grid(True, axis='y', alpha=0.3); plt.tight_layout()
    plt.savefig('diversity_bar.png', dpi=1200); plt.close()

    print("\n------------------- 多样性指标（Top-δ） -------------------")
    for alg in algorithms:
        print(f"{alg}: C_delta={C_delta.get(alg,0)}, D_delta_median={D_delta_med.get(alg,0.0):.4f}")








