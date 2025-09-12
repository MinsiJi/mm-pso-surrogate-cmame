import numpy as np
from scipy.integrate import solve_ivp
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from scipy.stats import qmc
import os
import time
os.chdir(os.path.dirname(__file__))

# 配置支持数学符号的字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# 已知参常数
m = 1.0  # 弹射物体质量
A = np.pi * 0.02** 2  # 面积
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
    if T <= 0:
        T = 1e-10
    if M <= 0:
        M = 1e-10
    V = V1 + A * x
    P = M * R * T / V
    Cd_n=cd_model(P, T, A1, V)
    leak = Cd_n * A1 * P * np.sqrt(gamma / (R * T)) * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
    dMdt = -leak
    dvdt = P * A / m if P > Patm else 0
    dTdt = (-R * T * leak - P * A * v) / (Cv * M) if M > 1e-6 else 0
    dxdt = v
    return [dvdt, dMdt, dTdt, dxdt]

def calculate_efficiency(P1, T1, A1, V1, L, Patm=101325, method='Radau'):
    M0 = P1 * V1 / (R * T1)
    v0 = 0
    T0 = T1
    x0 = 0
    initial_state = [v0, M0, T0, x0]
    
    def event(t, state, *args):
        return state[3] - L
    event.terminal = True
    
    solution = solve_ivp(
        unified_equations, 
        t_span=(0, 0.1),
        y0=initial_state, 
        args=(P1, T1, A1, V1, L, Patm), 
        method=method,
        events=event,
        dense_output=True
    )
    
    if solution.t_events[0].size > 0:
        t_end = solution.t_events[0][0]
        v_final, M_final, T_final, x_final = solution.sol(t_end)
    else:
        v_final, M_final, T_final, x_final = solution.y[:, -1]
    
    if T_final <= 0:
        T_final = 1e-10
    
    V_final = V1 + A * x_final
    P_final = M_final * R * T_final / V_final
    pressure_constraint = P_final >= Patm
    
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
                P_current = 1.0  # 数值保护
            
            # 计算当前泄漏率
            leak = Cd_current * A1 * P_current * np.sqrt(gamma / (R * T_current)) * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
            h = Cp * T_current
            leak_energy += leak * h * dt
        
        U0 = Cp * M0 * T1
        Uf = Cp * M_final * T_final
        W_total = U0 - Uf - leak_energy
        yita = 0.5 * m * v_final** 2 / W_total if W_total > 0 else 0
        yita_zong = 0.5 * m * v_final** 2 / U0
    else:
        yita = 1e-6
        yita_zong = 1e-6
        pressure_constraint = False
    
    return yita, yita_zong, v_final, x_final, pressure_constraint, P_final

def dynamic_penalty(iteration, max_iterations):
    return 1 + 10 * (iteration / max_iterations) ** 2

def objective_function(params, constants, iteration, max_iterations, alpha=1.0, beta=1.0, gamma_weight=0.0, v_ref=60.0):
    P1, T1, A1, V1, L = params
    m,gamma, R, Cv, Cp, T0, A, Patm = constants
    yita, yita_zong, v_final, x_final, pressure_ok, P_final = calculate_efficiency(P1, T1, A1, V1, L, Patm)
    
    obj_value = alpha * yita + beta * yita_zong + gamma_weight * (v_final / v_ref)
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
        self.model = make_pipeline(
            StandardScaler(),
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
        self.scaler_y = StandardScaler()
    
    def train(self, X, y):
        y_scaled = self.scaler_y.fit_transform(y)
        self.model.fit(X, y_scaled)
    
    def predict(self, X):
        y_pred_scaled = self.model.predict(X)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred  # [v, yita, yita_zong]

def generate_training_data(constants, num_samples=10000):
    sampler = qmc.LatinHypercube(d=5)
    params = sampler.random(num_samples)
    low = np.array([5e5, 300, 1e-5, 0.001, 0.6])
    high = np.array([10e6, 800, 5e-5, 0.005, 1.5])
    params = qmc.scale(params, low, high)
    
    X, y = [], []
    m, gamma, R, Cv, Cp, T0, A, Patm = constants
    for p in params:
        P1, T1, A1, V1, L = p
        yita, yita_zong, v_end, x_end, pressure_ok, P_final = calculate_efficiency(P1, T1, A1, V1, L, Patm)
        if pressure_ok and x_end >= L - 1e-3:
            X.append(p)
            y.append([v_end, yita, yita_zong])
    return np.array(X), np.array(y)

class Particle:
    def __init__(self, position, velocity, bounds):
        self.position = position
        self.velocity = velocity
        self.bounds = bounds
        self.fitness = float('inf')
        self.best_fitness = float('inf')
        self.best_position = position.copy()
    
    def update_velocity(self, global_best_position, w):
        c1 = 1.5
        c2 = 1.5
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social
    
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
        self.bounds = bounds
        self.constants = constants
        self.max_iter = max_iter
        self.global_best_position = None
        self.global_best_value = float('inf')
        self.particles = []
        self.convergence_history = []
        self.alpha = alpha
        self.beta = beta
        self.gamma_weight = gamma_weight
        self.v_ref = v_ref
        self.time_elapsed = 0.0
    
    def optimize(self, surrogate_model):
        start_time = time.time()
        for i in range(self.max_iter):
            for particle in self.particles:
                if np.random.rand() < 0.7:
                    pred = surrogate_model.predict([particle.position])[0]
                    pred_v, pred_yita, pred_yita_zong = pred
                    
                    P1, T1, A1, V1, L = particle.position
                    V_final = V1 + A * L
                    M_final = P1 * V1/(R*T1) - 0.62*A1*P1*np.sqrt(gamma/(R*T1)) * (2/(gamma+1))**((gamma+1)/(2*(gamma-1))) * 0.05
                    T_final = T1 * (V1/V_final)**(gamma-1)
                    P_final = M_final*R*T_final/V_final
                    pressure_ok = P_final >= self.constants[-1]
                    
                    if pressure_ok:
                        penalty = 0.0
                        if pred_v < self.v_ref:
                            penalty += dynamic_penalty(i, self.max_iter) * (self.v_ref - pred_v)
                        obj_value = self.alpha * pred_yita + self.beta * pred_yita_zong + self.gamma_weight * (pred_v / self.v_ref)
                        particle.fitness = -obj_value + penalty
                    else:
                        particle.fitness = float('inf')
                else:
                    particle.fitness = objective_function(
                        particle.position, self.constants, i, self.max_iter,
                        alpha=self.alpha, beta=self.beta, gamma_weight=self.gamma_weight, v_ref=self.v_ref
                    )
                
                particle.update_best()
                
                if particle.fitness < self.global_best_value:
                    self.global_best_value = particle.fitness
                    self.global_best_position = particle.position.copy()
            
            self.convergence_history.append(self.global_best_value)
            w = 0.9 - 0.5 * (i / self.max_iter)
            
            for particle in self.particles:
                particle.update_velocity(self.global_best_position, w)
                particle.apply_bounds()
            
            if i % 10 == 0:
                self.genetic_operation()
        
        # 最终评估
        yita, yita_zong, v_final, x_final, pressure_ok, P_final = calculate_efficiency(
            *self.global_best_position, Patm=self.constants[-1]
        )
        self.time_elapsed = time.time() - start_time
        
        return self.global_best_position, -self.global_best_value, self.convergence_history, {
            'efficiency': yita,
            'total_efficiency': yita_zong,
            'velocity': v_final,
            'pressure_ok': pressure_ok,
            'final_pressure': P_final
        }
    
    def genetic_operation(self):
        fitness_values = np.array([p.fitness for p in self.particles])
        probabilities = np.exp(-fitness_values) / np.sum(np.exp(-fitness_values))
        selected_indices = np.random.choice(self.n_particles, size=self.n_particles, p=probabilities)
        selected_particles = [self.particles[i] for i in selected_indices]
        
        new_particles = []
        for j in range(0, self.n_particles, 2):
            parent1 = selected_particles[j]
            parent2 = selected_particles[j + 1]
            child1, child2 = self.crossover(parent1, parent2)
            new_particles.extend([child1, child2])
        
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
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.constants = constants
        self.max_iter = max_iter
        self.global_best_position = None
        self.global_best_value = float('inf')
        self.particles = []
        self.convergence_history = []
        self.time_elapsed = 0.0
    
    def optimize(self):
        start_time = time.time()
        for i in range(self.max_iter):
            for particle in self.particles:
                particle.fitness = objective_function(particle.position, self.constants, i, self.max_iter)
                if particle.fitness < particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()
                if particle.fitness < self.global_best_value:
                    self.global_best_value = particle.fitness
                    self.global_best_position = particle.position.copy()
            
            self.convergence_history.append(self.global_best_value)
            w = 0.9 - 0.5 * (i / self.max_iter)
            
            for particle in self.particles:
                c1, c2 = 1.5, 1.5
                r1, r2 = np.random.rand(self.dimensions), np.random.rand(self.dimensions)
                cognitive = c1 * r1 * (particle.best_position - particle.position)
                social = c2 * r2 * (self.global_best_position - particle.position)
                particle.velocity = w * particle.velocity + cognitive + social
                particle.position += particle.velocity
                particle.apply_bounds()
        
        self.time_elapsed = time.time() - start_time
        return self.global_best_position, -self.global_best_value, self.convergence_history

class GeneticAlgorithm:
    def __init__(self, population_size, dimensions, bounds, constants, max_iter):
        self.pop_size = population_size  # 种群大小（50）
        self.dim = dimensions
        self.bounds = bounds
        self.constants = constants
        self.max_iter = max_iter
        self.best_position = None
        self.best_value = float('inf')
        self.global_best_value = float('inf')
        self.convergence_history = []
        self.time_elapsed = 0.0
        self.elitism_rate = 0.1  # 精英保留比例
        self.min_bounds = bounds[:, 0]
        self.max_bounds = bounds[:, 1]
    
    def initialize_population(self):
        return np.array([
            np.random.uniform(self.min_bounds, self.max_bounds, self.dim)
            for _ in range(self.pop_size)
        ]).clip(self.min_bounds, self.max_bounds)
    
    def check_bounds(self, individual):
        return np.all(individual >= self.min_bounds) and np.all(individual <= self.max_bounds)
    
    def selection(self, population, fitness):
        pop_size = len(population)
        elite_size = int(pop_size * self.elitism_rate)
        elite_size = min(elite_size, pop_size - 1)
        
        # 选择精英个体
        elite_indices = np.argsort(fitness)[:elite_size]
        elite_individuals = population[elite_indices]
        
        # 计算选择概率
        fitness = np.array(fitness, dtype=float)
        min_fitness = np.min(fitness)
        max_fitness = np.max(fitness)
        
        if max_fitness == min_fitness:
            probabilities = np.ones(pop_size) / pop_size
        else:
            normalized_fitness = fitness - min_fitness
            if np.sum(normalized_fitness) == 0:
                probabilities = np.ones(pop_size) / pop_size
            else:
                probabilities = normalized_fitness / np.sum(normalized_fitness)
        
        remaining_size = pop_size - elite_size
        selected_indices = list(elite_indices)
        
        while len(selected_indices) < pop_size:
            new_indices = np.random.choice(
                pop_size, 
                size=pop_size - len(selected_indices), 
                replace=True,
                p=probabilities
            )
            selected_indices.extend(new_indices)
        
        # 去重并截取到种群大小
        selected_indices = np.unique(selected_indices)[:pop_size]
        return population[selected_indices]
    
    def crossover(self, parents):
        pop_size = self.pop_size
        elite_size = int(pop_size * self.elitism_rate)
        elite_size = min(elite_size, pop_size - 1)
        children = []
        
        # 直接添加精英个体
        if len(parents) >= elite_size:
            children.extend(parents[:elite_size])
        else:
            children.extend(parents)
        
        # 生成交叉后代
        n_parents = len(parents)
        for i in range(0, n_parents, 2):
            if len(children) >= pop_size:
                break
            if i + 1 < n_parents:
                p1, p2 = parents[i], parents[i + 1]
                alpha = np.random.rand()
                child1 = alpha * p1 + (1 - alpha) * p2
                child2 = (1 - alpha) * p1 + alpha * p2
                child1 = np.clip(child1, self.min_bounds, self.max_bounds)
                child2 = np.clip(child2, self.min_bounds, self.max_bounds)
                children.extend([child1, child2])
        
        # 补充随机个体直到达到种群大小
        while len(children) < pop_size:
            rand_individual = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], self.dim)
            children.append(rand_individual)
        
        return np.array(children[:pop_size])
    
    def mutation(self, population, rate=0.1):
        """确保变异操作在合法范围内"""
        for i in range(len(population)):
            if np.random.rand() < rate:
                idx = np.random.randint(0, self.dim)
                mutation_step = 0.1 * (1 - self.current_iter / self.max_iter)
                population[i, idx] += np.random.normal(0, mutation_step) * (self.max_bounds[idx] - self.min_bounds[idx])
                population[i, idx] = np.clip(population[i, idx], self.min_bounds[idx], self.max_bounds[idx])
        return population
    
    def optimize(self):
        start_time = time.time()
        population = self.initialize_population()
        
        for i in range(self.max_iter):
            self.current_iter = i
            
            fitness = []
            for p in population:
                if not self.check_bounds(p):
                    fitness.append(1e10)
                    continue
                fitness_val = objective_function(p, self.constants, i, self.max_iter)
                fitness.append(fitness_val)
            
            fitness = np.array(fitness)
            best_idx = np.argmin(fitness)
            current_best = fitness[best_idx]
            
            if current_best < self.global_best_value:
                self.global_best_value = current_best
                self.best_position = population[best_idx].copy()
                            # 调试输出
                if i % 10 == 0:
                    print(f"迭代 {i}: 找到新的最优解 - 参数 = {self.best_position}, 适应度 = {self.global_best_value:.6f}")
            self.convergence_history.append(self.global_best_value)
            parents = self.selection(population, fitness)
            children = self.crossover(parents)
            population = self.mutation(children)
        
        self.time_elapsed = time.time() - start_time
        
        final_yita, final_yita_zong, final_v, _, final_pressure_ok, final_P = calculate_efficiency(
            *self.best_position, Patm
        )
        print(f"遗传算法优化完成 - 最佳参数: {self.best_position}")
        print(f"最终评估: 效率 = {final_yita:.6f}, 总效率 = {final_yita_zong:.6f}, 速度 = {final_v:.2f} m/s, 压力约束 = {final_pressure_ok}, 最终压力 = {final_P:.2f} Pa")
        
        return self.best_position, -self.global_best_value, self.convergence_history, {
            'efficiency': final_yita,
            'total_efficiency': final_yita_zong,
            'velocity': final_v,
            'pressure_ok': final_pressure_ok,
            'final_pressure': final_P
        }

def plot_algorithm_convergence(alg_name, convergence_histories, num_runs):
    histories = np.array(convergence_histories)
    max_iter = histories.shape[1]
    
    mean_history = np.mean(histories, axis=0)
    std_history = np.std(histories, axis=0)
    overall_std = np.mean(std_history)
    
    plt.figure(figsize=(8, 6))
    
    for i in range(num_runs):
        plt.plot(range(1, max_iter), histories[i][1:], alpha=0.2, color='gray', linestyle='-')
    
    plt.plot(range(1, max_iter), mean_history[1:], 'b-', linewidth=2, label='Mean')
    plt.fill_between(
        x=range(1, max_iter),
        y1=mean_history[1:] - std_history[1:],
        y2=mean_history[1:] + std_history[1:],
        color='blue',
        alpha=0.1,
        label='Standard Deviation'
    )
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Fitness (Negative Objective)', fontsize=12)
    plt.title(f'{alg_name} Convergence Curves (N={num_runs})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{alg_name.lower()}_convergence.png', dpi=1200)
    plt.close()
    
    print(f"\n------------------- {alg_name} 收敛统计 -------------------")
    print(f"平均收敛曲线均方差（标准差）: {overall_std:.6f}")
    print(f"各迭代步最大标准差: {np.max(std_history):.6f}")
    print(f"各迭代步最小标准差: {np.min(std_history):.6f}")

if __name__ == "__main__":
    BASE_SEED = 42
    num_runs = 20  
    algorithms = ['GeneticAlgorithm','MMPSO','StandardPSO']
    results = {alg: [] for alg in algorithms}
    convergence_histories = {alg: [] for alg in algorithms}
    run_times = {alg: [] for alg in algorithms}
    
    # 使用第一种权重组合
    alpha, beta, gamma_weight = 1.0, 1.0, 0.0
    v_ref = 60.0
    
    for alg in algorithms:
        print(f"\n------------------- 开始运行 {alg} 算法 -------------------")
        for run in range(num_runs):
            print(f"运行 {alg} {run + 1}/{num_runs}...")
            run_seed = BASE_SEED + run
            np.random.seed(run_seed)
            
            X, y = generate_training_data(constants, num_samples=5000)
            model = SurrogateModel()
            model.train(X, y)
            
            n_particles = 50
            dimensions = 5
            bounds = np.array([[5e5, 10e6], [300, 800], [1e-5, 5e-5], [0.001, 0.005], [0.6, 1.5]])
            max_iter = 100
            
            if alg == 'StandardPSO':
                pso = StandardPSO(n_particles, dimensions, bounds, constants, max_iter)
                for i in range(n_particles):
                    np.random.seed(run_seed + i)
                    position = np.random.uniform(bounds[:, 0], bounds[:, 1])
                    velocity = np.random.uniform(-1, 1, dimensions)
                    pso.particles.append(Particle(position, velocity, bounds))
                best_position, best_fitness, history = pso.optimize()
                final_yita, final_yita_zong, final_v, _, final_pressure_ok, final_P = calculate_efficiency(*best_position, Patm)
                run_time = pso.time_elapsed
            
            elif alg == 'GeneticAlgorithm':
                ga = GeneticAlgorithm(n_particles, dimensions, bounds, constants, max_iter)
                best_position, best_fitness, history, final_results = ga.optimize()
                final_yita = final_results['efficiency']
                final_yita_zong = final_results['total_efficiency']
                final_v = final_results['velocity']
                final_pressure_ok = final_results['pressure_ok']
                final_P = final_results['final_pressure']
                run_time = ga.time_elapsed
            
            elif alg == 'MMPSO':
                pso = MMPSO(
                    n_particles, dimensions, bounds, constants, max_iter,
                    alpha=alpha, beta=beta, gamma_weight=gamma_weight, v_ref=v_ref
                )
                for i in range(n_particles):
                    np.random.seed(run_seed + i)
                    position = np.random.uniform(bounds[:, 0], bounds[:, 1])
                    velocity = np.random.uniform(-1, 1, dimensions)
                    pso.particles.append(Particle(position, velocity, bounds))
                best_position, best_fitness, history, final_results = pso.optimize(model)
                final_yita = final_results['efficiency']
                final_yita_zong = final_results['total_efficiency']
                final_v = final_results['velocity']
                final_pressure_ok = final_results['pressure_ok']
                final_P = final_results['final_pressure']
                run_time = pso.time_elapsed
            
            P1, T1, A1, V1, L = best_position
            print(
                f"运行完成 | "
                f"参数: P1={P1/1e6:.2f} MPa, T1={T1:.0f} K, A1={A1*1e6:.2f} mm², V1={V1*1e3:.2f} mL, L={L:.2f} m | "
                f"效率: {final_yita*100:.2f}% | 总效率: {final_yita_zong*100:.2f}% | "
                f"速度: {final_v:.2f} m/s | "
                f"压力约束: {'满足' if final_pressure_ok else '不满足'} | "
                f"时间: {run_time:.2f}s"
            )
            
            results[alg].append({
                'efficiency': final_yita,
                'total_efficiency': final_yita_zong,
                'velocity': final_v,
                'pressure_ok': final_pressure_ok,
                'final_pressure': final_P
            })
            convergence_histories[alg].append(history)
            run_times[alg].append(run_time)
        
        # 计算算法平均性能
        avg_efficiency = np.mean([r['efficiency'] for r in results[alg]])
        avg_total_efficiency = np.mean([r['total_efficiency'] for r in results[alg]])
        avg_time = np.mean(run_times[alg])
        print(f"\n{alg} 算法总结：")
        print(f"  平均效率: {avg_efficiency*100:.2f}% (±{np.std([r['efficiency'] for r in results[alg]])*100:.2f}%)")
        print(f"  平均总效率: {avg_total_efficiency*100:.2f}% (±{np.std([r['total_efficiency'] for r in results[alg]])*100:.2f}%)")
        print(f"  平均运行时间: {avg_time:.2f}秒 (±{np.std(run_times[alg]):.2f}秒)\n")
        plot_algorithm_convergence(alg, convergence_histories[alg], num_runs)
    
    # 三种算法收敛对比
    plt.figure(figsize=(8, 6))
    for alg in algorithms:
        mean_history = np.mean(convergence_histories[alg], axis=0)
        plt.plot(range(0, len(mean_history)), mean_history, linewidth=2, label=f'{alg}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness (Negative Objective)')
    plt.title('Comparison of the average convergence distribution of the three algorithms')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('combined_convergence.png', dpi=1200)
    plt.close()
    
    # 效率对比
    # 第一个图表：Utilize Efficiency
    plt.figure(figsize=(8, 6))
    efficiencies = {alg: [r['efficiency']*100 for r in results[alg]] for alg in algorithms}
    plt.boxplot(efficiencies.values(), labels=algorithms)
    plt.xlabel('Algorithm')
    plt.ylabel('Utilize Efficiency (%)')
    plt.title('Comparison of the utilize efficiency distribution of the three algorithms')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('utilize_efficiency_comparison.png', dpi=1200)
    plt.close()

    # 第二个图表：Total Efficiency
    plt.figure(figsize=(8, 6))
    total_efficiencies = {alg: [r['total_efficiency']*100 for r in results[alg]] for alg in algorithms}
    plt.boxplot(total_efficiencies.values(), labels=algorithms)
    plt.xlabel('Algorithm')
    plt.ylabel('Total Efficiency (%)')
    plt.title('Comparison of the total efficiency distribution of the three algorithms')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('total_efficiency_comparison.png', dpi=1200)
    plt.close()
    
    # 输出最终对比统计
    print("\n------------------- 三种算法综合对比 -------------------")
    for alg in algorithms:
        eff = [r['efficiency']*100 for r in results[alg]]
        total_eff = [r['total_efficiency']*100 for r in results[alg]]
        time = run_times[alg]
        success_rate = np.mean([r['pressure_ok'] for r in results[alg]])*100
        print(f"{alg}:")
        print(f"  平均效率: {np.mean(eff):.2f}% ± {np.std(eff):.2f}%")
        print(f"  平均总效率: {np.mean(total_eff):.2f}% ± {np.std(total_eff):.2f}%")
        print(f"  平均运行时间: {np.mean(time):.2f}s ± {np.std(time):.2f}s")
        print(f"  压力约束满足率: {success_rate:.2f}%")



