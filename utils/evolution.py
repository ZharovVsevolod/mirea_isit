import numpy as np
from scipy.optimize import newton
from geneticalgorithm import geneticalgorithm
from pyswarm import pso
from bees_algorithm import BeesAlgorithm
from scipy.optimize import  differential_evolution
import time
from tqdm import tqdm

def rastirigin(x, A:int=10):
    x = np.array(x)
    n = len(x)
    return A * n + sum(x**2 - A * np.cos(2 * 3.14 * x))

def rastrigin_for_bees(x, A:int=10):
    return rastirigin(x, A) * (-1)

def rosenbrok(x):
    x = np.array(x)
    return sum(100 * (x[1:] - (x[:-1])**2)**2 + (x[:-1] - 1)**2)

def rosenbrok_for_bees(x):
    return rosenbrok(x) * (-1)

def bil(input_bil):
    x, y = input_bil
    return (1.5 - x - x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

def bil_for_bees(input_bil):
    return bil(input_bil) * (-1)

def make_newton(x:list, func=rosenbrok) -> (list, list):
    answers = []
    times = []
    for data in tqdm(x):
        start_time = time.time()
        root = newton(func, data, maxiter=1000000)
        answers.append(list(root))
        times.append(time.time() - start_time)
    return answers, times

def ga_100_run(
        x,
        algorithm_param = {
            'max_num_iteration': 100, 
            'population_size': 100, 
            'mutation_probability': 0.1, 
            'elit_ratio': 0.01, 
            'crossover_probability': 0.5, 
            'parents_portion': 0.3, 
            'crossover_type': 'uniform', 
            'max_iteration_without_improv': None
        },
        func = rosenbrok
    ):
    model = geneticalgorithm(
        func, 
        dimension=3, 
        variable_type="real", 
        variable_boundaries=x, 
        algorithm_parameters=algorithm_param,
        convergence_curve=False,
        progress_bar=False
    )
    
    answers = []
    times = []

    for _ in tqdm(range(100)):
        start_timer = time.time()
        model.run()
        answers.append(model.best_variable)
        times.append(time.time() - start_timer)

    return answers, times

def pso_100_run(x_bottom, x_top, func=rosenbrok):
    pso_answers = []
    pso_func_answers = []
    pso_time = []
    for _ in tqdm(range(100)):
        start_timer = time.time()
        x_opt, f_opt = pso(func, x_bottom, x_top, maxiter=10000)
        pso_answers.append(x_opt)
        pso_func_answers.append(f_opt)
        pso_time.append(time.time() - start_timer)
    return pso_answers, pso_func_answers, pso_time

def bees_100_run(x_bottom, x_top, func=rosenbrok_for_bees):
    bees_answers = []
    bees_func_answers = []
    bees_time = []
    for _ in tqdm(range(100)):
        alg = BeesAlgorithm(func, x_bottom, x_top)
        start_timer = time.time()
        alg.performFullOptimisation(max_iteration=100)
        bees_answers.append(alg.best_solution.values)
        bees_func_answers.append(alg.best_solution.score)
        bees_time.append(time.time() - start_timer)
    return bees_answers, bees_func_answers, bees_time

def de_100_run(x, func=rosenbrok):
    de_answers = []
    de_func_answers = []
    de_time = []
    for _ in tqdm(range(100)):
        start_timer = time.time()
        result = differential_evolution(func, x, updating="deferred")
        de_answers.append(result.x)
        de_func_answers.append(result.fun)
        de_time.append(time.time() - start_timer)
    return de_answers, de_func_answers, de_time

def all_in_one(x_bottom, x_top, func, bees_func):
    # Данные в разном виде
    print("Подготовка данных")
    # newton_random_data = np.random.randn(100, 3) * 1.15
    x = np.stack([x_bottom, x_top], axis=1)

    # Ньютон
    # print("Алгоритм оптимизации Ньютона")
    # n_asnwers, n_time = make_newton(newton_random_data, func=func)
    # print("-----")
    # Генетический
    print("Генетический алгоритм")
    ga_answers, ga_times = ga_100_run(x, func=func)
    print("-----")
    # Рой
    print("Алгоритм роя частиц")
    pso_answers, pso_func_answers, pso_time = pso_100_run(x_bottom, x_top, func=func)
    print("-----")
    # Пчёлы
    print("Пчелиный алгоритм")
    bees_answers, bees_func_answers, bees_time = bees_100_run(x_bottom, x_top, func=bees_func)
    print("-----")
    # Дифференциальная эволюция
    print("Алгоритм дифференциальной эволюции")
    de_answers, de_func_answers, de_time = de_100_run(x, func=func)
    print("-----")

    return (
        # n_asnwers, n_time, 
        ga_answers, ga_times, 
        pso_answers, pso_time, 
        bees_answers, bees_time, 
        de_answers, de_time
    )




