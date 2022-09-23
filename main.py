from scipy.optimize import rosen, differential_evolution, Bounds, NonlinearConstraint
import numpy as np
import matplotlib.pyplot as plt
import signal

design_factor = 7
population = 20
support_points = 20
theta = np.asanyarray((-0.4926, -0.6280, -0.3283, 0.4378, 0.5283, -0.6120, -0.6837, -0.2061))
y_plot = []

def save_plot(signum, frame):
    plt.plot(y_plot)
    plt.savefig("output.jpg")
    exit(0)

signal.signal(signal.SIGINT, save_plot)

def obj_func(x: tuple, *args):
    a = np.asanyarray(x) 
    a = np.reshape(a, (support_points, 1+design_factor)) # p x (1+factor)
    split = np.hsplit(a, np.array([1, 1+design_factor]))
    n = split[0] # p x 1
    n = np.rint(n)
    x = split[1] # p x factor
    r = np.insert(x, 0, [1], 1) # p x (factor + 1)

    # (factor+1) x (factor+1)
    information_matrix = np.sum([
        n[i].item()/support_points * \
        # 1/support_points * \
        np.exp(np.matmul(r_n, theta)) / (1+np.exp(np.matmul(r_n, theta)))**2 * \
        np.matmul(
            np.expand_dims(r_n, axis=-1), # (factor+1) x 1
            np.expand_dims(r_n, axis=0), # 1 x (factor+1)
        )
        for i, r_n in enumerate(r)
    ], axis=0)

    obj = -np.log(np.linalg.det(information_matrix))

    return obj

def termination_check(xk, convergence):
    a = np.asanyarray(xk) 
    a = np.reshape(a, (support_points, 1+design_factor)) # p x (1+factor)
    split = np.hsplit(a, np.array([1, 1+design_factor]))
    n = split[0] # p x 1
    n = np.rint(n)
    x = split[1] # p x factor
    r = np.insert(x, 0, [1], 1) # p x (factor + 1)
    
    # (factor+1) x (factor+1)
    information_matrix = np.sum([
        n[i].item()/support_points * \
        # 1/support_points * \
        np.exp(np.matmul(r_n, theta)) / (1+np.exp(np.matmul(r_n, theta)))**2 * \
        np.matmul(
            np.expand_dims(r_n, axis=-1), # (factor+1) x 1
            np.expand_dims(r_n, axis=0), # 1 x (factor+1)
        )
        for i, r_n in enumerate(r)
    ], axis=0)

    # scalar
    sensitivity_values = []

    for r_n in r:
        sensitivity_value = \
            np.exp(np.matmul(r_n, theta)) / \
            ( (1 + np.exp(np.matmul(r_n, theta))) **2 ) * \
            (
                # 1 x (factor+1)
                np.expand_dims(r_n, axis=0) @
                # (factor+1) x (factor+1)
                np.linalg.inv(information_matrix) @
                # (factor+1) x 1
                np.expand_dims(r_n, axis=-1)
            ) - \
            (design_factor + 1)
        sensitivity_values.append(sensitivity_value.item())
    
    obj = -np.log(np.linalg.det(information_matrix))
    y_plot.append(-obj)

    d_efficiency = np.exp(-max(sensitivity_values)/(design_factor + 1))
    # print(f"D-efficiency: {d_efficiency}")

    return False

def constr_f(x):
    a = np.asanyarray(x) 
    a = np.reshape(a, (support_points, 1+design_factor)) # p x (1+factor)
    split = np.hsplit(a, np.array([1, 1+design_factor]))
    n = split[0] # p x 1
    c = np.sum(n)
    # print(c)
    return c

constrain = NonlinearConstraint(constr_f, support_points, support_points)
bounds = np.full((support_points, design_factor+1), 0, dtype='i,i')
bounds[:,:] = (-1,1) # design factor bounds
bounds[:,0] = (0, 2) # replicates constrains
bounds = bounds.flatten()
bounds = bounds.tolist()

integrality = np.full((support_points, design_factor+1), False, dtype='b')
integrality[:,0] = True # replicates count are integer
integrality = integrality.flatten()
integrality = integrality.tolist()

result = differential_evolution(
    func=obj_func,
    bounds=bounds,
    polish=False,
    callback=termination_check,
    strategy="rand2bin",
    popsize=population,
    mutation=0.8,
    disp=True,
    recombination=0.9,
    maxiter=2000,
    constraints=(constrain),
    integrality=integrality,
    tol=0,
)

print(result)

save_plot()