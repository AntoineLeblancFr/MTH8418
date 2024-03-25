import torch
from torch.quasirandom import SobolEngine
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
import sys
import matplotlib.pyplot as plt
import time
from botorch.utils import standardize
import PyNomad

gamma = [0.0137, 0.0274, 0.0434, 0.0866, 0.137, 0.274, 0.434, 0.866, 1.37, 2.74, 4.34, 5.46, 6.88]
etha = [3220, 2190, 1640, 1050, 766, 490, 348, 223, 163, 104, 76.7, 68.1, 58.2]

def f_rhelogie(etha0, betha, lambda_, gamma_i):
    """
    Fonction f dont les paramètres sont à déterminer
    """
    res = 1 + (lambda_**2) * (gamma_i **2)
    res = res**((betha - 1)/2)
    return etha0 * res

def epsilon(i, etha0, lambda_, betha):
    """
    Calcul de l'erreur absolue entre une donnée et notre prédiction
    """
    return abs(f_rhelogie(etha0, betha, lambda_, gamma[i]) - etha[i])

def h(etha0, lambda_, betha):
    """
    Somme des valeurs absolues entre les données et notre prédiction
    """
    abs_err = 0
    for i in range(len(gamma)):
        abs_err += epsilon(i, etha0, lambda_, betha)
    return abs_err

def g(H, p):
    estimation = 0
    for i in range(len(H)):
        estimation += abs(H[i])**p
    return estimation**(1/p)
start_time2 = time.time()
times_ = []

# Definition of the Black Bow for Nomad
def bb(x):
    try:
        x0 = x.get_coord(0)
        x1 = x.get_coord(1)
        x2 = x.get_coord(2)
        H = [h(520*x0, 14*x1, 0.038*x2)]
        rawBBO = str(g(H,p))
        times_.append(time.time() - start_time2)
        x.setBBO(rawBBO.encode("UTF-8"))
    except:
        print("Unexpected eval error", sys.exc_info()[0])
        return 0
    return 1 # 1: success 0: failed evaluation

def LHS(bounds, n_samples):
    sobol = SobolEngine(dimension=len(bounds), scramble=True)
    return sobol.draw(n_samples).to(dtype=torch.float32) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

# Device Characteristics
seed = 0
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

# Hyperparameters
p = 2
n_evals = 150
n_random_points = 25
bounds = torch.tensor([(0, 20), (0, 20), (0, 20)], dtype=dtype)

# Random points generation
X_train = LHS(bounds, n_random_points)

# Random points evaluation
Y_train = torch.tensor([g([h(520*x[0], 14*x[1], 0.038*x[2])],p) for x in X_train], dtype=dtype)

# Extraction of the best point
min_idx = torch.argmin(Y_train)
h_best = Y_train[min_idx]
x_best = X_train[min_idx]

Y_graph = [-Y_train.min().item()]
times = [0.0]
start_time = time.time()

# Bayesian Optimization
for _ in range(n_evals - n_random_points):
    
    # Preprocessing 
    X_min, X_max = X_train.min(dim=0)[0], X_train.max(dim=0)[0]
    X_train2 = (X_train - X_min) / (X_max - X_min)
    
    Y_train2 = standardize(Y_train)
    Y_train2 = Y_train2.unsqueeze(-1) if Y_train2.dim() == 1 else Y_train2
    
    # Construction of the model
    model = SingleTaskGP(train_X=X_train2, train_Y=Y_train2)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)#, step_limit=1000, optimizer=partial(Adam, lr=0.1))

    # Choice for the next point
    EI = ExpectedImprovement(model=model, best_f=Y_train2.min(), maximize = False)
    candidate_nlzd, _ = optimize_acqf(EI, bounds=torch.stack([torch.zeros(3), torch.ones(3)]),
        q=1, num_restarts=5, raw_samples=20,)
    
    # Update
    candidate = candidate_nlzd * (X_max - X_min) + X_min
    candidate = candidate.squeeze()
    new_y = g([h(520*candidate[0].item(), 14*candidate[1].item(), 0.038*candidate[2].item())], p)
    X_train = torch.cat([X_train, candidate.unsqueeze(-2)])
    new_y = torch.tensor(new_y, dtype=dtype).clone().detach()
    Y_train = torch.cat([Y_train, new_y.unsqueeze(-1)])
    Y_graph.append(Y_train.min().item())
    times.append(time.time() - start_time)

best_idx = torch.argmin(Y_train)
h_best_BO = Y_train[min_idx]
x_best_BO = X_train[min_idx]
print("\nBO results \n" + "Best point : " + str(x_best_BO.tolist()) + " \nBest evaluation : " + str(h_best_BO.item()))

# Save the tensors to the specified file
file_path1 = "data_eval.txt"
file_path2 = "data_cpu.txt"
with open(file_path2, "w") as file:
    for x, y in zip(times, Y_graph):
        file.write(f"{x} {y}\n")
X = [i for i in range(0,126)]
with open(file_path1, "w") as file:
    for x, y in zip(X, Y_graph):
        file.write(f"{x} {y}\n")


X0 = x_best
# Nomad Optimization
params_basic = ["BB_OUTPUT_TYPE OBJ", "MAX_BB_EVAL 125", "LOWER_BOUND * 0", "UPPER_BOUND * 20", "DISPLAY_DEGREE 2", "QUAD_MODEL_SEARCH FALSE" ,"DISPLAY_ALL_EVAL false", "DISPLAY_STATS BBE OBJ",  "HISTORY_FILE BasicNomad.txt", "EVAL_STATS_FILE StatsBasic.txt"]
start_time2 = time.time()
result_basic = PyNomad.optimize(bb, X0, [] , [], params_basic)

fmt_basic = ["{} = {}".format(n,v) for (n,v) in result_basic.items()]
output_basic = "\n".join(fmt_basic)
print("\nNOMAD results \n" + output_basic + " \n")

# Nomad Optimization with quadratic models
params_quadratic = ["BB_OUTPUT_TYPE OBJ", "MAX_BB_EVAL 125", "LOWER_BOUND * 0", "UPPER_BOUND * 20", "DISPLAY_DEGREE 2", "DISPLAY_ALL_EVAL false", "DISPLAY_STATS BBE OBJ", "HISTORY_FILE QuadNomad.txt", "EVAL_STATS_FILE StatsQuad.txt"]

result_quadratic = PyNomad.optimize(bb, X0, [] , [], params_quadratic)
start_time2 = time.time()
fmt_quadratic = ["{} = {}".format(n,v) for (n,v) in result_quadratic.items()]
output_quadratic = "\n".join(fmt_quadratic)
print("\nNOMAD results \n" + output_quadratic + " \n")

with open("times_nomads.txt", "w") as file:
    for x in zip(times_):
        file.write(f"{x}\n")