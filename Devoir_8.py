# Devoir 8 - Projet chapitre 8
# Antoine Leblanc - 2310186
import numpy as np

########################################################################

# Données 
gamma = [0.0137, 0.0274, 0.0434, 0.0866, 0.137, 0.274, 0.434, 0.866, 1.37, 2.74, 4.34, 5.46, 6.88]
etha = [3220, 2190, 1640, 1050, 766, 490, 348, 223, 163, 104, 76.7, 68.1, 58.2]

def f(etha0, betha, lambda_, gamma_i):
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
    return abs(f(etha0, betha, lambda_, gamma[i]) - etha[i])

def g_hat(etha0, lambda_, betha):
    """
    Somme des valeurs absolues entre les données et notre prédiction
    """
    abs_err = 0
    for i in range(len(gamma)):
        abs_err += epsilon(i, etha0, lambda_, betha)
    return abs_err

########################################################################

# Question a
def randv(dim):
    """
    Fonction qui retourne un vecteur de appartenant à R**dim
    """
    vecteur = np.random.randn(dim)
    # On s'assure que les vecteur n'est pas nul
    while np.all(vecteur == 0):
        vecteur = np.random.randn(dim)
    # On le normalise en utilisant la norme 2
    norme = np.linalg.norm(vecteur)
    vecteur_normalise = vecteur / norme
    return vecteur_normalise

# Question b 
def set_poll_directions(deltak, DELTAk):
    B = np.zeros((3, 3))
    v = randv(3)
    In = np.eye(3)
    H = In - 2 * np.outer(v, v.T)
    for j in range(len(H[0,:])):
        norme_Hj = np.linalg.norm(H[:,j], ord=np.inf)
        B[:,j] = np.round(DELTAk * H[:,j]/(deltak * norme_Hj))
    # On prend la transposée pour directement récupérer les directions d lors du parcours de D
    D = np.transpose(np.concatenate([B, -B], axis=1))
    return D

# Question c
def MADS(f, x0, budget, f_0, seed):
    """
    Implémentation de MADS sans le step 2
    """
    np.random.seed(seed)
    DELTAk = 1
    xk     = x0
    f_xk   = f_0
    tau    = 1/2
    k      = 1 # Une première évaluation a été utilisée pour x_0
    
    while k < budget:
        moving = False
        deltak = min(DELTAk, DELTAk ** 2)
        D = set_poll_directions(deltak, DELTAk)
        
        for i in range(len(D)):
            d = D[i]
            t = xk + deltak * d
            f_eval = f(520*t[0],14*t[1],0.038*t[2])
            k += 1
            if f_xk > f_eval:
                xk = t
                f_xk = f_eval
                DELTAk *= (1/tau)
                moving = True
                break

            if k == budget:
                break
        if not moving:
            DELTAk *= tau
    return xk, f_xk

# Question d
n_evals = [125, 375, 875]
starting_point = [[15,20,10],
                  [8.172517606, 5.058263716, 5.444856567],
                  [13.04832254, 15.84400309, 9.950620587],
                  [12.31453665, 13.75028434, 9.557207957],
                  [11.36633281, 12.12935162, 8.906909739],
                  [9.690281657, 6.799833301, 5.904578444],
                  [12.20082785, 12.61627174, 8.890182552]]
starting_eval = [422.506, 
                 692.324,
                 379.782,
                 293.003,
                 309.718,
                 825.650,
                 188.710]
seeds = [1, 42, 38, 125]

for seed in seeds:
    np.random.seed(seed)
    print("--------------------------------")
    for i in range(len(starting_point)):
        for budget in n_evals : 
            x0 = starting_point[i]
            f_x0 = starting_eval[i]
            results = MADS(g_hat, x0, budget, f_x0, seed)
            print(f"Pour seed = {seed}, x0 = {x0} et budget = {budget} on a fbest = {results[1]:.3f}")