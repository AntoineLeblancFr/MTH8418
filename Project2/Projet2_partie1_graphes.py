import matplotlib.pyplot as plt

# Chemin du fichier de données
file_path1 = "data_eval.txt"
file_path2 = "data_cpu.txt"
file_path3 = "BasicNomad.0.txt"
file_path4 = "QuadNomad.0.txt"
file_path5 = "times_nomads.txt"

# Liste pour stocker les données
BO_cpu_x = []
BO_cpu_y = []
BO_eval_x = []
BO_eval_y = []
Basic_cpu_x = []
Basic_cpu_y = []
Basic_eval_x = []
Basic_eval_y = []
QUAD_cpu_x = []
QUAD_cpu_y = []
QUAD_eval_x = []
QUAD_eval_y = []

with open(file_path1, "r") as file:
    for line in file:
        elements = line.split()
        BO_eval_x.append(float(elements[0]))
        BO_eval_y.append(float(elements[1]))

with open(file_path2, "r") as file:
    for line in file:
        elements = line.split()
        BO_cpu_x.append(float(elements[0]))
        BO_cpu_y.append(float(elements[1]))

with open(file_path3, "r") as file:
    j = 0
    for line in file:
        elements = line.split()
        Basic_eval_x.append(j)
        if len(Basic_eval_y) == 0:
            Basic_eval_y.append(float(elements[-1]))
        elif float(elements[-1]) < Basic_eval_y[-1]:
            Basic_eval_y.append(float(elements[-1]))
        else:
            Basic_eval_y.append(Basic_eval_y[-1])
        j += 1

with open(file_path4, "r") as file:
    j = 0
    for line in file:
        elements = line.split()
        QUAD_eval_x.append(j)
        if len(QUAD_eval_y) == 0:
            QUAD_eval_y.append(float(elements[-1]))
        elif len(QUAD_eval_y) == 0 or float(elements[-1]) < QUAD_eval_y[-1]:
            QUAD_eval_y.append(float(elements[-1]))
        else:
            QUAD_eval_y.append(QUAD_eval_y[-1])
        j += 1
print(len(QUAD_eval_y))
with open(file_path5, "r") as file:
    j = 0
    for line in file:
        elements = line.split()
        if j < 125:
            Basic_cpu_x.append(float(elements[0].replace('(', '').replace(')', '').replace(',', '')))
        else:
            QUAD_cpu_x.append(float(elements[0].replace('(', '').replace(')', '').replace(',', '')))
        j += 1

QUAD_cpu_y = QUAD_eval_y
Basic_cpu_y = Basic_eval_y
BO_eval_y[0] *= -1
BO_cpu_y[0] *= -1
plt.plot(BO_eval_x, BO_eval_y, label='BO')
plt.plot(Basic_eval_x, Basic_eval_y, label='Basic Nomad')
plt.plot(QUAD_eval_x, QUAD_eval_y, label='QUAD Nomad')

plt.xlabel("Nombre d'évaluations")
plt.ylabel('Meilleur valeur trouvée')
plt.legend("Graphe de convergence pour p = 1")

plt.show()

plt.plot(BO_cpu_x, BO_cpu_y, label='BO')
plt.plot(Basic_cpu_x, Basic_cpu_y, label='Basic Nomad')
plt.plot(QUAD_cpu_x, QUAD_cpu_y, label='QUAD Nomad')

plt.xlabel("Tempas CPU")
plt.ylabel('Meilleur valeur trouvée')
plt.legend("Graphe de convergence pour p = 1")

plt.show()