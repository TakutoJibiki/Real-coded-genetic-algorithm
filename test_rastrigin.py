import matplotlib.pyplot as plt
from deap.benchmarks import rastrigin
import numpy as np
from RealCodedGeneticAlgorithm import RealCodecGA_JGG_AREX


def obj_func(inds):
    return np.array([rastrigin(ind)[0] for ind in inds])


if __name__ == '__main__':
    ga = RealCodecGA_JGG_AREX(
        gene_num=2,
        evaluation_func=obj_func,
        initial_min=-5,
        initial_max=5,
        population=2*20,
        seed=42,
    )

    evals = []
    for i in range(400):
        ga.generation_step()
        evals.append(ga.best_evaluation)
        print(f'gen {i+1} | {round(ga.best_evaluation, 3)} | {round(ga.best_gene[0], 3)}, {round(ga.best_gene[1], 3)}')

    # 収束履歴
    plt.plot(evals)
    plt.xlabel('generation')
    plt.ylabel('evaluation')
    plt.yscale('log')
    plt.show()

    # 最終世代
    div = [-5 + 0.1*i for i in range(100)]
    X, Y = np.meshgrid(div, div)
    Z = np.zeros(X.shape)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i, j] = rastrigin([X[i, j], Y[i, j]])[0]
    plt.cla()
    plt.clf()
    plt.pcolormesh(X, Y, Z)
    plt.scatter(ga.genes[:, 0], ga.genes[:, 1])
    plt.scatter(ga.best_gene[0], ga.best_gene[1], color='red')
    plt.xlabel('$x_{1}$')
    plt.ylabel('$x_{2}$')
    plt.show()
