from deap import base
from deap import creator
from deap import tools
import random

sizePopulation=100
probabilityMutation = 0.2
probabilityCrossover = 0.8
numberIteration=100

def individual(icls):
    genome = list()
    genome.append(random.uniform(-10, 10))
    genome.append(random.uniform(-10, 10))

    return icls(genome)

def fitnessFunction(individual):
    result = (individual[0] + 2* individual[1] - 7)**2 + (2*individual[0] + individual[1]-5)**2
    return result

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('individual', individual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitnessFunction)

toolbox.register("select", tools.selTournament,tournsize=3)
toolbox.register("mate", tools.cxOnePoint)

def cxOnePoint(ind1, ind2):
 size = min(len(ind1), len(ind2))
 cxpoint = random.randint(1, size - 1)
 ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]
 return ind1, ind2

toolbox.register("mutate", tools.mutGaussian, mu = 5, sigma=10)
