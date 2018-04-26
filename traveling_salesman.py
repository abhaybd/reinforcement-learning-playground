import numpy as np
from utils import Organism, tournament_select_single, update_fitness, create_offspring

SIZE = 1000
NUM_ORGANISMS = 500

points = [[0,2*i] for i in range(SIZE)]

def fitness(organism):
      last_point = points[organism.genome[0]]
      fitness = 0
      for point_index in organism.genome[1:]:
            point = points[point_index]
            dist = (point[0]-last_point[0])**2+(point[1]-last_point[1])**2
            fitness += dist
            last_point = point
      return fitness

def create_random_organism():
      genome = [i for i in range(len(points))]
      np.random.shuffle(genome)
      assert len(set(genome)) == len(genome)
      return Organism(genome)

organisms = [create_random_organism() for i in range(NUM_ORGANISMS)]

curr_gen = 0
best_gen = 0
best_dist = -1
print('Starting!')
while True:
      for i in range(len(organisms)):
            parent1 = tournament_select_single(organisms, 10, mode='min')
            parent2 = tournament_select_single(organisms, 10, mode='min')
            child1, child2 = create_offspring(parent1, parent2, crossover=False)
            parent1.replace_with(child1)
            parent2.replace_with(child2)
            assert len(set(parent1.genome)) == len(parent1.genome)
      update_fitness(organisms, fitness)
      curr_gen += 1
      best_in_gen = min(organisms, key=lambda x: x.fitness)
      min_dist = best_in_gen.fitness**0.5
      if min_dist < best_dist or best_dist == -1:
            best_dist = min_dist
            best_gen = curr_gen
            best_organism = best_in_gen
      print('\rBest distance: {:.2f}, Best generation: {}, Current generation: {}'.format(best_dist, best_gen, curr_gen), end='')