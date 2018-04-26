import numpy as np
import keras.backend as K
import struct

def model_to_organism(model):
      values = [K.get_value(w) for w in model.weights]
      shapes = [arr.shape for arr in values]
      values = np.hstack(arr.flatten() for arr in values)
      organism = Organism(values, shapes)
      return organism

def organism_to_model(model, organism):
      shapes = organism.shapes
      genome = organism.genome
      def product(nums):
          p = 1
          for i in nums:
              p *= i
          return p
      index = 0
      values = []
      for shape in shapes:
            num_values = product(shape)
            values.append(np.array(genome[index:index+num_values]).reshape(shape))
            index += num_values
      for w,v in zip(model.weights, values):
            K.set_value(w,v)
      return model

def create_offspring(organism1, organism2, crossover=True, mutate=True, scale=-1):
      if crossover:
            genome1 = organism1.genome
            genome2 = organism2.genome
            cross_index = np.random.randint(0,len(genome1))
            offspring1 = Organism(genome1[:cross_index]+genome2[cross_index:], organism1.shapes)
            offspring2 = Organism(genome2[:cross_index]+genome1[cross_index:], organism2.shapes)
      else:
            offspring1 = organism1
            offspring2 = organism2
      if mutate:
            offspring1.mutate(scale=scale)
            offspring2.mutate(scale=scale)
      return offspring1, offspring2

def update_fitness(organisms, fitness_func):
      for organism in organisms:
            organism.fitness = fitness_func(organism)

def tournament_selection(organisms, gen_size, k, mode='max'):
      selected = set()
      while len(selected) < gen_size:
            selected.add(tournament_select_single(organisms, k, mode, selected))
      return list(selected)

def tournament_select_single(organisms, k, mode='max', selected=None):
      if selected == None:
            selected = set()
      indexes = [i for i in range(len(organisms))]
      np.random.shuffle(indexes)
      indexes = indexes[:k]
      competing_organisms = [organisms[i] for i in list(indexes)]
      for organism in selected:
            if organism in competing_organisms:
                  competing_organisms.remove(organism)
      if mode == 'max':
            return max(competing_organisms, key=lambda x: x.fitness)
      elif mode == 'min':
            return min(competing_organisms, key=lambda x: x.fitness)
      else:
            raise ValueError('mode must be \'min\' or \'max\'!')

def float_to_bits(f):
      s = struct.pack('>f', f)
      return struct.unpack('>l', s)[0]

def bits_to_float(b):
      s = struct.pack('>l', b)
      return struct.unpack('>f', s)[0]      

class Organism(object):
      def from_model(model):
            return model_to_organism(model)
      
      def __init__(self, genome, shapes=None):
            self.genome = genome
            self.shapes = shapes
            self.fitness = 0
      
      def mutate(self, scale=2.0, shuffle_max_size=20):
            if scale != -1:
                  for i, gene in enumerate(self.genome):
                        self.genome[i] += np.random.normal(loc=gene, scale=scale)
            if np.random.random() < 1/len(self.genome): # Should shuffle?
                  start = np.random.randint(len(self.genome))
                  stop = start + shuffle_max_size
                  stop = np.clip(stop, 0, len(self.genome))
                  to_shuffle = self.genome[start:stop]
                  np.random.shuffle(to_shuffle)
                  self.genome[start:stop] = to_shuffle
      
      def replace_with(self, other):
            self.genome = list(other.genome)
            self.shapes = list(other.shapes) if other.shapes != None else None
            self.fitness = other.fitness
      
      def to_model(self, model):
            return organism_to_model(model, self)
      
      def create_offspring(self, organism):
            return create_offspring(self, organism)
      
      def __str__(self):
            return self.__repr__()
      
      def __repr__(self):
            return 'Organism(genome={}, fitness={})'.format(self.genome, self.fitness)