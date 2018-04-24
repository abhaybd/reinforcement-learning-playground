import numpy as np
import keras.backend as K
import struct

MUTATE_SPREAD = 2.0
MUTATE_SHUFFLE_MAX_SIZE = 4

def model_to_arr(model):
      values = [K.get_value(w) for w in model.weights]
      shapes = [arr.shape for arr in values]
      values = np.hstack((arr.flatten() for arr in values))
      return values, shapes

def model_from_arr(model, shapes, flattened):
      def product(*nums):
          p = 1
          for i in nums:
              p *= i
          return p
      index = 0
      values = []
      for shape in shapes:
            num_values = product(*shape)
            values.append(np.array(flattened[index:index+num_values]).reshape(shape))
            index += num_values
      for w,v in zip(model.weights, values):
            K.set_value(w,v)

def create_offspring(model1, model2):
      arr1 = model_to_arr(model1)
      arr2 = model_to_arr(model2)
      cross_index = np.random.randint(0,len(arr1))
      offspring1 = arr1[:cross_index]+arr2[cross_index:]
      offspring2 = arr2[:cross_index]+arr1[cross_index:]
      return mutate(offspring1), mutate(offspring2)

def float_to_bits(f):
      s = struct.pack('>f', f)
      return struct.unpack('>l', s)[0]

def bits_to_float(b):
      s = struct.pack('>l', b)
      return struct.unpack('>f', s)[0]

def mutate(genome):
      for i, gene in enumerate(genome):
            genome[i] += np.random.normal(loc=gene, scale=MUTATE_SPREAD)
      if np.random.random() < 1/len(genome): # Should shuffle?
            start = np.random.randint(len(genome))
            stop = start + np.random.randint(1,MUTATE_SHUFFLE_MAX_SIZE)
            stop = np.clip(stop, 0, len(genome)-1)
            to_shuffle = genome[start:stop]
            np.random.shuffle(to_shuffle)
            genome[start:stop] = to_shuffle