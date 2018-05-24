# -*- coding: utf-8 -*-

from numpy.random import *
import numpy as np
import random, math, sys


class GeneticAlgorithm():
	MUTATION_RATE        = 0.1
	GENOM_MUTATION_RATE  = 0.1
	SELECTION_TYPE       = 'ranking_select'  # roulette_select / ranking_select / tournament_select / elitist_select
	CROSSOVER_TYPE       = 'order_based_crossover' # two_point_crossover / order_based_crossover

	def __init__(self, population_size, genom_length, generations):
		self.genom_length = genom_length
		self.population_size = population_size
		self.current_group = randint(2, size=(population_size,genom_length))
		self.current_generation = 1
		self.generations = generations


	def evaluate(self, genom) -> float:
		return sum(genom)/self.genom_length


	def mutate(self, mutation_rate=0.1, genom_mutation_rate=0.1):
		for i in range(len(self.current_group)):
			if self.p_select(mutation_rate):
				for j in range(len(self.current_group[i])):
					if self.p_select(genom_mutation_rate):
						self.current_group[i][j] = ~self.current_group[i][j]+2


	def pickup_two_point(self):
		one = random.randint(0,self.genom_length)
		another = random.randint(0,self.genom_length-1)
		another += 1 if one <= another else 0

		return sorted([one,another])


	def pickup_crossover_pair(self,sample):
		pairs = random.sample(list(range(1,self.population_size*(self.population_size-1))),sample)
		pairs = np.array([ [ math.ceil( (math.ceil(p/self.population_size)+p) / self.population_size )-1, (math.ceil(p/self.population_size)+p) - math.floor( (math.ceil(p/self.population_size)+p) / self.population_size ) * self.population_size - 1 ] for p in pairs ])

		return pairs


	def two_point_crossover(self, rate=0.7):
		next_candidates = []
		pairs = self.pickup_crossover_pair(math.floor(self.population_size*rate))

		for x in pairs:
			points = self.pickup_two_point()

			for p in range(points[0],points[1]):
				tmp = self.current_group[x[0]][p]
				self.current_group[x[0]][p] = self.current_group[x[1]][p]
				self.current_group[x[1]][p] = tmp

			next_candidates.extend([self.current_group[x[0]],self.current_group[x[1]]])

		return np.array(next_candidates)


	def order_based_crossover(self, rate=0.7):
		next_candidates = []
		pairs = self.pickup_crossover_pair(math.floor(self.population_size*rate))

		for x in pairs:

			for p in range(self.genom_length):
				if self.p_select(0.5):
					tmp = self.current_group[x[0]][p]
					self.current_group[x[0]][p] = self.current_group[x[1]][p]
					self.current_group[x[1]][p] = tmp

			next_candidates.extend([self.current_group[x[0]],self.current_group[x[1]]])

		return np.array(next_candidates)


	def p_select(self,eps):
		if eps > random.randint(0,1):
			return True
		else:
			return False


	def roulette_select(self):
		evaluations = np.array([ self.evaluate(x) for x in self.current_group ])
		total_evaluation = sum(evaluations)
		rel_evaluations = np.array([ x/total_evaluation for x in evaluations ])
		probavilities = np.array([ sum(rel_evaluations[:x+1]) for x in range(len(rel_evaluations)) ])
		next_group = []
		for n in range(self.population_size):
			r = rand()
			for i,p in enumerate(probavilities):
				if r <= p:
					next_group.append(self.current_group[i])
					break

		self.current_group = np.array(next_group)


	def ranking_select(self):
		evaluations = np.array([ self.evaluate(x) for x in self.current_group ])
		ratios = np.array([ x+1 for x in range(len(self.current_group)) ])
		total = sum(ratios)
		rel = np.array([ x/total for x in ratios ])
		probavilities = np.array([ sum(rel[:x+1]) for x in range(len(rel)) ])

		order = evaluations.argsort()[::-1]
		probavilities = probavilities[order]
		sorted_group = self.current_group[order]

		next_group = []
		for n in range(self.population_size):
			r = rand()
			for i,p in enumerate(probavilities):
				if r <= p:
					next_group.append(sorted_group[i])
					break

		self.current_group = np.array(next_group)


	def tournament_select(self, rate=0.7):
		m = math.ceil(len(self.current_group)*rate)
		next_group = []
		for n in range(self.population_size):
			idx = random.sample(list(range(0,len(self.current_group)-1)),m)
			candidates = np.array([ self.current_group[x] for x in idx ])
			evaluations = np.array([ self.evaluate(x) for x in candidates ])

			order = evaluations.argsort()
			candidates = candidates[order]

			next_group.append(candidates[-1])

		self.current_group = np.array(next_group)


	def elitist_select(self):
		evaluations = np.array([ self.evaluate(x) for x in self.current_group ])
		order = evaluations.argsort()
		sorted_group = self.current_group[order]
		next_group = sorted_group[-self.population_size:]

		self.current_group = next_group


	def select(self):
		return eval('self.'+self.SELECTION_TYPE)()


	def crossover(self):
		return eval('self.'+self.CROSSOVER_TYPE)()


	def forward(self):
		self.current_group = np.r_[self.current_group,self.crossover()]
		self.mutate()
		self.select()

		print('[{0:04}] average : {1:.2f}'.format(self.current_generation, np.average(self.current_group)))

		self.current_generation+=1


	def run(self):
		while self.generations >= self.current_generation:
			self.forward()

		return self.current_group


# sample
if __name__ == '__main__':
	# 個体数
	N = 10

	# ゲノム長
	GL = 10

	# 最大世代数
	GN = 100

	ga = GeneticAlgorithm(N,GL,GN)
	ga.run()
