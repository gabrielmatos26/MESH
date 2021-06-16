###########################################################################
# Lucas Braga, MS.c. (email: lucas.braga.deo@gmail.com )
# Gabriel Matos Leite, PhD candidate (email: gmatos@cos.ufrj.br)
# Carolina Marcelino, PhD (email: carolimarc@ic.ufrj.br)
# June 16, 2021
###########################################################################
# Copyright (c) 2021, Lucas Braga, Gabriel Matos Leite, Carolina Marcelino
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in
#      the documentation and/or other materials provided with the
#      distribution
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS USING #THE  CREATIVE COMMONS LICENSE: CC BY-NC-ND "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE #IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import sys
import copy
from scipy.stats import truncnorm
from Particle import *
from tqdm import tqdm


class MC_DEEPSO_Params:
    def __init__(self,
                 objectives_dim,
                 otimizations_type,
                 max_iterations,
                 max_fitness_eval,
                 position_dim,
                 position_max_value,
                 position_min_value,
                 population_size,
                 memory_size,
                 memory_update_type,
                 global_best_attribution_type,
                 DE_mutation_type,
                 Xr_pool_type,
                 communication_probability,
                 mutation_rate,
                 personal_guide_array_size,
                 secondary_params = False,
                 initial_state = False):

        self.objectives_dim = objectives_dim
        self.otimizations_type = otimizations_type

        self.max_iterations = max_iterations
        self.max_fitness_eval = max_fitness_eval

        self.position_dim = position_dim
        self.position_max_value = position_max_value
        self.position_min_value = position_min_value

        self.velocity_min_value = list()
        self.velocity_max_value = list()
        for i in range(position_dim):
            self.velocity_min_value.append(-1*self.position_max_value[i] + self.position_min_value[i])
            self.velocity_max_value.append(-1*self.velocity_min_value[i])

        self.population_size = population_size

        self.memory_size = memory_size
        self.memory_update_type = memory_update_type

        self.global_best_attribution_type = global_best_attribution_type

        self.DE_mutation_type = DE_mutation_type
        self.Xr_pool_type = Xr_pool_type

        self.communication_probability = communication_probability
        self.mutation_rate = mutation_rate

        self.personal_guide_array_size = personal_guide_array_size

        self.secondary_params = secondary_params
        self.initial_state = initial_state

class MC_DEEPSO:
    def __init__(self,params,fitness_function):
        self.params = params
        self.stopping_criteria_reached = False
        self.generation_count = 0

        self.population = []
        self.population_copy = []

        self.memory = []
        self.fronts = []

        self.fitness_function = fitness_function
        self.fitness_eval_count = 0

        w1 = np.random.uniform(0.0, 1.0, [4, self.params.population_size])
        w2 = np.random.uniform(0.0, 0.5, [1, self.params.population_size])
        w3 = np.random.uniform(0.0, 2.0, [1, self.params.population_size])

        self.weights = np.concatenate((w1,w2,w3),axis=0)
        self.weights_copy = []

        self.update_from_differential_mutation = False
        self.log_memory = False
        self.copy_pop = True

    def init_population(self):
        for i in range(self.params.population_size):
            new_particle = Particle(self.params.position_min_value,self.params.position_max_value, self.params.position_dim,
                                    self.params.velocity_min_value, self.params.velocity_max_value,
                                    self.params.objectives_dim,self.params.otimizations_type,
                                    self.params.secondary_params)
            new_particle.init_random()
            self.population.append(new_particle)

    def particle_copy(self,particle):
        copy = Particle(self.params.position_min_value,self.params.position_max_value, self.params.position_dim,
                                    self.params.velocity_min_value, self.params.velocity_max_value,
                                    self.params.objectives_dim,self.params.otimizations_type,
                                    self.params.secondary_params)
        copy.position = particle.position
        copy.fitness = particle.fitness
        copy.velocity = particle.velocity
        if particle.personal_best is not None:
            copy_personal_best_list = []
            for pb in particle.personal_best:
                copy_pb = Particle(self.params.position_min_value,self.params.position_max_value, self.params.position_dim,
                                    self.params.velocity_min_value, self.params.velocity_max_value,
                                    self.params.objectives_dim,self.params.otimizations_type,
                                    self.params.secondary_params)
                copy_pb.position = pb.position
                copy_pb.fitness = pb.fitness
                copy_personal_best_list.append(copy_pb)
            copy.personal_best = copy_personal_best_list
        return copy

    def fitness_evaluation(self, function, *args):
        self.fitness_eval_count = self.fitness_eval_count + 1
        if self.params.initial_state:
            args_and_initial_state = []
            args_and_initial_state.append(args[0])
            args_and_initial_state.extend(self.params.initial_state)
            return function(args_and_initial_state)
        else:
            return function(*args)

    def fast_nondominated_sort(self, first_front_only=False, use_copy_population=False, specific_population=None):
        population = []
        fronts = []
        fronts.append([])

        if specific_population != None:
            for s in specific_population:
                population.append(s)
        else:
            for p in self.population:
                population.append(p)
            if use_copy_population:
                for o in self.population_copy:
                    population.append(o)

        for p in population:
            p.domination_counter = 0
            for q in population:
                if p == q:
                    continue
                if p >> q:
                    p.dominated_set.append(q)
                elif p << q:
                    p.domination_counter = p.domination_counter + 1
            if p.domination_counter == 0:
                fronts[0].append(p)
                p.rank = 0

        i = 0
        if not first_front_only:
            while len(fronts[i]) != 0:
                new_front = []
                for p in fronts[i]:
                    for s in p.dominated_set:
                        s.domination_counter = s.domination_counter - 1
                        if s.domination_counter == 0:
                            new_front.append(s)
                            s.rank = i+1
                i += 1
                fronts.append(list(new_front))
            fronts.pop()
            for p in population:
                p.dominated_set = []
            return fronts
        else:
            for p in population:
                p.dominated_set = []
            return fronts[0]


    def crowding_distance(self, front):
        for j in front:
            j.crowd_distance = 0
        for objective_index in range(self.params.objectives_dim):
            front.sort(key=lambda x: x.fitness[objective_index])
            front[0].crowd_distance = sys.maxsize
            front[-1].crowd_distance = sys.maxsize
            for p in range(1, len(front) - 1):
                if(front[p].crowd_distance == sys.maxsize):
                    continue
                if (front[-1].fitness[objective_index] - front[0].fitness[objective_index]) == 0:
                    continue
                front[p].crowd_distance += (front[p + 1].fitness[objective_index] - front[p - 1].fitness[objective_index]) / (front[-1].fitness[objective_index] - front[0].fitness[objective_index])

    def crowd_distance_selection(self, particle_A, particle_B):
        if particle_A.rank < particle_B.rank:
            return particle_A
        elif particle_B.rank < particle_A.rank:
            return particle_B
        elif particle_A.rank == particle_B.rank:
            if particle_A.crowd_distance > particle_B.crowd_distance:
                return particle_A
            elif particle_B.crowd_distance >= particle_A.crowd_distance:
                return particle_B

    def check_position_limits(self,position_input):
        position = position_input[:]
        for i in range(self.params.position_dim):
            if position[i] < self.params.position_min_value[i]:
                position[i] = self.params.position_min_value[i]
            if position[i] > self.params.position_max_value[i]:
                position[i] = self.params.position_max_value[i]
        return position

    def check_velocity_limits(self,velocity_input,position_input=None):
        velocity = velocity_input[:]
        if position_input is not None:
            position = position_input[:]
            for i in range(self.params.position_dim):
                if position[i] == self.params.position_min_value[i] and velocity[i] < 0:
                    velocity[i] = -1 * velocity[i]
                elif position[i] == self.params.position_max_value[i] and velocity[i] > 0:
                    velocity[i] = -1 * velocity[i]
        else:
            for i in range(self.params.position_dim):
                if velocity[i] < self.params.velocity_min_value[i]:
                    velocity[i] = self.params.velocity_min_value[i]
                if velocity[i] > self.params.velocity_max_value[i]:
                    velocity[i] = self.params.velocity_max_value[i]
        return velocity

    def euclidian_distance(self,a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        return np.linalg.norm(a - b)

    def sigma_eval(self,particle):
        squared_power = np.power(particle.fitness,2)
        denominator = np.sum(squared_power)
        numerator = []
        if self.params.objectives_dim == 2:
            numerator = squared_power[0] - squared_power[1]
        else:
            for i in range(self.params.objectives_dim):
                if i != self.params.objectives_dim-1:
                    numerator.append(squared_power[i] - squared_power[i+1])
                else:
                    numerator.append(squared_power[i] - squared_power[0])
        sigma = np.divide(numerator,denominator)
        particle.sigma_value = sigma

    def sigma_nearest(self,particle,search_pool):
        sigma_distance = sys.maxsize
        nearest_particle = None
        for p in search_pool:
            if particle != p:
                new_distance = self.euclidian_distance(particle.sigma_value, p.sigma_value)
                if sigma_distance > new_distance:
                    sigma_distance = new_distance
                    nearest_particle = p
        if nearest_particle is None:
            nearest_particle = particle
        nearest_particle = copy.deepcopy(nearest_particle)
        particle.global_best = nearest_particle

    def move_particle(self,particle,particle_index,is_copy):
        if is_copy:
            weights = self.weights_copy
        else:
            weights = self.weights

        personal_best_pos = particle.personal_best[np.random.choice(len(particle.personal_best))].position

        inertia_term = np.asarray(particle.velocity) * weights[0][particle_index]

        memory_term = weights[1][particle_index]*(np.asarray(personal_best_pos) - np.asarray(particle.position))

        communication = (np.random.uniform(0.0, 1.0, self.params.position_dim) < self.params.communication_probability) * 1
        cooperation_term = weights[2][particle_index] * (np.asarray(particle.global_best.position) * (1 + (weights[3][particle_index] * np.random.normal(0,1)) ) - np.asarray(particle.position))
        cooperation_term = cooperation_term * communication

        new_velocity = inertia_term + memory_term + cooperation_term
        new_velocity = self.check_velocity_limits(new_velocity)

        new_position = np.asarray(particle.position) + new_velocity
        new_position = self.check_position_limits(new_position)
        new_velocity = self.check_velocity_limits(new_velocity,new_position)

        particle.velocity = new_velocity
        particle.position = new_position

        if self.params.secondary_params:
            fit_eval = self.fitness_evaluation(self.fitness_function,particle.position)
            particle.fitness = fit_eval[0]
            particle.secondary_params = fit_eval[1:]
        else:
            particle.fitness = self.fitness_evaluation(self.fitness_function,particle.position)

    def mutate_weights(self):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                #weight[i] = weight[i] + np.random.normal(0,1)*self.params.mutation_rate
                if i < 4:
                    self.weights[i][j] = truncnorm.rvs(0,1) * self.params.mutation_rate
                    if self.weights[i][j] > 1:
                        self.weights[i][j] = 1
                    elif self.weights[i][j] < 0:
                        self.weights[i][j] = 0
                if i == 4:
                    self.weights[i][j] = truncnorm.rvs(0,0.5) * self.params.mutation_rate
                    if self.weights[i][j] > 0.5:
                        self.weights[i][j] = 0.5
                    elif self.weights[i][j] < 0:
                        self.weights[i][j] = 0
                if i == 5:
                    self.weights[i][j] = truncnorm.rvs(0, 2) * self.params.mutation_rate
                    if self.weights[i][j] > 2:
                        self.weights[i][j] = 2
                    elif self.weights[i][j] < 0:
                        self.weights[i][j] = 0
        if self.copy_pop:
            for i in range(len(self.weights_copy)):
                for j in range(len(self.weights_copy[i])):
                    #weight[i] = weight[i] + np.random.normal(0,1)*self.params.mutation_rate
                    if i < 4:
                        self.weights_copy[i][j] = truncnorm.rvs(0,1) * self.params.mutation_rate
                        if self.weights_copy[i][j] > 1:
                            self.weights_copy[i][j] = 1
                        elif self.weights_copy[i][j] < 0:
                            self.weights_copy[i][j] = 0
                    if i == 4:
                        self.weights_copy[i][j] = truncnorm.rvs(0,0.5) * self.params.mutation_rate
                        if self.weights_copy[i][j] > 0.5:
                            self.weights_copy[i][j] = 0.5
                        elif self.weights_copy[i][j] < 0:
                            self.weights_copy[i][j] = 0
                    if i == 5:
                        self.weights_copy[i][j] = truncnorm.rvs(0, 2) * self.params.mutation_rate
                        if self.weights_copy[i][j] > 2:
                            self.weights_copy[i][j] = 2
                        elif self.weights_copy[i][j] < 0:
                            self.weights_copy[i][j] = 0

    def differential_mutation(self,particle,particle_index):
        Xr_pool = []
        personal_best = particle.personal_best[np.random.choice(len(particle.personal_best))]

        if self.params.Xr_pool_type == 0: # Apenas Populacao
            for p in self.population:
                if not personal_best == p or not particle == p:
                    if not particle >> p:
                        Xr_pool.append(p)
        elif self.params.Xr_pool_type == 1: # Apenas Memoria
            for m in self.memory:
                if not personal_best == m or not particle == m:
                    if not particle >> m:
                        Xr_pool.append(m)
        elif self.params.Xr_pool_type == 2: # Combinacao Memoria e Populacao
            for m in self.memory:
                if not personal_best == m and not particle == m:
                    if not particle >> m:
                         Xr_pool.append(m)
            for p in self.population:
                if not personal_best == p and not particle == p and p not in Xr_pool and p.rank > particle.rank:
                    if not particle >> p:
                        Xr_pool.append(p)

        if self.params.DE_mutation_type == 0 and len(Xr_pool) >= 3: #DE\rand\1\Bin
            Xr_list = np.random.choice(Xr_pool, 3, replace=False)

            Xr1 = np.asarray(Xr_list[0].position)
            Xr2 = np.asarray(Xr_list[1].position)
            Xr3 = np.asarray(Xr_list[2].position)

            Xst = Xr1 + self.weights[5][particle_index] * (Xr2 - Xr3)
            Xst = Xst.tolist()
            Xst = self.check_position_limits(Xst)

            mutation_index = np.random.choice(self.params.position_dim)
            mutation_chance = np.random.uniform(0.0, 1.0,self.params.position_dim)

            for i in range(self.params.position_dim):
                if (mutation_chance[i] < self.weights[4][particle_index] or i == mutation_index):
                    Xst[i] = personal_best.position[i]

        elif self.params.DE_mutation_type == 1 and len(Xr_pool) >= 5: #DE\rand\2\Bin
            Xr_list = np.random.choice(Xr_pool, 5, replace=False)

            Xr1 = np.asarray(Xr_list[0].position)
            Xr2 = np.asarray(Xr_list[1].position)
            Xr3 = np.asarray(Xr_list[2].position)
            Xr4 = np.asarray(Xr_list[3].position)
            Xr5 = np.asarray(Xr_list[4].position)

            Xst = Xr1 + self.weights[5][particle_index] * ((Xr2 - Xr3) + (Xr4 - Xr5))
            Xst = Xst.tolist()
            Xst = self.check_position_limits(Xst)

            mutation_index = np.random.choice(self.params.position_dim)
            mutation_chance = np.random.uniform(0.0, 1.0,self.params.position_dim)

            for i in range(self.params.position_dim):
                if (mutation_chance[i] < self.weights[4][particle_index] or i == mutation_index):
                    Xst[i] = personal_best.position[i]

        elif self.params.DE_mutation_type == 2 and len(Xr_pool) >= 2: #DE/Best/1/Bin
            Xr_list = np.random.choice(Xr_pool, 2, replace=False)

            Xr1 = np.asarray(Xr_list[0].position)
            Xr2 = np.asarray(Xr_list[1].position)

            Xst = particle.global_best.position + self.weights[5][particle_index] * (Xr1 - Xr2)
            Xst = Xst.tolist()
            Xst = self.check_position_limits(Xst)

            mutation_index = np.random.choice(self.params.position_dim)
            mutation_chance = np.random.uniform(0.0, 1.0, self.params.position_dim)

            for i in range(self.params.position_dim):
                if not (mutation_chance[i] < self.weights[4][particle_index] or i == mutation_index):
                    Xst[i] = particle.global_best.position[i]

        elif self.params.DE_mutation_type == 3 and len(Xr_pool) >= 2:  # DE/Current-to-best/1/Bin
            Xr_list = np.random.choice(Xr_pool, 2, replace=False)

            Xr1 = np.asarray(Xr_list[0].position)
            Xr2 = np.asarray(Xr_list[1].position)

            Xst = np.asarray(personal_best.position) + self.weights[5][particle_index] * ((Xr1 - Xr2) + (np.asarray(particle.global_best.position) - np.asarray(personal_best.position)))
            Xst = Xst.tolist()
            Xst = self.check_position_limits(Xst)

            mutation_index = np.random.choice(self.params.position_dim)
            mutation_chance = np.random.uniform(0.0, 1.0, self.params.position_dim)

            for i in range(self.params.position_dim):
                if not (mutation_chance[i] < self.weights[4][particle_index] or i == mutation_index):
                    Xst[i] = particle.global_best.position[i]

        elif self.params.DE_mutation_type == 4 and len(Xr_pool) >= 3:  # DE/Current-to-rand/1/Bin
            Xr_list = np.random.choice(Xr_pool, 3, replace=False)

            Xr1 = np.asarray(Xr_list[0].position)
            Xr2 = np.asarray(Xr_list[1].position)
            Xr3 = np.asarray(Xr_list[2].position)

            Xst = np.asarray(personal_best.position) + self.weights[5][particle_index] * ((Xr1 - Xr2) + (Xr3 - np.asarray(personal_best.position)))
            Xst = Xst.tolist()
            Xst = self.check_position_limits(Xst)

            mutation_index = np.random.choice(self.params.position_dim)
            mutation_chance = np.random.uniform(0.0, 1.0, self.params.position_dim)

            for i in range(self.params.position_dim):
                if not (mutation_chance[i] < self.weights[4][particle_index] or i == mutation_index):
                    Xst[i] = particle.global_best.position[i]

        else:
            return

        if self.params.secondary_params:
            fit_eval = self.fitness_evaluation(self.fitness_function,Xst)
            Xst_fit = fit_eval[0]
        else:
            Xst_fit = self.fitness_evaluation(self.fitness_function,Xst)
        Xst_particle = Particle(self.params.position_min_value,self.params.position_max_value, self.params.position_dim,
                                    self.params.velocity_min_value, self.params.velocity_max_value,
                                    self.params.objectives_dim,self.params.otimizations_type,
                                    self.params.secondary_params)
        Xst_particle.fitness = Xst_fit
        Xst_particle.position = Xst
        if self.params.secondary_params:
            Xst_particle.secondary_params = fit_eval[1:]

        if Xst_particle >> particle:
            particle.fitness = Xst_fit
            particle.position = Xst
            self.update_from_differential_mutation = True
            self.update_personal_best(particle)

    def memory_update(self):
        new_memory_candidates = []
        for f in self.fronts[0]:
            new_memory_candidates.append(f)
        for m in self.memory:
            if m not in new_memory_candidates:
                new_memory_candidates.append(m)
        new_memory_front = self.fast_nondominated_sort(True, False, new_memory_candidates)
        new_memory = []
        if len(new_memory_front) <= self.params.memory_size:
            for f in new_memory_front:
                new_memory.append(f)
        else:
            self.crowding_distance(new_memory_front)
            new_memory_front.sort(key=lambda x: x.crowd_distance)
            i = len(new_memory_front) - 1
            while len(new_memory) < self.params.memory_size:
                new_memory.append(new_memory_front[i])
                i = i - 1
        self.memory = copy.deepcopy(new_memory)

    def update_personal_best(self,particle):
        i = len(particle.personal_best)
        if particle.personal_best[0] is None:
            new_personal_best = Particle(self.params.position_min_value,self.params.position_max_value, self.params.position_dim,
                                self.params.velocity_min_value, self.params.velocity_max_value,
                                self.params.objectives_dim,self.params.otimizations_type,
                                self.params.secondary_params)
            new_personal_best.position = particle.position
            new_personal_best.fitness = particle.fitness
            particle.personal_best = []
            particle.personal_best.append(new_personal_best)
        else:
            remove_list = []
            include_flag = False
            for s in particle.personal_best:
                if particle == s:
                    break
                if particle >> s:
                    include_flag = True
                    if s not in remove_list:
                        remove_list.append(s)
                if not particle << s:
                    i = i - 1
            if len(remove_list) > 0:
                for r in remove_list:
                    particle.personal_best.remove(r)
            if i == 0 or include_flag:
                new_personal_best = Particle(self.params.position_min_value,self.params.position_max_value, self.params.position_dim,
                                self.params.velocity_min_value, self.params.velocity_max_value,
                                self.params.objectives_dim,self.params.otimizations_type,
                                self.params.secondary_params)
                new_personal_best.position = particle.position
                new_personal_best.fitness = particle.fitness

                if self.params.personal_guide_array_size > 0 and len(particle.personal_best) == self.params.personal_guide_array_size:
                    particle.personal_best.pop(0)
                particle.personal_best.append(new_personal_best)

    def global_best_attribution(self, use_copy_population=False):

        if self.params.global_best_attribution_type == 0 or self.params.global_best_attribution_type == 1:
            for m in self.memory:
                self.sigma_eval(m)
            # Sigma com memoria apenas.
            if self.params.global_best_attribution_type == 0:
                for p in self.population:
                    self.sigma_eval(p)
                    self.sigma_nearest(p, self.memory)
                if use_copy_population:
                    for c in self.population_copy:
                        self.sigma_eval(c)
                        self.sigma_nearest(c, self.memory)
            #Sigma por fronteiras.
            if self.params.global_best_attribution_type == 1:
                for p in self.population:
                    self.sigma_eval(p)
                if use_copy_population:
                    for c in self.population_copy:
                        self.sigma_eval(c)
                for p in self.population:
                    if p.rank == 0:
                        self.sigma_nearest(p,self.memory)
                    else:
                        self.sigma_nearest(p,self.fronts[p.rank-1])
                for c in self.population_copy:
                    if c.rank == 0:
                        self.sigma_nearest(c,self.memory)
                    else:
                        self.sigma_nearest(c,self.fronts[c.rank-1])
            #Random na memoria
            if self.params.global_best_attribution_type == 2:
                for p in self.population:
                    p.global_best = self.memory[np.random.choice(len(self.memory))]
                if use_copy_population:
                    for c in self.population_copy:
                        c.global_best = self.memory[np.random.choice(len(self.memory))]
            #Random por fronteiras
            if self.params.global_best_attribution_type == 3:
                for p in self.population:
                    if p.rank == 0:
                        p.global_best = self.memory[np.random.choice(len(self.memory))]
                    else:
                        p.global_best = self.fronts[p.rank-1][np.random.choice(len(self.fronts[p.rank-1]))]
                if use_copy_population:
                    for c in self.population:
                        if c.rank == 0:
                            c.global_best = self.memory[np.random.choice(len(self.memory))]
                        else:
                            c.global_best = self.fronts[c.rank - 1][np.random.choice(len(self.fronts[c.rank - 1]))]

    def check_stopping_criteria(self):
        if self.params.max_fitness_eval != 0 and self.fitness_eval_count >= self.params.max_fitness_eval:
            self.stopping_criteria_reached = True
        if self.params.max_iterations != 0 and self.generation_count == self.params.max_iterations:
            self.stopping_criteria_reached = True

    def run(self):
        with tqdm(total=self.params.max_fitness_eval, leave=False) as pbar:
            ## Inicia populacao
            self.init_population()

            prev_fitness_eval = 0

            ## avalia fitness e sigma da populacao
            for p in self.population:
                if self.params.secondary_params:
                    fit_eval = self.fitness_evaluation(self.fitness_function, p.position)
                    p.fitness = fit_eval[0]
                    p.secondary_params = fit_eval[1:]
                else:
                    p.fitness = self.fitness_evaluation(self.fitness_function,p.position)
                self.update_personal_best(p)

            ## encontra fronteiras das populacao
            self.fronts = self.fast_nondominated_sort()

            ## atualiza memoria
            if len(self.fronts[0]) <= self.params.memory_size:
                for f in self.fronts[0]:
                    self.memory.append(f)
            else:
                self.crowding_distance(self.fronts[0])
                self.fronts[0].sort(key=lambda x: x.crowd_distance)
                j = len(self.fronts[0])-1
                while len(self.memory) < self.params.memory_size:
                    self.memory.append(self.fronts[0][j])
                    j = j - 1

            ## Main loop
            while self.stopping_criteria_reached == False:
                ## encontra os melhores globais de cada particula
                if self.params.DE_mutation_type == 2 or self.params.DE_mutation_type == 3 or self.params.DE_mutation_type == 4:#Somente se for necessario na mutação do DE
                    self.global_best_attribution()
                ## calcular Xst de cada particula.
                for i,p in enumerate(self.population):
                    self.differential_mutation(p,i)

                ## se alguma particula for substituida pelo seu Xst
                if self.update_from_differential_mutation:
                    self.fronts = self.fast_nondominated_sort()
                    self.memory_update()
                    self.update_from_differential_mutation = False

                ## copia pesos e particulas
                if self.copy_pop:
                    self.population_copy = copy.deepcopy(self.population)
                    self.weights_copy = copy.deepcopy(self.weights)

                ## muta os pesos de ambas populacoes
                self.mutate_weights()

                ## Atualizar melhores globais.
                if self.copy_pop:
                    self.global_best_attribution(True)
                else:
                    self.global_best_attribution()

                ## Aplica movimento em todas particulas
                for i,p in enumerate(self.population):
                    self.move_particle(p,i,False)
                    self.update_personal_best(p)
                if self.copy_pop:
                    for i,p in enumerate(self.population_copy):
                        self.move_particle(p,i,True)
                        self.update_personal_best(p)

                ## Separar particulas em fronteiras.
                if self.copy_pop:
                    self.fronts = self.fast_nondominated_sort(False,True)
                else:
                    self.fronts = self.fast_nondominated_sort(False)

                ## Seleciona para a proxima geracao
                if self.copy_pop:
                    next_generation = []
                    i = 0
                    while len(next_generation) < self.params.population_size:
                        if len(self.fronts[i]) + len(next_generation) <= self.params.population_size:
                            for p in self.fronts[i]:
                                next_generation.append(p)
                        else:
                            self.crowding_distance(self.fronts[i])
                            self.fronts[i].sort(key=lambda x: x.crowd_distance)
                            j = len(self.fronts[i])-1
                            while len(next_generation) < self.params.population_size:
                                next_generation.append(self.fronts[i][j])
                                j = j - 1
                        i = i + 1
                    self.population = next_generation

                ## Atualiza Memoria.
                self.memory_update()
                if self.log_memory:
                    file = open(self.log_memory+"fit.txt","a+")
                    memory_fitness = ""
                    for m in self.memory:
                        string = ""
                        for i in range(self.params.objectives_dim):
                            string += str(m.fitness[i]) + " "
                        string = string[:-1]
                        memory_fitness += string + ", "
                    memory_fitness = memory_fitness[:-2]
                    memory_fitness += "\n"
                    file.write(memory_fitness)
                    file.close()

                    file2 = open(self.log_memory + "pos.txt", "a+")
                    memory_position = ""
                    for m in self.memory:
                        string = ""
                        for i in range(self.params.position_dim):
                            string += str(m.position[i])+" "
                        string = string[:-1]
                        memory_position += string + ", "
                    memory_position = memory_position[:-2]
                    memory_position += "\n"
                    file2.write(memory_position)
                    file2.close()
                ## Fim do loop principal.
              
                delta_evals = self.fitness_eval_count - prev_fitness_eval
                pbar.update(delta_evals)
                prev_fitness_eval = self.fitness_eval_count


                self.generation_count = self.generation_count + 1
                self.check_stopping_criteria()
