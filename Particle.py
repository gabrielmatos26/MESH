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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS USING 
# THE CREATIVE COMMONS LICENSE: CC BY-NC-ND "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
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


class Particle:

    def __init__(self,
                 pos_min,pos_max,position_dim,
                 vel_min,vel_max,
                 objectives_dim,maximize,
                 secondary_params):
        self.maximize = maximize
        self.crowd_distance = 0
        self.rank = sys.maxsize
        self.pos_min = pos_min
        self.pos_max = pos_max
        self.vel_min = vel_min
        self.vel_max = vel_max
        self.fitness = [None] * objectives_dim
        self.position = [None] * position_dim
        self.velocity = [None] * position_dim
        self.objectives_dim = objectives_dim
        self.domination_counter = 0
        self.dominated_set = []
        self.sigma_value = 0
        self.global_best = None
        self.personal_best = [None]
        if secondary_params:
            self.secondary_params = [None]

    def init_random(self):
        '''
        Inicializa a particula de maneira aleatoria com distribuicao uniforme nos limites de posicao e velocidade.
        '''
        for i in range(len(self.position)):
            self.position[i] = self.pos_min[i] + (self.pos_max[i] - self.pos_min[i])*np.random.uniform(0.0,1.0)
            self.velocity[i] = self.vel_min[i] + (self.vel_max[i] - self.vel_min[i])*np.random.uniform(0.0,1.0)

    def __eq__(self, other):
        return np.array_equal(self.position,other.position)

    def __rshift__(self, other):
        '''
        Operador >>
        Se esta particula domina outra
        '''
        dominates = False
        for i in range(len(self.fitness)):
            if self.maximize[i]:
                if self.fitness[i] > other.fitness[i]:
                    dominates = True
                elif self.fitness[i] < other.fitness[i]:
                    return False
            else:
                if self.fitness[i] > other.fitness[i]:
                    return False
                elif self.fitness[i] < other.fitness[i]:
                    dominates = True
        return dominates

    def __lshift__(self, other):
        '''
        Operador <<
        Se esta particula e dominada por outro
        '''

        return other >> self



