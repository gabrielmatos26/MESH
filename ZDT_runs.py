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
import datetime
from ZDT import *
from MESH import *
import pygmo as pg
import pickle
from tqdm import tqdm
from pathlib import Path


def main():
    Path("result").mkdir(parents=False, exist_ok=True)

    num_runs = 30
    zdt = 4
    zdt_func = get_function(zdt)

    objectives_dim = 2
    otimizations_type = [False,False]
    max_iterations = 0
    max_fitness_eval = 15000
    position_dim = 5
    position_max_value = [1] * position_dim
    position_min_value = [0] * position_dim
    population_size = 50
    num_final_solutions = 50
    memory_size = 50
    memory_update_type = 0

    communication_probability = 0.7 #0.5
    mutation_rate = 0.9
    personal_guide_array_size = 3

    global_best_attribution_type = 1 #G
    Xr_pool_type = 1                 #V
    DE_mutation_type = 0             #M
    config = f"G{global_best_attribution_type}V{Xr_pool_type}M{DE_mutation_type}"
    
    print(f"Running E{global_best_attribution_type+1}V{Xr_pool_type+1}D{DE_mutation_type+1} on ZDT{zdt}")

    result = {}
    combined = None
    for i in tqdm(range(num_runs)):

        params = MESH_Params(objectives_dim,otimizations_type,max_iterations,max_fitness_eval,position_dim,position_max_value,position_min_value,population_size,memory_size,memory_update_type,global_best_attribution_type,DE_mutation_type,Xr_pool_type,communication_probability,mutation_rate,personal_guide_array_size)

        MCDEEPSO = MESH(params,zdt_func)
        MCDEEPSO.log_memory = f"result/{config}_{i}-ZDT{zdt}-"
        MCDEEPSO.run()
        
        F = open(MCDEEPSO.log_memory+"fit.txt", 'r').read().split("\n")[-2]
        F = np.array([v.split() for v in F.split(",")], dtype=np.float64)

        P = open(MCDEEPSO.log_memory+"pos.txt", 'r').read().split("\n")[-2]
        P = np.array([v.split() for v in P.split(",")], dtype=np.float64)

        result[i+1] = {"F":F, "P":P}
        if combined is None:
            combined = F
        else:
            combined = np.vstack((combined, F))

    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=combined)
    n = num_final_solutions
    if len(ndf[0]) < num_final_solutions:
        n = len(ndf[0])
    best_idx = pg.sort_population_mo(points = combined)[:n]
    result['combined'] = (best_idx, combined[best_idx])

    with open(f'result/{config}_zdt{zdt}.pkl', 'wb') as f:
        pickle.dump(result, f)

if __name__ == '__main__':
    main()