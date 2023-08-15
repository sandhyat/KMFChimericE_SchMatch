
import sys
import os
from itertools import product
from scipy.stats import qmc
import numpy as np
import math

# for ActFAST
LSF_vars = "LSF_DOCKER_VOLUMES='/storage1/fs1/christopherking/Active/sandhyat/data/ActFast_twoeras/:/input/ /storage1/fs1/christopherking/Active/sandhyat/Output-from-docker-CL/ACTFAST_all_views/:/output/ /home/sandhyat/ChimericEnc_Schema_Matching/CL_HP_tuning/:/codes/'"


poweroftwo_samples = 5

## bounds are inclusive for integers
## third element is 1 if is an integer, 2 binary (so no value is passed)
## fourth element 1 if on log scale
bounds = { 'encKnwMapDepth': [2,6,1,0],
           'encKnwMapWidth': [20,140,1,0],
           'encKnwMapWidthFinal':[15,30,1,0],
            'encUknwD1OrigDepth': [2,6,1,0],
           'encUknwD1OrigWidth': [20,140,1,0],
           'encUknwD2ReDepth': [2, 6, 1, 0],
           'encUknwD2ReWidth': [20, 140, 1, 0],
            'LRPatience': [2, 6, 1, 0],
           'batchSize':[32, 128, 1, 1 ],
           'learningRate': [.0001, .01, 0, 1],
           'learningRateFactor': [.1, .5, 0, 0],
           'epochs':[20, 50, 1, 0],
           'tau':[0.2,0.8,0,0],
           'masking_ratio':[0.1,0.5,0,0],
           'dropout_rate_CL':[0.1,0.8,0,0],
           'weightDirDecoder':[0.1,0.8,0,0],
           'weightCrossDecoder':[0.1,0.8,0,0],
           'weightCombDecoder':[0.1,0.8,0,0]
}

sampler = qmc.Sobol(d =len(bounds.keys()))
sample = sampler.random_base2(m=poweroftwo_samples)


def makeargstring(sample, bounds):
  out = "" ## if I was clever I would do this with a list comprehension
  for samplei, thisvar in zip(sample, bounds.keys() ):
    if(bounds[thisvar][3]==1):
      target = bounds[thisvar][0] * math.exp(samplei* (math.log(bounds[thisvar][1])-math.log(bounds[thisvar][0] + bounds[thisvar][2]) ) )
    else:
      target = bounds[thisvar][0] + samplei* (bounds[thisvar][1]-bounds[thisvar][0] + bounds[thisvar][2]) ## plus one to include outer limit
    if(bounds[thisvar][2]==1):
      target = " --" + thisvar+ "=" + str(math.floor(target))
    elif (bounds[thisvar][2]==2):
      if(target > 0.5):
        target = " --" + thisvar
      else:
        target = ''
    else:
      target = " --" + thisvar + "=" + str(round(target, 4)) ## the rounding is just to make it prettier
    out = out + target
  return out

def makefilestring(sample, bounds):
  out = "" ## if I was clever I would do this with a list comprehension
  for samplei, thisvar in zip(sample, bounds.keys() ):
    if(bounds[thisvar][3]==1):
      target = bounds[thisvar][0] * math.exp(samplei* (math.log(bounds[thisvar][1])-math.log(bounds[thisvar][0] + bounds[thisvar][2]) ) )
    else:
      target = bounds[thisvar][0] + samplei* (bounds[thisvar][1]-bounds[thisvar][0] + bounds[thisvar][2]) ## plus one to include outer limit
    if(bounds[thisvar][2]==1):
      target = str(math.floor(target))
    else:
      target = str(round(target,4))
      out = out + "_" + target
  return out


dataset_number =  'ACTFAST_bow_vs_preopsNlabs' # options  'ACTFAST_bow_vs_outcomes', 'ACTFAST_outcomes_preopsNlabs', 'ActFast_Outcomes_vs_PreopsLabs', 'ACTFAST_bow_vs_preopsNlabs'
if dataset_number == 'ACTFAST_preops_vs_Labs':
    list_of_number_mapped_variables = [20,40, 63]   # preops so possibility of subsets
    file_to_run = 'ActFast_preops_vs_Labs_HP_tuning.py'
elif dataset_number == 'ACTFAST_bow_vs_preopsNlabs':
    list_of_number_mapped_variables = [400]  # bow tokens so using all of them
    file_to_run = 'ActFast_ProcTxt_vs_PreopsNLabs_HP_tuning.py'
elif dataset_number == 'ACTFAST_bow_vs_outcomes':
    list_of_number_mapped_variables = [400]  # bow tokens so using all of them
    file_to_run = 'ActFast_ProcTxt_vs_Outcomes_HP_tuning.py'



bsub_filename = 'bsub_command_CL_Dataset' + str(dataset_number) + '.txt'

output_file_name = '/output/logs/' + str(dataset_number) +'-CL-withDec_HP_'

np.random.seed(100)
initial_seed_list = np.random.randint(10000, size=2)

with open(bsub_filename, 'w') as f:
    for i in range(np.power(2, poweroftwo_samples)):
        for seed in initial_seed_list:
            for num_map in list_of_number_mapped_variables:
                python_command = 'python /codes/' + \
                  file_to_run + \
                  " --nameinfo=testingargs_Dataset"+str(dataset_number)+"_CL --outputcsv=HP-CL-Tuning_dataset"+ str(dataset_number)+ ".csv --dataset_number="+str(dataset_number)+ \
                  makeargstring(sample[i,:], bounds) + " --randomSeed=" + str(seed) + " --mp_features=" + str(num_map) +' > ' \
                 + output_file_name + makefilestring(sample[i,:], bounds) + '_' + str(seed) +'.out'

                comTemp = LSF_vars+ " bsub -G 'compute-christopherking' -g '/sandhyat/largeNjob15hpsearchgroup' -n 8 -q general -R 'gpuhost' -gpu 'num=1:gmodel=TeslaV100_SXM2_32GB:gmem=6G' -R 'rusage[mem=256GB] span[hosts=1]' -a 'docker(docker121720/pytorch-for-ts:0.5)' " + "'" +str( python_command)+ "'"
                f.write(f"{comTemp}\n")
f.close()