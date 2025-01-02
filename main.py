import argparse
import datetime
import os
import random
import sys

import numpy as np

from algorithms import config_parameters as conf

from env import Arm, SupArm, User
from simulateExp import simulateExp

seeds_set = [2756048, 675510, 807110, 2165051, 9492253,114514, 927, 218, 495, 515, 452]

from utils.utils import set_random_seed

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    # in_folder: the folder containing input files, Google_syn,OpenAI_syn,Google_llm,OpenAI_llm
    parser.add_argument('--in_folder', dest='in_folder', default="input_data/Google_syn", help='folder with input files')
    # out_folder: the folder to output
    parser.add_argument('--out_folder', dest='out_folder', default="output_data", help='the folder to output')
    # poolSize: the number of arms for each arm set for each agent
    parser.add_argument('--poolsize', dest='poolsize', type=int, default=40, help='poolSize of each iteration')
    # seedIndex: the index of random seed
    parser.add_argument('--seedindex', dest='seedindex', type=int, default=0, help='seedIndex')
    # agentNum: the number of agents
    parser.add_argument('--agent_num', dest='agent_num', type=int, default=4, help='agentNum')
    # horizons: T
    parser.add_argument('--horizons', dest='horizons', type=int, default=100000, help='horizons')
    # thread_num: the number of threads
    parser.add_argument('--thread_num', dest='thread_num', type=int, default=20, help='thread_num')
    # user_num: the number of users 
    parser.add_argument('--user_num', dest='user_num', type=int, default=1, help='user_num')
    parser.add_argument('--user_id', dest='user_id', type=int, default=510, help='user_id')
    parser.add_argument('--algorithms', dest='algorithms', default="MACO", help='algorithms name')
    args = parser.parse_args()

    #set random seed
    set_random_seed(seeds_set[args.seedindex])
    #load arms
    AM = Arm.ArmManager(args.in_folder)
    AM.loadArms()
    print(f'[main] Finish loading arms: {AM.n_arms}')
    #load Suparms
    SAM = SupArm.SupArmManager(args.in_folder, AM)
    SAM.loadArmSuparmRelation()
    print(f'[main] Finish loading suparms: {SAM.num_suparm}')
    #load User
    UM = User.UserManager(args.in_folder)
    UM.loadUser()
    print(f'[main] Finish loading users: {UM.n_user}')

    # check which algorithms to run
    assert args.algorithms, "Please input the algorithms name"
    algorithms_name = args.algorithms.split(' ')
    algorithms = {}
    if "MACO" in algorithms_name:
        algorithms["MACO"] = conf.cadi_para
    assert len(algorithms.keys()) == len(algorithms_name), "Set up wrong algorithms"
    print(f'[main] Finish setting up algorithms: {algorithms.keys()}')
    # set up user id set
    userid_set = list(range(args.user_num))
    print(f'[main] Finish setting up users: {args.user_num} users')
    print(f'[main] Number of agents: {args.agent_num}, Poolsize: {args.poolsize}, Horizons: {args.horizons}')
    user_random_seed = np.random.randint(0, 1000000, size=len(userid_set))
    # check the dataset name
    try:
        dataset_name = args.in_folder.split('/')[-1]
        print(f'[main] dataset_name: {dataset_name}')
    except:
        dataset_name = args.in_folder
    # set up experiment
    simExperiment = simulateExp(UM.users, AM.arms, SAM.suparms, args.agent_num, args.out_folder, args.poolsize, args.horizons, dim=AM.dim)
    simExperiment.set_noise(conf.armNoiseScale)
    simExperiment.set_suparm_noise(conf.suparmNoiseScale)
    print(f'[main] Finish setting up noise: armNoiseScale: {conf.armNoiseScale}, suparmNoiseScale: {conf.suparmNoiseScale}')
    # run experiment 
    simExperiment.run_algorithm(algorithms, user_random_seed, dataset_name, thread_num=args.thread_num,userid_set=[args.user_id])
