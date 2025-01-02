import argparse
import csv
import datetime
import math
import os
import random
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import pandas as pd
from algorithms.MACO import MACO
from utils.utils import set_random_seed
from datetime import datetime

current_time = datetime.now()

time_string = current_time.strftime("%Y%m%d%H%M%S")

class simulateExp():

    def __init__(self, users, arms, suparms, agent_num, out_folder, pool_size, horizons, dim=50, batch_size=100):
        self.users = users
        self.all_arms = arms
        self.suparms = suparms
        self.agent_num = agent_num
        self.out_folder = out_folder
        self.pool_size = pool_size
        self.horizons = horizons
        self.dim = dim
        self.batch_size = batch_size
        self.arm_noise = 0.1
        self.suparm_noise = 0.1
        self.agent_arm_set = {}  # agent arm set for each agent, dict: {cid: arm_set}, where arm_set is a dict: {aid: arm}

    def set_noise(self, scale=0.1):
        # set noise for arms
        self.arm_noise = scale

    def set_suparm_noise(self, scale=0.1):
        # set noise for superarms
        self.suparm_noise = scale

    def noise(self):
        # generate noise for arms
        return np.random.normal(0, self.arm_noise)

    def superarm_noise(self):
        # generate noise for superarms
        return random.gauss(mu=0, sigma=self.suparm_noise)

    def generate_agent_arm_set(self):
        # generate arm sets for each agent
        self.agent_arm_set = {}
        for cid in range(self.agent_num):
            temp_agent_arm_set = {}
            all_index = range(0, len(self.all_arms))
            selected_pool_index = np.random.choice(all_index, self.pool_size, replace=False)
            # print("selected_pool_index: ", selected_pool_index)
            for aid in selected_pool_index:
                temp_agent_arm_set[aid] = self.all_arms[aid]
            assert len(temp_agent_arm_set) == self.pool_size
            self.agent_arm_set[cid] = temp_agent_arm_set

    def get_agent_optimal_reward(self, uid, cid):
        # calculate optimal reward and arm id for each agent
        max_reward = float('-inf')
        best_arm = None
        best_arm_id = None
        for aid, arm in self.agent_arm_set[cid].items():
            reward = np.dot(self.users[uid].theta.T, arm.fv)
            reward = float(reward)
            if reward > max_reward:
                best_arm = arm
                best_arm_id = aid
                max_reward = reward
        if best_arm is None:
            raise AssertionError
        return max_reward, best_arm_id, best_arm

    def simulationPerUser(self, uid, algorithms, rdseed):
        # set random seed
        set_random_seed(rdseed)
        # run simulation for each user
        process_id = os.getpid()
        print(f'[simulationPerUser] current uid: {uid:d}, random seed: {rdseed}, process_id: {process_id:d}')
        # generate agent arm set and calculate optimal reward and arm id for each agent
        self.generate_agent_arm_set()
        agent_opt_rewards_and_aid = {}
        for cid in range(self.agent_num):
            opt_reward, opt_aid, _ = self.get_agent_optimal_reward(uid, cid)
            agent_opt_rewards_and_aid[cid] = [opt_reward, opt_aid]
        total_opt_reward_per_round = 0
        for cid in range(self.agent_num):
            total_opt_reward_per_round += agent_opt_rewards_and_aid[cid][0]
        print(f'[simulationPerUser] current uid: {uid:d}, total optimal reward per round: {total_opt_reward_per_round:.2f}')
        # init regret_dict and theta_diff_dict
        debug_fw = None
        regret_dict = {}
        reward_dict = {}
        theta_diff_dict = {}
        keyterms_times_dict = {}
        # begin simulation for each round
        for t in range(self.horizons):
            try:
                regret_dict[t] = {key: 0 for key in algorithms.keys()}
                reward_dict[t] = {key: 0 for key in algorithms.keys()}
                theta_diff_dict[t] = {key: 0 for key in algorithms.keys()}
                keyterms_times_dict[t] = {key: 0 for key in algorithms.keys()}
            except:
                print("init regret_dict and theta_diff_dict error")
        if "MACO" in algorithms.keys():
            # * create MACO algorithm
            alg_MACO = MACO(uid, self.dim, self.agent_num, self.agent_arm_set, self.suparms, self.horizons, self.users[uid].theta,
                                    algorithms["MACO"]["const_c"], algorithms["MACO"]["sigma"])
        for t in tqdm(range(self.horizons)):
            # generate noise for current round t
            current_noise = self.noise()
            current_sp_noise = self.superarm_noise()
            # run each algorithm
            if "MACO" in algorithms.keys():
                # * run MACO algorithm and get regret
                reward, theta_diff = alg_MACO.round_proceed(current_noise, current_sp_noise)
                regret = total_opt_reward_per_round + current_noise * self.agent_num - float(reward)
                keyterms_times = alg_MACO.total_key_terms_times
                regret_dict[t]["MACO"] = regret
                reward_dict[t]["MACO"] = reward
                theta_diff_dict[t]["MACO"] = theta_diff
                keyterms_times_dict[t]["MACO"] = keyterms_times
        return regret_dict, theta_diff_dict, keyterms_times_dict,reward_dict

    def run_algorithm(self, algorithms, user_random_seed, dataset_name, userid_set=None, thread_num=1):
        # record start time
        starttime = datetime.now()
        time_string = starttime.strftime("%Y%m%d%H%M%S") 
        # init alg_regret and alg_thetad
        alg_regret = {}
        alg_reward = {}
        alg_thetad = {}
        alg_keyterms = {}
        for algname in algorithms.keys():
            alg_regret[algname] = []
            alg_reward[algname] = []
            alg_thetad[algname] = []
            alg_keyterms[algname] = []
        # run simulation for each user multi-threading
        pool = Pool(processes=thread_num)
        results = []
        if userid_set is None:
            userid_set = list(self.users.keys())
        for i, uid in enumerate(userid_set):
            rdseed = user_random_seed[i]
            rdseed = int(rdseed)
            result = pool.apply_async(self.simulationPerUser, args=(uid, algorithms, rdseed))
            results.append(result)
        pool.close()
        pool.join()
        # get regret and theta_diff for each algorithm
        all_user_regret = []
        all_theta_diff = []
        all_keyterms_times = []
        all_user_reward = []
        # pick up results
        for result in results:
            tmp_regret, tmp_theta_diff, tmp_keyterms_times,tmp_reward = result.get()
            all_user_regret.append(tmp_regret)
            all_theta_diff.append(tmp_theta_diff)
            all_keyterms_times.append(tmp_keyterms_times)
            all_user_reward.append(tmp_reward)
        # process data
        # put all users' regret and theta_diff together for each algorithm
        for t in range(self.horizons):
            for algname in algorithms.keys():
                temp_alg_reg = 0
                temp_alg_theta_diff = 0
                tmp_alg_keyterms_times = 0
                tmp_alg_reward = 0
                for user_regret in all_user_regret:
                    # sum up all users' regret at round t
                    temp_alg_reg += user_regret[t][algname]
                for user_theta_diff in all_theta_diff:
                    # sum up all users' theta_diff at round t
                    temp_alg_theta_diff += user_theta_diff[t][algname]
                for user_keyterms in all_keyterms_times:
                    tmp_alg_keyterms_times += user_keyterms[t][algname]
                for user_reward in all_user_reward:
                    tmp_alg_reward += user_reward[t][algname]
                alg_reward[algname].append(tmp_alg_reward)
                alg_regret[algname].append(temp_alg_reg)
                alg_thetad[algname].append(temp_alg_theta_diff / len(userid_set))
                alg_keyterms[algname].append(tmp_alg_keyterms_times)
        # calculate cumulative regret and theta_diff for each algorithm
        for algname in algorithms.keys():
            alg_regret[algname] = np.cumsum(alg_regret[algname])
            alg_reward[algname] = np.cumsum(alg_reward[algname])
            for t in range(self.horizons):
                alg_reward[algname][t] = alg_reward[algname][t] / (t + 1)
        # # write cumulative regret and theta_diff to file
        output_alias = f"_{dataset_name}_arms_{self.pool_size:d}_agents_{self.agent_num:d}_users_{len(userid_set):d}_T_{self.horizons:d}"
        out_regret_file = os.path.join(self.out_folder, "AccRegret" + output_alias +'_'+ time_string + '.csv')
        out_theta_file = os.path.join(self.out_folder, "AccTheta" + output_alias +'_'+ time_string + '.csv')
        out_keyterms_file = os.path.join(self.out_folder, "KeyTerms" + output_alias +'_'+ time_string + '.csv')
        out_reward_file = os.path.join(self.out_folder, "Rewards" + output_alias +'_'+ time_string + '.csv')
        # check if the output folder exists
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)
        # write cumulative regret and theta_diff to file
        algnames = list(algorithms.keys())
        header = ["round"] + algnames
        with open(out_regret_file, 'w') as fw:
            writer = csv.writer(fw)
            writer.writerow(header)
            for t in range(self.horizons):
                row = [t]
                for algname in algnames:
                    row.append(alg_regret[algname][t])
                writer.writerow(row)
        with open(out_theta_file, 'w') as fw:
            writer = csv.writer(fw)
            writer.writerow(header)
            for t in range(self.horizons):
                row = [t]
                for algname in algnames:
                    row.append(alg_thetad[algname][t])
                writer.writerow(row)
        with open(out_reward_file, 'w') as fw:
            writer = csv.writer(fw)
            writer.writerow(header)
            for t in range(self.horizons):
                row = [t]
                for algname in algnames:
                    row.append(alg_reward[algname][t])
                writer.writerow(row)
        with open(out_keyterms_file, 'w') as fw:
            writer = csv.writer(fw)
            writer.writerow(header)
            for t in range(self.horizons):
                row = [t]
                for algname in algnames:
                    row.append(alg_keyterms[algname][t])
                writer.writerow(row)
        endtime = datetime.now()
        used_time = endtime - starttime
        print(f'[runAlgorithms] Finish writing files. folder: {self.out_folder}, file: AccRegret/AccTheta/KeyTerms{output_alias}')
        starttime = starttime.strftime('%m_%d-%H:%M:%S')
        endtime = endtime.strftime('%m_%d-%H:%M:%S')
        
        print(f'[runAlgorithms] Starttime: {starttime}, Endtime: {endtime}, Used time: {used_time}')
