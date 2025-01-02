import numpy as np
from utils.utils import compute_t, find_optimal_design_distr

class MACO():
    def __init__(self, uid, dim, agent_num, arm_set, suparms_set, total_tounds, u_theta, const_c=1, delta=0.1):
        self.name = "MACO"
        self.uid = uid
        self.agent_num = agent_num
        self.dim = dim
        self.delta = delta
        self.initial_arm_set = arm_set
        self.superarms_set = suparms_set
        self.total_rounds = total_tounds
        self.theta_true = u_theta
        self.const_c = const_c
        self.agents = {}
        self.init_agents()
        self.thre = 100
        self.round = 0
        self.phase = 1
        self.num_rounds_current_phase = 0
        self.theta_hat = np.zeros(self.dim)
        self.total_v_matrix = 1 * np.identity(n=self.dim)
        self.total_w_vector = np.zeros((self.dim, 1))
        self.total_key_terms_times = 0

    def init_agents(self):
        for cid in range(self.agent_num):
            self.agents[cid] = Local_Agent(cid, self.dim, self.uid,
                                           self.initial_arm_set[cid],
                                           self.superarms_set,
                                           self.theta_true,
                                           self.agent_num,
                                           self.total_rounds,
                                           self.const_c,
                                           self.delta)

    def compute_theta_hat(self):
        try:
            self.theta_hat = np.dot(np.linalg.pinv(self.total_v_matrix), self.total_w_vector)
        except:
            self.theta_hat = self.theta_hat

    def broadcast_theta_hat(self):
        for cid in range(self.agent_num):
            self.agents[cid].theta_hat = self.theta_hat

    def compute_rounds_current_phase(self):
        rounds = compute_t(self.dim, 1, self.phase, self.agent_num,
                           len(self.initial_arm_set[0]),
                           self.total_rounds,
                           self.delta)
        rounds = int(rounds)
        print("compute_rounds_current_phase:", rounds)
        return rounds

    def phase_control(self):
        if self.round < self.total_rounds:
            num_rounds = self.compute_rounds_current_phase()
            for cid in range(self.agent_num):
                self.agents[cid].compute_simple_design()
                self.agents[cid].compute_pull_times(num_rounds)
                arm_list = [key for key, value in self.agents[cid].arm_pull_times.items()
                            for cnt in range(int(value))]
                assert len(arm_list) == num_rounds
                self.agents[cid].pull_arm_list_current_phase = arm_list
                v_pi = self.agents[cid].compute_Vpi()
                self.agents[cid].q_matrix += v_pi
                agent_underestimated_vector = self.agents[cid].compute_underestimated_vector()
                self.agents[cid].compute_key_terms(agent_underestimated_vector)

    def round_proceed(self, current_noise, current_sp_noise):
        if self.round < min(len(self.initial_arm_set[0]), self.thre):
            sum_round_rewards = 0
            for cid in range(self.agent_num):
                key_list = list(self.agents[cid].initial_arm_set.keys())
                if len(self.initial_arm_set[0]) < self.thre:
                    arm_id = key_list[self.round]
                else:
                    arm_id = np.random.choice(key_list)
                round_agent_reward = np.dot(self.theta_true.T,
                                            self.agents[cid].initial_arm_set[arm_id].fv) + current_noise
                round_agent_reward = float(round_agent_reward)
                self.agents[cid].w_vector += round_agent_reward * self.agents[cid].initial_arm_set[arm_id].fv
                self.agents[cid].v_matrix += np.outer(self.agents[cid].initial_arm_set[arm_id].fv,
                                                      self.agents[cid].initial_arm_set[arm_id].fv)
                self.total_w_vector += round_agent_reward * self.agents[cid].initial_arm_set[arm_id].fv
                self.total_v_matrix += np.outer(self.agents[cid].initial_arm_set[arm_id].fv,
                                                self.agents[cid].initial_arm_set[arm_id].fv)
                sum_round_rewards += round_agent_reward
                self.agents[cid].reward[self.agents[cid].round] = round_agent_reward
                self.agents[cid].round += 1
            self.compute_theta_hat()
            self.broadcast_theta_hat()
            theta_diff = float(np.linalg.norm(self.theta_hat - self.theta_true))
            self.round += 1
            return sum_round_rewards, theta_diff
        if self.num_rounds_current_phase == 0:
            if self.phase > 1:
                for cid in range(self.agent_num):
                    for spid, num_pulls in self.agents[cid].extra_arms_pulls.items():
                        for _ in range(num_pulls):
                            key_terms = self.superarms_set[spid]
                            round_agent_reward = np.dot(self.theta_true.T, key_terms.fv) + current_sp_noise
                            round_agent_reward = float(round_agent_reward)
                            self.agents[cid].w_vector += round_agent_reward * key_terms.fv
                            self.agents[cid].v_matrix += np.outer(key_terms.fv, key_terms.fv)
                            self.total_w_vector += round_agent_reward * key_terms.fv
                            self.total_v_matrix += np.outer(key_terms.fv, key_terms.fv)
                            self.agents[cid].key_terms_times += 1
                            self.total_key_terms_times += 1
            self.compute_theta_hat()
            self.broadcast_theta_hat()
            for cid in range(self.agent_num):
                self.agents[cid].eliminate_suboptimal_arms()
            self.phase_control()
            self.phase += 1
            for cid in range(self.agent_num):
                self.agents[cid].phase += 1
                assert self.agents[cid].phase == self.phase
        sum_round_rewards = 0
        for cid in range(self.agent_num):
            cnt = len(self.agents[cid].pull_arm_list_current_phase) - self.num_rounds_current_phase
            arm_id = self.agents[cid].pull_arm_list_current_phase[cnt]
            round_agent_reward = np.dot(self.theta_true.T,
                                        self.agents[cid].active_arm_set[arm_id].fv) + current_noise
            round_agent_reward = float(round_agent_reward)
            self.agents[cid].w_vector += round_agent_reward * self.agents[cid].active_arm_set[arm_id].fv
            self.agents[cid].v_matrix += np.outer(self.agents[cid].active_arm_set[arm_id].fv,
                                                  self.agents[cid].active_arm_set[arm_id].fv)
            self.total_w_vector += round_agent_reward * self.agents[cid].active_arm_set[arm_id].fv
            self.total_v_matrix += np.outer(self.agents[cid].active_arm_set[arm_id].fv,
                                            self.agents[cid].active_arm_set[arm_id].fv)
            sum_round_rewards += round_agent_reward
            self.agents[cid].reward[self.agents[cid].round] = round_agent_reward
            self.agents[cid].round += 1
        theta_diff = float(np.linalg.norm(self.theta_hat - self.theta_true))
        self.round += 1
        self.num_rounds_current_phase -= 1
        return sum_round_rewards, theta_diff

class Local_Agent():
    def __init__(self, cid, dim, uid, arm_set, suparms_set, true_theta,
                 agent_num, total_rounds, const_c=1, delta=0.1):
        self.cid = cid
        self.dim = dim
        self.uid = uid
        self.delta = delta
        self.initial_arm_set = arm_set
        self.superarms_set = suparms_set
        self.M = agent_num
        self.K = len(self.initial_arm_set)
        self.total_rounds = total_rounds
        self.theta_true = true_theta
        self.const_c = const_c
        self.round = 0
        self.phase = 1
        self.num_rounds_current_phase = 0
        self.pull_arm_list_current_phase = None
        self.active_arm_set = self.initial_arm_set
        self.pi_distr = None
        self.arm_pull_times = None
        self.extra_arms_pulls = None
        self.theta_hat = np.zeros(self.dim)
        self.v_matrix = np.zeros((self.dim, self.dim))
        self.w_vector = np.zeros((self.dim, 1))
        self.q_matrix = np.zeros((self.dim, self.dim))
        self.reward = np.zeros((self.total_rounds, 1))
        self.key_terms_times = 0

    def compute_simple_design(self):
        pi_distribution = {}
        if len(self.active_arm_set) == 1:
            pi_distribution[next(iter(self.active_arm_set))] = 1
            self.pi_distr = pi_distribution
            return
        arm_matrix = np.zeros((len(self.active_arm_set), self.dim))
        keys = list(self.active_arm_set.keys())
        for i, aid in enumerate(keys):
            arm_matrix[i, :] = self.active_arm_set[aid].fv.T
        distr = find_optimal_design_distr(arm_matrix)
        try:
            assert all(distr >= -1e-7)
        except:
            print(f"round: {self.round}, arms: {len(self.active_arm_set)}, negative probability, randomly select arms")
            distr = np.ones(len(self.active_arm_set)) / len(self.active_arm_set)
        for i, aid in enumerate(keys):
            pi_distribution[aid] = distr[i]
        self.pi_distr = pi_distribution

    def compute_pull_times(self, num_rounds):
        arms_pull_times = None
        pull_times = np.zeros(len(self.active_arm_set))
        arm_ids = [k for k, v in sorted(self.pi_distr.items(), key=lambda item: item[1], reverse=True)]
        for i in range(len(self.active_arm_set)):
            if i != len(self.active_arm_set) - 1:
                pull_times[i] = int(compute_t(self.dim,
                                              self.pi_distr[arm_ids[i]],
                                              self.phase,
                                              self.M,
                                              self.K,
                                              self.total_rounds,
                                              self.delta))
            else:
                pull_times[i] = int(num_rounds - np.sum(pull_times))
        try:
            assert all(pull_times >= 0)
        except:
            if num_rounds < len(self.active_arm_set):
                print("randomly select num_rounds arms, and distribute one round to each of them")
                pull_times = np.zeros(len(self.active_arm_set))
                selected_index = np.random.choice(range(len(arm_ids)), num_rounds, replace=False)
                for i in selected_index:
                    pull_times[i] += 1
            else:
                print("randomly distribute the rounds to arms")
                pull_times = np.ones(len(self.active_arm_set))
                remaining_rounds = num_rounds - np.sum(pull_times)
                for _ in range(remaining_rounds):
                    index = np.random.randint(0, len(arm_ids))
                    pull_times[index] += 1
        arms_pull_times = dict(zip(arm_ids, pull_times))
        self.arm_pull_times = arms_pull_times

    def compute_rounds_current_phase(self):
        rounds = compute_t(self.dim, 1, self.phase,
                           self.M, self.K, self.total_rounds, self.delta)
        return rounds

    def compute_theta_hat(self):
        self.theta_hat = np.dot(np.linalg.inv(self.v_matrix), self.w_vector)

    def compute_Vpi(self):
        V_pi = np.zeros((self.dim, self.dim))
        for k, v in self.active_arm_set.items():
            V_pi += self.pi_distr[k] * np.outer(v.fv, v.fv)
        return V_pi

    def compute_underestimated_vector(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.q_matrix)
        underestimated_vectors = []
        for i in range(min(len(eigenvalues), len(self.active_arm_set))):
            if eigenvalues[i] < 1 / self.dim:
                underestimated_vectors.append([eigenvectors[i], eigenvalues[i]])
        return underestimated_vectors

    def compute_key_terms(self, underestimated_vectors):
        extra_arms_pulls = {}
        for eig_pairs in underestimated_vectors:
            first_key = list(self.superarms_set.keys())[0]
            temp_sp_id = first_key
            for spid, sp in self.superarms_set.items():
                eig_vec = eig_pairs[0].reshape(1, self.dim)
                if np.dot(eig_vec, sp.fv) > np.dot(eig_vec, self.superarms_set[temp_sp_id].fv):
                    temp_sp_id = spid
            super_arm = temp_sp_id
            pull_times = int(0.3 * compute_t((1 - self.dim * np.real(eig_pairs[1])),
                                             1 / (self.const_c**2),
                                             self.phase,
                                             self.M,
                                             self.K,
                                             self.total_rounds,
                                             self.delta))
            if super_arm in extra_arms_pulls.keys():
                extra_arms_pulls[super_arm] += int(pull_times)
            else:
                extra_arms_pulls[super_arm] = int(pull_times)
            coef = (1 / self.dim - np.real(eig_pairs[1])) / (self.const_c**2)
            self.q_matrix += coef * np.outer(self.superarms_set[super_arm].fv,
                                             self.superarms_set[super_arm].fv)
        self.extra_arms_pulls = extra_arms_pulls

    def eliminate_suboptimal_arms(self):
        remaining_arms = {}
        maximum_value = max(np.dot(arm.fv.T, self.theta_hat) for arm in self.active_arm_set.values())
        for aid in self.active_arm_set.keys():
            diff = maximum_value - np.dot(self.active_arm_set[aid].fv.T, self.theta_hat)
            if diff < 2 * (np.sqrt(2)**(-self.phase)) / np.sqrt(self.M):
                remaining_arms[aid] = self.active_arm_set[aid]
        self.active_arm_set = remaining_arms

    def phase_control(self):
        if self.round < self.total_rounds:
            num_rounds = self.compute_rounds_current_phase()
            self.num_rounds_current_phase = num_rounds
            self.compute_simple_design()
            self.compute_pull_times(num_rounds)
            arm_list = [key for key, value in self.arm_pull_times.items() for cnt in range(int(value))]
            assert len(arm_list) == num_rounds
            self.pull_arm_list_current_phase = arm_list
            v_pi = self.compute_Vpi()
            self.q_matrix += v_pi
            underestimated_vector = self.compute_underestimated_vector()
            self.compute_key_terms(underestimated_vector)

    def round_proceed(self):
        if self.num_rounds_current_phase == 0:
            if self.phase > 1:
                for spid, num_pulls in self.extra_arms_pulls.items():
                    for _ in range(num_pulls):
                        key_terms = self.superarms_set[spid]
                        round_reward = np.dot(self.theta_true.T, key_terms.fv)
                        round_reward = float(round_reward)
                        self.w_vector += round_reward * key_terms.fv
                        self.v_matrix += np.outer(key_terms.fv, key_terms.fv)
                        self.key_terms_times += 1
                self.compute_theta_hat()
                self.eliminate_suboptimal_arms()
            self.phase_control()
            self.phase += 1
        cnt = len(self.pull_arm_list_current_phase) - self.num_rounds_current_phase
        arm_id = self.pull_arm_list_current_phase[cnt]
        round_reward = np.dot(self.theta_true.T, self.active_arm_set[arm_id].fv)
        round_reward = float(round_reward)
        self.w_vector += round_reward * self.active_arm_set[arm_id].fv
        self.v_matrix += np.outer(self.active_arm_set[arm_id].fv, self.active_arm_set[arm_id].fv)
        self.reward[self.round] = round_reward
        theta_diff = float(np.linalg.norm(self.theta_hat - self.theta_true))
        self.round += 1
        self.num_rounds_current_phase -= 1
        return round_reward, theta_diff
