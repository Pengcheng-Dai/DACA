import numpy as np
import trafficEnv
import math
from scipy import special
import trafficAgent
from tqdm import trange
import matplotlib.pyplot as plt
import time
import os



# give a global state in nd_array
# compute the global state code

# 把全局状态向量用数字进行编码
def global_state_encoder(global_state, state_num):
    agent_num = global_state.shape[0]
    global_state_code = 0
    for i in range(agent_num):
        global_state_code *= state_num
        global_state_code += global_state[i]
    return global_state_code

# give a global_state_code, which is a nonnegative integer
# compute the global state in nd_array

# 将数字解码为全局状态向量
def global_state_decoder(global_state_code, state_num, agent_num):
    global_state = np.zeros(agent_num, dtype=int)
    code = global_state_code
    for i in range(agent_num):
        global_state[agent_num-1-i] = code % state_num
        code = code//state_num
    return global_state


# 分布式算法
class DecentralizedOptimizer:
    def __init__(self, network, agent_list, state_num, action_num, init_states, goal_states, gamma, horizon, epsilon=0.5):
        self.network = network
        self.agent_list = agent_list
        self.agent_num = len(agent_list)
        self.state_num = state_num
        self.action_num = action_num
        self.init_states = init_states
        self.goal_states = goal_states
        self.gamma = gamma
        self.horizon = horizon
        self.epsilon = epsilon  # The cost for waiting one unit of time
        self.game_simulator = trafficEnv.TrafficGame(network=self.network, action_num=self.action_num, init_states=self.init_states, goal_states=self.goal_states, epsilon=self.epsilon)

        # hash tables to store w functions
        self.w_table = {}
        self.zeta_table = {}
        self.averaged_Q_table = {}
        self.stationary_dist_table = np.zeros((self.agent_num, self.state_num, self.horizon, self.state_num))
        # 定义每个智能体的观测邻居
        self.observation_list = self.construct_obervation_table(hop=2)
        self.reward_list = self.construct_obervation_table(hop=1)

    # 智能体的观测信息，根据初始的状态定义邻居的概念
    def construct_obervation_table(self, hop):
        observation_list = []
        for i in range(self.agent_num):
            local_list = []
            for j in range(self.agent_num):
                if abs(self.init_states[i] - self.init_states[j]) <= hop:
                    local_list.append(j)
            observation_list.append(local_list)
        return observation_list # [[], [], [], []...]

    # simulate the trajectory for one epsiode, and update the local Q functions
    # rate_w and rate_zeta are the learning rate of weights and eligibility vectors
    # rate_zeta is TD(0)
    def episode(self, rate_w, rate_zeta=0):
        self.game_simulator.reset()
        # run an episode and record the trajectory
        for t in range(self.horizon):
            global_action = np.zeros(self.agent_num, dtype=int)
            for i in range(self.agent_num):
                global_action[i] = self.agent_list[i].sample_action((-1, self.game_simulator.global_state[i]))
            self.game_simulator.step(global_action)

        # update the local Q functions
        for i in range(self.agent_num):
            for t in range(self.horizon-1):
                # first construct the feature vectors
                phi_1 = np.zeros(len(self.observation_list[i]) * self.state_num + self.action_num)
                phi_2 = np.zeros(len(self.observation_list[i]) * self.state_num + self.action_num)
                for c in range(len(self.observation_list[i])):
                    j = self.observation_list[i][c]
                    local_state = self.game_simulator.global_state_history[t][j]
                    phi_1[c * self.state_num + local_state] = 1.0
                    local_state = self.game_simulator.global_state_history[t+1][j]
                    phi_2[c * self.state_num + local_state] = 1.0
                local_action = self.game_simulator.global_action_history[t][i]
                phi_1[len(self.observation_list[i]) * self.state_num + local_action + 1] = 1.0
                local_action = self.game_simulator.global_action_history[t+1][i]
                phi_2[len(self.observation_list[i]) * self.state_num + local_action + 1] = 1.0

                if t == 0:
                    self.zeta_table[i] = phi_1

                w = self.w_table.get(i, np.zeros(len(self.observation_list[i]) * self.state_num + self.action_num))
                reward_neighbor = 0 # Dai
                for j in self.reward_list[i]: # Dai
                    reward_neighbor += self.game_simulator.global_reward_history[t][j] # Dai
                TD_err = np.dot(phi_1, w) - (reward_neighbor / self.agent_num) - np.dot(phi_2, w)  # Dai
                zeta = self.zeta_table.get(i)
                self.w_table[i] = w - rate_w * TD_err * zeta  # 参数更新
                self.zeta_table[i] = self.gamma * rate_zeta * zeta + phi_2

    # 更新策略参数
    def update_params(self, rate_theta):
        for i in range(self.agent_num):
            local_grad = np.zeros((self.state_num, self.action_num))
            discount_factor = 1.0
            for t in range(self.horizon):
                # first compute the Q function value
                phi = np.zeros(len(self.observation_list[i]) * self.state_num + self.action_num)
                for c in range(len(self.observation_list[i])):
                    j = self.observation_list[i][c]
                    local_state = self.game_simulator.global_state_history[t][j]
                    phi[c * self.state_num + local_state] = 1.0
                local_action = self.game_simulator.global_action_history[t][i]
                phi[len(self.observation_list[i]) * self.state_num + local_action + 1] = 1.0
                w = self.w_table.get(i, np.zeros(len(self.observation_list[i]) * self.state_num + self.action_num))
                # Q-function的估计，这里需要做一些修改
                Q_val = np.dot(phi, w)
                local_state = self.game_simulator.global_state_history[t][i]
                params = self.agent_list[i].invariant_policy[local_state,:]
                prob_vec = special.softmax(params)
                term1 = np.zeros(self.action_num)
                term1[local_action+1] = 1.0
                term1 -= prob_vec
                self.agent_list[i].invariant_policy[local_state,:] += (rate_theta * discount_factor * Q_val * term1)
                discount_factor *= self.gamma

    # 计算V^{\pi_{\theta}}_{i}(s)
    def local_objective(self, index, global_state):
        result = 0
        s = global_state[index]
        params = self.agent_list[index].invariant_policy[s, :]
        prob_vec = special.softmax(params)
        for a in range(self.action_num):
            result += (prob_vec[a] * self.averaged_Q(index, global_state, a-1))
        return result

    # 重置d^{\pi_{\theta}}_{\rho}(s)
    def reset(self, agent_list = None):
        if agent_list is not None:
            self.agent_list = agent_list
        self.averaged_Q_table = {}
        self.stationary_dist_table = np.zeros((self.agent_num, self.state_num, self.horizon, self.state_num))
        for i in range(self.agent_num):
            for s in range(self.state_num):
                init_dist = np.zeros(self.state_num)
                init_dist[s] = 1
                self.stationary_dist_table[i, s, :, :] = self.agent_list[i].stationary_dist(self.network, init_dist,
                                                                                            self.horizon, True)
    # 计算Q^{\pi_{\theta}}_{i}(s,a_{i})
    def averaged_Q(self, index, global_state, local_action):
        global_state_code = global_state_encoder(global_state, self.state_num)
        Q_val = self.averaged_Q_table.get((index, global_state_code, local_action))
        if Q_val is not None:
            return Q_val

        if global_state[index] == self.goal_states[index]:
            return 0

        # if the value is not stored in the table or the precision is not enough, we need to compute it
        # first handle the first fixed local action
        result = - self.epsilon
        discount_factor = 1.0

        next_local_state = self.network.get_neighbor(global_state[index], local_action)
        if next_local_state != global_state[index]: # 智能体i有移动
            # we need to check who else passed through the same edge
            for i in range(self.agent_num): # 除了智能体i之外的智能体有相同移动的概率
                # if i != index and global_state[i] == global_state[index]: # original
                if global_state[i] == global_state[index]:
                    p = self.stationary_dist_table[i, global_state[index], 1, next_local_state]
                    # result -= (discount_factor * p) # original
                    result -= (1 - self.epsilon) * (discount_factor * p) / self.agent_num  # Dai 包含了奖励的定义的概念

        for t in range(1, self.horizon):
            discount_factor *= self.gamma
            for s in range(self.state_num):
                if s == self.goal_states[index]:
                    continue
                # the probability "index" is at s
                # 智能体i从next_local_state通过t-1步到达s的概率
                p_s = self.stationary_dist_table[index, next_local_state, t-1, s]
                # first subtract the cost for time elapse
                result -= (discount_factor * p_s * self.epsilon)
                for next_s in range(self.state_num):
                    # if next_s == s: # original
                    #     continue
                    # the probability "index" goes to next_s from s
                    # 智能体i再从s通过一步到达next_s的概率
                    p_next_s = self.stationary_dist_table[index, s, 1, next_s]
                    for i in range(self.agent_num):
                        p1 = self.stationary_dist_table[i, global_state[i], t, s]
                        p2 = self.stationary_dist_table[i, s, 1, next_s]
                        if i != index:
                            # result -= (discount_factor * p_s * p_next_s * p1 * p2) # original
                            result -= (1 - self.epsilon) * (discount_factor * p_s * p_next_s * p1 * p2) / self.agent_num  # Dai
                        else:
                            # result -= (discount_factor * p_s * p_next_s) # original
                            result -= (1 - self.epsilon) * (discount_factor * p_s * p_next_s) / self.agent_num  # Dai
        self.averaged_Q_table[(index, global_state_code, local_action)] = result
        return result




if __name__ == '__main__':

#   seed_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
# for t_seed in seed_list:
#     # print("t_seed", t_seed)
#     np.random.seed(t_seed)
    np.random.seed(0)
    state_num = 6
    action_num = 3
    agent_num = 6 # 智能体的个数
    horizon = 10
    gamma = 0.9
    # T = 1001
    T = 10000
    # init_states = np.zeros(agent_num, dtype=int) # original
    # goal_states = np.ones(agent_num, dtype=int) # original
    init_states = np.array([0, 1, 2, 0, 1, 2], dtype=int) # Dai
    goal_states = np.ones(agent_num, dtype=int) * 5

    learning_rate = 1e-2
    rate_w = 1e-2
    rate_theta = 1e-2

    # 设置traffic network
    # global_state = np.zeros(agent_num, dtype=int)
    network = trafficEnv.TrafficNetwork(node_num=6)
    network.add_edge((0, 3))
    network.add_edge((1, 3))
    network.add_edge((1, 4))
    network.add_edge((2, 4))
    network.add_edge((3, 5))
    network.add_edge((4, 5))


    agent_list = []
    for i in range(agent_num):
        agent_list.append(trafficAgent.TrafficAgent(state_num, action_num, horizon, random_init=True))

    # print("optimizer.averaged_Q(0, global_state, -1)", optimizer.averaged_Q(0, global_state, -1))
    # print(optimizer.averaged_Q_table)

# ####################################################################################
    # 分布式算法
    Dis_optimizer = DecentralizedOptimizer(network, agent_list, state_num, action_num, init_states, goal_states,
                                                   gamma, horizon)
    # 输出开始代码的时间
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("开始时间：", formatted_time)
    sample_size = agent_num
    Dis_objective_list = []
    Dis_objective_perform = np.zeros(T)
    for j in range(sample_size):
        Dis_objective_list.append([Dis_optimizer.local_objective(j, init_states)])
    for m in trange(T):
        # print("m", m)
        Dis_optimizer.episode(rate_w)
        # update of the parameters
        Dis_optimizer.update_params(rate_theta)
        # Dis_optimizer.reset()  # reset the updated policy parameters
        # for j in range(sample_size):
        #     val = Dis_optimizer.local_objective(j, init_states)
        #     Dis_objective_list[j].append(val)
        #     # objective_list [[...], [...], [...]...]
        #     Dis_objective_perform[m] += val
        # Dis_objective_perform[m] = Dis_objective_perform[m]/agent_num

    # 保存数据
    # np.save("multi_network321/objective_perform_dis10000.npy", Dis_objective_perform)
    # np.save("./multi_network321/objective_perform_disnew_{}.npy".format(t_seed), Dis_objective_perform)
    # 画图
    # plt.figure()
    # plt.plot(Dis_objective_perform)
    # plt.savefig("./multi_network321/multi_bridge_dis.pdf")
    # plt.savefig("./multi_network321/multi_bridge_dis.png")

    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("结束时间：", formatted_time)




