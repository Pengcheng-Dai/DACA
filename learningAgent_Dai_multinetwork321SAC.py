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

# 对比算法--Scalable Actor Critic
class SACOptimizer:
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

        self.local_Q_table = {} # Q_{i}(s_{\mathcal{N}^{\kappa}_{i}},a_{\mathcal{N}^{\kappa}_{i}})
        self.averaged_Q_table = {}
        self.stationary_dist_table = np.zeros((self.agent_num, self.state_num, self.horizon, self.state_num))
        # 定义每个智能体的观测邻居
        self.observation_list = self.construct_obervation_table(hop=1)
        self.action_list = self.construct_obervation_table(hop=1)
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
    # 计算局部Q函数的值
    def local_episode(self, rate_w):
        self.game_simulator.reset()
        # run an episode and record the trajectory
        for t in range(self.horizon):
            global_action = np.zeros(self.agent_num, dtype=int)
            for i in range(self.agent_num): # 智能体选取动作
                global_action[i] = self.agent_list[i].sample_action((-1, self.game_simulator.global_state[i]))
            self.game_simulator.step(global_action)
        # update the local Q functions
        # 用样本数据更新Q函数
        for i in range(self.agent_num):
            # 设置Q函数的初值为0
            # print("len", len(self.observation_list[i]))

            local_Q_value = self.local_Q_table.get(i, np.zeros([self.state_num**len(self.observation_list[i]),
                                                                self.action_num**len(self.observation_list[i])]))
            # print(local_Q_value.shape)
            for t in range(self.horizon - 1):
                local_state_t = [] # s_{\mathcal{N}_{i},t}
                local_action_t = []  # a_{\mathcal{N}_{i},t}
                local_state_t_1 = []  # s_{\mathcal{N}_{i},t+1}
                local_action_t_1 = []  # a_{\mathcal{N}_{i},t+1}
                for c in range(len(self.observation_list[i])):
                    j = self.observation_list[i][c] # 第i个智能体的邻居j
                    # 邻居j的状态
                    local_state_t.append(self.game_simulator.global_state_history[t][j])
                    local_action_t.append(self.game_simulator.global_action_history[t][j])
                    local_state_t_1.append(self.game_simulator.global_state_history[t+1][j])
                    local_action_t_1.append(self.game_simulator.global_action_history[t+1][j])
                # print(local_action_t)
                # print(local_action_t_1)
                # 局部状态s_{\mathcal{N}^{\kappa}_{i}}的编码
                local_state_t_code = global_state_encoder(np.array(local_state_t), self.state_num)
                # print("local_action_t", local_action_t)
                local_action_t_code = global_state_encoder(np.array(local_action_t)+1, self.action_num)
                local_state_t_1_code = global_state_encoder(np.array(local_state_t_1), self.state_num)
                local_action_t_1_code = global_state_encoder(np.array(local_action_t_1)+1, self.action_num)
                # 更新局部Q函数的值
                # print(local_state_t_code, local_action_t_code)
                # print(local_state_t_1_code, local_action_t_1_code)
                local_Q_value[local_state_t_code, local_action_t_code] = (1-rate_w) * local_Q_value[local_state_t_code, local_action_t_code] + rate_w * (self.game_simulator.global_reward_history[t][i]+self.gamma * local_Q_value[local_state_t_1_code, local_action_t_1_code])
            self.local_Q_table[i] = local_Q_value

    # 更新策略参数
    def update_params(self, rate_theta):
        for i in range(self.agent_num):
            local_grad = np.zeros((self.state_num, self.action_num))
            discount_factor = 1.0
            for t in range(self.horizon):
                # first compute the Q function value
                # 根据样本的history取样
                local_state_t = []
                local_action_t = []
                for c in range(len(self.observation_list[i])):
                    j = self.observation_list[i][c] # 第i个智能体的邻居j
                    # 邻居j的状态
                    local_state_t.append(self.game_simulator.global_state_history[t][j])
                    local_action_t.append(self.game_simulator.global_action_history[t][j])
                # 局部状态s_{\mathcal{N}^{\kappa}_{i}}的编码
                local_state_t_code = global_state_encoder(np.array(local_state_t), self.state_num)
                local_action_t_code = global_state_encoder(np.array(local_action_t)+1, self.action_num)
                # 取对应的Q值
                localQi = self.local_Q_table.get(i, np.zeros([self.state_num**len(self.observation_list[i]),
                                                              self.action_num**len(self.observation_list[i])]))
                localQivalue = localQi[local_state_t_code, local_action_t_code]
                # 邻居智能体的局部Q值
                tot_localQvalue = [localQivalue]
                for j in self.observation_list[i]:
                    if j is not i: # 邻居中不包含i
                        local_state_t = []
                        local_action_t = []
                        for c in range(len(self.observation_list[j])):
                            k = self.observation_list[j][c]  # 第j个智能体的邻居k
                            # 邻居j的状态
                            local_state_t.append(self.game_simulator.global_state_history[t][k])
                            local_action_t.append(self.game_simulator.global_action_history[t][k])
                        # 局部状态s_{\mathcal{N}^{\kappa}_{i}}的编码
                        local_state_t_code = global_state_encoder(np.array(local_state_t), self.state_num)
                        local_action_t_code = global_state_encoder(np.array(local_action_t)+1, self.action_num)
                        # 取对应的Q值
                        localQj = self.local_Q_table.get(j, np.zeros([self.state_num**len(self.observation_list[j]),
                                                                      self.action_num**len(self.observation_list[j])]))
                        localQjvalue = localQj[local_state_t_code, local_action_t_code]
                        tot_localQvalue.append(localQjvalue)
                # 开始计算梯度，更新策略参数
                local_state = self.game_simulator.global_state_history[t][i]
                local_action = self.game_simulator.global_action_history[t][i]
                params = self.agent_list[i].invariant_policy[local_state, :]
                prob_vec = special.softmax(params)
                term1 = np.zeros(self.action_num)
                term1[local_action+1] = 1.0
                term1 -= prob_vec
                self.agent_list[i].invariant_policy[local_state,:] += (rate_theta * discount_factor * (np.sum(tot_localQvalue)/self.agent_num) * term1)
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

  seed_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
for t_seed in seed_list:
    np.random.seed(t_seed)
    # np.random.seed(0)
    state_num = 6
    action_num = 3
    agent_num = 6  # 智能体的个数
    # horizon = 10
    horizon = 10 # SAC中的样本取样都是10
    gamma = 0.9
    # T = 1001
    T = 10000
    # init_states = np.zeros(agent_num, dtype=int) # original
    # goal_states = np.ones(agent_num, dtype=int) # original
    init_states = np.array([0, 1, 2, 0, 1, 2], dtype=int)  # Dai
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

    # 智能体的list
    agent_list = []
    for i in range(agent_num):
        agent_list.append(trafficAgent.TrafficAgent(state_num, action_num, horizon, random_init=True))

####################################################################
####对比算法SAC
    SAC_optimizer = SACOptimizer(network, agent_list, state_num, action_num, init_states, goal_states,
                                                   gamma, horizon)
    # 输出开始代码的时间
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("开始时间：", formatted_time)
    sample_size = agent_num
    SAC_objective_list = []
    SAC_objective_perform = np.zeros(T)
    for j in range(sample_size):
        SAC_objective_list.append([SAC_optimizer.local_objective(j, init_states)])
    for m in trange(T):
        # print("m", m)
        SAC_optimizer.local_episode(rate_w)
        # update of the parameters
        SAC_optimizer.update_params(rate_theta)
        SAC_optimizer.reset()  # reset the updated policy parameters
        for j in range(sample_size):
            val = SAC_optimizer.local_objective(j, init_states)
            SAC_objective_list[j].append(val)
            # objective_list [[...], [...], [...]...]
            SAC_objective_perform[m] += val
        SAC_objective_perform[m] = SAC_objective_perform[m] / agent_num

    np.save("./multi_network321/objective_perform_SAC_{}_{}.npy".format(t_seed,horizon), SAC_objective_perform)
    # 画图
    # plt.figure()
    # plt.plot(SAC_objective_perform)
    # plt.savefig("./multi_network321/one_bridge_SAC{}_{}.pdf".format(5000, horizon))
    # plt.savefig("./multi_network321/one_bridge_SAC{}_{}.png".format(5000, horizon))

    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("结束时间：", formatted_time)


