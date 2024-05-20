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

# 集中式算法
class CentralizedOptimizer:
    def __init__(self, network, agent_list, state_num, action_num, init_states, goal_states, gamma, horizon, epsilon=0.5):
        self.network = network
        self.agent_list = agent_list # [agent, agent,...]
        self.agent_num = len(agent_list)
        self.state_num = state_num
        self.action_num = action_num
        self.init_states = init_states
        self.goal_states = goal_states
        self.gamma = gamma
        self.horizon = horizon
        self.epsilon = epsilon  # The cost for waiting one unit of time

        # hash tables to store local Q functions
        # it has the form (value, precision_level)
        self.averaged_Q_table = {}

        # we build a stationary distribution table to accelerate later computation
        self.stationary_dist_table = np.zeros((self.agent_num, self.state_num, self.horizon, self.state_num))

        for i in range(self.agent_num):
            for s in range(self.state_num):
                init_dist = np.zeros(self.state_num)
                init_dist[s] = 1
                self.stationary_dist_table[i, s, :, :] = self.agent_list[i].stationary_dist(self.network, init_dist, self.horizon, True)

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

    # here, precision_level is a non-negative integer.
    # We roll out the Q at least "precision_level" times to compute it.
    # Q^{\pi_{\theta}}_{i}(s,a_{i})
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

    # this function evaluates the exact local objective value under the current policy
    # V^{\pi_{\theta}}_{i}(s)
    def local_objective(self, index, global_state):
        result = 0
        s = global_state[index]
        params = self.agent_list[index].invariant_policy[s, :]
        prob_vec = special.softmax(params)
        for a in range(self.action_num):
            result += (prob_vec[a] * self.averaged_Q(index, global_state, a-1))
        return result

    # this function computes the exact local policy gradient under the current policy
    # exact policy gradient
    def local_gradient(self):
        gradient = np.zeros((self.agent_num, self.state_num, self.action_num))
        # we enumerate over all possible global states
        code_max = self.state_num ** self.agent_num
        for code in range(code_max):
            global_state = global_state_decoder(code, self.state_num, self.agent_num)
            p1 = 0
            discount_factor = 1.0
            for t in range(self.horizon):
                tmp = 1.0
                for i in range(self.agent_num):
                    v = self.stationary_dist_table[i, self.init_states[i], t, global_state[i]]
                    if v < 1e-7:
                        tmp = 0
                        break
                    tmp *= v
                p1 += (discount_factor * tmp)
                discount_factor *= self.gamma
            for i in range(self.agent_num):
                for a in range(self.action_num):
                    params = self.agent_list[i].invariant_policy[global_state[i], :]
                    prob_vec = special.softmax(params)
                    term1 = np.zeros(self.action_num)
                    term1[a] = 1.0
                    term1 -= prob_vec
                    term2 = self.averaged_Q(i, global_state, a-1)
                    for j in range(self.agent_num):
                        if j is not i:
                            term2 += self.local_objective(j, global_state)
                    # gradient[i, global_state[i], a] += (p1 * prob_vec[a] * term1[a] * term2) # Dai
                    gradient[i, global_state[i], :] += (p1 * prob_vec[a] * term1 * term2) # original
        return gradient

    # estimated policy gradient
    def mc_local_gradient(self, sample_num=100, agent_index=None):
        gradient = np.zeros((self.agent_num, self.state_num, self.action_num))
        if agent_index is not None:
            gradient = np.zeros((self.state_num, self.action_num))

        for s in range(sample_num):
            for t in range(self.horizon):
                flag = np.random.binomial(1, 1 - self.gamma)
                if flag == 1 or t == self.horizon-1:
                    # first sample a global state
                    global_state = np.zeros(self.agent_num, dtype = int)
                    for i in range(self.agent_num):
                        global_state[i] = np.random.choice(a=self.state_num, p=self.stationary_dist_table[i, self.init_states[i], t, :])
                        #print(global_state[i])

                    if agent_index is None: # 所有的智能体都计算梯度
                        # now we compute the gradient term at this global state
                        for i in range(self.agent_num):
                            for a in range(self.action_num):
                                params = self.agent_list[i].invariant_policy[global_state[i], :]
                                prob_vec = special.softmax(params)
                                term1 = np.zeros(self.action_num)
                                term1[a] = 1.0
                                term1 -= prob_vec
                                term2 = self.averaged_Q(i, global_state, a - 1)
                                for j in range(self.agent_num):
                                    if j is not i:
                                        term2 += self.local_objective(j, global_state)
                                # gradient[i, global_state[i], a] += (prob_vec[a] * term1[a] * term2) / (1 - self.gamma) # Dai
                                gradient[i, global_state[i], :] += (prob_vec[a] * term1 * term2)/(1 - self.gamma) # original
                    else:
                        for a in range(self.action_num):
                            params = self.agent_list[agent_index].invariant_policy[global_state[agent_index], :]
                            prob_vec = special.softmax(params)
                            term1 = np.zeros(self.action_num)
                            term1[a] = 1.0
                            term1 -= prob_vec
                            term2 = self.averaged_Q(agent_index, global_state, a - 1)
                            for j in range(self.agent_num):
                                if j is not agent_index:
                                    term2 += self.local_objective(j, global_state)
                            # gradient[global_state[agent_index], a] += (prob_vec[a] * term1[a] * term2) / (1 - self.gamma) # Dai
                            gradient[global_state[agent_index], :] += (prob_vec[a] * term1 * term2) / (1 - self.gamma) # original

                    break

        return gradient/sample_num


    # We implement the local regret by local policy gradient.
    # The gradient is provided by mc_local_gradient with sample num
    def local_regret(self, sample_num=100, lr=1e-3, acc=0.1, max_round=1000):
        self.reset()
        regret_list = []
        for i in range(self.agent_num):
            local_objective_current = self.local_objective(i, self.init_states)
            back_up_policy = self.agent_list[i].invariant_policy    # first store the original policy
            # update of policy parameters
            for _ in range(max_round):
                self.reset()
                local_grad = self.mc_local_gradient(sample_num, i)
                self.agent_list[i].invariant_policy += lr * local_grad
                if np.linalg.norm(local_grad) < acc:
                    break
            self.reset()
            local_objective_optimal = self.local_objective(i, self.init_states)
            gap = local_objective_optimal - local_objective_current
            if gap > 0:
                regret_list.append(gap)
            else:
                regret_list.append(0.0)
            self.agent_list[i].invariant_policy = back_up_policy
        return np.array(regret_list) # (agent_num, )




if __name__ == '__main__':

#   seed_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# for t_seed in seed_list:
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
    optimizer = CentralizedOptimizer(network, agent_list, state_num, action_num, init_states, goal_states, gamma, horizon)

    # print("optimizer.averaged_Q(0, global_state, -1)", optimizer.averaged_Q(0, global_state, -1))
    # print(optimizer.averaged_Q_table)
#####################################################################################
    # centralized algorithm
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("开始时间：", formatted_time)
    sample_size = agent_num
    objective_list = []
    objective_perform = np.zeros(T)
    regret_history = []
    for j in range(sample_size):
        objective_list.append([optimizer.local_objective(j, init_states)])
    for m in trange(T):
        # print("m", m)
        gradients = optimizer.mc_local_gradient(sample_num=300)
        # update of the parameters
        for i in range(agent_num):
            agent_list[i].invariant_policy += learning_rate * gradients[i, :, :]
        optimizer.reset() # reset the updated policy parameters
        for j in range(sample_size):
            val = optimizer.local_objective(j, init_states)
            objective_list[j].append(val)
            # objective_list [[...], [...], [...]...]
            objective_perform[m] += val
        objective_perform[m] = objective_perform[m]/agent_num

    # np.save("./multi_network321/objective_perform_cen500.npy", objective_perform)
    # np.save("./multi_network321/objective_perform_cen_{}.npy".format(t_seed), objective_perform)

    # 画图
    # plt.figure()
    # plt.plot(objective_perform)
    # plt.savefig("./multi_network321/multi_bridge_cen.pdf")
    # plt.savefig("./multi_network321/multi_bridge_cen.png")
    # 输出结束时间
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("结束时间：", formatted_time)






