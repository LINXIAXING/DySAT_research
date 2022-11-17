import numpy as np
import networkx as nx
import random

# DISCLAIMER:
# Parts of this code file are derived from
#  https://github.com/aditya-grover/node2vec

'''Random walk sampling code'''

class Graph_RandomWalk():
    def __init__(self, nx_G, is_directed, p, q):
        # nx_G: 图
        # is_directed: 是否是有向图
        # p: 影响边alias中指回出发节点的概率(1/p)
        # q: 影响边alias中指向二阶及以上邻居的概率(1/q)
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        '''
        计算随机游走,返回walk_length长度的节点列表

        Simulate a random walk starting from start node.
        '''
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            # 取出节点 及其邻居节点
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    # 第一次游走,从start_node邻居节点中取点
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    # 第二次及以后游走,借alias_edges从cur邻居取点.区别于首次游走的是,待取节点与prev点的跳数会影响概率
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
                        alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        num_walks:游走次数(影响walk长)
        walk_length: 游走取点数量

        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        # 循环num_walks次, 每次随机从一点出发取样walk_length个, 一起添加到walks中
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        '''
        计算边之间的alias采样概率
        
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        # 遍历 目标节点的全部邻居
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                # 处理指回出发节点的概率
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                # 处理指向一阶邻居的概率
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                # 处理指向二阶及以上邻居的概率
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        # 归一化
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
        # 计算alias取样概率
        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        预处理影响随机游走时的概率数值
        '''
        G = self.G
        is_directed = self.is_directed
        # alias采样方法计算节点
        alias_nodes = {}
        for node in G.nodes():
            # 将每个节点与其不同邻居节点的权重作为两点之间的非标概率
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            # 计算节点node与全部邻居weight之和 (用于归一化)
            norm_const = sum(unnormalized_probs)
            # 归一化概率 (node节点与全部邻居节点之间游走概率和为1)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            # 将节点node的alias采样概率进行存储
            alias_nodes[node] = alias_setup(normalized_probs)  # ([拼接的node index], [概率])

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            # 无向图不分边方向(需要计算两遍)
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])
        # 保存
        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    '''
    使用alias采样计算离散分布的概率
    probs是经过归一化的概率列表

    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    # alias方法中,index是取样的对象
    # q存储alias方法处理后的概率
    # J中元素对应q中概率未命中时,取到的index(取样的对象)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    # 记录过小或过大概率的下标 (方便进行裁减或填充)
    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        # q拷贝概率,并全部乘上概率长度k (使概率总面积=1*K)
        q[kk] = K*prob
        # 记录过大过小概率的index
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)
    # 裁减和填充概率
    while len(smaller) > 0 and len(larger) > 0:
        # 移除并获取最后一位值(index)
        small = smaller.pop()
        large = larger.pop()
        # j中更新过小的prob所拼接的prob的index
        J[small] = large
        # 被裁减prob更新自己的新prob, 不足1加入small, 超1加入large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    # q存储alias方法处理后的概率
    # J中元素对应q中概率未命中时,取到的index(取样的对象)
    K = len(J)
    # 取随机下标kk
    kk = int(np.floor(np.random.rand()*K))
    # 取kk下的随机数
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
