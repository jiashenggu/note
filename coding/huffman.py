import heapq


class Node:
    def __init__(self, v, l_node=None, r_node=None):
        self.v = v
        self.l_node = l_node
        self.r_node = r_node

    def __lt__(self, other):
        return self.v < other.v


def build_huffman_tree(values):
    h = []
    for value in values:
        heapq.heappush(h, Node(value))

    while len(h) > 1:
        l_node = heapq.heappop(h)
        r_node = heapq.heappop(h)
        merged = Node(l_node.v + r_node.v, l_node, r_node)
        heapq.heappush(h, merged)
    return h[0]


def print_huffman_tree(root):
    q = [root]
    cnt = 0
    while q:
        nq = len(q)
        line = str(cnt) + ": "
        for _ in range(nq):
            node = q.pop(0)
            line += str(node.v) + "|"
            if node.l_node:
                q.append(node.l_node)
            if node.r_node:
                q.append(node.r_node)
        print(line)
        cnt += 1


a = [5, 9, 11, 2, 8, 16, 20]

huffman_tree = build_huffman_tree(a)

print_huffman_tree(huffman_tree)

# 哈夫曼编码是一种广泛使用的**无损数据压缩算法**，它的核心思想是通过**变长编码**对数据进行高效压缩。以下是要使用哈夫曼编码的主要原因及其优势：

# ---

# ### 1. **高效压缩：基于频率的最优编码**
#    - **频率敏感的编码**：哈夫曼编码根据字符（或符号）在数据中出现的**频率**分配编码。高频字符用**短编码**，低频字符用**长编码**，从而显著减少整体数据的存储空间。
#    - **最小化平均编码长度**：在所有前缀编码中，哈夫曼编码的**平均编码长度最短**（数学上可证明），这意味着它在已知字符频率的情况下是**最优的**。

# ---

# ### 2. **无损压缩：数据完整保留**
#    - 哈夫曼编码是**无损压缩**，解压后能完全恢复原始数据，适用于对数据精度要求高的场景（如文本、代码、医学图像）。

# ---

# ### 3. **前缀编码：消除解码歧义**
#    - 哈夫曼编码是**前缀码（Prefix Code）**，即任意字符的编码都不是其他编码的前缀。这使得解码时无需分隔符，可以直接按位流逐次匹配唯一字符，解码高效且无歧义。

# ---

# ### 4. **应用场景广泛**
#    - **文件压缩**：如ZIP、GZIP等格式的压缩算法中常用哈夫曼编码。
#    - **多媒体编码**：JPEG图像、MPEG视频等标准中结合哈夫曼编码进行数据压缩。
#    - **通信协议**：减少网络传输的数据量，提升传输效率。

# ---

# ### 5. **算法简单且高效**
#    - 哈夫曼编码的实现基于**哈夫曼树（Huffman Tree）**，通过优先级队列（如最小堆）即可快速构建，时间复杂度为 \(O(n \log n)\)，适合实际应用。

# ---

# ### 示例说明
# 假设一个文本中字符频率如下：
# - `A: 50%` → 编码为 `0`（1位）
# - `B: 25%` → 编码为 `10`（2位）
# - `C: 15%` → 编码为 `110`（3位）
# - `D: 10%` → 编码为 `111`（3位）

# 相比定长编码（如ASCII需要8位/字符），哈夫曼编码的平均长度为：
# \(0.5 \times 1 + 0.25 \times 2 + 0.15 \times 3 + 0.1 \times 3 = 1.75\) 位/字符，压缩率高达 **78%**。

# ---

# ### 适用场景的限制
# - **静态数据**：需要预先知道字符频率，不适合动态变化的数据流（此时需用动态哈夫曼编码或算术编码）。
# - **额外存储开销**：需保存哈夫曼树或编码表，对小数据可能不划算。

# ---

# ### 总结
# 哈夫曼编码在**压缩效率**、**解码速度**和**通用性**之间达到了优秀的平衡，是数据压缩领域的经典算法。当需要减少存储或传输成本，同时保证数据完整性时，哈夫曼编码是一个理想选择。
