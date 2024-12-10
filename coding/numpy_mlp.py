import numpy as np

# 激活函数基类
class Activation:
    def forward(self, x):
        pass
    
    def backward(self, grad_output):
        pass

# ReLU激活函数
class ReLU(Activation):
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        return grad_output * (self.input > 0)

# 线性层
class Linear:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.b = np.zeros(output_dim)
        
    def forward(self, x):
        self.input = x
        return np.dot(x, self.W) + self.b
    
    def backward(self, grad_output):
        self.grad_W = np.dot(self.input.T, grad_output) / self.input.shape[0]
        self.grad_b = np.sum(grad_output, axis=0) / self.input.shape[0]
        return np.dot(grad_output, self.W.T)
    
    def update(self, lr):
        self.W -= lr * self.grad_W
        self.b -= lr * self.grad_b

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        # 初始化层
        self.linear1 = Linear(input_dim, hidden_dim)
        self.activation = ReLU()
        self.linear2 = Linear(hidden_dim, output_dim)
        self.lr = lr
        
    def forward(self, X):
        # 第一层线性变换
        self.z1 = self.linear1.forward(X)
        # ReLU激活
        self.a1 = self.activation.forward(self.z1)
        # 第二层线性变换
        self.z2 = self.linear2.forward(self.a1)
        # softmax输出
        self.output = self.softmax(self.z2)
        return self.output
    
    def backward(self, y):
        # 输出层梯度
        dz2 = self.output - y  # softmax的导数
        da1 = self.linear2.backward(dz2)
        
        # 隐藏层梯度
        dz1 = self.activation.backward(da1)
        self.linear1.backward(dz1)
        
    def update(self):
        self.linear1.update(self.lr)
        self.linear2.update(self.lr)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# 测试
if __name__ == "__main__":
    # 生成随机数据
    np.random.seed(0)
    X = np.random.randn(100, 2)
    y = np.array([1 if x1 + x2 > 0 else 0 for x1, x2 in X])
    y_onehot = np.zeros((y.shape[0], 2))
    y_onehot[np.arange(y.shape[0]), y] = 1
    
    # 初始化并训练MLP
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=2)
    
    # 训练1000轮
    for i in range(1000):
        pred = mlp.forward(X)
        mlp.backward(y_onehot)
        mlp.update()
        
        if i % 100 == 0:
            loss = -np.mean(np.sum(y_onehot * np.log(pred + 1e-8), axis=1))
            print(f"Iteration {i}, Loss: {loss:.4f}")
    
    # 预测并计算准确率
    pred = np.argmax(mlp.forward(X), axis=1)
    accuracy = np.mean(pred == y)
    print(f"\nAccuracy: {accuracy:.4f}")
