import numpy as np

# ReLU激活函数
class ReLU:
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * (self.input > 0)

# 线性层
class Linear:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
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

class SoftmaxCrossEntropyLoss:
    def forward(self, logits, y_true):
        """
        Calculates stable softmax and cross-entropy loss.
        logits: The raw output from the last linear layer (pre-activation).
        y_true: The one-hot encoded true labels.
        """
        # Store for backward pass
        self.y_true = y_true

        # Log-sum-exp trick for numerical stability
        max_logit = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logit)
        sum_exp_logits = np.sum(exp_logits, axis=1, keepdims=True)
        
        # Softmax probabilities
        self.y_pred = exp_logits / sum_exp_logits
        
        # Clip probabilities to avoid log(0)
        y_pred_clipped = np.clip(self.y_pred, 1e-8, 1 - 1e-8)

        # Calculate cross-entropy loss
        loss = -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
        return loss

    def backward(self):
        """
        Calculates the gradient of the loss with respect to the logits.
        The gradient is simply (softmax_output - true_labels).
        """
        return self.y_pred - self.y_true

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        self.linear1 = Linear(input_dim, hidden_dim)
        self.activation = ReLU()
        self.linear2 = Linear(hidden_dim, output_dim)
        self.lr = lr

    def forward(self, X):
        # First linear layer
        z1 = self.linear1.forward(X)
        # ReLU activation
        a1 = self.activation.forward(z1)
        # Second linear layer
        logits = self.linear2.forward(a1) # Note: We now call this 'logits'
        # Return the raw logits, DO NOT apply softmax here
        return logits

    def backward(self, grad_output):
        # Gradient from the output layer
        da1 = self.linear2.backward(grad_output)
        # Gradient through ReLU
        dz1 = self.activation.backward(da1)
        # Gradient for the first layer
        self.linear1.backward(dz1)

    def update(self):
        self.linear1.update(self.lr)
        self.linear2.update(self.lr)


# The Linear, ReLU classes remain the same.

# Test
if __name__ == "__main__":
    # Generate random data
    np.random.seed(0)
    X = np.random.randn(100, 2)
    y = np.array([1 if x1 + x2 > 0 else 0 for x1, x2 in X])
    y_onehot = np.zeros((y.shape[0], 2))
    y_onehot[np.arange(y.shape[0]), y] = 1

    # Initialize MLP and the new combined loss function
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=2)
    loss_fn = SoftmaxCrossEntropyLoss()

    # Train for 1000 epochs
    for i in range(1000):
        # --- FORWARD PASS ---
        # 1. Get logits from the network
        logits = mlp.forward(X)
        # 2. Calculate loss using the combined function
        loss = loss_fn.forward(logits, y_onehot)

        # --- BACKWARD PASS ---
        # 1. Get the initial gradient from the loss function
        grad = loss_fn.backward()
        # 2. Backpropagate the gradient through the network
        mlp.backward(grad)
        # 3. Update weights
        mlp.update()

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss:.4f}")

    # --- PREDICTION ---
    # For prediction, we still need to apply softmax to the final logits
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
    final_logits = mlp.forward(X)
    predictions = np.argmax(softmax(final_logits), axis=1) # Apply softmax here for probabilities
    
    accuracy = np.mean(predictions == y)
    print(f"\nAccuracy: {accuracy:.4f}")
