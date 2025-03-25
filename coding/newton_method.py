def newton_method(f, f_prime, x0, tol=1e-6, max_iter=100):
    """
    使用牛顿法寻找函数 f 的根。

    参数:
    f (function): 需要求根的函数。
    f_prime (function): 函数 f 的导数。
    x0 (float): 初始猜测值。
    tol (float, 可选): 容差，当函数值或步长小于该值时停止迭代。默认为 1e-6。
    max_iter (int, 可选): 最大迭代次数。默认为 100。

    返回:
    float: 近似根。

    异常:
    ValueError: 如果导数为零或迭代未收敛。
    """
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:  # 检查函数值是否足够小
            return x
        fp = f_prime(x)
        if abs(fp) < 1e-12:  # 避免除以零
            raise ValueError("导数为零，无法继续迭代。")
        delta_x = fx / fp
        x_new = x - delta_x
        if abs(delta_x) < tol:  # 检查步长是否足够小
            return x_new
        x = x_new
    raise ValueError("超过最大迭代次数，未收敛。")

# 示例用法
if __name__ == "__main__":
    import math

    # 示例1：求解 x^2 - 2 = 0 的根（sqrt(2)）
    f1 = lambda x: x**2 - 2
    f_prime1 = lambda x: 2 * x
    root1 = newton_method(f1, f_prime1, 1.0)
    print(f"根1: {root1:.6f}")  # 应接近 1.414214

    # 示例2：求解 cos(x) - x = 0 的根
    f2 = lambda x: math.cos(x) - x
    f_prime2 = lambda x: -math.sin(x) - 1
    root2 = newton_method(f2, f_prime2, 0.5)
    print(f"根2: {root2:.6f}")  # 应接近 0.739085
