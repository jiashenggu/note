bug举例：Tied Embedding场景下使用Distributed Optimizer会有问题

优化点举例：Distributed Optimizer随DP组可扩展性不足，DP组增大，overlap gradient all-reduce效果反而会变差

题主提到的两个例子，其实都是调试相关的问题。做系统的人，扒一扒源码，改一改配置，很快就能解决了。比如tied embedding在大模型里基本不会开启，zero stage1的bucket size开得大一点可以提高通信效率。如果dp group size太大导致参数切得太碎，还可以像fsdp一样将shard group和dp group解耦。

作者：Lin Zhang
链接：https://www.zhihu.com/question/633778272/answer/3388811917
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
