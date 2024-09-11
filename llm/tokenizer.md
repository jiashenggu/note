
BPE、WordPiece和Unigram都是用于子词分词(subword tokenization)的算法,主要用于自然语言处理中的词汇表构建。它们有一些相似之处,但也存在一些关键区别。我可以简要概括它们的主要区别:

BPE (Byte Pair Encoding):
1. 自底向上的合并策略
2. 每次迭代选择最频繁的相邻字符对合并
3. 将 word 以字符为单位进行拆分, 一个字符作为一个 subword。然后按照 合并规则列表 中的顺序, 如果出现了 subword pair, 就进行合并。


WordPiece:
1. 类似BPE,但使用不同的评分标准
2. 选择能最大化训练数据似然的合并操作，$score(sw1, sw2) = \frac{count(sw1, sw2)}{count(sw1), count(sw2)}$
3. 直接使用 词表 分词, 采用 正向最长匹配 的策略。也就是说, 不断寻找 word 中的最长前缀 subword。:


Unigram:
1. 自顶向下的分割策略
2. 从大词汇表开始,迭代删除对似然影响最小的token
3. 使用概率模型选择最佳分词


LLM 分词算法 (BPE, WordPiece, Unigram) 简介 - Soaring的文章 - 知乎
https://zhuanlan.zhihu.com/p/664717335

https://huggingface.co/learn/nlp-course/en/chapter6/7?fw=pt

SentencePiece支持BPE和Unigram两种主流算法，Unigram训练速度尚可，但压缩率会稍低一些，BPE的压缩率更高，但是训练速度要比Unigram慢上一个数量级！而且不管是BPE还是Unigram，训练过程都极费内存。总而言之，用较大的语料去训练一个SentencePiece模型真不是一种好的体验。
https://kexue.fm/archives/9752
