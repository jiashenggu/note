# 扩词表

不建议去轻易扩词表，满足以下两个条件，可以去尝试

1. 底座模型的词表跟你的domain的词表分布差距很大
2. 增训的domain语料足够多。

大部分词表都是有基础字的，比如原来 「北京」 -> [12, 15]。扩了词后，现在假设变成了「北京」-> [10233]。这种因为动了高频词，刘乾试过各种warmup，frozen，都是想要有正面作用，需要训练更久的时间。但多语言的continue pretrain，很多小语种的语料就这么点，还没变正向，样本就用完了。。还有一种情况，大家可以试试，就是你扩充的都是低频词，原有的高频字/词不被影响。大家还是选一个词表好的底座模型来做continue pretrain更合适，对比于底座训练不充分，词表的坑更大。

# Domain Continue Pretrain

1. **Replay**:采样pretrain阶段的数据。
2. **Learning rate**:保持continual pre-training总token数一样的情况下，原有domain 和新domain 的loss几乎都是可以预测的，本质上是「learning rate学习率」和「replay ratio 重放比例」的一个权衡。**在continual pre-training总token一样的情况下，学得越快，忘得越多，即使replay很多也一样如此。****
3. **比例控制**
4. **为什么语言的continue pretrain，比例不能剧烈变动？**  
  三点原因
  
  1. **不同的知识，集中在不同的transformer层**
    
  
  之前内部实验  
  发现transformer越往上，最后一层的知识往往就越具体，越底层的知识反而越基础。  
  类似cnn做人脸识别，第一层抽取的特征是线条，到了最后一层就变成了鼻子，人脸这些特征。  
  语义这些知识，是最基础的知识，往往是在最底层，更新起来影响的层数更多。  
  domain知识是最后几层，更新起来影响的层数相对更小一些。
  
  2. **扩词表**  
    新词的embedding是随机初始化的，是transformer最底层了。同理，影响面更大。
    
  3. **learing rate**
    
  
  不合适的learning rate会导致general能力”受损“。以及learning rate大小带来的影响，跟增训中文，一点点提高中文比例，有点异曲同工。从刘乾的反馈来看，他们不扩词表，先找到合适的learning rate，再找到合适的比例，直接continue pretrain，loss就能稳定持续下降了

continue pretrain分成三大类  
领域知识，语言类，long context

受到词表，知识难度，attention分布的影响，这几类知识的学习都会有不少的差距。其中领域知识增强类的domain更容易学习，因为基座llm中存在这样的知识，所以起始loss会更低，遗忘程度低，最优的配比低。  
语言类的domain和long context的数据更难学习，前者是因为语言的gap导致初始loss偏高，但随着不断的训练，loss会稳定下降，但遗忘程度高，最优配比高，后者对资源的消耗更高，遗忘程度高，最优配比高。

### **领域知识Continue Pretrain**

**难点**  
比例的控制

## **语言类Continue Pretrain**

**难点**  
去年大家的常用做法，就是已知llama的中文占比5%，那么我一点点增大中文的训练样本比例。  
而不是算好一个比例，直接硬train，这很容易导致loss直接崩掉。

### **Long Context Continue Pretrain**

**3.3.1 continue pretrain学了什么**  
拿long context举例子，根据我们的一些分析，LLM本身就具有long context的能力，或者说是已经学到了文本的框架。  
而之所以外推不好，其中一个猜测就是attention分布导致的。

而long context的continue pretrain某种程度上是让attention分布的调整。  
<u><a rel="nofollow noreferrer" class="external" href="https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2404.15574"><span class="invisible">https://</span><span class="visible">arxiv.org/abs/2404.1557</span><span class="invisible">4</span><span class="ellipsis"></span></a></u>(例如这篇文章）  
知识的重新学习并不是大头。

# Reference

https://zhuanlan.zhihu.com/p/707751901
