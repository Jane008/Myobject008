# 一个名词块就是一个chunking
# sentiment 情绪

# 以下都是例子
# sentiment_dictionary = {}

# for line in open('G:/资料汇总/nlp/AFINN/AFINN-111.txt'):
    #word, score = line.split('\t')
    #sentiment_dictionary[word] = int(score)

# 把这个打分表记录在一个dict上以后
# 跑遍整个句子把对应的值相加
# total_score = sum(sentiment_dictionary.get(word,0) for word in words)
# 有值就是dict中的值，没有就是0

# 配上ML的情感分析，会用到贝叶斯NaiveBayesClassifier
from nltk.classify import NaiveBayesClassifier

# 创建几个句子
s1 = 'this is a good book'
s2 = 'this is a awesome book'
s3 = 'this is a bad book'
s4 = 'this is a terrible book'

def preprocess(s):
    # func句子处理
    # 这里简单的用了split(),把句子中每个单词分开
    # 显然 还有更多的preprocessing method可以用
    return {word:True for word in s.lower().split()}
    # return这步是把出现过的词变成true和false的向量
    
#把训练集做成标准形式
training_data = [[preprocess(s1),'pos'],
                 [preprocess(s2),'pos'],
                 [preprocess(s3),'neg'],
                 [preprocess(s4),'neg']]
# 模型开始训练
model = NaiveBayesClassifier.train(training_data)

print(model.classify(preprocess('this is a good book')))
# 结果是pos意思就是积极的

# Frequency频率统计器:FreqDist

from nltk import FreqDist
import nltk

# 先做个语料库:corpus 
corpus = "this is my sentence"\
"this is my life"\
"this is the day"

# 先tonkenize一下
tokens = nltk.word_tokenize(corpus)
print(tokens)
