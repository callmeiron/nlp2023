import os
import math
import jieba
import logging
import numpy as np
import matplotlib as mpl

# 一元词频统计
def  get_tf(tf_dic,  words) :
    for  i  in  range(len(words) ):
        tf_dic[words[i]]  =  tf_dic.get(words[i],  0)  + 1
 # 二元模型词频统计

def  get_bigram_tf(tf_dic,  words) :
    for  i  in  range(len(words)-1):
        tf_dic[(words[i],  words[i+1])]  =  tf_dic.get((words[i],  words[i+1]),  0)   +1
# 三元模型词频统计
def  get_trigram_tf(tf_dic,  words) :
    for  i  in  range(len(words)-2):
        tf_dic[((words[i], words[i+1]), words[i+2])] = tf_dic.get(((words[i], words[i+1]), words[i+2]),  0)  +1

with  open('all_sentence.txt',  'r',  encoding='utf-8')  as  f :
    corpus  =  []
    count_all=   0
    for  line  in   f:
        if  line  !='\n':
            corpus.append(line.strip())
            count_all  +=  len(line.strip() )

split_words  =  []
words_len  =   0
line_count  =   0
words_tf  =  {}
bigram_tf  =  {}
trigram_tf  =  {}
mode  =  'jieba'
# mode  =  'singleword'
for  line  in  corpus:
    if  mode  ==  'jieba':
        for  x  in  jieba.cut(line):
            split_words.append(x)
            words_len  +=   1
    else:
        for  x  in  line:
            split_words.append(x)
            words_len  +=   1

    get_tf(words_tf,  split_words)
    get_bigram_tf(bigram_tf,  split_words )
    get_trigram_tf(trigram_tf,  split_words )
    split_words  =  [ ]
    line_count  +=   1

print("语料库字数:",  count_all)
print("分词个数:",  words_len)
print("平均词长:",  round(count_all  /  words_len,  5 ))

# 打印部分词频
tf_dic_list  =  sorted(words_tf.items(),  key  =  lambda  x:x[1],  reverse=True)
for  i  in  range(0,5 ):
    print(tf_dic_list[i][0],tf_dic_list[i][1])
bigram_tf_list  =  sorted(bigram_tf.items(),  key  =  lambda  x:x[1],  reverse=True)
for  i  in  range(0,5 ):
    print(bigram_tf_list[i][0],bigram_tf_list[i][1])
trigram_tf_list  =  sorted(trigram_tf.items(),  key  =  lambda  x:x[1],  reverse=True)
for  i  in  range(0,5 ):
    print(trigram_tf_list[i][0],trigram_tf_list[i][1])

words_len  =  sum([dic[1]  for  dic  in  words_tf.items ()])
bigram_len  =  sum([dic[1]  for  dic  in  bigram_tf.items ()])
trigram_len  =  sum([dic[1]  for  dic  in  trigram_tf.items()])
print("一元模型长度:",words_len)
print("二元模型长度:",bigram_len)
print("三元模型长度:",trigram_len)

entropy  =  []
entropy = [-(uni_word[1]/words_len)*math.log(uni_word[1]/words_len, 2) for uni_word in words_tf.items()]
print("基于jieba分割的一元模型的中文信息熵为:",  round(sum(entropy),  5),  "比特/词")
print("基于jieba分割的一元模型的中文平均信息熵为:",  round(sum(entropy)/len(entropy),  5),  "比特/词")

entropy  =  []
for  bi_word  in  bigram_tf.items( ):
    jp_xy  =  bi_word[1]  /  bigram_len#    计算联合概率p(x,y)
    cp_xy  =  bi_word[1]  /  words_tf[bi_word[0][0]]#    计算条件概率p(x|y)
    entropy.append(-jp_xy  *  math.log(cp_xy,  2)) #   计算二元模型的信息熵
print("基于jieba分割的二元模型的中文信息熵为:", round(sum(entropy) , 5), "比特/词")
print("基于jieba分割的二元模型的中文平均信息熵为:",  round(sum(entropy)/len(entropy),  5),  "比特/词")

entropy  =  []
for  tri_word  in  trigram_tf.items( ):
    jp_xy  =  tri_word[1]  /  trigram_len
    cp_xy  =  tri_word[1]  /  bigram_tf[tri_word[0][0]]
    entropy.append(-jp_xy  *  math.log(cp_xy,  2))
print("基于jieba分割的三元模型的中文信息熵为:",  round(sum(entropy),  5),  "比特/词")
print("基于jieba分割的三元模型的中文平均信息熵为:",  round(sum(entropy)/len(entropy),  5),  "比特/词")
