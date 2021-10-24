# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
This file should not be submitted - it is only meant to test your implementation of the Viterbi algorithm. 

See Piazza post @650 - This example is intended to show you that even though P("back" | RB) > P("back" | VB), 
the Viterbi algorithm correctly assigns the tag as VB in this context based on the entire sequence. 
"""
from utils import read_files, get_nested_dictionaries
import math

def main():
    test, emission, transition, output = read_files()
    emission, transition = get_nested_dictionaries(emission, transition)
    initial = transition["START"]
    prediction = []
    sheet={'NNP':0,'MD':1,'VB':2,'JJ':3,'NN':4,'RB':5,'DT':6}
    """WRITE YOUR VITERBI IMPLEMENTATION HERE"""
    v = len(sheet)
    for sentence in test:
        temp_result=[]
        length = len(sentence)
        # 回溯列表
        space = [0 for i in range(v)]
        # 存放现在的number of tags 个点, 顺序按 sheet
        temp = [0 for i in range(v)]
        # 存放现在的number of tags 个概率，顺序按 sheet
        hash = {}
        # 回溯字典
        for i in range( length ):
            word = sentence[i]
            if i == 0:
                # t = 1 时，temp 初始化为 P(W1|tag), 顺序按sheet; info_node 为当前时间t下，标签为tag的点
                # t = 1 时， info_node回溯的点均为START
                for tag in sheet:

                    temp[sheet[tag]] = emission[tag][word]*initial[tag]
                    info_node = (tag, i)
                    hash[info_node] = ('START', 0)
                    space[sheet[tag]] = info_node
            else:
                # t =k+1 时, temp[0-16] 分别为 temp在时间k时，所有标签转移到temp[0-16]的最大值
                temp_k = [0 for i in range(v)]
                for tag in sheet:
                    e = emission[tag][word]
                    result = find_max(temp, transition, e, tag, sheet)
                    info_node = (tag, i)
                    hash[info_node] = (result[1], i - 1)
                    space[sheet[tag]] = info_node
                    temp_k[sheet[tag]] = result[0]
                temp = temp_k
        final = space[temp.index(max(temp))]
        while final in hash:
            temp_result.append(final[0])
            final = hash[final]
        temp_result.reverse()
        for i in range(length):
            temp_result[i] = [sentence[i], temp_result[i]]
        prediction.append(temp_result)





    print('Your Output is:',prediction,'\n Expected Output is:',output)

def find_max(temp, tt_table, emission, tag, sheet):
    """
    input: tk到tk+1的转移表，temp 在时间k时的各个值，emission 概率，当前计算的tag,tag sheet
    output： 最大值以及造成最大值的前一个标签。
    """
    m = {}
    for t in sheet:
        transition = tt_table[tag][t]
        value = transition * emission * temp[sheet[t]]

        m[value] = t
    print(m)
    value = max(m.keys())
    print(m[value])
    return [value, m[value]]

if __name__=="__main__":
    main()