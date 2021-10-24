"""
Part 4: Here should be your best version of viterbi, 
with enhancements such as dealing with suffixes/prefixes separately
"""
import math


import math


def viterbi_3(train, test):
    """
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    """
    court = count_pair(train)

    tag_table = smooth(court[0], 0.00001)
    tt_table = smooth(court[1], 0.00001)
    sheet = {}
    nm = 0
    for i in tag_table[0]:
        sheet[i] = nm
        nm += 1

    ans = []
    v = len(sheet)
    for sentence in test:
        length = len(sentence)
        temp_result = ['END']
        # 回溯列表
        space = [0 for i in range(v)]
        # 存放现在的number of tags 个点, 顺序按 sheet
        temp = [0 for i in range(v)]
        # 存放现在的number of tags 个概率，顺序按 sheet
        hash = {}
        # 回溯字典
        for i in range(1, length - 1):
            word = sentence[i]
            if i == 1:
                # t = 1 时，temp 初始化为 P(W1|tag), 顺序按sheet; info_node 为当前时间t下，标签为tag的点
                # t = 1 时， info_node回溯的点均为START
                for tag in sheet:
                    if tag in tt_table[0]['START']:
                        ini = tt_table[0]['START'][tag]
                    else:
                        ini = tt_table[1][tag]
                    if word in tag_table[0][tag]:
                        temp[sheet[tag]] = tag_table[0][tag][word] + ini
                    else:
                        temp[sheet[tag]] = tag_table[1][tag] + ini
                    info_node = (tag, i)
                    hash[info_node] = ('START', 0)
                    space[sheet[tag]] = info_node
            else:
                # t =k+1 时, temp[0-16] 分别为 temp在时间k时，所有标签转移到temp[0-16]的最大值
                temp_k = [0 for i in range(v)]
                for tag in sheet:
                    if word in tag_table[0][tag]:
                        emission = tag_table[0][tag][word]
                    else:
                        emission = tag_table[1][tag]
                    result = find_max(temp, tt_table, emission, tag, sheet)
                    info_node = (tag, i)
                    hash[info_node] = (result[1], i - 1)
                    space[sheet[tag]] = info_node
                    temp_k[sheet[tag]] = result[0]
                temp = temp_k
        final = space[temp.index(max(temp))]
        while final in hash:
            temp_result.append(final[0])
            final = hash[final]
        temp_result.append('START')
        temp_result.reverse()
        for i in range(length):
            temp_result[i] = [sentence[i], temp_result[i]]
        ans.append(temp_result)
    return ans


def find_max(temp, tt_table, emission, tag, sheet):
    """
    input: tk到tk+1的转移表，temp 在时间k时的各个值，emission 概率，当前计算的tag,tag sheet
    output： 最大值以及造成最大值的前一个标签。
    """
    m = {}
    for t in sheet:
        if tag in tt_table[0][t]:
            transition = tt_table[0][t][tag]
        else:
            transition = tt_table[1][t]
        value = transition + emission + temp[sheet[t]]

        m[value] = t
    value = max(m.keys())
    return [value, m[value]]


def smooth(table, laplace):
    """
    input: one table with keys(tag) and value (dict in form of "word : court"), a laplace parameter
    output: a list in form [smooth table,unseen list]
    """
    court_table = {}
    once_table = {}
    unseen_table = {}
    total_once = 0
    for key in table:
        temp = table[key]
        court = 0
        once = 0
        for entity in temp:
            court += temp[entity]
            if temp[entity] == 1:
                once += 1
                total_once += 1
        court_table[key] = court
        once_table[key] = once

    for key in once_table:
        once_table[key] = (once_table[key]+0.1) / (total_once+len(court_table))

    for key in table:
        line = table[key]
        n = court_table[key]
        l = laplace * once_table[key]
        for each in line:
            court = line[each]
            v = len(line)
            line[each] = math.log((court + l) / (n + l * (v + 1)))
        table[key] = line
        unseen_table[key] = math.log(l / (n + l * (v + 1)))

    ans = [table, unseen_table]
    return ans


def count_pair(train):
    """
    input : training data E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]

    output: return a list of two table [tag_table, tt_table]
    tag_table is a court of words in given tag
    tt_table is a court of next tags in given tag
    """
    tag_table = {}
    tt_table = {}
    for sentence in train:
        for i in range(0, len(sentence) - 1):
            word = sentence[i]
            if word[1] not in tag_table.keys():
                word_table = {word[0]: 1}
                word_next = sentence[i + 1]
                tt_table[word[1]] = {word_next[1]: 1}
                tag_table[word[1]] = word_table
            else:
                word_table = tag_table[word[1]]
                if word[0] in word_table.keys():
                    word_table[word[0]] += 1
                else:
                    word_table[word[0]] = 1
                tag_table[word[1]] = word_table
                word_next = sentence[i + 1]
                if word_next[1] in tt_table[word[1]].keys():
                    tt_table[word[1]][word_next[1]] += 1
                else:
                    tt_table[word[1]][word_next[1]] = 1

    return [tag_table, tt_table]
