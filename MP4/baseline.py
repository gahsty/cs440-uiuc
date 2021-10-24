"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""


def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    words_table = {}
    unseen = ['NOUN', 0]
    unseen_table={}
    for sentence in train:
        for word in sentence:
            if word[0] not in words_table.keys():
                tag_table = {word[1]: 1}
                words_table[word[0]] = [[word[1], 1], tag_table]
            else:
                temp = words_table.get(word[0])
                tag_table = temp[1]
                if word[1] not in tag_table.keys():
                    tag_table[word[1]] = 1
                else:
                    tag_table[word[1]] += 1
                    if tag_table[word[1]] > temp[0][1]:
                        temp[0][0] = word[1]
                        temp[0][1] = tag_table[word[1]]

    for keys in words_table.keys():
        only_table = words_table[keys][1]
        if len(only_table) == 1:
            unseen_type = list(only_table.keys())[0]
            if unseen_type not in unseen_table.keys():
                unseen_table[unseen_type] = 1
            else:
                unseen_table[unseen_type] += 1
            if unseen_table[unseen_type]>unseen[1]:
                unseen=[unseen_type,unseen_table[unseen_type]]

    ans =[]
    for sentence in test:
        result = []
        for word in sentence:
            if word in words_table.keys():
                result.append((word,words_table[word][0][0]))
            else:
                result.append((word,unseen[0]))

        ans.append(result)
    return ans
