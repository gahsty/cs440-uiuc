# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)


# Feel free to use the code below as you wish
# Initialize it with a list/tuple of objectives
# Call compute_mst_weight to get the weight of the MST with those objectives
# TODO: hint, you probably want to cache the MST value for sets of objectives you've already computed...
import copy
import heapq
import itertools
import queue


def DISTANCE(i, j):
    return abs(i[0]-j[0])+abs(i[1]-j[1])


class MST:
    def __init__(self, objectives):
        self.elements = {key: None for key in objectives}

        # TODO: implement some distance between two objectives
        # ... either compute the shortest path between them, or just use the manhattan distance between the objectives
        self.distances = {
            (i, j): DISTANCE(i, j)
            for i, j in self.cross(objectives)
        }

    # Prim's algorithm adds edges to the MST in sorted order as long as they don't create a cycle
    def compute_mst_weight(self):
        weight = 0
        for distance, i, j in sorted((self.distances[(i, j)], i, j) for (i, j) in self.distances):
            if self.unify(i, j):
                weight += distance
        return weight

    # helper checks the root of a node, in the process flatten the path to the root
    def resolve(self, key):
        path = []
        root = key
        while self.elements[root] is not None:
            path.append(root)
            root = self.elements[root]
        for key in path:
            self.elements[key] = root
        return root

    # helper checks if the two elements have the same root they are part of the same tree
    # otherwise set the root of one to the other, connecting the trees
    def unify(self, a, b):
        ra = self.resolve(a)
        rb = self.resolve(b)
        if ra == rb:
            return False
        else:
            self.elements[rb] = ra
            return True

    # helper that gets all pairs i,j for a list of keys
    def cross(self, keys):
        return (x for y in (((i, j) for j in keys if i < j) for i in keys) for x in y)


class State:
    def __init__(self, posi, notv):
        self.posi = posi
        self.notv = notv
        self.preval = None


class SLinkedList:
    def __init__(self):
        self.headval = None


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    temp = [[0 for i in range(maze.size.x)] for j in range(maze.size.y)]
    # 开始与结束点
    start = maze.start
    end = maze.waypoints[0]
    s = set()
    from collections import deque
    d = deque()
    d.append(start)
    s.add(start)

    while d:
        cur_node = d.popleft()
        if cur_node == (end[0], end[1]):
            break
        o = maze.neighbors(cur_node[0], cur_node[1])

        for nb in o:

            if (nb[0], nb[1]) in s:
                continue
            d.append(nb)
            s.add((nb[0], nb[1]))

            temp[nb[0]][nb[1]] = cur_node

    ans = [end]
    pre = temp[end[0]][end[1]]

    while 1:

        ans.append(pre)
        pre = temp[pre[0]][pre[1]]
        if pre == 0:
            break
    ans.reverse()

    return ans


def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # 开始与结束点
    start = maze.start
    end = maze.waypoints[0]
    pq = []  # list of entries arranged in a heap
    entry_finder = {}  # mapping of tasks to entries
    REMOVED = '<removed-task>'  # placeholder for a removed task
    counter = itertools.count()  # unique sequence count
    cost = {}
    pre = {}

    def add_task(task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in entry_finder:
            remove_task(task)
        count = next(counter)
        entry = [priority, count, task]
        entry_finder[task] = entry
        heapq.heappush(pq, entry)

    def remove_task(task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = entry_finder.pop(task)
        entry[-1] = REMOVED

    def pop_task():
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while pq:
            priority, count, task = heapq.heappop(pq)
            if task is not REMOVED:
                del entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')

    add_task(start, 0)
    cost[start] = 0
    while pq:
        cur = pop_task()
        if cur[0] == end[0] and cur[1] == end[1]:
            break
        for n in maze.neighbors(cur[0], cur[1]):
            n_cost = cost[cur] + 1
            if n not in cost:
                priority = n_cost + abs(n[0] - end[0]) + abs(n[1] - end[1])
                cost[n] = n_cost
                add_task(n, priority)
                pre[n] = cur

    ans = []
    prenode = cur
    while prenode != start:
        ans.insert(0, prenode)
        prenode = pre[prenode]
    ans.insert(0, start)
    return ans


def astar_multiple(maze):
    """
    Runs A star for part 4 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # 开始与结束点
    start = maze.start
    end_t = maze.waypoints
    pq = []  # list of entries arranged in a heap
    entry_finder = {}  # mapping of tasks to entries
    REMOVED = '<removed-task>'  # placeholder for a removed task
    counter = itertools.count()  # unique sequence count
    cost = {}

    def add_task(task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in entry_finder:
            remove_task(task)
        count = next(counter)
        entry = [priority, count, task]
        entry_finder[task] = entry
        heapq.heappush(pq, entry)

    def remove_task(task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = entry_finder.pop(task)
        entry[-1] = REMOVED

    def pop_task():
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while pq:
            priority, count, task = heapq.heappop(pq)
            if task is not REMOVED:
                del entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')

    '计算与当前点最近的且未被访问的路径点，返回与这点的距离'

    def close_point(human, not_visit):
        if len(not_visit) == 0:
            return 0
        distance = 99999
        for i in not_visit:
            if abs(i[0] - human[0]) + abs(i[1] - human[1]) < distance:
                distance = abs(i[0] - human[0]) + abs(i[1] - human[1])
                close = i
        return abs(close[0] - human[0]) + abs(close[1] - human[1])

    '计算mst，返回mst值'

    def cal_mst(task):
        if len(task) == 0:
            return 0

        return MST(task).compute_mst_weight()

    # 储存状态，cur坐标，未访问的点,cur花销

    s = (start, end_t, 0)
    add_task(s, 0)
    tail = ()
    pre = {}
    visited = {(start, end_t): s}

    MST_lengths = {}
    i = 0
    while pq:
        i += 1
        if i > 999999:
            return []
        cur = pop_task()
        not_visit = cur[1]
        if len(not_visit) == 0:
            tail = cur
            break

        if not_visit in MST_lengths:
            MST_cost = MST_lengths[not_visit]

        else:
            MST_cost = cal_mst(not_visit)
            MST_lengths[not_visit] = MST_cost

        for n in maze.neighbors(cur[0][0], cur[0][1]):
            n_cost = cur[2] + 1
            not_visit_n = list(copy.deepcopy(not_visit))

            if n in not_visit_n:
                not_visit_n.remove(n)
            if (n, tuple(not_visit_n)) not in visited:
                priority = n_cost + MST_cost + close_point(n, not_visit)
                n_s = (n, tuple(not_visit_n), n_cost)
                pre[n_s] = cur
                add_task(n_s, priority)
                visited[(n, tuple(not_visit_n))] = cur

    ans = []
    pre_n = tail
    while pre_n != s:
        ans.insert(0, pre_n[0])
        pre_n = pre[pre_n]
    ans.insert(0, start)

    return ans


def fast(maze):
    """
    Runs suboptimal search algorithm for part 5.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    """
        Runs A star for part 4 of the assignment in the case where there are
        multiple objectives.

        @param maze: The maze to execute the search on.

        @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # 开始与结束点
    start = maze.start
    end_t = maze.waypoints
    pq = []  # list of entries arranged in a heap
    entry_finder = {}  # mapping of tasks to entries
    REMOVED = '<removed-task>'  # placeholder for a removed task
    counter = itertools.count()  # unique sequence count
    cost = {}

    def add_task(task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in entry_finder:
            remove_task(task)
        count = next(counter)
        entry = [priority, count, task]
        entry_finder[task] = entry
        heapq.heappush(pq, entry)

    def remove_task(task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = entry_finder.pop(task)
        entry[-1] = REMOVED

    def pop_task():
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while pq:
            priority, count, task = heapq.heappop(pq)
            if task is not REMOVED:
                del entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')

    '计算与当前点最近的且未被访问的路径点，返回与这点的距离'

    def close_point(human, not_visit):
        if len(not_visit) == 0:
            return 0
        distance = 99999
        for i in not_visit:
            if abs(i[0] - human[0]) + abs(i[1] - human[1]) < distance:
                distance = abs(i[0] - human[0]) + abs(i[1] - human[1])
                close = i
        return abs(close[0] - human[0]) + abs(close[1] - human[1])

    '计算mst，返回mst值'

    def cal_mst(task):
        if len(task) == 0:
            return 0

        return MST(task).compute_mst_weight()

    # 储存状态，cur坐标，未访问的点,cur花销

    s = (start, end_t, 0)
    add_task(s, 0)
    tail = ()
    pre = {}
    visited = {(start, end_t): s}

    MST_lengths = {}
    i = 0
    while pq:
        i += 1
        if i > 999999:
            return []
        cur = pop_task()
        not_visit = cur[1]
        if len(not_visit) == 0:
            tail = cur
            break

        if not_visit in MST_lengths:
            MST_cost = MST_lengths[not_visit]

        else:
            MST_cost = cal_mst(not_visit)
            MST_lengths[not_visit] = MST_cost

        for n in maze.neighbors(cur[0][0], cur[0][1]):
            n_cost = cur[2] + 1
            not_visit_n = list(copy.deepcopy(not_visit))

            if n in not_visit_n:
                not_visit_n.remove(n)
            if (n, tuple(not_visit_n)) not in visited:
                priority = n_cost + (MST_cost + close_point(n, not_visit))*3
                n_s = (n, tuple(not_visit_n), n_cost)
                pre[n_s] = cur
                add_task(n_s, priority)
                visited[(n, tuple(not_visit_n))] = cur

    ans = []
    pre_n = tail
    while pre_n != s:
        ans.insert(0, pre_n[0])
        pre_n = pre[pre_n]
    ans.insert(0, start)

    return ans
