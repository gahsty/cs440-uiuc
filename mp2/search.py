# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze
import queue
from collections import deque
from heapq import heappop, heappush


def search(maze, searchMethod):
    return {
        "bfs": bfs,
    }.get(searchMethod, [])(maze)


def bfs(maze, ispart1=False):
    # Write your code here
    """
    This function returns optimal path in a list, which contains start and objective.
    If no path found, return None. 

    Args:
        maze: Maze instance from maze.py
        ispart1: pass this variable when you use functions such as getNeighbors and isObjective. DO NOT MODIFY THIS
    """
    # 开始与结束点
    start = maze.getStart()
    end = maze.getObjectives()
    s = set()
    from collections import deque
    d = deque()
    d.append(start)
    s.add(start)
    pre={}
    while d:
        cur_node = d.popleft()
        if cur_node in end:
            break
        o = maze.getNeighbors(cur_node[0],cur_node[1],cur_node[2],ispart1)

        for nb in o:

            if nb in s:
                continue
            d.append(nb)
            s.add(nb)
            pre [nb] =cur_node
    ans = []
    if cur_node not in end:
        return None
    prenode = cur_node
    while prenode != start:
        ans.insert(0, prenode)
        prenode = pre[prenode]
    ans.insert(0, start)
    return ans


