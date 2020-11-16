import numpy as np
from coverage import get_coverage_medians
import random
import math
import tsp

def generate_task_times(end_time, lam):
    cur_time = 0
    task_times = []
    while (cur_time < end_time):
        p = random.random()
        inter_arrival_time = -math.log(1.0 - p)/lam
        cur_time += inter_arrival_time
        task_times.append(cur_time)
    return task_times[0:-1]

def generate_task_locs(T, dim):
    tx = []
    ty = []
    for i in range(T):
        tx.append(random.random()*dim)
        ty.append(random.random()*dim)
    return (tx, ty)

def euc_dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))

def which_region(x, y, mx, my):
    BIG = 1000000
    mindist = BIG
    mini = -1
    for i in range(len(mx)):
        dist = euc_dist(mx[i], my[i], x, y)
        if dist < mindist:
            mindist = dist
            mini = i
    return mini

def simulate(mx, my, tasks):
    total_wait_time = 0
    curtime = 0
    tdone = -1 # most recent task done
    tavailable = -1 # latest task available to be done
    curx = mx
    cury = my
    while (tdone + 1 < len(tasks)):
        print("tdone:", tdone)
        print("tavailable:", tavailable)
        if tavailable == tdone: # no tasks available
            if curx != mx or cury != my: # not at median, so move to median
                dist = euc_dist(curx, cury, mx, my)
                curtime += dist / 1.0
                curx = mx
                cury = my
            else: # already at median, fast forward to next task
                tavailable += 1
                curtime = tasks[tavailable][0]
        else:
            # execute TSP on tdone+1 through tavailable
            nodes = [(curx, cury)]
            for i in range(tdone+1, tavailable+1):
                nodes.append((tasks[i][1], tasks[i][2]))
            # nodes = [(tasks[i][1], tasks[i][2]) for i in range(tdone+1, tavailable+1)]
            # nodes.append((curx, cury))
            (dist, ind) = tsp.tsp3(nodes)
            assert ind[0] == 0
            d1 = euc_dist(nodes[0][0], nodes[0][1], nodes[ind[1]][0], nodes[ind[1]][1])
            d2 = euc_dist(nodes[0][0], nodes[0][1], nodes[ind[-1]][0], nodes[ind[-1]][1])
            if d2 < d1: # want to visit last node first
                ind.reverse()
                del ind[-1]
                ind.insert(0, 0)

            for i in range(1, len(ind)):
                x1 = nodes[ind[i-1]][0]
                y1 = nodes[ind[i-1]][1]
                t2 = tasks[ind[i] + tdone][0]
                x2 = tasks[ind[i] + tdone][1]
                y2 = tasks[ind[i] + tdone][2]
                curtime += euc_dist(x1, y1, x2, y2)/1.0 # travel time
                curtime += 1.0 # task completion time
                total_wait_time += curtime - t2

            # TSPtime = dist/1.0 + (len(ind)-1) * 1.0
            # curtime += TSPtime
            curx = nodes[ind[-1]][0]
            cury = nodes[ind[-1]][1]
            tdone = tavailable
        while (tavailable+1 < len(tasks) and tasks[tavailable+1][0] <= curtime): # check new available tasks
            tavailable += 1
    return (total_wait_time, curtime)

if __name__ == "__main__":
    n = 4 # number of robots
    dim = 10 # length of side length of square world
    end_time = 100
    lam = 0.2 # new task every 5 steps
    init_x = [2, 4, 5, 9]
    init_y = [6, 4, 3, 8]

    coverage_iters = 10
    (mx, my) = get_coverage_medians(n, init_x, init_y, dim, coverage_iters)

    task_times = generate_task_times(end_time, lam)
    T = len(task_times) # number of tasks
    print("Arrival times of", T, "tasks:", task_times)
    (tx, ty) = generate_task_locs(T, dim)
    rtasks = [[] for _ in range(n)]
    for i in range(T):
        which_robot = which_region(tx[i], ty[i], mx, my)
        rtasks[which_robot].append((task_times[i], tx[i], ty[i]))
    print("Tasks assigned to each robot:", rtasks)

    # simulate robots
    total_wait_time = 0
    for i in range(n):
        print("Simulating robot", i, "with", len(rtasks[i]), "tasks")
        this_wait_time, end_time = simulate(mx[i], my[i], rtasks[i])
        total_wait_time += this_wait_time
        print("Average wait time for robot", i, "tasks:", this_wait_time/len(rtasks[i]))
    print("Average wait time for all tasks:", total_wait_time/T)