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
    for _ in range(T):
        tx.append(random.random()*dim)
        ty.append(random.random()*dim)
    return (tx, ty)

def generate_task_urgencies(T, low=0.5, high=2.0):
    tu = []
    for _ in range(T):
        # tu.append(random.random()*(high-low) + low)
        tu.append(high if random.random() > 0.5 else low)
    return tu

def euc_dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))

# Identifies which robot is closest to assign the task to
def which_region(x, y, mx, my, robot_speeds):
    BIG = 1000000
    mindist = BIG
    mini = -1
    for i in range(len(mx)):
        dist = euc_dist(mx[i], my[i], x, y)
        # dist = euc_dist(mx[i], my[i], x, y) / np.sqrt(robot_speeds[i])
        if dist < mindist:
            mindist = dist
            mini = i
    return mini

# Returns cutoff of last task to include
def get_cutoff(todo):
    # cutoff = len(todo) - 1 # complete all

    # cutoff = random.randrange(len(todo)) # uniform random

    val = random.random()*max(todo)[0] # uniform random over urgencies
    cutoff = -1
    while cutoff < len(todo)-1 and todo[cutoff+1][0] > val:
        cutoff += 1

    return cutoff

def simulate(mx, my, tasks, robot_speed, task_completion_time):
    total_wait_time = 0
    curtime = 0
    numdone = 0 # number of tasks completed
    tavailable = -1 # latest task available to be done
    curx = mx
    cury = my
    todo = [] # contains pairs of urgencies and task indices of todo tasks
    while (numdone < len(tasks)):
        # print("numdone:", numdone)
        # print("tavailable:", tavailable)
        if len(todo) == 0: # no tasks available
            if curx != mx or cury != my: # not at median, so move to median
                dist = euc_dist(curx, cury, mx, my)
                curtime += dist / robot_speed
                curx = mx
                cury = my
            else: # already at median, fast forward to next task
                tavailable += 1
                todo.append((tasks[tavailable][3], tavailable))
                curtime = tasks[tavailable][0]
        else:
            # sort todo by highest urgency first
            todo.sort(reverse=True)
            cutoff = get_cutoff(todo)

            # execute TSP on list of todo tasks up to and including cutoff index
            nodes = [(curx, cury)]
            for i in range(cutoff + 1):
                taski = todo[i][1]
                nodes.append((tasks[taski][1], tasks[taski][2]))

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
                t2 = tasks[todo[ind[i]-1][1]][0]
                x2 = tasks[todo[ind[i]-1][1]][1]
                y2 = tasks[todo[ind[i]-1][1]][2]
                u2 = tasks[todo[ind[i]-1][1]][3]
                curtime += euc_dist(x1, y1, x2, y2) / robot_speed # travel time
                curtime += task_completion_time # task completion time
                total_wait_time += (curtime - t2) * u2

            curx = nodes[ind[-1]][0]
            cury = nodes[ind[-1]][1]
            numdone += cutoff+1
            todo = todo[cutoff+1:]

        while (tavailable+1 < len(tasks) and tasks[tavailable+1][0] <= curtime): # check new available tasks
            tavailable += 1
            todo.append((tasks[tavailable][3], tavailable))
    return (total_wait_time, curtime)

if __name__ == "__main__":
    n = 4 # number of robots
    dim = 10 # length of side length of square world
    end_time = 1000
    lam = 0.7 # rate parameter lambda, the expected number of new tasks per timestep
    robot_speed = 1.0 # robot speed
    # robot_speeds = [robot_speed] * n # identical robot speeds
    robot_speeds = [0.5, 1.0, 1.5, 2.0] # different robot speeds
    task_completion_time = 1.0 # time to complete a task
    init_x = [2, 4, 5, 9]
    init_y = [6, 4, 3, 8]

    # init_x = [2, 4, 9]
    # init_y = [6, 4, 8]

    # init_x = [2, 4, 5, 9, 8]
    # init_y = [6, 4, 3, 8, 1]

    coverage_iters = 40
    # (mx, my) = get_coverage_medians(n, init_x, init_y, dim, coverage_iters, [1.0] * n) # equal voronoi regions
    (mx, my) = get_coverage_medians(n, init_x, init_y, dim, coverage_iters, robot_speeds) # adjust voronoi based on speeds

    ans = 0
    num_trials = 5
    for trial in range(num_trials):
        task_times = generate_task_times(end_time, lam)
        T = len(task_times) # number of tasks
        # print("Arrival times of", T, "tasks:", task_times)
        (tx, ty) = generate_task_locs(T, dim)
        tu = generate_task_urgencies(T, low=0.5, high=10.0)
        rtasks = [[] for _ in range(n)]
        for i in range(T):
            which_robot = which_region(tx[i], ty[i], mx, my, robot_speeds)
            rtasks[which_robot].append((task_times[i], tx[i], ty[i], tu[i]))
        # print("Tasks assigned to each robot:", rtasks)

        # simulate robots
        total_wait_time = 0
        output = ""
        for i in range(n):
            output += "Simulating robot " + str(i) + " with " + str(len(rtasks[i])) + " tasks\n"
            # print("Simulating robot", i, "with", len(rtasks[i]), "tasks")
            this_wait_time, end_time = simulate(mx[i], my[i], rtasks[i], robot_speeds[i], task_completion_time)
            total_wait_time += this_wait_time
            output += "Average wait time for robot " + str(i) + " tasks: " + str(this_wait_time/len(rtasks[i])) + "\n"
            # print("Average wait time for robot", i, "tasks:", this_wait_time/len(rtasks[i]))
        print(output)
        print("Average wait time for all tasks:", total_wait_time/T)
        ans += total_wait_time/T
    print("\nAverage over all trials:", ans/num_trials)