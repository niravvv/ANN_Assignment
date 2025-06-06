import numpy as np
import random
import csv
from synthetic_data import tasks, employees  # Make sure these are defined

#  Penalty Weights 
ALPHA = 0.20
BETA = 0.20
DELTA = 0.20
GAMMA = 0.20
SIGMA = 0.20  # Not used

NUM_TASKS = len(tasks)
NUM_EMPLOYEES = len(employees)

#  Fitness Function 
def f(assignment):
    overloadPenalty = 0
    skillMismatchPenalty = 0
    difficultyPenalty = 0
    deadlinePenalty = 0

    employeeWorkloads = {i: [] for i in range(NUM_EMPLOYEES)}

    for taskIdx, employeeIndex in enumerate(assignment):
        task = tasks[taskIdx]
        employee = employees[employeeIndex]

        taskTime = task["Estimated Time (hrs)"]
        taskDifficulty = task["Difficulty"]
        taskDeadline = task["Deadline (hrs from now)"]
        taskSkills = task["Required Skill"].split()

        employeeHours = employee["Available Hours"]
        employeeSkills = employee["Skills"].replace(" ", "").split(",")
        employeeSkillLevel = employee["Skill Level"]

        employeeWorkloads[employeeIndex].append((taskIdx, taskTime, taskDeadline))

        if not any(skill in employeeSkills for skill in taskSkills):
            skillMismatchPenalty += 1

        if employeeSkillLevel < taskDifficulty:
            difficultyPenalty += 1

    for employeeIndex, taskList in employeeWorkloads.items():
        totalTime = sum(t[1] for t in taskList)
        if totalTime > employees[employeeIndex]["Available Hours"]:
            overloadPenalty += (totalTime - employees[employeeIndex]["Available Hours"])

        taskList.sort(key=lambda t: t[1])  # Sort by task duration
        finishTime = 0
        for _, taskTime, taskDeadline in taskList:
            finishTime += taskTime
            delay = max(0, finishTime - taskDeadline)
            deadlinePenalty += delay

    totalCost = (
        ALPHA * overloadPenalty +
        BETA * skillMismatchPenalty +
        DELTA * difficultyPenalty +
        GAMMA * deadlinePenalty
    )

    return -totalCost  # Return negative penalty for PSO to maximize fitness

#  PSO Algorithm 
def particleSwarmOptimisation(f, n, N, T_max, w, c1, c2, x_min, x_max):
    x = [[0 for _ in range(n)] for _ in range(N)]
    v = [[0.0 for _ in range(n)] for _ in range(N)]
    pBest = [[0 for _ in range(n)] for _ in range(N)]
    fitness_pBest = [0] * N

    for i in range(N):
        x[i] = [random.randint(x_min[d], x_max[d]) for d in range(n)]
        for d in range(n):
            v[i][d] = random.uniform(-(x_max[d]-x_min[d]), (x_max[d]-x_min[d]))
        pBest[i] = x[i][:]
        fitness_pBest[i] = f(pBest[i])

    best_index = max(range(N), key=lambda i: fitness_pBest[i])
    gBest = pBest[best_index][:]
    fitness_gBest = fitness_pBest[best_index]

    for iteration in range(T_max):
        for i in range(N):
            for d in range(n):
                r1 = random.uniform(0, 1)
                r2 = random.uniform(0, 1)
                v[i][d] = (
                    w * v[i][d] +
                    c1 * r1 * (pBest[i][d] - x[i][d]) +
                    c2 * r2 * (gBest[d] - x[i][d])
                )

            for d in range(n):
                candidate = int(round(x[i][d] + v[i][d]))
                candidate = max(x_min[d], min(candidate, x_max[d]))

                if candidate in x[i][:d]:
                    available = [val for val in range(x_min[d], x_max[d]+1) if val not in x[i][:d]]
                    if available:
                        candidate = random.choice(available)
                x[i][d] = candidate

            fitness_current = f(x[i])
            if fitness_current > fitness_pBest[i]:
                pBest[i] = x[i][:]
                fitness_pBest[i] = fitness_current

        best_index = max(range(N), key=lambda i: fitness_pBest[i])
        if fitness_pBest[best_index] > fitness_gBest:
            gBest = pBest[best_index][:]
            fitness_gBest = fitness_pBest[best_index]

    return gBest, fitness_gBest

#  PSO Hyperparameters 
dims = 10
pop_size = 20
max_iter = 5001
w = 0.5
c1 = 1.5
c2 = 1.5
x_min = [0] * dims
x_max = [NUM_EMPLOYEES - 1] * dims

#  Generate 100 Mappings 
records = []

for _ in range(100):
    best_solution, best_fitness = particleSwarmOptimisation(
        f, dims, pop_size, max_iter, w, c1, c2, x_min, x_max
    )
    row = best_solution + [best_fitness]
    records.append(row)

#  Export to CSV 
with open("mappings_with_penalty_pso.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([f"T{i+1}" for i in range(dims)] + ["Penalty"])
    writer.writerows(records)

print("CSV file 'mappings_with_penalty_pso.csv' has been generated.")
