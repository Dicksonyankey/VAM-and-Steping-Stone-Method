import numpy as np
import pulp as plp
plp.LpSolverDefault.msg = 0


def vogel_approximation_method(costs, supply, demand):


    INF = 10 ** 3
    n, m = costs.shape
    ans = 0
    allocations = np.zeros((n, m))

    def find_diff(grid):
        row_diff = np.diff(np.sort(grid, axis=1), axis=1)[:, 0]
        col_diff = np.diff(np.sort(grid, axis=0), axis=0)[0, :]
        return row_diff, col_diff

    while max(supply) != 0 or max(demand) != 0:
        row, col = find_diff(costs)
        maxi1 = max(row)
        maxi2 = max(col)

        if maxi1 >= maxi2:
            ind = np.argmax(row)
            mini1 = np.min(costs[ind])
            ind2 = np.argmin(costs[ind])
            mini2 = min(supply[ind], demand[ind2])
            ans += mini2 * mini1
            supply[ind] -= mini2
            demand[ind2] -= mini2
            allocations[ind, ind2] += mini2
            if demand[ind2] == 0:
                costs[:, ind2] = INF
            else:
                costs[ind, :] = INF
        else:
            ind = np.argmax(col)
            mini1 = np.min(costs[:, ind])
            ind2 = np.argmin(costs[:, ind])
            mini2 = min(supply[ind2], demand[ind])
            ans += mini2 * mini1
            supply[ind2] -= mini2
            demand[ind] -= mini2
            allocations[ind2, ind] += mini2
            if demand[ind] == 0:
                costs[:, ind] = INF
            else:
                costs[ind2, :] = INF

    return ans, allocations


def solve_stepping_stone(supply, demand, costs):
    # Define the LP problem and decision variables
    model = plp.LpProblem("steppingstone", plp.LpMinimize)
    allocation = {}
    for i in range(len(supply)):
        for j in range(len(demand)):
            allocation[(i, j)] = plp.LpVariable(f"x{i}{j}", lowBound=0)

    # Define the objective function
    model += plp.lpSum(costs[i][j] * allocation[(i, j)] for i in range(len(supply)) for j in range(len(demand)))

    # Define the supply and demand constraints
    for i in range(len(supply)):
        model += plp.lpSum(allocation[(i, j)] for j in range(len(demand))) == supply[i]
    for j in range(len(demand)):
        model += plp.lpSum(allocation[(i, j)] for i in range(len(supply))) == demand[j]

    # Solve the LP problem and print the results
    status = model.solve()
    if status == plp.LpStatusOptimal:
        return plp.value(model.objective), {(i, j): plp.value(allocation[(i, j)]) for i in range(len(supply)) for j in range(len(demand))}
    else:
        return None, None


# Define the cost matrix and supply/demand arrays
costs = np.array([[24, 28, 32], [14, 32, 24], [0, 0, 0]])
supply = np.array([4000, 6000, 600])
demand = np.array([4400, 3400, 2800])

obj_value, allocation = solve_stepping_stone(supply, demand, costs)
print(f"Objective Function Value: ${obj_value}")
print("OptimalSolution:")
for i in range(len(supply)):
    for j in range(len(demand)):
        print(f"x{i}{j} = {allocation[(i, j)]}")

total_cost, allocations = vogel_approximation_method(costs, supply, demand)
print(f"The initial alloctions for the VAM is \n{allocations}")
print(f"The total initial cost for the VAM is ${total_cost}")




