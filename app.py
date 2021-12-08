import pandas as pd
import numpy as np 
from pulp import *
stock = 3800
# d = {"demand": [45, 70, 50, 150, 50, 80, 55, 100], "length": [60, 80, 160, 260, 360, 410, 520, 600]}
d = {"demand": [245, 55, 10, 158, 16], "length": [4, 7, 123, 2001, 2122]}
# d = {"demand": np.random.randint(5,50, size= 10), "length": np.random.randint(50,1500, size= 10)}

parts = pd.DataFrame(data = d)
source_length = len(parts.demand)
parts.shape[0] 
m = 0
rem = []
for i in range(parts.shape[0]):
    resources = [[np.inf for x in range(2)] for y in range(4)]
    # i kesim sonrası
    resources[0][0] = stock % parts.length[i]
    resources[0][1] = np.zeros(source_length)
    resources[0][1][i] = int(stock / parts.length[i])
    # i-1 kesim sonrası
    resources[1][0] = resources[0][0] + parts.length[i]
    resources[1][1] = resources[0][1].copy()
    resources[1][1][i] = resources[0][1][i] - 1 
    for m in range(parts.shape[0]):
        if m != i:
            # i kesiminden sonra m kesim
            if resources[0][0] >= parts.length[m]:
                resources[2][0]  = resources[0][0] % parts.length[m]
                resources[2][1] = resources[0][1].copy()
                resources[2][1][m] = int(resources[0][0] / parts.length[m])
            # i-1 kesim sonrası m kesimi
            if resources[1][0] >= parts.length[m]:
                resources[3][0] = resources[1][0] % parts.length[m]
                resources[3][1] = resources[1][1].copy()
                resources[3][1][m] = int(resources[1][0] / parts.length[m])
            min_remainder = min([row[0] for row in resources])
            index = np.argmin([row[0] for row in resources])
            data = {"pattern": resources[index][1], "remainder": min_remainder}
            rem.append(data)
for i in range(parts.shape[0]-1,0,-1):
    resources = [[np.inf for x in range(2)] for y in range(4)]
    # i kesim sonrası
    resources[0][0] = stock % parts.length[i]
    resources[0][1] = np.zeros(source_length)
    resources[0][1][i] = int(stock / parts.length[i])
    # i-1 kesim sonrası
    resources[1][0] = resources[0][0] + parts.length[i]
    resources[1][1] = resources[0][1].copy()
    resources[1][1][i] = resources[0][1][i] - 1 
    for m in range(parts.shape[0]-1,0,-1):
        if m != i:
            # i kesiminden sonra m kesim
            if resources[0][0] >= parts.length[m]:
                resources[2][0]  = resources[0][0] % parts.length[m]
                resources[2][1] = resources[0][1].copy()
                resources[2][1][m] = int(resources[0][0] / parts.length[m])
            # i-1 kesim sonrası m kesimi
            if resources[1][0] >= parts.length[m]:
                resources[3][0] = resources[1][0] % parts.length[m]
                resources[3][1] = resources[1][1].copy()
                resources[3][1][m] = int(resources[1][0] / parts.length[m])
            min_remainder = min([row[0] for row in resources])
            index = np.argmin([row[0] for row in resources])
            data = {"pattern": resources[index][1], "remainder": min_remainder}
            rem.append(data)
rem = pd.DataFrame.from_records([s for s in rem])
rem = rem[~rem.pattern.duplicated(keep='first')].reset_index(drop= True)




from pulp import *

prob = LpProblem("One Dimensional Cutting Stock Problem ", LpMinimize)
variables = []
for i in range(len(rem.remainder)):
    variables.append(LpVariable(str(i), 0, None, LpInteger))

prob += lpSum([rem.loc[i, "remainder"]* variables[i] for i in range(0,len(variables))]), "total waste"

for i in range(parts.shape[0]):
    prob += lpSum([rem.loc[j, "pattern"][i]* variables[j] for j in range(0, len(variables))]) >= parts.loc[i, "demand"]
    prob += lpSum([rem.loc[j, "pattern"][i]* variables[j] for j in range(0, len(variables))]) <= 1.10*(parts.loc[i, "demand"])

prob.solve()
print("Status:", LpStatus[prob.status])
cutting_model = []
for v in prob.variables():
    print(v.name, "=", v.varValue)
    if v.varValue != 0:
        data = {"index": int(v.name), "amount": int(v.varValue)}
        cutting_model.append(data)
print("Total waste = ", value(prob.objective))

total_pieces = np.zeros(len(parts.demand))
for item in cutting_model:
    total_pieces = np.sum([total_pieces, np.array(rem.loc[item["index"],"pattern"])*item["amount"]], axis=0)
extras = np.sum([total_pieces, np.array(parts["demand"])*-1], axis = 0)

while extras.any():
    cutting_patterns = []
    for item in cutting_model:     
        objective = rem.iloc[item["index"]]
        total_sub = 0
        cutted_extras = []
        for i in range(len(parts.demand)):
            sub_amount = min(objective.pattern[i], extras[i])
            total_sub += sub_amount*parts.length[i]
            cutted_extras.append(sub_amount)
        total_sub += objective.remainder 
        if np.array(cutted_extras).any():
            cutting_patterns.append([item["index"], cutted_extras, total_sub])
    deleting_index = np.argmax([row[2] for row in cutting_patterns])
    delete_from = cutting_patterns[deleting_index][0]
    subtract_this = cutting_patterns[deleting_index][1]
    new_rem = cutting_patterns[deleting_index][2]
    new_pattern = np.sum([rem.iloc[delete_from]["pattern"], np.array(subtract_this)*-1], axis = 0)
    data = {"pattern": list(new_pattern), "remainder": new_rem}
    rem = rem.append(data, ignore_index = True)
    data = {"index": len(rem.remainder)-1, "amount": 1}
    cutting_model.append(data)
    cutting_model[deleting_index]["amount"] -= 1
    extras = np.sum([extras, np.array(subtract_this)*-1], axis = 0)

total_waste = np.sum([rem.iloc[cutting_model[i]["index"]].remainder*cutting_model[i]["amount"] for i in range(len(cutting_model))])
total_cut = np.sum([parts.iloc[i]["demand"]*parts.iloc[i]["length"] for i in range(len(parts))])
waste_percentage = (total_waste / total_cut) * 100
print(waste_percentage)