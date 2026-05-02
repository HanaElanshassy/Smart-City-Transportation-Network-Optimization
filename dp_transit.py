routes = ["B1","B2","B3","B4","B5","B6","B7","B8","B9","B10"]

demand = [35000,42000,28000,31000,25000,33000,21000,17000,39000,28000]

buses = 203
capacity_per_bus = 1000
n = len(routes)
dp = [[0]*(buses+1) for _ in range(n+1)]

for i in range(1, n+1):
    for j in range(buses+1):
        for k in range(j+1):
            covered = min(k*capacity_per_bus, demand[i-1])
            dp[i][j] = max(dp[i][j],
                           dp[i-1][j-k] + covered)

print("Max passengers served:", dp[n][buses])
