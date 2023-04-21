features = ["year", "type", "odometer", "manufacturer", "fuel", "title_status", "transmission", "state"]
dt = {}
dt["year"] = 8
dt["type"] = 7
dt["odometer"] = 6
dt["manufacturer"] = 5
dt["fuel"] = 4
dt["title_status"] = 3
dt["transmission"] = 2
dt["state"] = 1

rf = {}
rf["year"] = 8
rf["odometer"] = 7
rf["type"] = 6
rf["manufacturer"] = 5
rf["fuel"] = 4
rf["state"] = 3
rf["transmission"] = 2
rf["title_status"] = 1

xgb = {}
xgb["odometer"] = 8
xgb["manufacturer"] = 7
xgb["type"] = 6
xgb["year"] = 5
xgb["state"] = 4
xgb["fuel"] = 3
xgb["transmission"] = 2
xgb["title_status"] = 1

nn = {}
nn["manufacturer"] = 8
nn["year"] = 7
nn["title_status"] = 6
nn["transmission"] = 5
nn["type"] = 4
nn["fuel"] = 3
nn["state"] = 2
nn["odometer"] = 1

all = {}
for key in dt:
    all[key] = dt[key] + rf[key] + xgb[key] + nn[key]

# sort by value
all = sorted(all.items(), key=lambda x: x[1], reverse=True)
print(all)
