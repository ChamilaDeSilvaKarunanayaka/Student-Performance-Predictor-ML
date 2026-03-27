import pandas as pd
import random

data = []

for i in range(1000):   #  change 500 → 1000
    hours = random.randint(1, 10)
    attendance = random.randint(30, 100)

    # better logic
    if hours >= 6 and attendance >= 70:
        result = "pass"
    elif hours >= 4 and attendance >= 60:
        result = "pass"
    else:
        result = "fail"

    data.append([hours, attendance, result])

df = pd.DataFrame(data, columns=["hours", "attendance", "result"])

df.to_csv("data/student_data.csv", index=False)

print("Dataset created ✅")