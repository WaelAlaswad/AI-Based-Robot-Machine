import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import seaborn as sns


while True:
    # Reading the data from Files
    startTime = time.time()
    df = pd.read_csv("Vision_Inspection_Result.csv")
    speed = pd.read_csv("Machine Speed.csv")
    df.head()

    Machine_Speed = open("Machine Speed.csv", "r")
    Machine_Speed = Machine_Speed.readlines()
    speed = Machine_Speed[-1][:2]

    Machine_Status_File = open("Machine Status.csv", "r")
    Machine_Status_File = Machine_Status_File.readlines()
    Machine_Status = Machine_Status_File[-1][0]

    Machine_Running_Hours_File = open("Running Hours.csv", "r")
    Machine_Running_Hours_File = Machine_Running_Hours_File.readlines()
    Running_Hours = Machine_Running_Hours_File[-1]

    Production_Remaining_File = open("Production Remaining.csv", "r")
    Production_Remaining_File = Production_Remaining_File.readlines()
    Production_Remaining = Production_Remaining_File[-1]

    Production_Hours_File = open("Production Hours.csv", "r")
    Production_Hours_File = Production_Hours_File.readlines()
    Production_Hours = Production_Hours_File[-1]

    Maintenance_Hours_File = pd.read_csv("Maintenance Hours.csv")
    Maintenance_Hours_File.drop_duplicates()
    Maintenance_Hours_File.dropna()
    Maintenance_Hours = Maintenance_Hours_File.Maintenance_Hours.sum()
    

    year = pd.read_csv("Year.csv")
    month = pd.read_csv("Month.csv")
    maintenance_hours = pd.read_csv("Maintenance Hours.csv")
    BreakdownTable = pd.DataFrame(
        {
            "Hours": maintenance_hours["Maintenance_Hours"],
            "Month": month["Month"],
            "Year": year["Year"],
        }
    )
    #BreakdownTable = BreakdownTable.dropna()
    #BreakdownTable["Month"] = BreakdownTable["Month"].drop_duplicates()
    #BreakdownTable = BreakdownTable.dropna()

    df.head()
    df.Inspection.unique()
    df.Inspection.value_counts()

    goodCounter = df.Inspection.value_counts()["Accept"]
    goodCounter
    badCounter = df.Inspection.value_counts()["Reject"]
    badCounter
    totalCounter = goodCounter + badCounter
    totalCounter
    goodCountenp = np.array([goodCounter])
    badCounternp = np.array([badCounter])

    totalCounternp = np.array([totalCounter])

    GoodPartProductivity = goodCounter / totalCounter
    badPartProducctivity = badCounter / totalCounter

    GoodPartProductivityPer = GoodPartProductivity * 100
    badPartProductivityPer = badPartProducctivity * 100

    goodCounter = df.Inspection.value_counts()["Accept"]
    badCounter = df.Inspection.value_counts()["Reject"]
    totalCounter = goodCounter + badCounter

    GoodPartProductivity = goodCounter / totalCounter
    badPartProducctivity = badCounter / totalCounter

    GoodPartProductivityPer = GoodPartProductivity * 100
    badPartProductivityPer = badPartProducctivity * 100

    #

    ProductivityPer = pd.DataFrame(
        {
            "Good Part Productivity %": GoodPartProductivityPer,
            "Bad Part Productivity %": badPartProductivityPer,
        },
        index=(0,),
    )

    # Measures Calculation
    CycleTime = [int(Machine_Speed[-1][:2])]
    Running_Hours = [int(Machine_Running_Hours_File[-1])]
    Status = [Machine_Status]
    counter = [goodCounter, badCounter]
    Utilization = [float(Machine_Running_Hours_File[-1]) / 8000]
    Production_Remaing = [int(Production_Remaining_File[-1]) * (-1)]
    Expected_Time_Complete = [float(Production_Remaing[0]) * int(24) / int(220000)]
    OEE = [
        float(Utilization[0])
        * (float(CycleTime[0]) / int(18))
        * float(GoodPartProductivityPer)
    ]
    Maintenance_Hours = [float(Maintenance_Hours)]
    Production_Hours = [float(Production_Hours)]
    Machine_Breakdown = [
        float(Maintenance_Hours[0]) / float(Production_Hours_File[-1]) * 100
    ]

    x = np.array([GoodPartProductivityPer, badPartProductivityPer])
    myExplode = [0.0, 0.2]

    # Plot The Measures
    # Good and Bad counters
    fig = plt.figure(figsize=(5, 4))
    fig.patch.set_facecolor("#090235")
    plt.rcParams["text.color"] = "white"
    a = plt.barh(["Good", "Bad"], counter, color=["g", "r"], height=0.2, align="center")
    plt.title("Good / Bad Product Counter", fontsize=16)
    plt.xticks(color="white")
    plt.yticks(color="white")
    for i1, v1 in enumerate(counter):
        plt.text(v1 / 1000, i1, str(v1))

    plt.savefig("static/Report1.jpg")
    plt.close()

    # Quality%
    fig = plt.figure(figsize=(5, 4))
    fig.patch.set_facecolor("#090235")
    plt.rcParams["text.color"] = "white"
    b = plt.pie(
        x,
        labels=["Good", "bad"],
        autopct="%.2f%%",
        explode=myExplode,
        colors=["g", "r"],
        textprops={"fontsize": 15},
    )
    plt.title("Quality %", fontsize=18)

    plt.savefig("static/Report2.jpg")
    plt.close()

    # Cycle Time
    fig = plt.figure(figsize=(4, 4))
    fig.patch.set_facecolor("#090235")
    plt.rcParams["text.color"] = "white"
    c = plt.pie(
        CycleTime,
        colors=["#090235"],
        autopct=str(f"{CycleTime[0]:.2f}"),
        pctdistance=0.0,
        radius=1,
        textprops={"fontsize": 48},
    )
    plt.title("Cycle Time", fontsize=18)
    plt.savefig("static/Report3.jpg")
    plt.close()

    # Machine Status
    fig = plt.figure(figsize=(2, 4))
    fig.patch.set_facecolor("#090235")
    plt.rcParams["text.color"] = "white"

    if Status == ["1"]:
        plt.pie(
            Status,
            colors=["g"],
            textprops={"fontsize": 15},
            radius=1,
            autopct=str("Running"),
            pctdistance=0.0,
        )
        plt.title("Machine Status", fontsize=18)

    elif Status == ["2"]:
        plt.pie(
            Status,
            colors=["r"],
            textprops={"fontsize": 15},
            radius=1,
            autopct=str("Stopped"),
            pctdistance=0.0,
        )
        plt.title("Machine Status", fontsize=18)
    plt.savefig("static/Report4.jpg")
    plt.close()

    # Running Hours
    fig = plt.figure(figsize=(4, 4))
    fig.patch.set_facecolor("#090235")
    plt.rcParams["text.color"] = "white"

    plt.pie(
        Running_Hours,
        colors=["#090235"],
        textprops={"fontsize": 48},
        radius=2,
        autopct=str(f"{Running_Hours[0]:.2f}"),
        pctdistance=0.0,
    )
    plt.title("Running Hours", fontsize=18)
    plt.savefig("static/Report5.jpg")
    plt.close()

    # Utilization %
    fig = plt.figure(figsize=(4, 4))
    fig.patch.set_facecolor("#090235")
    plt.rcParams["text.color"] = "white"

    plt.pie(
        Utilization,
        colors=["#090235"],
        textprops={"fontsize": 48},
        radius=2,
        autopct=str(f"{Utilization[0]:.2f}"),
        pctdistance=0.0,
    )
    plt.title("Yearly Utilization %", fontsize=18)
    plt.savefig("static/Report6.jpg")
    plt.close()

    # Production Remaining Quantity
    fig = plt.figure(figsize=(4, 5))
    fig.patch.set_facecolor("#090235")
    plt.rcParams["text.color"] = "white"

    plt.pie(
        Production_Remaing,
        colors=["#090235"],
        textprops={"fontsize": 48},
        radius=2,
        autopct=str(f"{Production_Remaing[0]:.2f}"),
        pctdistance=0.0,
    )
    plt.title("Production Engaged \nPCS", fontsize=18)
    plt.savefig("static/Report7.jpg")
    plt.close()

    # Expected Time To Complete The Order
    fig = plt.figure(figsize=(4, 5))
    fig.patch.set_facecolor("#090235")
    plt.rcParams["text.color"] = "white"

    plt.pie(
        Expected_Time_Complete,
        colors=["#090235"],
        textprops={"fontsize": 48},
        radius=2,
        autopct=str(f"{Expected_Time_Complete[0]:.2f}"),
        pctdistance=0.0,
    )
    plt.title("Expected Hours \nTo Complete", fontsize=18)
    plt.savefig("static/Report8.jpg")
    plt.close()

    # OEE %
    fig = plt.figure(figsize=(4, 5))
    fig.patch.set_facecolor("#090235")
    plt.rcParams["text.color"] = "white"

    plt.pie(
        OEE,
        colors=["#090235"],
        textprops={"fontsize": 48},
        radius=2,
        autopct=str(f"{OEE[0]:.2f}"),
        pctdistance=0.0,
    )
    plt.title("Overall Equipement \nEfficiency %", fontsize=18)
    plt.savefig("static/Report9.jpg")
    plt.close()

    # Production Hours
    fig = plt.figure(figsize=(4, 5))
    fig.patch.set_facecolor("#090235")
    plt.rcParams["text.color"] = "white"

    plt.pie(
        Production_Hours,
        colors=["#090235"],
        textprops={"fontsize": 48},
        radius=2,
        autopct=str(f"{Production_Hours[0]:.2f}"),
        pctdistance=0.0,
    )
    plt.title("Production Time \nHours", fontsize=18)
    plt.savefig("static/Report10.jpg")
    plt.close()

    # Maintenance Hours
    fig = plt.figure(figsize=(4, 5))
    fig.patch.set_facecolor("#090235")
    plt.rcParams["text.color"] = "white"

    plt.pie(
        Maintenance_Hours,
        colors=["#090235"],
        textprops={"fontsize": 48},
        radius=2,
        autopct=str(f"{Maintenance_Hours[0]:.2f}"),
        pctdistance=0.0,
    )
    plt.title("Maintenance Time \nHours", fontsize=18)
    plt.savefig("static/Report11.jpg")
    plt.close()

    # Machine Breakdown %
    fig = plt.figure(figsize=(4, 5))
    fig.patch.set_facecolor("#090235")
    plt.rcParams["text.color"] = "white"

    plt.pie(
        Machine_Breakdown,
        colors=["#090235"],
        textprops={"fontsize": 48},
        radius=2,
        autopct=str(f"{Machine_Breakdown[0]:.2f}"),
        pctdistance=0.0,
    )
    plt.title("Machine Breakdown %", fontsize=18)
    plt.savefig("static/Report12.jpg")

    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor("#090235")
    sns.barplot(x=BreakdownTable.Month, y=BreakdownTable.Hours, data=BreakdownTable,hue= BreakdownTable.Year)
    plt.title("Breakdown Hours", fontsize=18, color="white")
    plt.xlabel("Month", color="white",fontsize=18)
    plt.ylabel("Hours", color="white",fontsize=18)
    plt.xticks(color="white")
    plt.yticks(color="white")
    #plt.legend(fontsize=18)
    #plt.legend(labels=["2023","2024"])
    plt.savefig("static/Report13.jpg")

    plt.close("all")
    #matplotlib.pyplot.close()
    
