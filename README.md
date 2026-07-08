# Industrial Machine Failure Simulation

A Python simulation project for modeling industrial machine wear, sensor drift, and failure events over time. The project generates synthetic machine data that can be used for industrial engineering analysis, predictive maintenance experiments, and future machine learning models.

## Overview

This project simulates a factory with multiple machines running over time. Each machine has its own wear rate, quality level, operating sensor values, and failure conditions. As the simulation runs, each machine gradually loses health, sensor readings drift, and occasional sudden faults can happen.

The generated data is saved to a CSV file so it can be analyzed later or used as training data for predictive maintenance models.

## Features

- Simulates multiple industrial machines at once
- Tracks machine health over time
- Generates realistic sensor readings:
  - temperature
  - pressure
  - vibration
  - RPM
  - electrical current
- Models gradual wear and random sudden faults
- Marks machines as failed when critical thresholds are reached
- Saves time-series machine data to `data/machine_data.csv`
- Prints factory health summaries during the simulation

## Project Structure

```text
ie proj/
├── main.py
├── data/
│   └── machine_data.csv
├── simulator/
│   ├── machine.py
│   ├── factory.py
│   ├── simulator.py
│   └── __init__.py
├── ml/
│   └── train_model.py
├── models/
└── README.md
```

## How It Works

### Machine Simulation

Each `Machine` starts with:

- `health = 100`
- randomized maintenance cost
- randomized production value
- randomized wear multiplier
- randomized machine quality
- starting sensor values for temperature, pressure, vibration, RPM, and current

Every simulated minute, the machine:

1. Gets older
2. Loses a small amount of health
3. Updates its sensor readings
4. Becomes more unstable as health drops
5. Has a small chance of a sudden fault
6. Fails if health or sensor values cross critical thresholds

### Factory Simulation

The `Factory` class creates a group of machines and updates all of them each minute. It also writes each machine's state to `data/machine_data.csv`.

The current simulation in `main.py` creates 25 machines and runs them for 500 simulated minutes.

## Dataset

The generated CSV contains one row per machine per minute.

Columns:

```text
MachineID
Minute
Health
Temperature
Pressure
Vibration
RPM
Current
Failed
```

Example row:

```text
1,0,99.95,64.8,101.5,0.18,1781,8.98,0
```

`Failed` is stored as:

- `0` = machine is still running
- `1` = machine has failed

## Requirements

This project uses Python and standard libraries for the simulator. The analysis script in `ml/train_model.py` uses pandas.

Install dependencies with:

```powershell
pip install pandas
```

If you are using the included virtual environment in PyCharm, install packages inside that environment.

## Run The Simulation

From the project folder:

```powershell
cd "C:\Users\jaymo\PycharmProjects\ie proj"
python main.py
```

Or, using the PyCharm virtual environment:

```powershell
.\.venv\Scripts\python.exe main.py
```

The script prints a summary every 50 minutes:

```text
Minute 0
Healthy Machines : 25
Warning Machines : 0
Failed Machines  : 0
```

The generated data is written to:

```text
data/machine_data.csv
```

## Machine Failure Logic

A machine fails if any of these conditions become true:

- health drops to 5 or below
- temperature reaches 110 or higher
- vibration reaches 1.0 or higher
- current reaches 20 or higher

As health gets worse, temperature, vibration, and current become more unstable, making failure more likely.

## ML / Analysis Direction

The `ml/train_model.py` file is currently a starter script for loading the generated CSV with pandas. A future version could train a model to predict whether a machine is likely to fail based on sensor readings.

Possible next steps:

- clean and label the CSV data
- train a classification model to predict `Failed`
- predict future failure risk before a machine actually fails
- compare maintenance cost vs. production value
- add charts for machine health and sensor trends
- build a dashboard for factory status

## Notes

This is a simulation, not real factory data. The sensor behavior is intentionally simplified, but it creates a useful dataset for learning about industrial monitoring, predictive maintenance, and machine learning workflows.
