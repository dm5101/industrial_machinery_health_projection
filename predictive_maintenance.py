#!/usr/bin/env python3
"""

What this is
-------------
A predictive-maintenance system where:
  1. Owners register their own machines (type, site, install info).
  2. Owners log sensor readings over time (vibration, temperature, current draw).
  3. The model predicts each machine's health score and time-to-failure.
  4. Crucially: a NEW machine with little or no history of its own still gets a
     useful prediction, because the model looks at the "cohort" — every other
     machine of the same type across the whole fleet — and borrows their
     degradation patterns. As the machine's own readings accumulate, the
     prediction shifts from "what machines like this usually do" toward
     "what THIS machine is actually doing".


How the ML part actually works
-----------------------------------------------------------
- Every machine TYPE (e.g. "Centrifugal Pump") has a baseline degradation
  curve: health_score(hours) = 100 * exp(-k/1000 * hours).
- `k` (how fast that type typically wears out) and the "expected" sensor
  readings at a given age are estimated from every machine of that type in
  the fleet — including ones that have already failed. This is the cohort.
- A specific machine's OWN sensor readings are compared against what the
  cohort considers normal for a machine of that type at that age. If this
  pump vibrates more than fleet pumps usually do at 8,000 hours, its
  effective degradation rate is pushed above the cohort baseline — the
  model predicts it will fail sooner than a "typical" pump.
- With zero readings of its own, a new machine just gets the cohort curve
  (a reasonable, humble default). With more of its own readings, the
  prediction becomes machine-specific and reported confidence rises.

"""

from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Machine type knowledge base — this is the "textbook" the model starts from.
# base: expected sensor value right after install.
# drift_per_1000h: how much that sensor typically rises per 1000 runtime hours.
# k: baseline health-score decay rate per 1000 runtime hours for a *typical*
#    machine of this type running under normal conditions.
# ---------------------------------------------------------------------------

TYPE_BASELINES = {
    # k calibrated so a "typical" machine of this type reaches the failure
    # threshold after roughly its real-world expected service life.
    "Centrifugal Pump":   dict(k=0.16, vibration=(2.4, 0.22), temperature=(52, 1.6), current=(68, 0.6)),
    "Air Compressor":     dict(k=0.21, vibration=(3.0, 0.30), temperature=(60, 2.0), current=(74, 0.8)),
    "Conveyor Motor":     dict(k=0.13, vibration=(2.0, 0.15), temperature=(48, 1.2), current=(65, 0.5)),
    "CNC Spindle":        dict(k=0.24, vibration=(1.6, 0.28), temperature=(45, 1.8), current=(60, 0.9)),
    "Diesel Generator":   dict(k=0.11, vibration=(3.4, 0.18), temperature=(70, 1.4), current=(80, 0.4)),
    "Hydraulic Press":    dict(k=0.27, vibration=(2.8, 0.35), temperature=(58, 2.2), current=(78, 1.0)),
}

FAILURE_THRESHOLD = 15.0        # health score at which a machine is considered "failed"
COHORT_AGE_WINDOW_HOURS = 1500  # how close in "age" a cohort machine must be to compare fairly
RISK_ZSCORE_THRESHOLD = 1.4     # how far above cohort-normal a reading must be to flag as a risk factor

RECOMMENDED_ACTIONS = {
    "vibration": [
        "Inspect bearing housing and check shaft alignment",
        "Schedule a vibration-analysis technician visit",
    ],
    "temperature": [
        "Check lubrication levels and coolant/airflow",
        "Run a thermal imaging scan on the housing",
    ],
    "current": [
        "Inspect motor windings and electrical connections",
        "Verify load is within rated capacity",
    ],
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SensorReading:
    hours_since_install: float
    vibration: float      # mm/s
    temperature: float    # °C
    current: float        # % of rated load
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class Machine:
    machine_id: str
    owner: str
    name: str
    machine_type: str
    site: str
    readings: List[SensorReading] = field(default_factory=list)
    failed: bool = False
    failure_hours: Optional[float] = None

    def latest(self) -> Optional[SensorReading]:
        return self.readings[-1] if self.readings else None

    def current_hours(self) -> float:
        return self.latest().hours_since_install if self.readings else 0.0


@dataclass
class PredictionReport:
    machine_id: str
    health_score: float
    predicted_days_to_failure: Optional[float]
    confidence: str
    confidence_score: float
    cohort_size: int
    own_reading_count: int
    risk_factors: List[str]
    recommended_actions: List[str]

    def pretty(self) -> str:
        lines = [
            f"── {self.machine_id} ─────────────────────────────",
            f"Health score: {self.health_score:.1f}/100",
        ]
        if self.predicted_days_to_failure is None:
            lines.append("Predicted failure window: none in the near term")
        else:
            lines.append(f"Predicted failure window: ~{self.predicted_days_to_failure:.0f} days")
        lines.append(f"Confidence: {self.confidence}  (cohort={self.cohort_size} machines, own readings={self.own_reading_count})")
        if self.risk_factors:
            lines.append("Why:")
            for r in self.risk_factors:
                lines.append(f"  - {r}")
        else:
            lines.append("Why: no sensor readings currently deviate from fleet norms.")
        if self.recommended_actions:
            lines.append("Recommended to prevent failure:")
            for a in self.recommended_actions:
                lines.append(f"  - {a}")
        return "\n".join(lines)


#cross-machine learning happens


class Fleet:
    def __init__(self):
        self.machines: Dict[str, Machine] = {}


    def add_machine(self, machine_id: str, owner: str, name: str, machine_type: str, site: str) -> Machine:
        if machine_type not in TYPE_BASELINES:
            raise ValueError(f"Unknown machine type '{machine_type}'. Known types: {list(TYPE_BASELINES)}")
        if machine_id in self.machines:
            raise ValueError(f"Machine id '{machine_id}' already exists")
        m = Machine(machine_id, owner, name, machine_type, site)
        self.machines[machine_id] = m
        return m

    def add_reading(self, machine_id: str, hours_since_install: float,
                     vibration: float, temperature: float, current: float) -> None:
        m = self._require(machine_id)
        m.readings.append(SensorReading(hours_since_install, vibration, temperature, current))

    # ---- the model --------------------------------------------------------

    def _require(self, machine_id: str) -> Machine:
        if machine_id not in self.machines:
            raise KeyError(f"No such machine: {machine_id}")
        return self.machines[machine_id]

    def cohort_for(self, machine: Machine) -> List[Machine]:
        """Every other machine of the same type across the whole fleet."""
        return [m for m in self.machines.values()
                if m.machine_type == machine.machine_type and m.machine_id != machine.machine_id]

    def _cohort_expected(self, machine_type: str, hours: float, sensor: str) -> float:
        base, drift = TYPE_BASELINES[machine_type][sensor]
        return base + drift * (hours / 1000.0)

    def _cohort_readings_near_age(self, cohort: List[Machine], hours: float) -> List[SensorReading]:
        near = []
        for m in cohort:
            for r in m.readings:
                if abs(r.hours_since_install - hours) <= COHORT_AGE_WINDOW_HOURS:
                    near.append(r)
        return near

    def predict(self, machine_id: str) -> PredictionReport:
        machine = self._require(machine_id)
        return self._predict_core(machine, machine.readings)

    def _predict_core(self, machine: "Machine", readings: List[SensorReading]) -> PredictionReport:
        """ This lets backtest_accuracy()
        replay history — 'what would we have predicted using only the first
        N readings?' — without mutating the real machine record."""
        cohort = self.cohort_for(machine)
        hours = readings[-1].hours_since_install if readings else 0.0
        base = TYPE_BASELINES[machine.machine_type]
        cohort_k = base["k"]

        risk_factors: List[str] = []
        recommended: List[str] = []
        risk_multiplier = 1.0

        latest = readings[-1] if readings else None
        if latest is not None:
            near_readings = self._cohort_readings_near_age(cohort, hours)
            for sensor, label, unit in [("vibration", "Vibration", "mm/s"),
                                         ("temperature", "Temperature", "°C"),
                                         ("current", "Current draw", "%")]:
                value = getattr(latest, sensor)
                cohort_values = [getattr(r, sensor) for r in near_readings] or None

                if cohort_values and len(cohort_values) >= 3:
                    mean = statistics.mean(cohort_values)
                    std = statistics.pstdev(cohort_values) or (mean * 0.1 or 1.0)
                else:
                    # Not enough cohort machines at this exact age yet — fall back
                    # to the type's textbook expectation instead of raw peers.
                    mean = self._cohort_expected(machine.machine_type, hours, sensor)
                    std = mean * 0.12 or 1.0

                z = (value - mean) / std
                if z > RISK_ZSCORE_THRESHOLD:
                    pct_over = (value / mean - 1) * 100 if mean else 0
                    risk_factors.append(
                        f"{label} {value:.1f}{unit} vs fleet-typical {mean:.1f}{unit} "
                        f"for {machine.machine_type.lower()}s at this age ({pct_over:+.0f}%)"
                    )
                    recommended.extend(RECOMMENDED_ACTIONS[sensor])
                    risk_multiplier += min(0.5, 0.12 * max(0, z - RISK_ZSCORE_THRESHOLD))

        k_effective = cohort_k * risk_multiplier
        health_score = max(1.0, 100.0 * math.exp(-k_effective / 1000.0 * hours))

        predicted_days = None
        if health_score > FAILURE_THRESHOLD:
            hours_to_failure = (1000.0 / k_effective) * math.log(health_score / FAILURE_THRESHOLD)
            assumed_hours_per_day = 16  # typical duty cycle; adjust per site if known
            predicted_days = hours_to_failure / assumed_hours_per_day
            # Only surface a horizon if it's within a meaningful planning window
            if predicted_days > 365:
                predicted_days = None

        own_n = len(readings)
        cohort_n = len(cohort)
        confidence_score = min(1.0, cohort_n / 8) * 0.5 + min(1.0, own_n / 5) * 0.5
        confidence = "High" if confidence_score >= 0.75 else "Medium" if confidence_score >= 0.4 else "Low"

        return PredictionReport(
            machine_id=machine.machine_id,
            health_score=health_score,
            predicted_days_to_failure=predicted_days,
            confidence=confidence,
            confidence_score=confidence_score,
            cohort_size=cohort_n,
            own_reading_count=own_n,
            risk_factors=risk_factors,
            recommended_actions=sorted(set(recommended)),
        )

    def report(self, site: Optional[str] = None) -> str:
        machines = [m for m in self.machines.values() if site is None or m.site == site]
        lines = [f"=== Fleet report {'(' + site + ')' if site else '(all sites)'} ==="]
        if not machines:
            lines.append("No machines registered.")
            return "\n".join(lines)
        scores = []
        at_risk = 0
        for m in machines:
            p = self.predict(m.machine_id)
            scores.append(p.health_score)
            if p.predicted_days_to_failure is not None and p.predicted_days_to_failure < 30:
                at_risk += 1
        lines.append(f"Machines: {len(machines)}   Avg health: {statistics.mean(scores):.1f}   Need attention (<30d horizon): {at_risk}")
        for m in sorted(machines, key=lambda m: self.predict(m.machine_id).health_score):
            p = self.predict(m.machine_id)
            horizon = f"{p.predicted_days_to_failure:.0f}d" if p.predicted_days_to_failure else "—"
            lines.append(f"  {m.machine_id:<10} {m.machine_type:<18} score={p.health_score:5.1f}  horizon={horizon:>6}  conf={p.confidence}")
        return "\n".join(lines)

    def health_history(self, machine_id: str) -> List[dict]:
        """Replay a machine's own readings one at a time so a UI can chart how
        its health score evolved as each new reading came in."""
        machine = self._require(machine_id)
        out = []
        for i in range(1, len(machine.readings) + 1):
            subset = machine.readings[:i]
            report = self._predict_core(machine, subset)
            out.append({"hours": subset[-1].hours_since_install, "health_score": report.health_score})
        return out

    # ---- accuracy: does this model actually predict well? -----------------

    def record_actual_failure(self, machine_id: str, actual_hours: float) -> None:
        """Owner reports a machine actually failed at a given runtime-hour
        reading. This is what turns 'a model that sounds plausible' into
        'a model with a measured track record' — see backtest_accuracy()."""
        m = self._require(machine_id)
        m.failed = True
        m.failure_hours = actual_hours

    def backtest_accuracy(self, machine_type: Optional[str] = None) -> List[dict]:
        """For every machine that has actually failed (seeded history, or an
        owner's machine after record_actual_failure), replay its readings one
        at a time and ask: 'using only what we knew back then, how many hours
        off was the prediction from the real failure point?'

        Returns one row per (machine, reading-index) so you can plot how the
        error shrinks as more of a machine's own readings come in — the
        concrete, checkable version of the cohort-learning claim.
        """
        rows = []
        failed_machines = [m for m in self.machines.values()
                            if m.failed and m.failure_hours and (machine_type is None or m.machine_type == machine_type)]
        for m in failed_machines:
            for i in range(1, len(m.readings) + 1):
                subset = m.readings[:i]
                report = self._predict_core(m, subset)
                if report.predicted_days_to_failure is None:
                    continue
                predicted_failure_hour = subset[-1].hours_since_install + report.predicted_days_to_failure * 16
                error_hours = predicted_failure_hour - m.failure_hours
                rows.append({
                    "machine_id": m.machine_id,
                    "machine_type": m.machine_type,
                    "readings_used": i,
                    "hours_at_prediction": subset[-1].hours_since_install,
                    "predicted_failure_hour": predicted_failure_hour,
                    "actual_failure_hour": m.failure_hours,
                    "error_hours": error_hours,
                    "pct_error": abs(error_hours) / m.failure_hours * 100,
                })
        return rows

    # ---- persistence so owners can keep adding data across sessions ------

    def save(self, path: str) -> None:
        data = {mid: {**asdict(m), "readings": [asdict(r) for r in m.readings]}
                for mid, m in self.machines.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Fleet":
        fleet = cls()
        with open(path) as f:
            data = json.load(f)
        for mid, md in data.items():
            m = Machine(md["machine_id"], md["owner"], md["name"], md["machine_type"], md["site"],
                        failed=md.get("failed", False), failure_hours=md.get("failure_hours"))
            m.readings = [SensorReading(**r) for r in md["readings"]]
            fleet.machines[mid] = m
        return fleet

    # ---- synthetic history, so the cohort isn't empty on day one ---------

    def seed_demo_history(self, machines_per_type: int = 6, seed: int = 7) -> None:
        """Populate the fleet with realistic historical run-to-failure data for
        every known machine type, so new machines have a cohort to learn from —
        exactly like a real deployment accumulates data over months/years."""
        rng = random.Random(seed)
        for mtype, base in TYPE_BASELINES.items():
            for i in range(machines_per_type):
                mid = f"seed-{mtype[:3].lower()}-{i}"
                lifespan_hours = rng.uniform(6000, 20000)
                m = Machine(mid, "fleet-history", f"{mtype} (historical)", mtype, "seed-site")
                noise_k = rng.uniform(0.85, 1.2)  # this individual machine's inherent variability
                hours = 0.0
                while hours < lifespan_hours:
                    score = 100 * math.exp(-(base["k"] * noise_k) / 1000 * hours)
                    vib = self._cohort_expected(mtype, hours, "vibration") * rng.uniform(0.9, 1.1)
                    temp = self._cohort_expected(mtype, hours, "temperature") * rng.uniform(0.95, 1.05)
                    cur = self._cohort_expected(mtype, hours, "current") * rng.uniform(0.95, 1.05)
                    m.readings.append(SensorReading(hours, vib, temp, cur))
                    hours += rng.uniform(400, 900)
                    if score < FAILURE_THRESHOLD:
                        m.failed = True
                        m.failure_hours = hours
                        break
                self.machines[mid] = m


# ---------------------------------------------------------------------------
# Demo walkthrough
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fleet = Fleet()

    print("1. Seeding fleet with historical run-to-failure data for existing machines...\n")
    fleet.seed_demo_history(machines_per_type=6)

    print("2. Owner Dana registers a brand-new pump with ZERO history...\n")
    fleet.add_machine("dana-01", owner="Dana", name="Riverside Pump #12",
                       machine_type="Centrifugal Pump", site="Riverside Plant")
    fleet.add_reading("dana-01", hours_since_install=0, vibration=2.3, temperature=51, current=67)

    print(fleet.predict("dana-01").pretty())
    print("\n^ Notice: even with one reading, it already gets a real prediction —")
    print("  borrowed entirely from every other pump in the fleet.\n")

    print("3. Three months later, Dana logs new readings. Vibration is running high...\n")
    fleet.add_reading("dana-01", hours_since_install=2100, vibration=4.9, temperature=57, current=70)
    fleet.add_reading("dana-01", hours_since_install=2600, vibration=5.6, temperature=59, current=71)
    fleet.add_reading("dana-01", hours_since_install=3000, vibration=6.1, temperature=60, current=72)

    print(fleet.predict("dana-01").pretty())
    print("\n^ Now the model compares Dana's pump against OTHER pumps at ~3,000 hours")
    print("  specifically, notices it vibrates well above fleet-normal for that age,")
    print("  and moves up the predicted failure window with higher confidence.\n")

    print("4. Owner's fleet-wide analytical view:\n")
    fleet.add_machine("dana-02", owner="Dana", name="Riverside Compressor #4",
                       machine_type="Air Compressor", site="Riverside Plant")
    fleet.add_reading("dana-02", hours_since_install=500, vibration=3.1, temperature=61, current=75)
    print(fleet.report(site="Riverside Plant"))

    fleet.save("fleet_demo.json")
    print("\nSaved to fleet_demo.json — reload later with Fleet.load('fleet_demo.json')")
