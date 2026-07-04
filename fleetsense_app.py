#!/usr/bin/env python3
"""
FleetSense Desktop — a real GUI app on top of predictive_maintenance.py

Run with:      python3 fleetsense_app.py
Requirement:   Python's built-in tkinter (already included on Windows/macOS
               Python installs; on Linux you may need: sudo apt install python3-tk)

Put this file in the SAME FOLDER as predictive_maintenance.py — it imports
the Fleet/prediction engine from there rather than duplicating the logic.

What you get: an actual window, not a terminal printout —
  - A live table of your machines, colored by health status
  - Buttons to add a new machine or log a new sensor reading
  - A detail panel with the failure-horizon gauge, the reasons a machine was
    flagged, and what to do about it
  - Save / Load so your fleet persists between sessions
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from predictive_maintenance import Fleet, TYPE_BASELINES

STATUS_COLORS = {
    "healthy": "#34D3AC",
    "watch": "#5EC8E8",
    "warning": "#F2A93B",
    "critical": "#F0524A",
}


def status_for_score(score: float) -> str:
    if score >= 80:
        return "healthy"
    if score >= 60:
        return "watch"
    if score >= 35:
        return "warning"
    return "critical"


class AddMachineDialog(tk.Toplevel):
    def __init__(self, master, on_submit):
        super().__init__(master)
        self.title("Add Machine")
        self.configure(bg="#0F141A")
        self.resizable(False, False)
        self.on_submit = on_submit

        fields = [
            ("Machine ID", "id"), ("Owner name", "owner"), ("Display name", "name"),
        ]
        self.vars = {}
        row = 0
        for label, key in fields:
            tk.Label(self, text=label, bg="#0F141A", fg="#8B98A5", font=("Segoe UI", 9)).grid(
                row=row, column=0, sticky="w", padx=12, pady=(10, 2))
            var = tk.StringVar()
            tk.Entry(self, textvariable=var, width=30, bg="#161B22", fg="#E6EDF3",
                      insertbackground="#E6EDF3", relief="flat").grid(row=row + 1, column=0, padx=12)
            self.vars[key] = var
            row += 2

        tk.Label(self, text="Machine type", bg="#0F141A", fg="#8B98A5", font=("Segoe UI", 9)).grid(
            row=row, column=0, sticky="w", padx=12, pady=(10, 2))
        self.type_var = tk.StringVar(value=list(TYPE_BASELINES)[0])
        ttk.Combobox(self, textvariable=self.type_var, values=list(TYPE_BASELINES),
                     state="readonly", width=28).grid(row=row + 1, column=0, padx=12)
        row += 2

        tk.Label(self, text="Site", bg="#0F141A", fg="#8B98A5", font=("Segoe UI", 9)).grid(
            row=row, column=0, sticky="w", padx=12, pady=(10, 2))
        self.site_var = tk.StringVar()
        tk.Entry(self, textvariable=self.site_var, width=30, bg="#161B22", fg="#E6EDF3",
                  insertbackground="#E6EDF3", relief="flat").grid(row=row + 1, column=0, padx=12)
        row += 2

        btn = tk.Button(self, text="Add Machine", command=self.submit, bg="#5EC8E8", fg="#0D1117",
                         relief="flat", font=("Segoe UI", 10, "bold"))
        btn.grid(row=row, column=0, pady=16, padx=12, sticky="ew")

    def submit(self):
        mid = self.vars["id"].get().strip()
        owner = self.vars["owner"].get().strip()
        name = self.vars["name"].get().strip()
        site = self.site_var.get().strip()
        if not all([mid, owner, name, site]):
            messagebox.showerror("Missing info", "Every field is required.")
            return
        try:
            self.on_submit(mid, owner, name, self.type_var.get(), site)
        except Exception as e:
            messagebox.showerror("Couldn't add machine", str(e))
            return
        self.destroy()


class LogReadingDialog(tk.Toplevel):
    def __init__(self, master, machine_ids, on_submit):
        super().__init__(master)
        self.title("Log Sensor Reading")
        self.configure(bg="#0F141A")
        self.resizable(False, False)
        self.on_submit = on_submit

        tk.Label(self, text="Machine", bg="#0F141A", fg="#8B98A5", font=("Segoe UI", 9)).grid(
            row=0, column=0, sticky="w", padx=12, pady=(10, 2))
        self.machine_var = tk.StringVar(value=machine_ids[0] if machine_ids else "")
        ttk.Combobox(self, textvariable=self.machine_var, values=machine_ids,
                     state="readonly", width=28).grid(row=1, column=0, padx=12)

        fields = [
            ("Runtime hours since install", "hours"),
            ("Vibration (mm/s)", "vibration"),
            ("Temperature (°C)", "temperature"),
            ("Current draw (% of rated)", "current"),
        ]
        self.vars = {}
        row = 2
        for label, key in fields:
            tk.Label(self, text=label, bg="#0F141A", fg="#8B98A5", font=("Segoe UI", 9)).grid(
                row=row, column=0, sticky="w", padx=12, pady=(10, 2))
            var = tk.StringVar()
            tk.Entry(self, textvariable=var, width=30, bg="#161B22", fg="#E6EDF3",
                      insertbackground="#E6EDF3", relief="flat").grid(row=row + 1, column=0, padx=12)
            self.vars[key] = var
            row += 2

        btn = tk.Button(self, text="Log Reading", command=self.submit, bg="#5EC8E8", fg="#0D1117",
                         relief="flat", font=("Segoe UI", 10, "bold"))
        btn.grid(row=row, column=0, pady=16, padx=12, sticky="ew")

    def submit(self):
        mid = self.machine_var.get()
        if not mid:
            messagebox.showerror("No machine", "Add a machine first.")
            return
        try:
            hours = float(self.vars["hours"].get())
            vibration = float(self.vars["vibration"].get())
            temperature = float(self.vars["temperature"].get())
            current = float(self.vars["current"].get())
        except ValueError:
            messagebox.showerror("Invalid input", "Hours, vibration, temperature and current must be numbers.")
            return
        self.on_submit(mid, hours, vibration, temperature, current)
        self.destroy()


class FleetSenseApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FleetSense — Predictive Maintenance")
        self.geometry("1060x620")
        self.configure(bg="#0D1117")

        self.fleet = Fleet()
        self.fleet.seed_demo_history(machines_per_type=6)  # gives new machines a cohort to learn from

        self._build_style()
        self._build_layout()
        self.refresh()

    # ---- styling ----------------------------------------------------

    def _build_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Treeview", background="#10151B", fieldbackground="#10151B",
                         foreground="#E6EDF3", rowheight=27, borderwidth=0, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", background="#161B22", foreground="#8B98A5",
                         font=("Segoe UI", 9, "bold"), relief="flat")
        style.map("Treeview", background=[("selected", "#1B2733")])
        for status, color in STATUS_COLORS.items():
            style.configure(f"{status}.Treeview")

    # ---- layout -------------------------------------------------------

    def _build_layout(self):
        top = tk.Frame(self, bg="#0D1117")
        top.pack(fill="x", padx=16, pady=12)
        tk.Label(top, text="FleetSense", bg="#0D1117", fg="#5EC8E8",
                 font=("Segoe UI", 15, "bold")).pack(side="left")
        self.summary_lbl = tk.Label(top, text="", bg="#0D1117", fg="#8B98A5", font=("Consolas", 10))
        self.summary_lbl.pack(side="left", padx=20)

        tk.Button(top, text="Save Fleet", command=self.save_fleet, bg="#161B22", fg="#E6EDF3",
                  relief="flat", padx=10).pack(side="right", padx=4)
        tk.Button(top, text="Load Fleet", command=self.load_fleet, bg="#161B22", fg="#E6EDF3",
                  relief="flat", padx=10).pack(side="right", padx=4)
        tk.Button(top, text="Log Reading", command=self.open_log_reading, bg="#161B22", fg="#E6EDF3",
                  relief="flat", padx=10).pack(side="right", padx=4)
        tk.Button(top, text="+ Add Machine", command=self.open_add_machine, bg="#5EC8E8", fg="#0D1117",
                  relief="flat", padx=10, font=("Segoe UI", 9, "bold")).pack(side="right", padx=4)

        body = tk.Frame(self, bg="#0D1117")
        body.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        left = tk.Frame(body, bg="#0D1117")
        left.pack(side="left", fill="both", expand=True)

        cols = ("type", "site", "score", "horizon", "confidence")
        self.tree = ttk.Treeview(left, columns=cols, show="tree headings", selectmode="browse")
        self.tree.heading("#0", text="Machine")
        self.tree.column("#0", width=150)
        headers = {"type": "Type", "site": "Site", "score": "Health", "horizon": "Horizon", "confidence": "Confidence"}
        widths = {"type": 150, "site": 130, "score": 70, "horizon": 80, "confidence": 90}
        for c in cols:
            self.tree.heading(c, text=headers[c])
            self.tree.column(c, width=widths[c], anchor="center")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        for status, color in STATUS_COLORS.items():
            self.tree.tag_configure(status, foreground=color)

        right = tk.Frame(body, bg="#0F141A", width=340)
        right.pack(side="right", fill="y", padx=(16, 0))
        right.pack_propagate(False)

        self.detail_title = tk.Label(right, text="Select a machine", bg="#0F141A", fg="#5EC8E8",
                                      font=("Segoe UI", 13, "bold"), anchor="w")
        self.detail_title.pack(anchor="w", padx=14, pady=(14, 0), fill="x")
        self.detail_sub = tk.Label(right, text="", bg="#0F141A", fg="#5B6673",
                                    font=("Consolas", 8), anchor="w")
        self.detail_sub.pack(anchor="w", padx=14, pady=(2, 12), fill="x")

        self.horizon_canvas = tk.Canvas(right, width=300, height=34, bg="#0F141A", highlightthickness=0)
        self.horizon_canvas.pack(padx=14)

        self.detail_text = tk.Text(right, width=40, height=22, bg="#131920", fg="#C9D1D9",
                                    insertbackground="#E6EDF3", relief="flat", wrap="word",
                                    font=("Segoe UI", 9), padx=10, pady=10)
        self.detail_text.pack(padx=14, pady=14, fill="both", expand=True)
        self.detail_text.configure(state="disabled")

    # ---- data actions ---------------------------------------------------

    def owned_machines(self):
        return [m for m in self.fleet.machines.values() if m.owner != "fleet-history"]

    def refresh(self):
        self.tree.delete(*self.tree.get_children())
        machines = self.owned_machines()
        scores, at_risk = [], 0
        for m in machines:
            p = self.fleet.predict(m.machine_id)
            scores.append(p.health_score)
            status = status_for_score(p.health_score)
            horizon = f"{p.predicted_days_to_failure:.0f}d" if p.predicted_days_to_failure else "—"
            if p.predicted_days_to_failure is not None and p.predicted_days_to_failure < 30:
                at_risk += 1
            self.tree.insert("", "end", iid=m.machine_id, text=m.name,
                              values=(m.machine_type, m.site, f"{p.health_score:.0f}", horizon, p.confidence),
                              tags=(status,))
        if scores:
            self.summary_lbl.configure(
                text=f"{len(machines)} machines · avg health {sum(scores)/len(scores):.0f} · {at_risk} need attention soon")
        else:
            self.summary_lbl.configure(text="No machines yet — add one to get started")

    def on_select(self, _event=None):
        sel = self.tree.selection()
        if not sel:
            return
        self.render_detail(sel[0])

    def render_detail(self, machine_id):
        machine = self.fleet.machines[machine_id]
        p = self.fleet.predict(machine_id)
        status = status_for_score(p.health_score)
        color = STATUS_COLORS[status]

        self.detail_title.configure(text=machine.name)
        self.detail_sub.configure(
            text=f"{machine.machine_type.upper()} · {machine.site.upper()} · OWNER: {machine.owner.upper()}")

        self.draw_horizon(p.predicted_days_to_failure)

        lines = [f"Health score: {p.health_score:.1f}/100  ({status})", ""]
        if p.predicted_days_to_failure is not None:
            lines.append(f"Predicted failure window: ~{p.predicted_days_to_failure:.0f} days")
        else:
            lines.append("Predicted failure window: none in the near term")
        lines.append(f"Confidence: {p.confidence}  (cohort of {p.cohort_size} similar machines, "
                      f"{p.own_reading_count} of your own readings)")
        lines.append("")
        if p.risk_factors:
            lines.append("Why it was flagged:")
            for r in p.risk_factors:
                lines.append(f"  • {r}")
        else:
            lines.append("Why: no readings currently deviate from fleet norms.")
        lines.append("")
        if p.recommended_actions:
            lines.append("Recommended to prevent failure:")
            for a in p.recommended_actions:
                lines.append(f"  • {a}")
        else:
            lines.append("No action needed right now.")

        self.detail_text.configure(state="normal")
        self.detail_text.delete("1.0", "end")
        self.detail_text.insert("1.0", "\n".join(lines))
        self.detail_text.configure(state="disabled")

    def draw_horizon(self, days):
        c = self.horizon_canvas
        c.delete("all")
        w, h = 300, 12
        zones = [(0, 0.22, "#34D3AC"), (0.22, 0.52, "#5EC8E8"), (0.52, 0.80, "#F2A93B"), (0.80, 1.0, "#F0524A")]
        for start, end, color in zones:
            c.create_rectangle(w * start, 8, w * end, 8 + h, fill=color, outline="", stipple="gray50")
        capped = 150 if days is None else min(days, 150)
        frac = capped / 150.0
        x = w * (1 - frac)
        c.create_rectangle(x - 2, 4, x + 2, 8 + h + 4, fill="#E6EDF3", outline="")
        labels = ["Stable", "Monitor", "Elevated", "Imminent"]
        for i, (start, end, _) in enumerate(zones):
            c.create_text(w * (start + end) / 2, 8 + h + 12, text=labels[i], fill="#5B6673", font=("Consolas", 7))

    # ---- dialogs --------------------------------------------------------

    def open_add_machine(self):
        AddMachineDialog(self, self.add_machine)

    def add_machine(self, mid, owner, name, mtype, site):
        self.fleet.add_machine(mid, owner, name, mtype, site)
        self.refresh()
        self.tree.selection_set(mid)
        self.render_detail(mid)

    def open_log_reading(self):
        ids = [m.machine_id for m in self.owned_machines()]
        if not ids:
            messagebox.showinfo("No machines", "Add a machine first.")
            return
        LogReadingDialog(self, ids, self.log_reading)

    def log_reading(self, mid, hours, vibration, temperature, current):
        self.fleet.add_reading(mid, hours, vibration, temperature, current)
        self.refresh()
        self.tree.selection_set(mid)
        self.render_detail(mid)

    # ---- persistence ------------------------------------------------------

    def save_fleet(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", initialfile="fleet.json")
        if path:
            self.fleet.save(path)
            messagebox.showinfo("Saved", f"Fleet saved to {os.path.basename(path)}")

    def load_fleet(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            self.fleet = Fleet.load(path)
            self.refresh()


if __name__ == "__main__":
    FleetSenseApp().mainloop()
