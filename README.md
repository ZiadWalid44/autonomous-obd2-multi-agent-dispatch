# 🚗 Autonomous OBD-II Vehicle Diagnostic & Smart Dispatch System

**A Collaborative Multi-Agent AI Framework for Automotive Triage**  
*Artificial Intelligence Course Project — Badr University in Cairo (BUC)*  
**Supervised by:** Dr. Samar M. Nour  

---

## 📌 Project Team
* **Ziad Walid Ali Mahdy** (ID: 2022003974)
* **Mostafa Mahmoud Elsayed** (ID: 2022015868)
* **Youssef Emad** (ID: 2022008365)

---

## 🧠 Academic Architecture (Russell & Norvig Paradigm)

This system implements a sequential, cooperative **Multi-Agent Architecture** designed to resolve the ambiguity of vehicle dashboard warnings and optimize repair dispatch decisions across Greater Cairo:

1. **Agent 1 — Diagnostic & Triage Agent (Model-Based, Goal-Based Agent):**
   * **Internal State Estimation:** Fuses 13 live OBD-II telemetry streams (RPM, Coolant Temp, MAF, STFT, Throttle Position, Battery Voltage, Alternator Output & Load, and 4-wheel TPMS tire pressures) with historical preventive maintenance records (`vehicle_maintenance_history.csv`).
   * **Perception Model:** A trained `RandomForestClassifier` operating under **Partially Observable** and **Stochastic** environment properties, achieving **95.71% accuracy** across 14 automotive states despite overlapping DTC codes.
   * **Goal Test:** Evaluates `is_goal_achieved()` against `SAFE_OPERATIONAL_STATE` to trigger either standard repair routing or emergency tow/mobile rescue dispatch.

2. **Agent 2 — Service & Dispatch Agent (Utility-Based Agent):**
   * **Multi-Attribute Utility Optimization:** Evaluates 14 regional service centers across East Cairo, Obour, Gesr El-Suez (El-Herafieen), El-Shorouk, New Cairo, Nasr City, Maadi, and 6th of October using:
     $$\text{Utility} = w_1(\text{ExpertiseMatch}) - w_2(\text{Distance}_{\text{norm}}) - w_3(\text{Cost}_{\text{norm}}) + w_4(\text{Rating}_{\text{norm}})$$
   * **Context-Aware Dynamic Weighting:** Automatically adapts weights and spare-parts pricing multipliers based on **Vehicle Origin** (German, European, Japanese, Korean, Chinese), **Model Year**, and **Active Dealership Warranty Status**.

---

## 📂 Repository Structure

```text
autonomous-obd2-multi-agent-dispatch/
├── data/
│   ├── obd2_telemetry_dataset.csv
│   ├── service_centers_registry.csv
│   └── vehicle_maintenance_history.csv
├── notebooks/
│   └── OBD2_MultiAgent_Architecture.ipynb
├── docs/
│   └── Autonomous_OBD-II_Diagnostic_System.pptx
├── app.py
├── requirements.txt
├── .gitignore
└── README.md