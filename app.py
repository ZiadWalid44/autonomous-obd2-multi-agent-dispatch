import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from agents_engine import DiagnosticTriageAgent, SmartDispatchUtilityAgent, RepairManualRAG

st.set_page_config(page_title="Autonomous OBD-II Digital Cockpit", page_icon="🚘", layout="wide", initial_sidebar_state="expanded")

CAR_PROFILES = {
    "Daewoo / Chevrolet Nubira (Korean - 2006)": {
        "origin": "Korean", "default_year": 2006, "warranty": False, "accent": "#00f5d4",
        "os_name": "GM-DAEWOO TECH-II DIAGNOSTIC OS",
        "logo_svg": """<svg width="85" height="45" viewBox="0 0 120 50" fill="none"><path d="M35 5H85V18H115V32H85V45H35V32H5V18H35V5Z" fill="#00f5d4" fill-opacity="0.2" stroke="#00f5d4" stroke-width="3"/><text x="60" y="30" fill="#ffffff" font-family="Arial" font-weight="bold" font-size="13" text-anchor="middle">DAEWOO</text></svg>""",
        "bg_image": "https://images.unsplash.com/photo-1549399542-7e3f8b79c341?auto=format&fit=crop&w=1920&q=85"
    },
    "BMW 320i M-Sport (German - 2025)": {
        "origin": "German", "default_year": 2025, "warranty": True, "accent": "#00a8ff",
        "os_name": "BMW iDRIVE 8.5 M-TELEMETRY OS",
        "logo_svg": """<svg width="68" height="68" viewBox="0 0 100 100"><circle cx="50" cy="50" r="46" fill="#0f172a" stroke="#e2e8f0" stroke-width="4"/><circle cx="50" cy="50" r="32" fill="#1e293b" stroke="#94a3b8" stroke-width="2"/><path d="M50 18 A32 32 0 0 1 82 50 L50 50 Z" fill="#00a8ff"/><path d="M50 82 A32 32 0 0 1 18 50 L50 50 Z" fill="#00a8ff"/><path d="M18 50 A32 32 0 0 1 50 18 L50 50 Z" fill="#ffffff"/><path d="M82 50 A32 32 0 0 1 50 82 L50 50 Z" fill="#ffffff"/><text x="50" y="13" fill="#ffffff" font-family="Arial" font-weight="bold" font-size="10" text-anchor="middle">B M W</text></svg>""",
        "bg_image": "https://images.unsplash.com/photo-1555215695-3004980ad54e?auto=format&fit=crop&w=1920&q=85"
    },
    "Mercedes-Benz C200 AMG (German - 2024)": {
        "origin": "German", "default_year": 2024, "warranty": True, "accent": "#c084fc",
        "os_name": "MERCEDES-BENZ MBUX HYPERSCREEN OS",
        "logo_svg": """<svg width="68" height="68" viewBox="0 0 100 100"><circle cx="50" cy="50" r="45" fill="none" stroke="#e2e8f0" stroke-width="4"/><polygon points="50,7 55,46 88,72 50,55 12,72 45,46" fill="#e2e8f0"/></svg>""",
        "bg_image": "https://images.unsplash.com/photo-1618843479313-40f8afb4b4d8?auto=format&fit=crop&w=1920&q=85"
    },
    "MG RX5 Plus Turbo (Chinese - 2024)": {
        "origin": "Chinese", "default_year": 2024, "warranty": True, "accent": "#ff4757",
        "os_name": "SAIC MG i-SMART PILOTED COCKPIT",
        "logo_svg": """<svg width="68" height="68" viewBox="0 0 100 100"><polygon points="30,6 70,6 94,30 94,70 70,94 30,94 6,70 6,30" fill="#ff4757" fill-opacity="0.18" stroke="#ff4757" stroke-width="4"/><text x="50" y="60" fill="#ffffff" font-family="Arial" font-weight="900" font-size="30" text-anchor="middle">MG</text></svg>""",
        "bg_image": "https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?auto=format&fit=crop&w=1920&q=85"
    },
    "Toyota Corolla GR (Japanese - 2019)": {
        "origin": "Japanese", "default_year": 2019, "warranty": False, "accent": "#38bdf8",
        "os_name": "TOYOTA SAFETY SENSE TELEMATICS",
        "logo_svg": """<svg width="82" height="56" viewBox="0 0 120 80"><ellipse cx="60" cy="40" rx="52" ry="33" fill="none" stroke="#38bdf8" stroke-width="4"/><ellipse cx="60" cy="40" rx="16" ry="32" fill="none" stroke="#38bdf8" stroke-width="3.5"/><ellipse cx="60" cy="24" rx="34" ry="13" fill="none" stroke="#38bdf8" stroke-width="3.5"/></svg>""",
        "bg_image": "https://images.unsplash.com/photo-1621007947382-bb3c3994e3fb?auto=format&fit=crop&w=1920&q=85"
    }
}

CAIRO_LOCATIONS = {
    "Obour City / Ismailia Desert Rd": (30.1650, 31.4250),
    "New Cairo (5th Settlement)": (30.0155, 31.4380),
    "Nasr City / Heliopolis": (30.0560, 31.3450),
    "Maadi / Autostrad": (29.9730, 31.2840),
    "Badr City / El-Shorouk": (30.1400, 31.6500),
    "6th of October / Sheikh Zayed": (30.0440, 31.0690)
}

PRESET_CASES = {
    "P0300: Ignition Misfire": (2450, 92.0, 14.2, 18.5, 21.0, 12.6, 14.1, 42.0, 32.5, "P0300"),
    "P0562: No-Start / Alternator": (850, 88.0, 6.5, 1.0, 10.0, 11.1, 11.4, 94.0, 33.0, "P0562"),
    "P0217: Engine Overheat": (2650, 113.5, 15.2, -2.0, 24.0, 12.5, 14.0, 50.0, 32.5, "P0217"),
    "C0750: Flat Tire (TPMS)": (2100, 89.5, 12.1, 0.5, 18.0, 12.7, 14.2, 38.0, 18.5, "C0750"),
    "P0700: Gearbox Fault": (4050, 100.5, 22.0, 3.0, 35.0, 12.6, 14.1, 48.0, 33.0, "P0700"),
    "System Normal": (1400, 89.0, 8.2, 0.5, 12.0, 12.7, 14.2, 34.0, 33.0, "NONE")
}

if "active_case" not in st.session_state:
    st.session_state.active_case = "P0300: Ignition Misfire"
if "chat_query" not in st.session_state:
    st.session_state.chat_query = ""

@st.cache_resource
def init_ai_agents():
    os.makedirs("data", exist_ok=True)
    v2_path = "data/service_centers_registry_v2.csv"
    if not os.path.exists(v2_path):
        pd.DataFrame([
            {"Name": "Mega Dealership (Obour Industrial Zone)", "Type": "Authorized_Dealership", "Specialty": "All_Systems", "Supported_Origins": "German,Japanese,Korean,Chinese,European", "City_Zone": "Obour City", "Address": "Block 13004, Industrial Zone A, Obour City", "Phone": "+20 19980 / 02-44891200", "Base_Labor_Cost_EGP": 2200, "Rating_Score": 9.3, "Reviews_Count": 1420, "Top_Comment": "توكيل شامل بأجهزة كشف أصلية وتفعيل ضمان فوري لجميع الأنظمة.", "Lat": 30.1982, "Lon": 31.4915},
            {"Name": "Bavaria EuroTech Dealership (New Cairo)", "Type": "Authorized_Dealership", "Specialty": "All_Systems", "Supported_Origins": "German,European", "City_Zone": "New Cairo (5th Settlement)", "Address": "South 90th St, Industrial Triangle, 5th Settlement", "Phone": "+20 19222 / 01223450987", "Base_Labor_Cost_EGP": 2800, "Rating_Score": 9.5, "Reviews_Count": 2150, "Top_Comment": "توكيل معتمد متخصص في الألماني وبرمجة الكنترولات والفتيس.", "Lat": 30.0155, "Lon": 31.4380},
            {"Name": "AutoWorld Dealer (Ismailia Desert Rd)", "Type": "Authorized_Dealership", "Specialty": "All_Systems", "Supported_Origins": "Korean,Chinese,Japanese", "City_Zone": "Ismailia Desert Rd", "Address": "KM 28 Cairo-Ismailia Desert Road, Next to Toll Station", "Phone": "+20 16550 / 01009823411", "Base_Labor_Cost_EGP": 1950, "Rating_Score": 8.9, "Reviews_Count": 980, "Top_Comment": "توكيل معتمد للصيني والكوري على طريق الإسماعيلية وقطع الغيار متوفرة.", "Lat": 30.1590, "Lon": 31.4980},
            {"Name": "GearMaster Automatic Transmission Lab (El-Herafieen)", "Type": "Specialized_Center", "Specialty": "Gearbox_Transmission", "Supported_Origins": "German,Japanese,Korean,Chinese,European", "City_Zone": "Gesr El-Suez (El-Herafieen)", "Address": "Block 9, Transmission Sector, El-Herafieen", "Phone": "01005544332 / 01122998877", "Base_Labor_Cost_EGP": 1100, "Rating_Score": 9.2, "Reviews_Count": 890, "Top_Comment": "متخصصين فتيس أوتوماتيك وCVT وDSG فقط—تغيير زيت وفالف بودي بضمان.", "Lat": 30.1440, "Lon": 31.3975},
            {"Name": "BrakePro & Suspension Specialist (Obour 1st District)", "Type": "Specialized_Center", "Specialty": "Brakes_Suspension", "Supported_Origins": "German,Japanese,Korean,Chinese,European", "City_Zone": "Obour City", "Address": "Service Axis, 1st District, Near Carrefour Obour", "Phone": "01066554433 / 01554433221", "Base_Labor_Cost_EGP": 450, "Rating_Score": 8.9, "Reviews_Count": 740, "Top_Comment": "متخصصين فرامل وعفشة فقط: خرط طنابير على العربية، إصلاح سيرفو وماستر عمومي.", "Lat": 30.1810, "Lon": 31.4610},
            {"Name": "ArcticCool Auto A/C & Radiators (Nasr City)", "Type": "Specialized_Center", "Specialty": "AC_Cooling", "Supported_Origins": "German,Japanese,Korean,Chinese,European", "City_Zone": "Nasr City", "Address": "El-Hay El-Asher, Auto Complex, Nasr City", "Phone": "01022113344 / 01200998877", "Base_Labor_Cost_EGP": 600, "Rating_Score": 8.8, "Reviews_Count": 680, "Top_Comment": "متخصصين تكييف ودورة تبريد وريداتير وثرموستات فقط بأحدث أجهزة شحن الفريون.", "Lat": 30.0510, "Lon": 31.3650},
            {"Name": "VoltTech Auto Electric & Alternator Hub (Shubra)", "Type": "Specialized_Center", "Specialty": "Auto_Electric", "Supported_Origins": "Korean,Japanese,European,Chinese,German", "City_Zone": "Shubra / Gesr El-Suez", "Address": "Aghakhan Strip, Cairo", "Phone": "01277665544 / 02-22056789", "Base_Labor_Cost_EGP": 520, "Rating_Score": 8.7, "Reviews_Count": 720, "Top_Comment": "متخصصين كهرباء سيارات ودينامو ومارش وضفيرة وبطاريات فقط.", "Lat": 30.1320, "Lon": 31.3250},
            {"Name": "Fix-It General Mechanical Hub (El-Herafieen)", "Type": "Local_Workshop", "Specialty": "All_Systems", "Supported_Origins": "Korean,Japanese,Chinese,European", "City_Zone": "Gesr El-Suez (El-Herafieen)", "Address": "Block 14, Mosque Square, El-Herafieen", "Phone": "01287654321 / 01099887766", "Base_Labor_Cost_EGP": 500, "Rating_Score": 8.1, "Reviews_Count": 1120, "Top_Comment": "ميكانيكا عامة وصيانة دورية ممتازة للنوبيرا واللانوس والكوري بأسعار مناسبة.", "Lat": 30.1425, "Lon": 31.3960}
        ]).to_csv(v2_path, index=False)

    df_tel = pd.read_csv("data/obd2_telemetry_dataset.csv")
    df_shop = pd.read_csv(v2_path)
    return DiagnosticTriageAgent(df_tel), SmartDispatchUtilityAgent(df_shop), RepairManualRAG(), df_shop
agent1, agent2, rag_engine, df_workshops = init_ai_agents()
# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## 🚘 Vehicle & Location Setup")
selected_car_name = st.sidebar.selectbox("Select Vehicle Model", list(CAR_PROFILES.keys()))
car_cfg = CAR_PROFILES[selected_car_name]
selected_loc_name = st.sidebar.selectbox("Current GPS Location", list(CAIRO_LOCATIONS.keys()))
driver_gps = CAIRO_LOCATIONS[selected_loc_name]

col_s1, col_s2 = st.sidebar.columns(2)
model_year = col_s1.number_input("Model Year", 2000, 2026, car_cfg["default_year"])
in_warranty = col_s2.checkbox("In Warranty", value=car_cfg["warranty"])

s_rpm, s_temp, s_maf, s_stft, s_tps, s_batt, s_alt, s_load, s_rr, s_dtc = PRESET_CASES[st.session_state.active_case]

with st.sidebar.expander("⚙️ Advanced Sensor Overrides (Engineers)"):
    current_odo = st.number_input("Odometer (KM)", value=148500 if model_year <= 2015 else 24500, step=1000)
    last_oil_odo = st.number_input("Last Oil Change (KM)", value=current_odo - 9300, step=500)
    oil_interval = st.selectbox("Oil Interval (KM)", [5000, 10000, 15000], index=1)
    rpm = st.slider("Engine RPM", 650, 5500, s_rpm)
    temp = st.slider("Coolant Temp (°C)", 70.0, 125.0, s_temp)
    maf = st.slider("MAF Flow (g/s)", 1.0, 50.0, s_maf)
    stft = st.slider("Fuel Trim STFT (%)", -25.0, 25.0, s_stft)
    batt_v = st.slider("Battery Voltage (V)", 9.5, 14.8, s_batt)
    alt_v = st.slider("Alternator Output (V)", 9.5, 15.5, s_alt)
    tire_rr = st.slider("Rear-Right Tire (PSI)", 12.0, 36.0, s_rr)

# ---------------- STYLING & HEADER ----------------
accent, bg_img = car_cfg["accent"], car_cfg["bg_image"]
st.markdown(f"""
<style>
    .stApp {{ background: linear-gradient(135deg, rgba(8, 11, 18, 0.80) 0%, rgba(15, 23, 42, 0.74) 50%, rgba(8, 11, 18, 0.85) 100%), url("{bg_img}"); background-size: cover; background-position: center; background-attachment: fixed; color: #f8fafc; }}
    .cockpit-header {{ background: rgba(15, 23, 42, 0.74); backdrop-filter: blur(14px); border: 1px solid {accent}55; border-bottom: 3px solid {accent}; border-radius: 18px; padding: 16px 24px; margin-bottom: 14px; display: flex; align-items: center; gap: 20px; }}
    .glass-panel {{ background: rgba(15, 23, 42, 0.78); backdrop-filter: blur(14px); border: 1px solid rgba(255, 255, 255, 0.12); border-top: 2px solid {accent}; border-radius: 16px; padding: 18px; margin-bottom: 14px; }}
    .workshop-card {{ background: rgba(17, 24, 39, 0.85); backdrop-filter: blur(12px); border-left: 5px solid {accent}; border: 1px solid rgba(255,255,255,0.1); padding: 16px; border-radius: 12px; margin-bottom: 10px; }}
    .tpms-box {{ background: rgba(15, 23, 42, 0.85); border: 1px solid {accent}66; border-radius: 10px; padding: 8px; text-align: center; font-weight: bold; }}
    .status-pill {{ display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 12px; font-weight: 700; background: {accent}22; color: {accent}; border: 1px solid {accent}; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="cockpit-header">
    <div style="filter: drop-shadow(0 0 12px {accent}aa);">{car_cfg['logo_svg']}</div>
    <div>
        <span class="status-pill">{car_cfg['os_name']} • AI ACCURACY: {agent1.accuracy}%</span>
        <h2 style="margin:6px 0 0 0; font-size:25px; color:#ffffff;">{selected_car_name}</h2>
        <p style="margin:3px 0 0 0; color:#cbd5e1; font-size:13.5px;">Origin: <b style="color:{accent};">{car_cfg['origin']}</b> | Model Year: <b>{model_year}</b> | Warranty: <b>{"ACTIVE" if in_warranty else "EXPIRED"}</b> | Location: <b>{selected_loc_name}</b></p>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- PRESET CASES & DRIVER COPILOT ----------------
st.markdown("##### ⚡ Quick Telemetry Scenarios (Click to switch live sensor state):")
btn_cols = st.columns(len(PRESET_CASES))
for i, case_label in enumerate(PRESET_CASES.keys()):
    if btn_cols[i].button(case_label, width="stretch", key=f"case_btn_{i}"):
        st.session_state.active_case = case_label
        st.session_state.chat_query = ""
        st.rerun()

user_complaint = st.text_input(
    "💬 Driver Copilot (Describe unmeasured symptoms like brake squeal, hard pedal, or no-start):",
    value=st.session_state.chat_query,
    placeholder="جرب تكتب: العربية مبتدورش... أو: صفير مع الفرامل... أو: دواسة الفرامل بتسقط لتحت... أو: دواسة الفرامل محجرة"
)

# ---------------- RUN AGENTS ----------------
sensor_percept = {
    'Engine_RPM': rpm, 'Coolant_Temp_C': temp, 'MAF_Air_Flow_gs': maf, 'Short_Term_Fuel_Trim_Pct': stft,
    'Throttle_Pos_Pct': s_tps, 'Battery_Voltage_V': batt_v, 'Alternator_Output_V': alt_v, 'Alternator_Load_Pct': s_load,
    'Tire_FL_PSI': 33.0, 'Tire_FR_PSI': 32.8, 'Tire_RL_PSI': 32.5, 'Tire_RR_PSI': tire_rr,
    'Min_Tire_PSI': min(33.0, 32.8, 32.5, tire_rr), 'DTC_Code': s_dtc
}
pred_root, pred_sev = agent1.perceive_and_diagnose(sensor_percept)

if user_complaint.strip():
    rag_entry, rag_sim = rag_engine.retrieve(user_complaint, batt_v, alt_v)
    effective_fault, required_specialty = rag_entry["fault"], rag_entry["specialty"]
    effective_sev = "NONE" if effective_fault == "System_Normal" else "HIGH"
else:
    rag_entry, rag_sim = rag_engine.retrieve(f"{pred_root} {s_dtc}", batt_v, alt_v)
    effective_fault, required_specialty, effective_sev = pred_root, rag_entry["specialty"], pred_sev

goal_achieved = agent1.is_goal_achieved(effective_fault, alt_v, tire_rr, temp)

# ---------------- GAUGES ----------------
g_col1, g_col2, g_col3, g_col4 = st.columns([1.1, 1.1, 1.1, 1.2])
def make_gauge(title, value, min_v, max_v, unit, warn_thresh, bar_color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number={'suffix': f" {unit}", 'font': {'size': 22, 'color': '#ffffff'}},
        title={'text': title, 'font': {'size': 13, 'color': '#cbd5e1'}},
        gauge={'axis': {'range': [min_v, max_v]}, 'bar': {'color': '#ff4757' if value >= warn_thresh else bar_color}, 'bgcolor': "rgba(15,23,42,0.6)"}
    ))
    fig.update_layout(height=180, margin=dict(l=15, r=15, t=32, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#ffffff"})
    return fig

with g_col1: st.plotly_chart(make_gauge("TACHOMETER", rpm, 0, 6000, "RPM", 4200, accent), width="stretch")
with g_col2: st.plotly_chart(make_gauge("COOLANT TEMP", temp, 50, 130, "°C", 104, accent), width="stretch")
with g_col3: st.plotly_chart(make_gauge("ALTERNATOR OUT", alt_v, 9.0, 15.5, "V", 14.8, "#22c55e" if alt_v >= 13.2 else "#ff4757"), width="stretch")
with g_col4:
    rr_color = "#ff4757" if tire_rr < 26.0 else "#22c55e"
    st.markdown(f"""
    <div class="glass-panel" style="height:180px; display:flex; flex-direction:column; justify-content:space-between;">
        <div style="text-align:center; font-size:13px; color:#cbd5e1; font-weight:bold;">🛞 TPMS TIRE PRESSURE</div>
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:6px;">
            <div class="tpms-box">FL: <span style="color:#22c55e;">33.0 PSI</span></div>
            <div class="tpms-box">FR: <span style="color:#22c55e;">32.8 PSI</span></div>
            <div class="tpms-box">RL: <span style="color:#22c55e;">32.5 PSI</span></div>
            <div class="tpms-box" style="border-color:{rr_color};">RR: <span style="color:{rr_color};">{tire_rr} PSI</span></div>
        </div>
        <div style="font-size:12px; text-align:center; color:#94a3b8;">🔋 Battery: <b>{batt_v}V</b> | DTC: <b style="color:{accent};">{s_dtc}</b></div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- TABS ----------------
tab_agents, tab_maint = st.tabs(["🤖 Live Diagnostic Triage & Smart Dispatch", "🛢️ Preventive Maintenance & Oil Lifecycle Log"])

with tab_agents:
    left_c, right_c = st.columns([1.05, 1.95])
    with left_c:
        status_color = "#22c55e" if goal_achieved else ("#ff4757" if effective_sev == "HIGH" or tire_rr < 26 or alt_v < 13.2 else "#f59e0b")
        st.markdown(f"""
        <div class="glass-panel" style="border-top: 4px solid {status_color};">
            <h4 style="margin:0 0 8px 0; color:{status_color};">🧠 Agent 1: Diagnostic Triage & RAG</h4>
            <p style="font-size:14px; margin:6px 0;"><b>Status:</b> <span style="color:{status_color}; font-weight:bold;">{"✅ VEHICLE SAFE TO DRIVE" if goal_achieved else "🚨 CRITICAL — STOP VEHICLE"}</span></p>
            <p style="font-size:13.5px; margin:6px 0;"><b>Fault Tag:</b> <code>{effective_fault}</code> | <b>Specialty:</b> <code>{required_specialty}</code></p>
            <hr style="border-color:rgba(255,255,255,0.12);">
            <div dir="rtl" style="text-align:right; background:rgba(2,6,23,0.72); padding:12px; border-radius:10px; border-right:4px solid {status_color};">
                <p style="margin:0 0 6px 0; font-size:14.5px; color:{accent}; font-weight:bold;">🔍 التشخيص: {rag_entry['title_ar']}</p>
                <p style="margin:0; font-size:13.5px; line-height:1.65; color:#f1f5f9;">📖 <b>المفروض تعمله دلوقتي:</b><br>{rag_entry['action_ar']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right_c:
        df_rank = agent2.rank_workshops(effective_fault, required_specialty, car_cfg["origin"], model_year, in_warranty, driver_gps)
        st.markdown(f"#### 🧭 Agent 2: Top Service Center Recommendations ({car_cfg['origin']} {model_year} • {required_specialty})")
        for idx, (_, row) in enumerate(df_rank.head(3).iterrows()):
            badge = "🛡️ AUTHORIZED DEALERSHIP (WARRANTY)" if (in_warranty and row["Type"]=="Authorized_Dealership") else f"🔧 {row['Type'].replace('_', ' ').upper()} ({row['Specialty']})"
            st.markdown(f"""
            <div class="workshop-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h4 style="margin:0; color:#ffffff;">🏆 {row['Name']}</h4>
                    <span class="status-pill">UTILITY SCORE: {row['Utility_Score']}</span>
                </div>
                <p style="margin:6px 0 4px 0; font-size:12.5px; color:{accent}; font-weight:bold;">{badge} • Distance: {row['Distance_km']} km • Est. Cost: {row['Estimated_Cost_EGP']:,.0f} EGP</p>
                <p style="margin:4px 0; font-size:13px;">📍 {row['City_Zone']} — {row['Address']} | ⭐ {row['Rating_Score']}/10 | 📞 <code>{row['Phone']}</code></p>
                <p style="margin:6px 0 0 0; font-size:12.5px; color:#cbd5e1; background:rgba(0,0,0,0.35); padding:6px 10px; border-radius:6px;">💬 <i>"{row['Top_Comment']}"</i></p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"📲 Dispatch / Book Service at {row['Name'].split('(')[0]}", key=f"book_{idx}"):
                st.success(f"✅ Instant Dispatch Ticket confirmed for **{row['Name']}**! Direct Hotline: **{row['Phone']}** — Estimated Cost: **{row['Estimated_Cost_EGP']:,.0f} EGP**.")

    st.markdown("#### 🗺️ Greater Cairo Service Centers GPS Map")
    st.map(df_rank.rename(columns={"Lat": "lat", "Lon": "lon"}))

with tab_maint:
    st.markdown(f"### 📋 Preventive Maintenance & Consumables Log ({selected_car_name})")
    origin_mult = agent2.ORIGIN_COST_MULT[car_cfg["origin"]]

    # Initialize maintenance records in session_state so user additions persist live
    if "maint_db" not in st.session_state:
        st.session_state.maint_db = [
            {"Component": "Engine Oil & Filter (5W-40)", "Last_Changed_Odo": last_oil_odo, "Interval_KM": oil_interval, "Est_Cost_EGP": int(1850 * origin_mult)},
            {"Component": "Timing Belt & Tensioner Kit", "Last_Changed_Odo": current_odo - 56500, "Interval_KM": 60000, "Est_Cost_EGP": int(3200 * origin_mult)},
            {"Component": "Rear Brake Pads", "Last_Changed_Odo": current_odo - 1700, "Interval_KM": 35000, "Est_Cost_EGP": int(1050 * origin_mult)},
            {"Component": "Front Brake Pads", "Last_Changed_Odo": current_odo - 30500, "Interval_KM": 30000, "Est_Cost_EGP": int(1550 * origin_mult)},
            {"Component": "Spark Plugs Set", "Last_Changed_Odo": current_odo - 26500, "Interval_KM": 25000, "Est_Cost_EGP": int(1300 * origin_mult)},
            {"Component": "ATF Transmission Fluid", "Last_Changed_Odo": current_odo - 36500, "Interval_KM": 40000, "Est_Cost_EGP": int(2700 * origin_mult)}
        ]

    # Interactive Form to Add / Update a Maintenance Record
    with st.expander("➕ Log New Maintenance / Add Service Record", expanded=False):
        with st.form("add_maint_form", clear_on_submit=True):
            f_c1, f_c2, f_c3, f_c4 = st.columns([2, 1.2, 1.2, 1])
            new_comp = f_c1.text_input("Component / Service Name", placeholder="e.g., Air Filter, Battery, Front Brake Pads...")
            new_odo = f_c2.number_input("Odometer at Service (KM)", min_value=0, max_value=1000000, value=int(current_odo), step=500)
            new_interval = f_c3.number_input("Lifespan Interval (KM)", min_value=1000, max_value=200000, value=10000, step=1000)
            new_cost = f_c4.number_input("Cost (EGP)", min_value=0, max_value=100000, value=1200, step=100)

            submitted = st.form_submit_button("✅ Save Maintenance Record", width="stretch")
            if submitted and new_comp.strip():
                # Update existing component if same name, otherwise append new record
                updated = False
                for rec in st.session_state.maint_db:
                    if rec["Component"].lower() == new_comp.strip().lower():
                        rec["Last_Changed_Odo"] = new_odo
                        rec["Interval_KM"] = new_interval
                        rec["Est_Cost_EGP"] = new_cost
                        updated = True
                        break
                if not updated:
                    st.session_state.maint_db.insert(0, {
                        "Component": new_comp.strip(),
                        "Last_Changed_Odo": new_odo,
                        "Interval_KM": new_interval,
                        "Est_Cost_EGP": new_cost
                    })
                st.success(f"✅ Logged **{new_comp.strip()}** at **{new_odo:,} KM**!")

    # Build live display table with dynamic remaining KM & status
    display_rows = []
    for item in st.session_state.maint_db:
        rem = item["Interval_KM"] - (current_odo - item["Last_Changed_Odo"])
        status = "🚨 OVERDUE - CHANGE NOW" if rem <= 0 else ("⚠️ DUE SOON" if rem <= 3500 else "✅ HEALTHY")
        display_rows.append({
            "Component": item["Component"],
            "Last_Changed_Odo": item["Last_Changed_Odo"],
            "Interval_KM": item["Interval_KM"],
            "KM_Remaining": rem,
            "Est_Cost_EGP": item["Est_Cost_EGP"],
            "Status": status
        })

    st.dataframe(pd.DataFrame(display_rows), width="stretch")