import os
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# 1. KNOWLEDGE BASE & SENSOR-FUSED RAG RETRIEVER (TF-IDF + COSINE SIMILARITY)
# ==============================================================================
REPAIR_MANUAL_KB = [
    {
        "id": "NO_START", "fault": "Alternator_Charging_Failure", "specialty": "Auto_Electric", "dtc": "P0562",
        "keywords": "مبتدورش مش بتدور مبتشتغلش بطارية دينامو تكتكة مارش كهرباء no start crank battery alternator",
        "title_ar": "مشكلة في دائرة بدء التشغيل (البطارية / الدينامو)",
        "action_ar": "DYNAMIC_NO_START"
    },
    {
        "id": "BRAKE_PADS_SQUEAL", "fault": "Brake_Pads_Wear", "specialty": "Brakes_Suspension", "dtc": "MANUAL_SYMPTOM",
        "keywords": "صفير فرامل تصفير صوت مع الفرامل تيل فرامل حكة brake squeal noise pads",
        "title_ar": "تآكل تيل الفرامل (وصول لمؤشر التحذير المعدني)",
        "action_ar": "الصفير المستمر عند الضغط على الفرامل فقط سببه انتهاء سمك تيل الفرامل واحتكاك ريشة التحذير بالطنابير. يجب تغيير طقم التيل فوراً قبل تجريح سطح الطنابير."
    },
    {
        "id": "BRAKE_ROTORS_WARP", "fault": "Brake_Rotors_Warping", "specialty": "Brakes_Suspension", "dtc": "MANUAL_SYMPTOM",
        "keywords": "رجة في الفرامل رعشة في الدركسيون مع الفرامل طنابير نبضة brake vibration shaking rotors",
        "title_ar": "اعوجاج في طنابير الفرامل (Brake Rotors Warping)",
        "action_ar": "وجود رجة أو نبضة في دواسة الفرامل وعجلة القيادة عند التهدئة من سرعة عالية يعني وجود عدم استواء (اعوجاج) في الطنابير. الحل: خرط الطنابير أو تغييرها مع طقم تيل جديد."
    },
    {
        "id": "BRAKE_MASTER_CYLINDER", "fault": "Brake_Master_Cylinder_Leak", "specialty": "Brakes_Suspension", "dtc": "MANUAL_SYMPTOM",
        "keywords": "دواسة الفرامل بتسقط بتغوص لتحت مفيش فرامل ماستر عمومي زيت فرامل brake pedal sinking soft master cylinder",
        "title_ar": "تسريب داخلي في ماستر الفرامل العمومي (Master Cylinder)",
        "action_ar": "خطير جداً: غوص دواسة الفرامل ببطء لأسفل تحت قدمك يعني تلف جلد ماستر الفرامل العمومي أو نقص زيت الباكم. لا تقُد السيارة على سرعة واطلب ونش إنقاذ فوراً."
    },
    {
        "id": "BRAKE_SERVO_BOOSTER", "fault": "Brake_Servo_Booster_Failure", "specialty": "Brakes_Suspension", "dtc": "MANUAL_SYMPTOM",
        "keywords": "دواسة الفرامل محجرة ناشفة تقيلة جداً سيرفو باكم brake pedal hard stiff servo booster",
        "title_ar": "تلف السيرفو أو خرطوم الفاكيوم (Brake Servo Booster)",
        "action_ar": "تحجير دواسة الفرامل وصعوبة الضغط عليها رغم عدم سقوطها سببه انقطاع هواء الفاكيوم عن السيرفو. ستحتاج لقوة ضغط مضاعفة لإيقاف السيارة—توجه بحذر شديد لأقرب مركز عفشة وفرامل."
    },
    {
        "id": "IGNITION_MISFIRE", "fault": "Spark_Plug_Ignition_Misfire", "specialty": "All_Systems", "dtc": "P0300",
        "keywords": "تقطيع تنتيش بوجيهات موبينة رجة اهتزاز misfire spark plug ignition coil vibration",
        "title_ar": "خلل في الإشعال (تلف البوجيهات أو المباين)",
        "action_ar": "بناءً على ارتفاع قراءة Fuel Trim مع كود P0300، افحص فيش المباين وأسلاك البوجيهات. تجنب الضغط بقوة على البنزين لأن البنزين غير المحترق يتلف علبة البيئة."
    },
    {
        "id": "ENGINE_OVERHEAT", "fault": "Cooling_Thermostat_Overheat", "specialty": "AC_Cooling", "dtc": "P0217",
        "keywords": "سخونة حرارة عالية بتسخن مية بتغلي ريداتير ثرموستات مروحة تكييف overheat hot temp cooling ac",
        "title_ar": "ارتفاع خطير في درجة حرارة المحرك / دورة التبريد",
        "action_ar": "أغلق التكييف فوراً، شغّل سخان المقصورة على أقصى درجة لسحب الحرارة من المحرك، واركن في مكان آمن. ممنوع فتح غطاء الريداتير والمحرك ساخن."
    },
    {
        "id": "FLAT_TIRE", "fault": "TPMS_Rapid_Tire_Deflation", "specialty": "Brakes_Suspension", "dtc": "C0750",
        "keywords": "كاوتش فردة نايمة بنشر إطار ضغط الهواء عجلة tire flat puncture tpms",
        "title_ar": "انخفاض حاد في ضغط الإطار الخلفي الأيمن",
        "action_ar": "اركن على جانب الطريق فوراً. السير على ضغط أقل من 22 PSI يمزق جوانب الإطار ويتلف الجنط. اطلب خدمة تغيير إطارات متنقلة."
    },
    {
        "id": "GEARBOX_FAULT", "fault": "Transmission_Solenoid_Stuck", "specialty": "Gearbox_Transmission", "dtc": "P0700",
        "keywords": "فتيس نقلات غيارات نتعة جيربوكس زيت الفتيس transmission gear solenoid",
        "title_ar": "عطل في صمامات / حساسات الفتيس الأوتوماتيك",
        "action_ar": "الفتيس دخل وضع الحماية (Limp Mode) وثبت على النقلة الثالثة لحماية الكلاتشات. تجنب السرعات العالية وتوجه لمركز متخصص في الفتيس الأوتوماتيك."
    },
    {
        "id": "SYSTEM_NORMAL", "fault": "System_Normal", "specialty": "All_Systems", "dtc": "NONE",
        "keywords": "طبيعي سليم مفيش مشكلة كويسه ممتاز normal good fine healthy",
        "title_ar": "السيارة في حالة فنية ممتازة",
        "action_ar": "جميع قراءات الحساسات، حرارة المحرك، شحن الدينامو، وضغط الإطارات في المعدل الطبيعي. يمكنك استكمال رحلتك بأمان."
    }
]

class RepairManualRAG:
    def __init__(self):
        docs = [self._normalize(f"{x['fault']} {x['dtc']} {x['keywords']} {x['title_ar']}") for x in REPAIR_MANUAL_KB]
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        self.doc_matrix = self.vectorizer.fit_transform(docs)

    @staticmethod
    def _normalize(text: str) -> str:
        text = re.sub(r'[إأآا]', 'ا', text.lower().strip())
        return re.sub(r'ة', 'ه', re.sub(r'ى', 'ي', text))

    def retrieve(self, query_text: str, batt_v: float, alt_v: float):
        q_vec = self.vectorizer.transform([self._normalize(query_text)])
        sims = cosine_similarity(q_vec, self.doc_matrix).flatten()
        idx = int(sims.argmax())
        entry = REPAIR_MANUAL_KB[idx].copy()

        # Sensor-Fused Reasoning for No-Start / Electrical Issues
        if entry["id"] == "NO_START":
            if alt_v < 13.0 and batt_v < 11.8:
                entry["action_ar"] = f"بناءً على قياسات الحساسات: فولت البطارية منخفض ({batt_v}V) وخرج الدينامو متوقف ({alt_v}V)؛ هناك احتمال كبير لقطع سير الدينامو أو تلف الكتاوت."
            elif batt_v < 11.8 and alt_v >= 13.2:
                entry["action_ar"] = f"بناءً على قياسات الحساسات: الدينامو يشحن بكفاءة ({alt_v}V) لكن فولت البطارية واقع ({batt_v}V)؛ أنت بحاجة لتغيير البطارية."
            else:
                entry["action_ar"] = f"قراءات البطارية ({batt_v}V) والدينامو ({alt_v}V) سليمة؛ افحص كتاوت المارش أو طرمبة البنزين."

        return entry, round(float(sims[idx]) * 100, 1)


# ==============================================================================
# 2. AGENT 1: MODEL-BASED, GOAL-BASED DIAGNOSTIC AGENT (RANDOM FOREST)
# ==============================================================================
class DiagnosticTriageAgent:
    FEATURE_COLS = [
        'Engine_RPM', 'Coolant_Temp_C', 'MAF_Air_Flow_gs', 'Short_Term_Fuel_Trim_Pct',
        'Throttle_Pos_Pct', 'Battery_Voltage_V', 'Alternator_Output_V', 'Alternator_Load_Pct',
        'Tire_FL_PSI', 'Tire_FR_PSI', 'Tire_RL_PSI', 'Tire_RR_PSI', 'Min_Tire_PSI'
    ]

    def __init__(self, telemetry_df: pd.DataFrame):
        X = pd.get_dummies(telemetry_df[self.FEATURE_COLS + ['DTC_Code']])
        y = telemetry_df['Root_Cause'] + " | " + telemetry_df['Severity']
        X_tr, X_te, y_tr, y_te = X.iloc[:2800], X.iloc[2800:], y.iloc[:2800], y.iloc[2800:]

        self.model = RandomForestClassifier(n_estimators=160, max_depth=14, random_state=42)
        self.model.fit(X_tr, y_tr)
        self.encoded_cols = X.columns
        self.accuracy = round(self.model.score(X_te, y_te) * 100, 2)

    def perceive_and_diagnose(self, sensor_percept: dict):
        X_in = pd.get_dummies(pd.DataFrame([sensor_percept])).reindex(columns=self.encoded_cols, fill_value=0)
        root_cause, severity = self.model.predict(X_in)[0].split(" | ")
        return root_cause, severity

    @staticmethod
    def is_goal_achieved(fault: str, alt_v: float, tire_rr: float, temp: float) -> bool:
        return (fault == "System_Normal" and alt_v >= 13.2 and tire_rr >= 27.0 and temp <= 103.0)


# ==============================================================================
# 3. AGENT 2: UTILITY-BASED SMART DISPATCH AGENT (MAUT + HAVERSINE DISTANCE)
# ==============================================================================
class SmartDispatchUtilityAgent:
    ORIGIN_COST_MULT = {"German": 2.40, "European": 1.75, "Japanese": 1.40, "Korean": 1.05, "Chinese": 1.00}
    FAULT_PARTS_EGP = {
        "System_Normal": 0, "Spark_Plug_Ignition_Misfire": 1400, "Brake_Pads_Wear": 1200,
        "Brake_Rotors_Warping": 2600, "Brake_Master_Cylinder_Leak": 2100, "Brake_Servo_Booster_Failure": 2800,
        "Cooling_Thermostat_Overheat": 1900, "Alternator_Charging_Failure": 3600,
        "TPMS_Rapid_Tire_Deflation": 450, "Transmission_Solenoid_Stuck": 5500
    }

    def __init__(self, workshops_df: pd.DataFrame):
        self.registry = workshops_df.copy()

    @staticmethod
    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
        a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
        return np.round(2 * R * np.arcsin(np.sqrt(a)) * 1.25, 1)

    def rank_workshops(self, fault: str, specialty: str, origin: str, year: int, in_warranty: bool, driver_gps: tuple):
        df = self.registry.copy()
        lat, lon = driver_gps
        df["Distance_km"] = self.haversine_km(lat, lon, df["Lat"], df["Lon"])

        origin_mult = self.ORIGIN_COST_MULT.get(origin, 1.0)
        base_parts = self.FAULT_PARTS_EGP.get(fault, 1400)

        def calc_cost(row):
            if in_warranty and row["Type"] == "Authorized_Dealership":
                return 350.0
            shop_mult = 1.65 if row["Type"] == "Authorized_Dealership" else (1.15 if row["Type"] == "Specialized_Center" else 0.85)
            return round((base_parts * origin_mult * shop_mult) + (row["Base_Labor_Cost_EGP"] * (1.3 if origin == "German" else 1.0)), -1)

        df["Estimated_Cost_EGP"] = df.apply(calc_cost, axis=1)

        def calc_expertise(row):
            origin_bonus = 0.12 if origin in row["Supported_Origins"].split(",") else -0.35
            spec_bonus = 0.25 if row["Specialty"] == specialty else (0.10 if row["Specialty"] == "All_Systems" else -0.30)
            if in_warranty:
                base = 0.85 if row["Type"] == "Authorized_Dealership" else 0.15
            elif year <= 2015:
                base = 0.80 if row["Type"] in ["Specialized_Center", "Local_Workshop"] else 0.25
            else:
                base = 0.82 if row["Type"] == "Specialized_Center" else 0.65
            return np.clip(base + origin_bonus + spec_bonus, 0.05, 1.0)

        df["Expertise_Match"] = df.apply(calc_expertise, axis=1)

        d_norm = (df["Distance_km"] - df["Distance_km"].min()) / (df["Distance_km"].max() - df["Distance_km"].min() + 1e-5)
        c_norm = (df["Estimated_Cost_EGP"] - df["Estimated_Cost_EGP"].min()) / (df["Estimated_Cost_EGP"].max() - df["Estimated_Cost_EGP"].min() + 1e-5)
        r_norm = df["Rating_Score"] / 10.0

        w1, w2, w3, w4 = (0.48, 0.22, 0.08, 0.22) if in_warranty else ((0.36, 0.24, 0.18, 0.22) if year <= 2015 else (0.38, 0.20, 0.18, 0.24))
        raw_u = (w1 * df["Expertise_Match"]) - (0.55 * w2 * d_norm) - (0.55 * w3 * c_norm) + (w4 * r_norm)
        df["Utility_Score"] = np.round(np.clip(raw_u * 142.2, 30.0, 94.2), 1)

        return df.sort_values(by="Utility_Score", ascending=False)