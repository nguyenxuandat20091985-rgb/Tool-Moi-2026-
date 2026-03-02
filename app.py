# ================= IMPORT THƯ VIỆN =================
import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime

# ================= CẤU HÌNH HỆ THỐNG =================
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    st.error("⚠️ Chưa cấu hình API Key trong Secrets!")
    st.stop()

DB_FILE = "titan_permanent_v32.json"

# ================= QUẢN LÝ QUOTA API =================
@st.cache_resource
def init_quota():
    return {'count': 0, 'last_reset': datetime.now().date(), 'limit': 18, 'error': None}

quota = init_quota()

def check_quota():
    today = datetime.now().date()
    if quota['last_reset'] != today:
        quota['count'] = 0
        quota['last_reset'] = today
        quota['error'] = None
    return quota['limit'] - quota['count']

def use_quota():
    quota['count'] += 1

# ================= KHỞI TẠO AI =================
@st.cache_resource
def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        preferred = ['models/gemini-1.5-flash', 'models/gemini-pro', 'models/gemini-1.0-pro']
        selected = next((m for m in preferred if m in models), models[0] if models else None)
        if selected:
            return genai.GenerativeModel(selected), selected
        return None, None
    except Exception as e:
        return None, None

neural_engine, model_used = setup_neural()

# ================= QUẢN LÝ DỮ LIỆU =================
def load_db():
    if "history" in st.session_state and st.session_state.history:
        return st.session_state.history
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except:
            return []
    return []

def save_db(data):
    try:
        with open(DB_FILE, "w") as f:
            json.dump(data[-3000:], f)
    except:
        pass

def clean_and_validate_data(raw_text, existing_history):
    cleaned = re.sub(r'[\s\t]+', ' ', raw_text.strip())
    matches = re.findall(r'\b\d{5}\b', cleaned)
    unique = list(dict.fromkeys(matches))
    new_nums = [n for n in unique if n not in existing_history]
    rejected = [m for m in re.findall(r'\b\d{3,7}\b', cleaned) if len(re.sub(r'\D','',m)) != 5]
    return {
        'found': len(matches), 'unique': len(unique), 'new': len(new_nums),
        'dup': len(matches) - len(unique), 'rejected': list(set(rejected))[:5],
        'numbers': new_nums
    }

def load_data_from_json(uploaded_file):
    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            return data if isinstance(data, list) else []
        except:
            return []
    return []

def convert_df_to_json(data):
    return json.dumps(data[-3000:], ensure_ascii=False).encode('utf-8')

# ================= THUẬT TOÁN =================
def detect_risk(history, window=20):
    if len(history) < window:
        return {"score": 0, "warnings": [], "level": "OK"}
    recent = history[-window:]
    all_d = "".join(recent)
    freq = Counter(all_d)
    warnings = []
    score = 0
    
    most = freq.most_common(1)
    if most and most[0][1] > 15:
        warnings.append(f"Số {most[0][0]} ra {most[0][1]}/{window} kỳ")
        score += 30
    
    for pos in range(5):
        seq = [int(n[pos]) if len(n)>pos else 0 for n in recent]
        streak = max_streak = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 1
        if max_streak >= 4:
            warnings.append(f"Vị {pos} bệt {max_streak}")
            score += 25
    
    total = len(all_d)
    if total > 0:
        entropy = -sum((c/total)*np.log2(c/total) for c in freq.values() if c>0)
        if entropy < 2.8:
            warnings.append(f"Pattern quá đều")
            score += 25
    
    level = "🔴 DỪNG" if score >= 60 else "🟡 CẨN THẬN" if score >= 40 else "🟢 OK"
    return {"score": score, "warnings": warnings, "level": level}

def fallback_predict(history):
    all_d = "".join(history[-50:] if len(history)>=50 else history)
    freq = Counter(all_d)
    top = [str(x[0]) for x in freq.most_common(10)]
    main_3 = "".join(list(dict.fromkeys(top))[:3])
    support = list(dict.fromkeys(top))[3:7]
    return {
        "main_3": main_3 if len(main_3)==3 else main_3.ljust(3,'0')[:3],
        "support_4": "".join(support) if len(support)>=4 else "".join(support).ljust(4,'0')[:4],
        "decision": "THEO DÕI",
        "logic": "Thống kê tần suất",
        "color": "Yellow",
        "conf": 70,
        "is_fallback": True
    }

# ================= KHỞI TẠO SESSION STATE =================
if "history" not in st.session_state:
    st.session_state.history = load_db()
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_clean" not in st.session_state:
    st.session_state.last_clean = None

# ================= GIAO DIỆN =================
st.set_page_config(page_title="TITAN v32.0 PRO", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .prediction-card {
        background: #0d1117; border: 1px solid #30363d;
        border-radius: 12px; padding: 20px; margin-top: 20px;
    }
    .num-box {
        font-size: 70px; font-weight: 900; color: #ff5858;
        text-align: center; letter-spacing: 10px; border-right: 2px solid #30363d;
    }
    .lot-box {
        font-size: 50px; font-weight: 700; color: #58a6ff;
        text-align: center; letter-spacing: 5px; padding-left: 20px;
    }
    .status-bar { padding: 10px; border-radius: 8px; text-align: center; font-weight: bold; margin-bottom: 10px; }
    .quota-ok { background: #064e3b; border: 1px solid #10b981; padding: 8px; border-radius: 6px; text-align: center; margin: 10px 0; }
    .quota-warn { background: #422006; border: 1px solid #9a6700; padding: 8px; border-radius: 6px; text-align: center; margin: 10px 0; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: #58a6ff;'>🎯 TITAN v32.0 PRO</h2>", unsafe_allow_html=True)

remaining = check_quota()
if remaining > 0:
    st.markdown(f"<div class='quota-ok'>✅ Quota: <strong>{remaining}/{quota['limit']}</strong> | Model: <code>{model_used.split('/')[-1] if model_used else 'None'}</code></div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div class='quota-warn'>⚠️ <strong>HẾT QUOTA!</strong></div>", unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:
    st.header("💾 Database")
    uploaded_db = st.file_uploader("📂 Nạp DB", type="json")
    if uploaded_db:
        st.session_state.history = load_data_from_json(uploaded_db)
        save_db(st.session_state.history)
        st.success(f"✅ {len(st.session_state.history)} kỳ")
        st.rerun()
    
    st.divider()
    if st.session_state.history:
        json_data = convert_df_to_json(st.session_state.history)
        st.download_button("💾 Tải DB", json_data, f"titan_{datetime.now().strftime('%Y%m%d')}.json", "application/json")
    
    st.divider()
    st.metric("📊 Tổng kỳ", len(st.session_state.history))
    st.metric("🔌 API calls", quota['count'])
    
    if st.button("🗑️ Xóa"):
        st.session_state.history = []
        st.session_state.last_prediction = None
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
        st.rerun()

# ================= NHẬP LIỆU =================
with st.container():
    col_in, col_st = st.columns([2, 1])
    with col_in:
        raw_input = st.text_area("📡 Dán dữ liệu (5 số mỗi kỳ):", height=100, placeholder="32880\n21808...")
    with col_st:
        st.write(f"📊 Dữ liệu: **{len(st.session_state.history)} kỳ**")
        c1, c2 = st.columns(2)
        btn_save = c1.button("🚀 GIẢI MÃ")
        btn_reset = c2.button("🗑️ RESET")

if st.session_state.last_clean:
    st.markdown("### 📊 KẾT QUẢ LÀM SẠCH")
    c1, c2, c3, c4 = st.columns(4)
    d = st.session_state.last_clean
    c1.metric("🔍 Tìm", d['found'])
    c2.metric("✅ Riêng", d['unique'])
    c3.metric("➕ Mới", d['new'])
    c4.metric("🗑️ Trùng", d['dup'])

if btn_reset:
    st.session_state.history = []
    st.session_state.last_prediction = None
    st.session_state.last_clean = None
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    st.rerun()

if btn_save and raw_input:
    clean_result = clean_and_validate_data(raw_input, st.session_state.history)
    st.session_state.last_clean = clean_result
    
    if clean_result['new'] > 0:
        st.session_state.history.extend(clean_result['numbers'])
        st.session_state.history = list(dict.fromkeys(st.session_state.history))
        save_db(st.session_state.history)
        
        risk = detect_risk(st.session_state.history)
        
        if risk['score'] >= 60:
            st.session_state.last_prediction = {
                "main_3": "000", "support_4": "0000", "decision": "DỪNG",
                "logic": f"Risk cao: {' | '.join(risk['warnings'])}",
                "color": "Red", "conf": 99, "is_fallback": True, "risk": risk
            }
            st.rerun()
        
        if check_quota() <= 0 or neural_engine is None:
            result = fallback_predict(st.session_state.history)
            result['risk'] = risk
            st.session_state.last_prediction = result
            st.rerun()
        
        with st.spinner("🤖 Đang phân tích..."):
            prompt = f"""TITAN v32.0 - DATA: {st.session_state.history[-100:]} | Risk: {risk['score']}/100
            JSON: {{"main_3":"123","support_4":"4567","decision":"ĐÁNH","logic":"Ngắn","color":"Green","conf":85}}"""
            try:
                response = neural_engine.generate_content(prompt)
                text = response.text.strip()
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                    result['main_3'] = str(result.get('main_3',''))[:3].ljust(3,'0')
                    result['support_4'] = str(result.get('support_4',''))[:4].ljust(4,'0')
                    result['risk'] = risk
                    result['is_fallback'] = False
                    st.session_state.last_prediction = result
                    use_quota()
                    st.rerun()
            except Exception as e:
                result = fallback_predict(st.session_state.history)
                result['risk'] = risk
                st.session_state.last_prediction = result
                st.rerun()

# ================= HIỂN THỊ KẾT QUẢ - ✅ ĐÃ SỬA LỖI =================
if st.session_state.last_prediction is not None:
    res = st.session_state.last_prediction
    
    # ✅ KIỂM TRA AN TOÀN TRƯỚC KHI TRUY CẬP
    if isinstance(res, dict):
        # Hiển thị risk
        if 'risk' in res and isinstance(res['risk'], dict):
            risk = res['risk']
            if risk.get('score', 0) >= 60:
                st.markdown(f"<div class='status-bar' style='background:#da3633'>🚨 {risk.get('score',0)}/100 - {risk.get('level','')}</div>", unsafe_allow_html=True)
            elif risk.get('score', 0) >= 40:
                st.markdown(f"<div class='status-bar' style='background:#d29922'>⚠️ {risk.get('score',0)}/100</div>", unsafe_allow_html=True)
        
        # Status bar
        color = res.get('color', 'green').lower()
        bg = "#238636" if color == "green" else "#da3633" if color == "red" else "#d29922"
        badge = "🔄" if res.get('is_fallback', False) else "🤖"
        st.markdown(f"<div class='status-bar' style='background:{bg}'>{res.get('decision','')} ({res.get('conf',0)}%) {badge}</div>", unsafe_allow_html=True)

        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.markdown("<p style='color:#8b949e'>🔥 3 SỐ CHÍNH</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='num-box'>{res.get('main_3','000')}</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<p style='color:#8b949e'>🛡️ 4 SỐ LÓT</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='lot-box'>{res.get('support_4','0000')}</div>", unsafe_allow_html=True)
        
        st.divider()
        st.write(f"💡 **LOGIC:** {res.get('logic','')}")
        
        full_dan = "".join(sorted(set(str(res.get('main_3','')) + str(res.get('support_4','')))))
        st.text_input("📋 DÀN 7 SỐ:", full_dan)
        st.markdown("</div>", unsafe_allow_html=True)

# Thống kê
if st.session_state.history:
    with st.expander("📊 Thống kê"):
        all_d = "".join(st.session_state.history[-50:])
        st.bar_chart(pd.Series(Counter(all_d)).sort_index())

st.caption(f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M')} | TITAN v32.0")