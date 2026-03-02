# ================= IMPORT THƯ VIỆN =================
import streamlit as st
import google.generativeai as genai
import re
import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import time

# ================= CẤU HÌNH =================
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    st.error("⚠️ Chưa cấu hình API Key!")
    st.stop()

st.set_page_config(page_title="TITAN v33.0 FAST", layout="wide", initial_sidebar_state="collapsed")

# ================= QUOTA & CACHE =================
@st.cache_resource
def init_quota():
    return {'count': 0, 'last_reset': datetime.now().date(), 'limit': 18}

@st.cache_data(ttl=300)
def cached_analysis(history_str, window):
    """Cache kết quả phân tích 5 phút"""
    return None

quota = init_quota()

def check_quota():
    today = datetime.now().date()
    if quota['last_reset'] != today:
        quota['count'] = 0
        quota['last_reset'] = today
    return quota['limit'] - quota['count']

# ================= AI SETUP =================
@st.cache_resource
def setup_ai():
    try:
        genai.configure(api_key=API_KEY)
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        selected = next((m for m in ['models/gemini-1.5-flash', 'models/gemini-pro'] if m in models), None)
        return genai.GenerativeModel(selected) if selected else None, selected
    except:
        return None, None

ai_engine, ai_model = setup_ai()

# ================= THUẬT TOÁN NHANH =================
def fast_frequency_analysis(history, top_n=10):
    """Phân tích tần suất nhanh với trọng số thời gian"""
    if not history:
        return {}
    recent = history[-50:] if len(history) >= 50 else history
    all_digits = "".join(recent)
    weights = np.linspace(0.5, 1.0, len(all_digits))
    freq = defaultdict(float)
    for i, d in enumerate(all_digits):
        freq[d] += weights[i]
    return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n])

def fast_pattern_detection(history):
    """Phát hiện pattern nhanh"""
    if len(history) < 10:
        return {"bệt": [], "nhịp": [], "lạnh": []}
    recent = history[-20:]
    patterns = {"bệt": [], "nhịp": [], "lạnh": []}
    
    for pos in range(5):
        seq = [n[pos] if len(n) > pos else '0' for n in recent]
        # Bệt
        for i in range(len(seq)-1):
            if seq[i] == seq[i+1]:
                patterns["bệt"].append(f"V{pos}:{seq[i]}")
                break
        # Nhịp 2
        for i in range(len(seq)-2):
            if seq[i] == seq[i+2] and seq[i] != seq[i+1]:
                patterns["nhịp"].append(f"V{pos}:{seq[i]}")
                break
    
    # Số lạnh
    all_d = "".join(recent)
    for d in '0123456789':
        if d not in all_d:
            patterns["lạnh"].append(d)
    
    return patterns

def fast_risk_check(history):
    """Kiểm tra risk nhanh"""
    if len(history) < 15:
        return {"score": 0, "level": "✅ OK"}
    recent = history[-20:]
    all_d = "".join(recent)
    freq = Counter(all_d)
    score = 0
    
    # Số ra nhiều
    if freq.most_common(1)[0][1] > 12:
        score += 30
    
    # Bệt
    for pos in range(5):
        seq = [n[pos] if len(n) > pos else '0' for n in recent]
        streak = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i-1]:
                streak += 1
            else:
                streak = 1
            if streak >= 4:
                score += 25
                break
    
    return {"score": score, "level": "🔴 DỪNG" if score >= 60 else "🟡 CẨN THẬN" if score >= 40 else "✅ OK"}

def ensemble_prediction(history, patterns, risk):
    """✅ AI KÉP: Thống kê + Rule-based"""
    if not history:
        return {"main_3": "000", "support_4": "0000", "conf": 50}
    
    # 1. Thống kê tần suất
    freq = fast_frequency_analysis(history)
    hot_nums = list(freq.keys())[:7]
    
    # 2. Pattern-based
    pattern_nums = []
    for p in patterns.get("bệt", []) + patterns.get("nhịp", []):
        if ":" in p:
            num = p.split(":")[1]
            if num not in pattern_nums:
                pattern_nums.append(num)
    
    # 3. Kết hợp: Ưu tiên số từ pattern + hot
    combined = pattern_nums + [n for n in hot_nums if n not in pattern_nums]
    main_3 = "".join(combined[:3]).ljust(3, '0')[:3]
    support_4 = "".join(combined[3:7]).ljust(4, '0')[:4]
    
    # Confidence dựa trên risk
    conf = max(50, 95 - risk['score'])
    
    return {"main_3": main_3, "support_4": support_4, "conf": conf, "method": "Ensemble"}

def ai_prediction(history, patterns, risk):
    """Gemini AI prediction"""
    if not ai_engine or risk['score'] >= 60:
        return None
    
    try:
        prompt = f"""TITAN v33.0 FAST - Phân tích nhanh:
DATA: {history[-50:] if len(history)>=50 else history}
Patterns: {patterns}
Risk: {risk['score']}/100

JSON: {{"main_3":"123","support_4":"4567","logic":"Ngắn","conf":85}}"""
        
        response = ai_engine.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.3, max_output_tokens=256)
        )
        
        text = response.text.strip()
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            result['method'] = "Gemini AI"
            return result
    except:
        pass
    return None

# ================= SESSION STATE =================
for key in ["history", "prediction", "active_tab", "last_update"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "history" else (0 if key == "active_tab" else (None if key == "prediction" else None))

# ================= UI CSS =================
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .main-card { background: #0d1117; border: 1px solid #30363d; border-radius: 12px; padding: 15px; margin: 10px 0; }
    .num-display { font-size: 60px; font-weight: 900; color: #ff5858; text-align: center; letter-spacing: 8px; }
    .status-badge { padding: 8px 15px; border-radius: 20px; text-align: center; font-weight: bold; display: inline-block; width: 100%; }
    .quick-stat { background: #1f2937; padding: 10px; border-radius: 8px; text-align: center; }
    .stat-value { font-size: 24px; font-weight: bold; color: #58a6ff; }
    .stat-label { font-size: 12px; color: #8b949e; }
    [data-testid="stTab"] { background: #161b22; border-radius: 8px; padding: 10px; }
    </style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("<h2 style='text-align:center;color:#58a6ff;margin:10px 0'>🎯 TITAN v33.0 FAST</h2>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center;color:#8b949e;margin:-10px 0 15px 0'>⚡ Xử lý <1s | 🤖 AI Kép | 📱 Mobile Optimized</p>", unsafe_allow_html=True)

# ================= TABS =================
tabs = st.tabs(["📊 TỔNG QUAN", "🔮 DỰ ĐOÁN", "📈 PHÂN TÍCH", "⚙️ CÀI ĐẶT"])

# ================= TAB 1: TỔNG QUAN =================
with tabs[0]:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Tổng kỳ", len(st.session_state.history))
    with col2:
        risk = fast_risk_check(st.session_state.history)
        st.metric("⚠️ Risk", f"{risk['score']}/100")
    with col3:
        st.metric("🕐 Cập nhật", datetime.now().strftime("%H:%M"))
    
    # Quick stats
    if st.session_state.history:
        st.markdown("### 📈 XU HƯỚNG NHANH")
        patterns = fast_pattern_detection(st.session_state.history)
        freq = fast_frequency_analysis(st.session_state.history, 5)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class='quick-stat'>
                <div class='stat-value'>{list(freq.keys())[0] if freq else '-'}</div>
                <div class='stat-label'>Số nóng nhất</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class='quick-stat'>
                <div class='stat-value'>{len(patterns['bệt'])}</div>
                <div class='stat-label'>Cầu bệt</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class='quick-stat'>
                <div class='stat-value'>{len(patterns['lạnh'])}</div>
                <div class='stat-label'>Số lạnh</div>
            </div>""", unsafe_allow_html=True)

# ================= TAB 2: DỰ ĐOÁN =================
with tabs[1]:
    if st.button("🔮 DỰ ĐOÁN NHANH", type="primary", use_container_width=True):
        with st.spinner("⚡ Đang phân tích..."):
            start_time = time.time()
            
            patterns = fast_pattern_detection(st.session_state.history)
            risk = fast_risk_check(st.session_state.history)
            
            # Ensemble prediction (nhanh)
            ensemble_result = ensemble_prediction(st.session_state.history, patterns, risk)
            
            # AI prediction (chậm hơn nhưng thông minh hơn)
            ai_result = None
            if check_quota() > 0:
                ai_result = ai_prediction(st.session_state.history, patterns, risk)
                if ai_result:
                    quota['count'] += 1
            
            # Kết hợp kết quả
            if ai_result and ai_result.get('conf', 0) > ensemble_result['conf']:
                final_result = ai_result
            else:
                final_result = ensemble_result
            
            final_result['patterns'] = patterns
            final_result['risk'] = risk
            final_result['time'] = f"{(time.time() - start_time)*1000:.0f}ms"
            
            st.session_state.prediction = final_result
            st.rerun()
    
    if st.session_state.prediction:
        pred = st.session_state.prediction
        
        # Risk indicator
        risk_level = pred.get('risk', {}).get('level', '✅ OK')
        risk_score = pred.get('risk', {}).get('score', 0)
        
        if risk_score >= 60:
            st.markdown(f"<div class='status-badge' style='background:#da3633'>🚨 {risk_level}</div>", unsafe_allow_html=True)
        elif risk_score >= 40:
            st.markdown(f"<div class='status-badge' style='background:#d29922'>⚠️ {risk_level}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='status-badge' style='background:#238636'>✅ {risk_level}</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<p style='text-align:center;color:#8b949e'>🔥 3 SỐ CHÍNH</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='num-display'>{pred.get('main_3','000')}</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<p style='text-align:center;color:#8b949e'>🛡️ 4 SỐ LÓT</p>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:40px;font-weight:700;color:#58a6ff;text-align:center'>{pred.get('support_4','0000')}</div>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style='margin-top:15px;padding:10px;background:#1f2937;border-radius:8px'>
        💡 <b>Phương pháp:</b> {pred.get('method','Ensemble')}<br>
        🎯 <b>Độ tin:</b> {pred.get('conf',0)}% | ⏱️ <b>Thời gian:</b> {pred.get('time','N/A')}
        </div>
        """, unsafe_allow_html=True)
        
        # Dàn 7 số
        full_dan = "".join(sorted(set(pred.get('main_3','') + pred.get('support_4',''))))
        st.text_input("📋 Dàn 7 số:", full_dan)
        
        st.markdown("</div>", unsafe_allow_html=True)

# ================= TAB 3: PHÂN TÍCH =================
with tabs[2]:
    if st.session_state.history:
        patterns = fast_pattern_detection(st.session_state.history)
        freq = fast_frequency_analysis(st.session_state.history, 10)
        
        st.markdown("### 🔍 PATTERN PHÁT HIỆN")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("**📌 Cầu bệt:**")
            for p in patterns['bệt'][:5]:
                st.write(f"- {p}")
        with c2:
            st.write("**🔁 Cầu nhịp:**")
            for p in patterns['nhịp'][:5]:
                st.write(f"- {p}")
        with c3:
            st.write("**❄️ Số lạnh:**")
            for p in patterns['lạnh'][:5]:
                st.write(f"- {p}")
        
        st.markdown("### 📊 TẦN SUẤT TOP 10")
        freq_df = pd.DataFrame({'Số': list(freq.keys()), 'Tần suất': list(freq.values())})
        st.bar_chart(freq_df.set_index('Số'), color="#58a6ff")

# ================= TAB 4: CÀI ĐẶT =================
with tabs[3]:
    st.markdown("### 📡 NHẬP DỮ LIỆU")
    raw_input = st.text_area("Dán kết quả (5 số/dòng)", height=100, placeholder="32880\n21808...")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 LƯU DỮ LIỆU", use_container_width=True):
            if raw_input:
                numbers = re.findall(r'\b\d{5}\b', raw_input)
                new_nums = [n for n in numbers if n not in st.session_state.history]
                if new_nums:
                    st.session_state.history.extend(new_nums)
                    st.session_state.history = st.session_state.history[-3000:]
                    st.success(f"✅ Lưu {len(new_nums)} kỳ mới!")
                    st.rerun()
                else:
                    st.warning("⚠️ Số đã có trong DB")
    
    with col2:
        if st.button("🗑️ XÓA DB", use_container_width=True):
            st.session_state.history = []
            st.session_state.prediction = None
            st.rerun()
    
    st.markdown("### 📂 IMPORT/EXPORT")
    if st.session_state.history:
        json_data = json.dumps(st.session_state.history[-3000:], ensure_ascii=False).encode('utf-8')
        st.download_button("💾 TẢI DB", json_data, f"titan_{datetime.now().strftime('%Y%m%d')}.json", "application/json")

# ================= FOOTER =================
st.markdown("---")
st.caption(f"⚡ {check_quota()} API calls còn lại | 🕐 {datetime.now().strftime('%H:%M:%S')} | TITAN v33.0 FAST")