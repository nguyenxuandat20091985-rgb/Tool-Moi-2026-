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

st.set_page_config(page_title="TITAN v29.1 DEBUG", layout="wide", page_icon="🔍")

@st.cache_resource
def get_available_models():
    try:
        genai.configure(api_key=API_KEY)
        models = genai.list_models()
        available = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                available.append(model.name)
        return available
    except:
        return []

@st.cache_resource
def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        
        available_models = get_available_models()
        
        preferred_models = [
            'models/gemini-1.5-flash',
            'models/gemini-1.5-pro',
            'models/gemini-pro',
            'models/gemini-1.0-pro'
        ]
        
        selected_model = None
        for model in preferred_models:
            if model in available_models:
                selected_model = model
                break
        
        if not selected_model and available_models:
            selected_model = available_models[0]
        
        if not selected_model:
            st.error("❌ Không tìm thấy model!")
            return None, None
        
        model_instance = genai.GenerativeModel(selected_model)
        st.sidebar.success(f"✅ Model: {selected_model.split('/')[-1]}")
        
        return model_instance, selected_model
        
    except Exception as e:
        st.error(f"❌ Lỗi AI: {str(e)}")
        return None, None

neural_engine, model_name = setup_neural()

# ================= QUẢN LÝ DỮ LIỆU =================
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

if "history" not in st.session_state:
    st.session_state.history = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "show_debug" not in st.session_state:
    st.session_state.show_debug = False

# ================= THUẬT TOÁN =================
def detect_scam_patterns(history, window=20):
    if len(history) < window:
        return {"risk_score": 0, "warnings": [], "level": "UNKNOWN"}
    
    warnings = []
    risk_score = 0
    recent = history[-window:]
    
    all_nums = "".join(recent)
    digit_freq = Counter(all_nums)
    most_common = digit_freq.most_common(1)
    if most_common and most_common[0][1] > 15:
        warnings.append(f"Số {most_common[0][0]} ra {most_common[0][1]}/{window}")
        risk_score += 25
    
    for pos in range(5):
        pos_seq = [int(num[pos]) if len(num) > pos else 0 for num in recent]
        max_streak = current = 1
        for i in range(1, len(pos_seq)):
            if pos_seq[i] == pos_seq[i-1]:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 1
        if max_streak >= 4:
            warnings.append(f"Vị {pos} bệt {max_streak}")
            risk_score += 20
    
    freq = Counter(all_nums)
    total = len(all_nums)
    if total > 0:
        entropy = -sum((c/total) * np.log2(c/total) for c in freq.values() if c > 0)
        if entropy < 2.8:
            warnings.append(f"Entropy thấp {entropy:.2f}")
            risk_score += 25
    
    if risk_score >= 60:
        level = "🔴 HIGH"
    elif risk_score >= 40:
        level = "🟡 MEDIUM"
    elif risk_score >= 20:
        level = "🟢 LOW"
    else:
        level = "✅ NORMAL"
    
    return {"risk_score": risk_score, "warnings": warnings, "level": level}

def analyze_bridge_rhythm(history):
    if len(history) < 15:
        return {"patterns": [], "trend": "UNKNOWN", "hot_numbers": []}
    
    patterns = []
    recent = history[-20:]
    
    for pos in range(5):
        pos_seq = [int(num[pos]) if len(num) > pos else 0 for num in recent]
        
        for i in range(len(pos_seq)-1):
            if pos_seq[i] == pos_seq[i+1]:
                patterns.append(f"Vị{pos}:Bệt-{pos_seq[i]}")
                break
        
        for i in range(len(pos_seq)-2):
            if pos_seq[i] == pos_seq[i+2] and pos_seq[i] != pos_seq[i+1]:
                patterns.append(f"Vị{pos}:Nhịp2-{pos_seq[i]}")
                break
    
    all_digits = "".join(recent)
    hot = [str(x[0]) for x in Counter(all_digits).most_common(3)]
    
    return {
        "patterns": list(set(patterns))[:5],
        "trend": f"Nóng: {', '.join(hot)}",
        "hot_numbers": hot
    }

def advanced_stats(history):
    if not history:
        return {}
    
    stats = {}
    recent = history[-50:]
    
    for pos, name in enumerate(['Chục ngàn', 'Ngàn', 'Trăm', 'Chục', 'Đơn vị']):
        pos_data = [int(num[pos]) for num in recent if len(num) > pos]
        weights = np.linspace(0.5, 1.0, len(pos_data))
        weighted_freq = defaultdict(float)
        for i, val in enumerate(pos_data):
            weighted_freq[val] += weights[i]
        
        top_3 = sorted(weighted_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        stats[name] = [str(x[0]) for x in top_3]
    
    totals = [sum(int(d) for d in num) for num in recent if len(num) == 5]
    stats['total'] = {
        'avg': round(np.mean(totals), 1) if totals else 0,
        'hot': [t for t, _ in Counter(totals).most_common(3)] if totals else []
    }
    
    return stats

# ================= GIAO DIỆN =================
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .prediction-card {
        background: #0d1117; border: 1px solid #30363d;
        border-radius: 12px; padding: 25px; margin: 15px 0;
    }
    .num-box {
        font-size: 80px; font-weight: 900; color: #ff5858;
        text-align: center; letter-spacing: 12px;
    }
    .lot-box {
        font-size: 55px; font-weight: 700; color: #58a6ff;
        text-align: center; letter-spacing: 8px;
    }
    .status-bar { 
        padding: 15px; border-radius: 10px; text-align: center; 
        font-weight: bold; font-size: 20px; margin: 15px 0; 
    }
    .risk-high { background: #7c2d12; border-left: 5px solid #fbbf24; }
    .risk-med { background: #451a03; border-left: 5px solid #f59e0b; }
    .risk-low { background: #064e3b; border-left: 5px solid #10b981; }
    .debug-box { 
        background: #1e1e1e; border: 1px solid #444; 
        border-radius: 8px; padding: 15px; margin: 10px 0;
        font-family: monospace; font-size: 12px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🔍 TITAN v29.1 DEBUG</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: #8b949e;'>Model: {model_name.split('/')[-1] if model_name else 'None'}</p>", unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ Cài đặt")
    
    st.session_state.show_debug = st.checkbox("🐛 Debug Mode", value=False)
    
    st.divider()
    
    uploaded_db = st.file_uploader("📂 Nạp DB", type="json")
    if uploaded_db:
        st.session_state.history = load_data_from_json(uploaded_db)
        st.success(f"✅ {len(st.session_state.history)} kỳ")
    
    st.divider()
    
    if st.session_state.history:
        json_data = convert_df_to_json(st.session_state.history)
        st.download_button(
            label="💾 Tải DB",
            data=json_data,
            file_name=f"titan_db_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
    
    st.divider()
    st.metric("📊 Tổng kỳ", len(st.session_state.history))
    
    if st.button("🗑️ Xóa"):
        st.session_state.history = []
        st.session_state.last_prediction = None
        st.rerun()

# ================= NHẬP LIỆU =================
col1, col2 = st.columns([3, 1])
with col1:
    raw_input = st.text_area("📡 Dán kết quả (5 số/dòng)", height=120, placeholder="32880\n21808...")
with col2:
    st.metric("Kỳ", len(st.session_state.history))
    if st.button("🚀 LƯU", type="primary", use_container_width=True):
        if raw_input:
            clean = re.findall(r"\b\d{5}\b", raw_input)
            if clean:
                new_data = list(dict.fromkeys(clean))
                st.session_state.history.extend(new_data)
                st.session_state.history = st.session_state.history[-3000:]
                st.success(f"✅ {len(new_data)} kỳ")
                st.rerun()

# ================= PHÂN TÍCH =================
st.markdown("---")
st.subheader("🔬 PHÂN TÍCH")

if st.session_state.history and len(st.session_state.history) >= 15:
    if st.button("🎯 CHẠY AI", type="secondary", use_container_width=True):
        if neural_engine is None:
            st.error("❌ AI không khả dụng")
        else:
            with st.spinner("🤖 Đang phân tích..."):
                
                scam_detect = detect_scam_patterns(st.session_state.history)
                bridge_rhythm = analyze_bridge_rhythm(st.session_state.history)
                adv_stats = advanced_stats(st.session_state.history)
                
                # ✅ PROMPT CẢI TIẾN - YÊU CẦU JSON RÕ RÀNG
                prompt = f"""Bạn là chuyên gia xổ số. Phân tích và trả về JSON.

DATA:
- Lịch sử: {st.session_state.history[-50:]}
- Risk: {scam_detect['risk_score']}/100 ({scam_detect['level']})
- Patterns: {bridge_rhythm['patterns']}
- Hot: {bridge_rhythm['hot_numbers']}

YÊU CẦU:
1. Nếu Risk >= 60: decision = "DỪNG"
2. Nếu Risk < 60: chọn 3 số chính + 4 số lót
3. Decision: "ĐÁNH", "THEO DÕI", hoặc "DỪNG"

TRẢ VỀ JSON (KHÔNG giải thích thêm):
{{
    "main_3": "123",
    "support_4": "4567",
    "decision": "ĐÁNH",
    "confidence": 85,
    "logic": "Giải thích ngắn"
}}"""
                
                try:
                    response = neural_engine.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.2,  # Thấp hơn để ổn định
                            top_p=0.9,
                            max_output_tokens=1024
                        )
                    )
                    
                    raw_text = response.text.strip()
                    
                    # ✅ DEBUG: Hiển thị raw response
                    if st.session_state.show_debug:
                        st.markdown("### 📄 Raw AI Response:")
                        st.code(raw_text, language="text")
                    
                    # ✅ CẢI TIẾN JSON PARSING
                    result = None
                    
                    # Cách 1: Tìm trong markdown code block
                    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        try:
                            result = json.loads(json_str)
                        except:
                            pass
                    
                    # Cách 2: Tìm JSON object
                    if not result:
                        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*\}', raw_text, re.DOTALL)
                        if json_match:
                            try:
                                result = json.loads(json_match.group(0))
                            except:
                                pass
                    
                    # Cách 3: Thử parse toàn bộ
                    if not result:
                        try:
                            result = json.loads(raw_text)
                        except:
                            pass
                    
                    if result:
                        # Validate required fields
                        required = ['main_3', 'support_4', 'decision', 'confidence', 'logic']
                        if all(k in result for k in required):
                            result['risk_assessment'] = scam_detect
                            st.session_state.last_prediction = result
                            st.success("✅ OK!")
                            st.rerun()
                        else:
                            missing = [k for k in required if k not in result]
                            st.error(f"❌ Thiếu fields: {missing}")
                            if st.session_state.show_debug:
                                st.write("Result:", result)
                    else:
                        st.error("❌ Không parse được JSON")
                        if st.session_state.show_debug:
                            st.warning("AI trả về:")
                            st.code(raw_text[:500], language="text")
                        
                        # Fallback
                        all_n = "".join(st.session_state.history[-50:])
                        top = [str(x[0]) for x in Counter(all_n).most_common(7)]
                        st.session_state.last_prediction = {
                            "main_3": "".join(top[:3]),
                            "support_4": "".join(top[3:]),
                            "decision": "THEO DÕI",
                            "confidence": 70,
                            "logic": "Fallback - Thống kê",
                            "risk_assessment": scam_detect
                        }
                        st.info("⚠️ Dùng fallback")
                
                except Exception as e:
                    st.error(f"❌ Lỗi: {str(e)}")
                    if st.session_state.show_debug:
                        import traceback
                        st.code(traceback.format_exc(), language="text")

elif st.session_state.history:
    st.info(f"💡 Cần ≥15 kỳ (có {len(st.session_state.history)})")

# ================= HIỂN THỊ =================
if st.session_state.last_prediction:
    res = st.session_state.last_prediction
    risk = res.get('risk_assessment', {})
    risk_score = risk.get('risk_score', 0)
    
    st.markdown("---")
    
    if risk_score >= 60:
        st.markdown(f"<div class='status-bar risk-high'>🚨 {risk_score}/100 - {risk.get('level')}</div>", unsafe_allow_html=True)
        st.error("**DỪNG!**")
    elif risk_score >= 40:
        st.markdown(f"<div class='status-bar risk-med'>⚠️ {risk_score}/100 - {risk.get('level')}</div>", unsafe_allow_html=True)
        st.warning("**CẨN THẬN**")
    else:
        st.markdown(f"<div class='status-bar risk-low'>✅ {risk_score}/100 - {risk.get('level')}</div>", unsafe_allow_html=True)
    
    if risk_score < 60:
        decision = res.get('decision', 'N/A')
        conf = res.get('confidence', 0)
        
        colors = {"ĐÁNH": ("#238636", "✅"), "THEO DÕI": ("#d29922", "⏳"), "DỪNG": ("#da3633", "🛑")}
        bg, icon = colors.get(decision, ("#30363d", "❓"))
        
        st.markdown(f"<div class='status-bar' style='background:{bg}'>{icon} {decision} ({conf}%)</div>", unsafe_allow_html=True)

        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
        
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.markdown("<p style='text-align:center;color:#8b949e;'>🔥 3 SỐ CHÍNH</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='num-box'>{res.get('main_3', '???')}</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<p style='text-align:center;color:#8b949e;'>🛡️ 4 SỐ LÓT</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='lot-box'>{res.get('support_4', '???')}</div>", unsafe_allow_html=True)
        
        st.divider()
        st.write(f"💡 **Logic:** {res.get('logic', 'N/A')}")
        
        full_dan = "".join(sorted(set(str(res.get('main_3', '')) + str(res.get('support_4', '')))))
        st.text_input("📋 Dàn 7 số:", full_dan)
        
        st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("---")
st.caption("⚠️ Tool tham khảo. Quản lý vốn chặt!")
st.caption(f"🕐 {datetime.now().strftime('%d/%m/%Y %H:%M')} | TITAN v29.1 DEBUG")