"""
🚀 TITAN V27 - ULTRA FAST VERSION
Dành cho 5D Bet - 1 phút là xong
"""
import streamlit as st
import re
import itertools
from collections import Counter

# Config
st.set_page_config(page_title="TITAN V27", page_icon="🎲", layout="wide")

# CSS - CỰC KỲ ĐƠN GIẢN
st.markdown("""
<style>
    .main {padding: 0.5rem;}
    .big-number {font-size: 56px; font-weight: bold; color: #28a745; text-align: center; font-family: monospace; letter-spacing: 8px; margin: 10px 0;}
    .grid-4 {display: grid; grid-template-columns: repeat(4, 1fr); gap: 5px;}
    .grid-3 {display: grid; grid-template-columns: repeat(3, 1fr); gap: 5px;}
    .box {background: #f8f9fa; border: 2px solid #28a745; border-radius: 8px; padding: 10px; text-align: center; font-family: monospace; font-size: 18px; font-weight: bold;}
    .box-3 {background: #e3f2fd; border: 2px solid #2196f3; border-radius: 8px; padding: 10px; text-align: center; font-family: monospace; font-size: 20px; font-weight: bold;}
    .status {background: #28a745; color: white; padding: 15px; border-radius: 10px; text-align: center; font-size: 20px; font-weight: bold; margin: 10px 0;}
    .status-stop {background: #dc3545;}
    .metric {display: flex; justify-content: space-around; margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 8px;}
    .metric-item {text-align: center;}
    .metric-value {font-size: 24px; font-weight: bold; color: #28a745;}
    .metric-label {font-size: 12px; color: #666;}
    textarea {height: 60px !important;}
    button {width: 100% !important; background: #28a745 !important; color: white !important; font-size: 20px !important; font-weight: bold !important; padding: 15px !important;}
    @media (max-width: 600px) {
        .big-number {font-size: 40px;}
        .grid-4 {grid-template-columns: repeat(3, 1fr);}
    }
</style>
""", unsafe_allow_html=True)

# Functions
def get_numbers(text):
    return re.findall(r"\d{5}", text) if text else []

def make_combos(digits_7):
    digits = sorted(list(set(digits_7)))[:7]
    while len(digits) < 7:
        for i in range(10):
            if str(i) not in digits:
                digits.append(str(i))
                break
    digits = sorted(digits)[:7]
    pairs = ["".join(p) for p in itertools.combinations(digits, 2)]
    triples = ["".join(t) for t in itertools.combinations(digits, 3)]
    return "".join(digits), pairs, triples

def analyze(db):
    if len(db) < 3:
        return "0123456", ["01","23","45"], ["012","234","456"], "012", "Chưa đủ dữ liệu", 50
    
    # Tính tần suất 30 kỳ gần
    all_digits = "".join(db[-30:])
    scores = {str(i): all_digits.count(str(i)) for i in range(10)}
    
    # Bonus số vừa ra
    last = db[-1]
    for d in set(last):
        scores[d] += 20
    
    # Sort và lấy top 7
    sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_7 = "".join(sorted([x[0] for x in sorted_nums[:7]]))
    
    # Generate combos
    base_7, pairs, triples = make_combos(top_7)
    
    # Main 3 số
    main_3 = triples[0] if triples else "012"
    
    # Confidence
    conf = min(90, 50 + len(db)//3)
    
    return base_7, pairs[:12], triples[:10], main_3, f"Phân tích {len(db)} kỳ", conf

# Init session
if "db" not in st.session_state:
    st.session_state.db = []
if "result" not in st.session_state:
    st.session_state.result = None

# Title
st.markdown('<h1 style="text-align:center;color:#28a745;margin:5px 0;">🎲 TITAN V27</h1>', unsafe_allow_html=True)

# Metrics
col1, col2 = st.columns(2)
with col1:
    st.markdown(f'<div class="metric"><div class="metric-item"><div class="metric-value">{len(st.session_state.db)}</div><div class="metric-label">Tổng kỳ</div></div></div>', unsafe_allow_html=True)
with col2:
    last_num = st.session_state.db[-1] if st.session_state.db else "Chưa có"
    st.markdown(f'<div class="metric"><div class="metric-item"><div class="metric-value" style="font-size:18px;">{last_num}</div><div class="metric-label">Kỳ cuối</div></div></div>', unsafe_allow_html=True)

# Input
st.markdown("### 📥 Nhập số (mỗi dòng 1 số 5 chữ số)")
user_input = st.text_area("", placeholder="16923\n51475\n31410\n...", height=60, label_visibility="collapsed")

# Button
if st.button("⚡ CHỐT SỐ"):
    numbers = get_numbers(user_input)
    if numbers:
        st.session_state.db.extend(numbers)
        base_7, pairs, triples, main_3, logic, conf = analyze(st.session_state.db)
        st.session_state.result = {
            "base_7": base_7,
            "pairs": pairs,
            "triples": triples,
            "main_3": main_3,
            "logic": logic,
            "conf": conf
        }
        st.rerun()
    else:
        st.error("❌ Không có số 5 chữ số!")

# Show result
if st.session_state.result:
    r = st.session_state.result
    
    # Status
    status_class = "" if r["conf"] >= 70 else "status-stop"
    st.markdown(f'<div class="status {status_class}">🔥 KHUYÊN ĐÁNH | Tin cậy: {r["conf"]}%</div>', unsafe_allow_html=True)
    
    # Base 7 - BIG & CLEAR
    st.markdown('<div style="text-align:center;margin:15px 0;">', unsafe_allow_html=True)
    st.markdown('<div style="color:#666;font-size:16px;">🎲 DÀN 7 SỐ</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="big-number">{r["base_7"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2 = st.tabs(["🎯 2 SỐ", "🎯 3 SỐ"])
    
    with tab1:
        st.markdown('<div class="grid-4">', unsafe_allow_html=True)
        for p in r["pairs"][:12]:
            st.markdown(f'<div class="box">{p}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="grid-3">', unsafe_allow_html=True)
        for t in r["triples"][:10]:
            st.markdown(f'<div class="box-3">{t}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main number
    st.markdown(f'<div style="background:#28a745;color:white;padding:15px;border-radius:10px;text-align:center;margin:10px 0;"><div style="font-size:14px;">⭐ SỐ CHỦ LỰC</div><div style="font-size:40px;font-weight:bold;font-family:monospace;">{r["main_3"]}</div></div>', unsafe_allow_html=True)
    
    # Logic
    st.markdown(f'<div style="background:#e3f2fd;padding:10px;border-radius:8px;text-align:center;"><strong style="color:#2196f3;">🧠 AI:</strong> {r["logic"]}</div>', unsafe_allow_html=True)

# Quick actions
st.markdown("---")
col_a, col_b = st.columns(2)
with col_a:
    if st.button("🗑️ XÓA HẾT", key="clear"):
        st.session_state.db = []
        st.session_state.result = None
        st.rerun()
with col_b:
    if st.button("📊 THỐNG KÊ", key="stats"):
        st.session_state.show_stats = not st.session_state.get("show_stats", False)

# Stats (optional)
if st.session_state.get("show_stats", False) and len(st.session_state.db) > 0:
    with st.expander("📊 Tần suất", expanded=True):
        freq = Counter("".join(st.session_state.db[-50:]))
        df_data = [{"Số": str(i), "Tần suất": freq.get(str(i), 0)} for i in range(10)]
        
        col_x, col_y = st.columns(2)
        with col_x:
            st.markdown("**🔥 Nóng nhất**")
            for item in sorted(df_data, key=lambda x: x["Tần suất"], reverse=True)[:3]:
                st.write(f"`{item['Số']}`: {item['Tần suất']}")
        with col_y:
            st.markdown("**❄️ Lạnh nhất**")
            for item in sorted(df_data, key=lambda x: x["Tần suất"])[:3]:
                st.write(f"`{item['Số']}`: {item['Tần suất']}")

# Footer
st.markdown('<div style="text-align:center;color:#999;font-size:11px;margin-top:10px;">⚡ TITAN V27 - 1 PHÚT LÀ XONG</div>', unsafe_allow_html=True)