import streamlit as st
import re
import json
import pandas as pd
import google.generativeai as genai
from collections import Counter
from datetime import datetime
from pathlib import Path

# ================= CONFIG & API =================
st.set_page_config(page_title="TITAN v1500 HYBRID AI", layout="wide")

# Thiết lập Gemini API
API_KEY = "AIzaSyDyyGUWbrxYlBq4X1RDzOVgL9cZiwp0KeY"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-pro')

DATA_FILE = "titan_v1500_dataset.json"

# ================= STYLE =================
st.markdown("""
    <style>
    .stApp { background-color: #050a0f; color: #e0e0e0; }
    .ai-box { border: 2px dashed #00bfff; padding: 15px; border-radius: 10px; background: #0b1622; }
    .titan-result { border: 2px solid #ff4b4b; padding: 20px; border-radius: 15px; background: #1a0a0a; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# ================= CORE LOGIC (GIỮ NGUYÊN BẢN GỐC) =================
def load_data():
    if Path(DATA_FILE).exists():
        with open(DATA_FILE, "r") as f: return json.load(f)
    return []

def save_data(data):
    with open(DATA_FILE, "w") as f: json.dump(data, f)

if "dataset" not in st.session_state:
    st.session_state.dataset = load_data()

def get_titan_score(digits_list):
    freq = Counter(digits_list)
    recent = Counter(digits_list[-30:])
    score = {str(i): 0 for i in range(10)}
    for i in score:
        score[i] += freq.get(i, 0) * 1.0
        score[i] += recent.get(i, 0) * 1.5
        if recent.get(i, 0) == 0: score[i] += 8
    return sorted(score, key=score.get, reverse=True), score

# ================= AI HYBRID ENGINE =================
def ask_gemini(history, current_predict, patterns):
    prompt = f"""
    Bạn là chuyên gia phân tích xác suất LotoBet. 
    Dữ liệu lịch sử: {history[-15:]}
    Mẫu hình hiện tại: {patterns}
    Hệ thống TITAN đang đề xuất 3 số: {current_predict}
    
    Hãy phân tích:
    1. Tỉ lệ nổ của 3 số này trong kỳ tới (%)?
    2. Có dấu hiệu nhà cái đảo cầu (cầu lừa) không?
    3. Lời khuyên đi vốn (Ví dụ: Đánh mạnh, đánh nhẹ, hoặc bỏ qua).
    Trả lời ngắn gọn, quyết đoán.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        return "⚠️ Không kết nối được bộ não AI. Hãy kiểm tra lại API Key hoặc kết nối mạng."

# ================= UI LAYOUT =================
st.title("🛡️ TITAN v1500 HYBRID AI CORE")
st.subheader("Sự kết hợp giữa Thống kê v1300 và Trí tuệ Gemini")

with st.sidebar:
    st.header("⚙️ Control Panel")
    manual_input = st.text_area("Nhập kết quả mới (Ví dụ: 12345):", height=150)
    run_btn = st.button("🚀 PHÂN TÍCH HYBRID", use_container_width=True)
    if st.button("Reset Data"):
        st.session_state.dataset = []
        save_data([])
        st.rerun()

col1, col2 = st.columns([1, 1])

if run_btn and manual_input:
    # 1. Xử lý dữ liệu
    nums = re.findall(r"\d{1,5}", manual_input)
    new_data = [n for n in nums if n not in st.session_state.dataset]
    st.session_state.dataset += new_data
    save_data(st.session_state.dataset)
    
    all_digits = list("".join(st.session_state.dataset))
    
    if len(all_digits) > 20:
        # 2. Chạy TITAN CORE
        ranked, full_scores = get_titan_score(all_digits)
        p1 = ranked[:3]
        
        # 3. Giả lập detect patterns
        patterns = "Bệt/Nhảy xen kẽ" # Có thể nâng cấp hàm này
        
        with col1:
            st.markdown(f"""
            <div class="titan-result">
                <h3 style='color: white;'>🎯 TITAN DỰ ĐOÁN</h3>
                <h1 style='color: #ff4b4b; font-size: 70px;'>{" - ".join(p1)}</h1>
                <p>Top dự phòng: {", ".join(ranked[3:6])}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("📊 **Bảng điểm Score chi tiết:**")
            st.bar_chart(pd.Series(full_scores))

        with col2:
            st.markdown("<div class='ai-box'>", unsafe_allow_html=True)
            st.subheader("🧠 PHÂN TÍCH TỪ GEMINI AI")
            with st.spinner('AI đang đọc vị nhà cái...'):
                ai_advice = ask_gemini(st.session_state.dataset, p1, patterns)
                st.write(ai_advice)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Lưu lịch sử dự đoán
            if "history" not in st.session_state: st.session_state.history = []
            st.session_state.history.append({"time": datetime.now().strftime("%H:%M:%S"), "predict": p1})

    else:
        st.warning("Cần thêm ít nhất 20 con số dữ liệu để AI phân tích chuẩn xác.")

# ================= HISTORY =================
st.divider()
st.subheader("📜 Nhật ký soi cầu")
if "history" in st.session_state:
    for h in st.session_state.history[-5:]:
        st.write(f"🕒 {h['time']} -> TITAN chốt: **{h['predict']}**")
