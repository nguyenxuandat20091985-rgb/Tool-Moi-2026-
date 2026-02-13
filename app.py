import streamlit as st
import google.generativeai as genai
import re
import json
from collections import Counter

# ================= KÍCH HOẠT HỆ THỐNG TITAN v16.0 =================
# API Key mới tinh của anh
API_KEY = "AIzaSyBRo51DqVoC7BSv3ipUrY8GaEVfi0cVQxc"

def init_system():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except:
        return None

model = init_system()

# ================= GIAO DIỆN DARK MODE CHỐNG ĐỨNG =================
st.set_page_config(page_title="TITAN v16.0 SUPER-ULTRA", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #0b1118; color: #00ffcc; }
    .status-tag { padding: 8px; border-radius: 20px; text-align: center; font-weight: bold; font-size: 12px; border: 1px solid #00ffcc; }
    .result-box { background: #16212e; border: 2px solid #00ffcc; border-radius: 15px; padding: 25px; margin-top: 20px; }
    .number-text { font-size: 50px; font-weight: 900; color: #ffffff; text-shadow: 0 0 20px #00ffcc; text-align: center; letter-spacing: 5px; }
    .reason-text { font-style: italic; color: #8899aa; margin-bottom: 20px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# Hiển thị trạng thái API
if model:
    st.markdown("<div class='status-tag'>● HỆ THỐNG NEURAL TRỰC TUYẾN (API LIVE)</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='status-tag' style='color:red; border-color:red;'>● LỖI KẾT NỐI API - HÃY KIỂM TRA LẠI GITHUB</div>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🧠 TITAN v16.0</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>BẢN SIÊU CẤP - CHUYÊN TRỊ CẦU BỆT & ĐẢO SỐ</p>", unsafe_allow_html=True)

# ================= NHẬP DỮ LIỆU & SOI CẦU =================
input_data = st.text_area("📡 DÁN DỮ LIỆU KỲ VỪA VỀ:", height=150, placeholder="Dán các kỳ như: 51875, 78733...")

if st.button("🔥 KÍCH HOẠT TƯ DUY AI"):
    # Tách lấy các kỳ số
    history = re.findall(r"\d{5}", input_data)
    
    if len(history) < 5:
        st.error("Anh cần dán ít nhất 5-10 kỳ để AI nhận diện được chu kỳ bệt!")
    else:
        with st.spinner('AI đang quét dữ liệu nhà cái...'):
            # Lệnh Prompt tối ưu nhất cho Gemini
            prompt = f"""
            Bạn là máy chủ dự đoán 5D. Lịch sử: {history}.
            Yêu cầu:
            1. Tìm các con số đang có xu hướng lặp lại (Bệt) trong các kỳ gần nhất.
            2. Tính toán tỷ lệ xuất hiện của các số từ 0-9.
            3. Trả về đúng định dạng JSON: {{"dan4": [], "dan3": [], "tu_duy": ""}}
            4. Lưu ý: "dan4" là 4 số mạnh nhất, "dan3" là 3 số lót.
            """
            
            try:
                response = model.generate_content(prompt)
                res_text = response.text
                
                # Bóc tách JSON
                json_match = re.search(r'\{.*\}', res_text, re.DOTALL)
                data = json.loads(json_match.group())
                
                # HIỂN THỊ KẾT QUẢ CỰC ĐẸP
                st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                st.markdown(f"<div class='reason-text'>💡 {data['tu_duy']}</div>", unsafe_allow_html=True)
                
                st.write("🎯 **DÀN 4 CHỦ LỰC (VÀO TIỀN MẠNH):**")
                st.markdown(f"<div class='number-text'>{' - '.join(map(str, data['dan4']))}</div>", unsafe_allow_html=True)
                
                st.write("🛡️ **DÀN 3 LÓT (BẢO TOÀN VỐN):**")
                st.markdown(f"<div class='number-text' style='color:#ffaa00; text-shadow: 0 0 20px #ffaa00;'>{' - '.join(map(str, data['dan3']))}</div>", unsafe_allow_html=True)
                
                # Dòng copy nhanh
                full_7 = "".join(map(str, data['dan4'])) + "".join(map(str, data['dan3']))
                st.text_input("📋 COPY NHANH DÀN 7 SỐ:", full_7)
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                # Thuật toán dự phòng nếu API lỗi
                all_nums = "".join(history)
                counts = Counter(all_nums)
                fallback = [n for n, c in counts.most_common(7)]
                st.warning("⚠️ AI ĐANG QUÁ TẢI. DÀN BỆT DỰ PHÒNG TỪ TOÁN HỌC:")
                st.markdown(f"<div class='number-text'>{' - '.join(fallback[:4])} | {' - '.join(fallback[4:])}</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Lưu ý: Nếu thấy nhà cái ra bệt (ví dụ 8-8-8), AI sẽ tự động bám sát con 8 cho anh.")
