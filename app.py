import streamlit as st
import google.generativeai as genai
import re
import json
import numpy as np
from collections import Counter

# ================= KÍCH HOẠT HỆ THỐNG =================
# API Key mới anh vừa tạo (Đã tích hợp)
API_KEY = "AIzaSyC7jzb0MiGy05zLaKnt4-3ribPxXzC73YQ"

# Cấu hình AI Gemini
def load_neural_engine():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except:
        return None

model = load_neural_engine()

# ================= GIAO DIỆN TITAN v15.0 =================
st.set_page_config(page_title="TITAN v15.0 NEURAL-PRO", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #04090f; color: #00ffcc; }
    .main-box {
        background: rgba(0, 255, 204, 0.05); border: 2px solid #00ffcc;
        border-radius: 15px; padding: 20px;
    }
    .digit-card {
        background: #111b27; border-left: 5px solid #0055ff;
        padding: 15px; border-radius: 10px; margin: 10px 0;
    }
    .big-num { font-size: 45px; font-weight: 900; color: #fff; text-shadow: 0 0 15px #00ffcc; text-align: center; }
    .status-active { color: #00ffcc; font-size: 14px; text-align: center; border: 1px solid #222; padding: 5px; }
    </style>
""", unsafe_allow_html=True)

# Hiển thị trạng thái kết nối
if model:
    st.markdown("<div class='status-active'>● HỆ THỐNG NEURAL: ĐANG HOẠT ĐỘNG (API LIVE)</div>", unsafe_allow_html=True)
else:
    st.error("● LỖI KẾT NỐI API: HÃY KIỂM TRA LẠI KEY TRÊN GOOGLE AI STUDIO")

st.markdown("<h2 style='text-align: center; color: #00ffcc;'>🧠 TITAN v15.0 NEURAL-PRO</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>CHUYÊN GIA NHẬN DIỆN CẦU BỆT & ĐẢO SỐ</p>", unsafe_allow_html=True)

# ================= THUẬT TOÁN NHẬN DIỆN BỆT (FALLBACK) =================
def manual_streak_check(history):
    # Nếu AI lỗi, dùng toán học đếm số hay về nhất (Bệt)
    all_digits = "".join(history[-10:])
    counts = Counter(all_digits)
    top_7 = [str(num) for num, count in counts.most_common(7)]
    return top_7 if len(top_7) == 7 else ["0","1","2","3","4","5","6"]

# ================= NHẬP DỮ LIỆU & XỬ LÝ =================
input_data = st.text_area("📡 DÁN DỮ LIỆU KỲ VỪA VỀ (VÍ DỤ: 32880...):", height=120)

if st.button("🔥 KÍCH HOẠT TƯ DUY AI"):
    # Tách dữ liệu thành các kỳ 5D
    history = re.findall(r"\d{5}", input_data)
    
    if len(history) < 5:
        st.warning("⚠️ Anh cần dán tối thiểu 5-10 kỳ để AI bắt được luồng cầu bệt!")
    else:
        with st.spinner('Gemini đang soi cầu bệt...'):
            # Lệnh ép AI phân tích sâu
            prompt = f"""
            Dữ liệu thực tế nhà cái: {history}. 
            Yêu cầu chuyên gia:
            1. Phân tích các số đang có xu hướng lặp lại (BỆT) hoặc hồi số.
            2. Nếu nhà cái đảo cầu mạnh, tính toán bước nhảy để chặn đầu.
            3. Trả về đúng định dạng JSON: {{"dan4_chuluc": [], "dan3_lot": [], "ly_do": ""}}
            """
            
            try:
                response = model.generate_content(prompt)
                res_text = response.text
                # Trích xuất JSON từ phản hồi AI
                json_match = re.search(r'\{.*\}', res_text, re.DOTALL)
                data = json.loads(json_match.group())
                
                # HIỂN THỊ KẾT QUẢ
                st.markdown(f"<div style='background:rgba(0,85,255,0.1); padding:10px; border-radius:5px;'>💡 <b>AI Tư Duy:</b> {data['ly_do']}</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='main-box'>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<div class='digit-card'>", unsafe_allow_html=True)
                    st.write("🎯 **DÀN 4 CHỦ LỰC**")
                    st.markdown(f"<div class='big-num'>{' - '.join(map(str, data['dan4_chuluc']))}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown("<div class='digit-card' style='border-left-color: #ffaa00;'>", unsafe_allow_html=True)
                    st.write("🛡️ **DÀN 3 LÓT**")
                    st.markdown(f"<div class='big-num' style='color:#ffaa00;'>{' - '.join(map(str, data['dan3_lot']))}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Copy dàn 7
                full_7 = "".join(map(str, data['dan4_chuluc'])) + "".join(map(str, data['dan3_lot']))
                st.text_input("📋 COPY NHANH DÀN 7 SỐ:", full_7)

            except Exception as e:
                # Nếu API gặp lỗi (Hết lượt/Bị chặn), tự động dùng thuật toán đếm số (Fallback)
                fallback_7 = manual_streak_check(history)
                st.error("⚠️ AI đang bận. Đang dùng thuật toán nhận diện Bệt dự phòng!")
                st.markdown(f"<div class='big-num'>{' - '.join(fallback_7[:4])} | {' - '.join(fallback_7[4:])}</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#555;'>TITAN v15.0 - Kết nối trực tiếp Google AI Studio</p>", unsafe_allow_html=True)
