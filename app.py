import streamlit as st
import google.generativeai as genai
import re
import json

# ================= KÍCH HOẠT NÃO BỘ GEMINI =================
# Em đã dán sẵn Key anh vừa gửi vào đây
GEMINI_API_KEY = "AIzaSyCF4AFrKTI8xs3uFX7OJwWcApa5dbRTIxA"

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    st.session_state.api_status = "✅ ĐÃ KẾT NỐI GEMINI"
except:
    st.session_state.api_status = "❌ LỖI KẾT NỐI API"

# ================= GIAO DIỆN CHUYÊN NGHIỆP =================
st.set_page_config(page_title="TITAN v13.5 STREAK MASTER", layout="centered")
st.markdown("""
    <style>
    .stApp { background-color: #050a10; color: #00ffcc; }
    .status-bar { padding: 10px; border-radius: 5px; background: #111b27; text-align: center; font-weight: bold; }
    .number-card { font-size: 40px; font-weight: 900; color: #ffffff; text-shadow: 0 0 10px #00ffcc; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 TITAN v13.5 - STREAK MASTER")
st.markdown(f"<div class='status-bar'>{st.session_state.api_status}</div>", unsafe_allow_html=True)

# ================= XỬ LÝ DỮ LIỆU & BỆT =================
input_data = st.text_area("📡 DÁN DỮ LIỆU KỲ VỪA VỀ:", height=100, placeholder="Ví dụ: 70938...")

if st.button("🔥 KÍCH HOẠT TƯ DUY AI"):
    history = re.findall(r"\d{5}", input_data)
    
    if len(history) < 5:
        st.error("Anh cần dán ít nhất 5-10 kỳ gần nhất để AI thấy được cầu bệt!")
    else:
        # Prompt mới: Ép AI nhận diện bệt (số lặp lại)
        prompt = f"""
        Bạn là chuyên gia toán xác suất 5D. 
        Dữ liệu thực tế: {history}.
        Yêu cầu:
        1. Tìm các số đang có xu hướng lặp lại (BỆT) ở 5 vị trí.
        2. Nếu nhà cái đảo cầu, hãy chọn 7 số có biên độ ổn định nhất.
        3. Chia thành 2 dàn: Dàn 4 (Chủ lực) và Dàn 3 (Lót).
        Trả về JSON duy nhất: {{"dan4": [], "dan3": [], "tu_duy": "giải thích ngắn gọn"}}
        """
        
        try:
            response = model.generate_content(prompt)
            # Trích xuất JSON từ phản hồi
            res_text = response.text
            json_match = re.search(r'\{.*\}', res_text, re.DOTALL)
            data = json.loads(json_match.group())
            
            st.success("AI ĐÃ PHÂN TÍCH XONG!")
            st.markdown(f"**💡 Tư duy AI:** {data['tu_duy']}")
            
            c1, c2 = st.columns(2)
            with c1:
                st.info("🎯 DÀN 4 (CHỦ LỰC)")
                st.markdown(f"<div class='number-card'>{' - '.join(map(str, data['dan4']))}</div>", unsafe_allow_html=True)
            with c2:
                st.warning("🛡️ DÀN 3 (LÓT)")
                st.markdown(f"<div class='number-card' style='color:#ffaa00;'>{' - '.join(map(str, data['dan3']))}</div>", unsafe_allow_html=True)
            
            st.text_input("📋 COPY NHANH DÀN 7 SỐ:", "".join(map(str, data['dan4'])) + "".join(map(str, data['dan3'])))
            
        except Exception as e:
            st.error(f"Lỗi khi AI tư duy: {e}. Anh kiểm tra xem đã bật Gemini 1.5 trong Google AI Studio chưa nhé!")
