import streamlit as st
import google.generativeai as genai
import re
import json
from collections import Counter

# ================= CẤU HÌNH API MỚI =================
# Em dán mã mới của anh vào đây
GEMINI_API_KEY = "AIzaSyB29CfRv79fqzOtCSvhTqMURyw9sB1xUIA"

def init_gemini():
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except:
        return None

model = init_gemini()

# ================= GIAO DIỆN PHẲNG CHỐNG ĐẢO =================
st.set_page_config(page_title="TITAN v13.9 STREAK", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #050a10; color: #00ffcc; }
    .status-active { color: #00ffcc; font-weight: bold; border: 1px solid #00ffcc; padding: 5px; border-radius: 5px; }
    .streak-box { background: rgba(255, 0, 85, 0.1); border-left: 5px solid #ff0055; padding: 15px; margin: 10px 0; }
    .number-highlight { font-size: 45px; font-weight: 900; color: #ffffff; text-shadow: 0 0 15px #00ffcc; }
    </style>
""", unsafe_allow_html=True)

# Hiển thị trạng thái kết nối thực tế
if model:
    st.markdown("<div class='status-active'>● GEMINI ĐÃ THÔNG NÃO (API LIVE)</div>", unsafe_allow_html=True)
else:
    st.error("● LỖI API: HÃY KIỂM TRA LẠI KEY TRÊN GOOGLE AI STUDIO")

st.title("🧠 TITAN v13.9 - CHUYÊN GIA BẮT BỆT")

# ================= XỬ LÝ DỮ LIỆU =================
input_data = st.text_area("📡 DÁN DỮ LIỆU NHÀ CÁI (VÍ DỤ: 70938...):", height=120)

if st.button("🚀 KÍCH HOẠT TƯ DUY AI"):
    # Tách lấy các kỳ số
    history = re.findall(r"\d{5}", input_data)
    
    if len(history) < 3:
        st.warning("Anh cần dán ít nhất 3-5 kỳ để AI thấy được cầu!")
    else:
        # 1. Thuật toán nhận diện bệt thủ công (Phòng hờ)
        all_nums = "".join(history)
        count_map = Counter(all_nums)
        top_streaks = [num for num, count in count_map.most_common(4)]
        
        # 2. Gửi lệnh cho Gemini tư duy sâu
        prompt = f"""
        Bạn là chuyên gia phân tích cầu 5D. Dữ liệu thực tế: {history}.
        Yêu cầu:
        1. Nhận diện các số đang BỆT (xuất hiện liên tục).
        2. Nếu nhà cái đảo cầu, hãy tính toán bước nhảy để chặn đầu.
        3. Trả về JSON: {{"dan4_chuluc": [], "dan3_lot": [], "ly_do": ""}}
        """
        
        try:
            response = model.generate_content(prompt)
            res_text = response.text
            json_match = re.search(r'\{.*\}', res_text, re.DOTALL)
            data = json.loads(json_match.group())
            
            # HIỂN THỊ KẾT QUẢ
            st.markdown(f"<div class='streak-box'><b>💡 Phân tích bệt:</b> {data['ly_do']}</div>", unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🎯 DÀN 4 CHỦ LỰC")
                st.markdown(f"<div class='number-highlight'>{' - '.join(map(str, data['dan4_chuluc']))}</div>", unsafe_allow_html=True)
            with c2:
                st.subheader("🛡️ DÀN 3 LÓT")
                st.markdown(f"<div class='number-highlight' style='color:#ffaa00;'>{' - '.join(map(str, data['dan3_lot']))}</div>", unsafe_allow_html=True)
            
            full_7 = "".join(map(str, data['dan4_chuluc'])) + "".join(map(str, data['dan3_lot']))
            st.text_input("📋 COPY DÀN 7 SỐ NHANH:", full_7)
            
        except Exception as e:
            st.error(f"AI đang bận hoặc Key bị giới hạn. Dàn bệt dự phòng: {top_streaks}")

st.markdown("---")
st.caption("Mẹo: Nếu thấy nhà cái ra số lặp (ví dụ 1-1, 9-9), hãy dán ngay vào để AI bắt cầu bệt.")
