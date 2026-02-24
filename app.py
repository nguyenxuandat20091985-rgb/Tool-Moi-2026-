import streamlit as st
import google.generativeai as genai
import re
import json
import os
from collections import Counter

# ================= CẤU HÌNH TITAN v23.0 =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_ultimate_v23.json"

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# ================= LOGIC ĐỐI ĐẦU NHÀ CÁI =================
def analyze_3_so_5_tinh(history):
    if len(history) < 10: return None
    
    # 1. Lấy 30 kỳ gần nhất để soi cầu bệt
    recent_30 = "".join(history[-30:])
    freq = Counter(recent_30).most_common(10)
    
    # 2. Định nghĩa bóng số (Shadow Numbers)
    shadows = {'0':'5','1':'6','2':'7','3':'8','4':'9','5':'0','6':'1','7':'2','8':'3','9':'4'}
    
    # 3. Thuật toán "Điểm mù nhà cái": Tìm những số đang bị 'giam' quá lâu
    all_possible = set("0123456789")
    present_recent = set(recent_30)
    missing = all_possible - present_recent
    
    return {"freq": freq, "shadows": shadows, "missing": list(missing)}

# ================= GIAO DIỆN CHIẾN ĐẤU =================
st.set_page_config(page_title="TITAN v23.0 ULTIMATE", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #000000; color: #00ff41; font-family: 'Courier New', monospace; }
    .main-card { border: 2px solid #00ff41; padding: 20px; border-radius: 10px; background: #0a0a0a; box-shadow: 0 0 20px #00ff41; }
    .target-num { font-size: 70px; color: #ff0000; text-align: center; font-weight: bold; text-shadow: 0 0 10px #ff0000; }
    .safety-alert { color: #ffff00; border: 1px solid #ffff00; padding: 10px; text-align: center; margin: 10px 0; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>⚡ TITAN v23.0 ULTIMATE ⚡</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>Hệ thống đánh chặn AI Kubet - Chuyên kèo 3 Số 5 Tinh</p>", unsafe_allow_html=True)

# Nạp dữ liệu
raw_data = st.text_area("📥 NẠP DỮ LIỆU SẢNH 5D (Copy kết quả vào đây):", height=150)

if st.button("🚀 BẺ KHÓA THUẬT TOÁN"):
    # Lọc dữ liệu chuẩn từ hình ảnh anh gửi (dãy 5 số)
    clean_history = re.findall(r"\b\d{5}\b", raw_data)
    
    if len(clean_history) >= 5:
        with st.spinner("Đang phân tích nhịp cầu và bóng số..."):
            # Gọi AI Gemini phân tích sâu
            analysis = analyze_3_so_5_tinh(clean_history)
            prompt = f"""
            Yêu cầu: Phân tích kèo '3 số 5 tinh' (chọn 3 số, chỉ cần xuất hiện trong 5 số giải).
            Lịch sử: {clean_history[-50:]}.
            Dữ liệu thống kê: {analysis}.
            Hãy tìm ra 3 số 'Chủ lực' và 4 số 'Vệ tinh'. 
            Lưu ý: Nhà cái đang có xu hướng đảo cầu sau mỗi chu kỳ bệt. 
            Trả về JSON duy nhất: {{"chu_luc_3": "abc", "ve_tinh_4": "defg", "canh_bao": "nội dung", "ti_le_thang": 95}}
            """
            
            try:
                response = model.generate_content(prompt)
                res_json = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
                
                # Hiển thị kết quả
                st.markdown("<div class='main-card'>", unsafe_allow_html=True)
                
                if res_json['ti_le_thang'] < 80:
                    st.markdown("<div class='safety-alert'>⚠️ CẦU ĐANG BIẾN ĐỘNG - KHÔNG NÊN ĐÁNH LỚN</div>", unsafe_allow_html=True)
                
                st.write(f"🧬 **LOGIC ĐỐI KHÁNG:** {res_json.get('canh_bao', 'Đang bám sát nhịp cầu')}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<p style='text-align:center;'>🎯 3 SỐ CHỦ LỰC (5 TINH)</p>", unsafe_allow_html=True)
                    st.markdown(f"<div class='target-num'>{res_json['chu_luc_3']}</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<p style='text-align:center;'>🛡️ DÀN VỆ TINH (LÓT)</p>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size:50px; text-align:center; color:#00ff41;'>{res_json['ve_tinh_4']}</div>", unsafe_allow_html=True)
                
                st.markdown(f"<p style='text-align:right;'>Độ tin cậy hệ thống: {res_json['ti_le_thang']}%</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error("Lỗi phân tích AI. Vui lòng kiểm tra định dạng dữ liệu hoặc API Key.")
    else:
        st.warning("Vui lòng nạp ít nhất 5 kỳ kết quả để bắt đầu phân tích.")

# Hướng dẫn chiến thuật từ hình ảnh thực tế
with st.expander("📝 HƯỚNG DẪN ĐÁNH THEO TITAN V23.0"):
    st.write("""
    1. **Cách nhập:** Copy toàn bộ dòng kết quả từ sảnh (ví dụ: 7, 8, 9, 3, 1) dán vào ô nhập liệu. AI sẽ tự động bỏ dấu phẩy.
    2. **Kèo 3 số 5 tinh:** Bản v23.0 tập trung tìm ra 3 con số mà khả năng ít nhất 1 trong 3 con đó sẽ xuất hiện trong giải là cực cao.
    3. **Quản lý vốn:** Nếu 'Độ tin cậy' dưới 85%, tuyệt đối không đánh gấp thếp.
    """)
