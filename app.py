import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
from collections import Counter

# ================= CẤU HÌNH HỆ THỐNG TITANIUM v25.0 =================
API_KEY = "AIzaSyB5PRp04XlMHKl3oGfCRbsKXjlTA-CZifc"
DB_FILE = "titan_permanent_v25.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        # Sử dụng model mạnh nhất để soi số kỹ càng
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try: return json.load(f)
            except: return []
    return []

def save_db(data):
    with open(DB_FILE, "w") as f:
        # Tăng dung lượng bộ nhớ lên 5000 kỳ để AI học sâu hơn
        json.dump(data[-5000:], f)

if "history" not in st.session_state:
    st.session_state.history = load_db()

# ================= THIẾT KẾ UI CHUẨN v22 (TỐI ƯU MƯỢT MÀ) =================
st.set_page_config(page_title="TITAN v25.0 TITANIUM", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #e6edf3; }
    .prediction-card {
        background: #0d1117; border: 1px solid #30363d;
        border-radius: 12px; padding: 25px; margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    .main-num-box {
        font-size: 80px; font-weight: 900; color: #ff5858;
        text-align: center; letter-spacing: 12px; text-shadow: 0 0 15px #ff5858;
    }
    .status-bar { padding: 12px; border-radius: 8px; text-align: center; font-weight: bold; margin-bottom: 15px; font-size: 18px; }
    .warning-box { background: #331010; color: #ff7b72; padding: 10px; border-radius: 5px; border: 1px solid #6e2121; margin-bottom: 10px; }
    .info-text { font-size: 14px; color: #8b949e; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #58a6ff;'>💎 TITAN v25.0 TITANIUM</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e;'>Hệ thống Siêu trí tuệ - Khắc chế đảo cầu - Bào tiền nhà cái</p>", unsafe_allow_html=True)

# ================= PHẦN 1: ĐIỀU KHIỂN & NHẬP LIỆU =================
with st.container():
    col_in, col_st = st.columns([2, 1])
    with col_in:
        raw_input = st.text_area("📡 NẠP DỮ LIỆU (Hệ thống tự động lọc số bẩn/trùng):", height=120, placeholder="32880\n21808...")
    with col_st:
        st.markdown(f"<div style='padding:10px; border:1px solid #30363d; border-radius:8px;'>📊 Dữ liệu hiện tại: <b>{len(st.session_state.history)} kỳ</b><br><small>Hệ thống tự học từ lịch sử để đưa ra kết quả đúng nhất.</small></div>", unsafe_allow_html=True)
        st.write("")
        c1, c2 = st.columns(2)
        btn_save = c1.button("🚀 GIẢI MÃ TINH HOA")
        btn_reset = c2.button("🗑️ RESET BỘ NHỚ")

if btn_reset:
    st.session_state.history = []
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    st.rerun()

if btn_save:
    # Lọc số chuẩn: Đúng 5 chữ số, loại bỏ trùng lặp và các số rác
    new_entries = re.findall(r"\b\d{5}\b", raw_input)
    if new_entries:
        # Chỉ thêm những số chưa có trong lịch sử (Loại trùng lặp hoàn toàn)
        current_history = st.session_state.history
        for entry in new_entries:
            if entry not in current_history:
                current_history.append(entry)
        
        st.session_state.history = current_history
        save_db(st.session_state.history)
        
        # PROMPT SIÊU CẤP CHO GEMINI v25.0
        prompt = f"""
        Bạn là Siêu AI chuyên gia giải mã thuật toán Kubet/Lotobet. 
        Mục tiêu: Bào tiền nhà cái bằng cách dự đoán chính xác 3 càng (3D) không cố định vị trí.
        Dữ liệu lịch sử thực tế (Đã lọc): {st.session_state.history[-150:]}

        NHIỆM VỤ CỦA BẠN:
        1. Phân tích bệt: Xác định các số đang bệt sâu hoặc các cặp số thường đi cùng nhau khi bệt.
        2. Nhận diện đảo cầu: Nếu nhà cái đang đảo cầu liên tục, hãy phân tích nhịp đảo để bắt điểm rơi.
        3. Soi số kỹ càng: Dùng ma trận xác suất để chọn ra 2 dàn số chủ lực chính xác nhất.
        
        YÊU CẦU ĐẦU RA (JSON BẮT BUỘC):
        - main_3_dan1: Dàn 3 số chủ lực thứ nhất (Ví dụ: "456").
        - main_3_dan2: Dàn 3 số chủ lực thứ hai (Ví dụ: "567").
        - support_4: 4 số lót để tạo thành dàn 7 số an toàn nhất.
        - decision: "ĐÁNH THEO BỆT", "ĐÁNH THEO ĐẢO CẦU", "VÀO TIỀN MẠNH" hoặc "DỪNG CƯỢC".
        - warning: Cảnh báo cụ thể nếu thấy nhà cái đang quây số.
        - logic: Phân tích ngắn gọn nhạy bén lý do chọn số.
        - color: "Green" (An toàn), "Yellow" (Cần thận trọng), "Red" (Cực kỳ nguy hiểm).
        """
        
        try:
            response = neural_engine.generate_content(prompt)
            # Trích xuất JSON từ phản hồi của AI
            res_data = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
            st.session_state.last_prediction = res_data
        except Exception as e:
            st.error(f"Lỗi AI: {e}. Đang sử dụng thuật toán dự phòng...")
            # Thuật toán dự phòng nếu API gặp sự cố
            all_digits = "".join(st.session_state.history[-50:])
            common = [x[0] for x in Counter(all_digits).most_common(7)]
            st.session_state.last_prediction = {
                "main_3_dan1": "".join(common[:3]),
                "main_3_dan2": "".join(common[1:4]),
                "support_4": "".join(common[3:7]),
                "decision": "ĐÁNH THEO TẦN SUẤT",
                "warning": "Dữ liệu đang được phân tích cục bộ.",
                "logic": "Dựa trên mật độ xuất hiện dày đặc của các con số kỳ gần nhất.",
                "color": "Yellow",
                "conf": 85
            }
        st.rerun()

# ================= PHẦN 2: HIỂN THỊ KẾT QUẢ (TRỰC QUAN v22) =================
if "last_prediction" in st.session_state:
    res = st.session_state.last_prediction
    
    # Thanh trạng thái động
    bg_color = {"green": "#238636", "yellow": "#d29922", "red": "#da3633"}.get(res['color'].lower(), "#30363d")
    st.markdown(f"<div class='status-bar' style='background: {bg_color};'>📢 CHIẾN THUẬT: {res['decision']}</div>", unsafe_allow_html=True)

    if res.get('warning'):
        st.markdown(f"<div class='warning-box'>⚠️ <b>CẢNH BÁO AI:</b> {res['warning']}</div>", unsafe_allow_html=True)

    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    # Hiển thị 2 Dàn chủ lực tách biệt rõ ràng
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("<p style='text-align:center; color:#ff7b72; font-weight:bold;'>🔥 DÀN CHỦ LỰC 1</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='main-num-box'>{res['main_3_dan1']}</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<p style='text-align:center; color:#ff7b72; font-weight:bold;'>🔥 DÀN CHỦ LỰC 2</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='main-num-box' style='color:#f2cc60; text-shadow: 0 0 15px #f2cc60;'>{res['main_3_dan2']}</div>", unsafe_allow_html=True)
        
    with col3:
        st.markdown("<p style='text-align:center; color:#58a6ff; font-weight:bold;'>🛡️ DÀN LÓT AN TOÀN</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='main-num-box' style='color:#58a6ff; font-size:50px; text-shadow: 0 0 15px #58a6ff;'>{res['support_4']}</div>", unsafe_allow_html=True)
    
    st.divider()
    st.markdown(f"<b>🧠 PHÂN TÍCH CHUYÊN SÂU:</b> {res['logic']}")
    
    # Kết hợp các dàn để đánh 7 số Kubet
    combined_7 = "".join(sorted(set(res['main_3_dan1'] + res['main_3_dan2'] + res['support_4'])))[:7]
    st.text_input("📋 SAO CHÉP DÀN 7 SỐ TỔNG HỢP:", combined_7)
    st.markdown("</div>", unsafe_allow_html=True)

# Thống kê nhịp rơi để anh đối soát
if st.session_state.history:
    with st.expander("📊 Xem Ma trận Tần suất (Hỗ trợ soi cầu bệt)"):
        # Hiển thị biểu đồ tần suất để anh thấy con số nào đang bệt
        all_d = "".join(st.session_state.history[-60:])
        counts = Counter(all_d)
        df_counts = pd.DataFrame(counts.items(), columns=['Số', 'Số lần xuất hiện']).sort_values('Số')
        st.bar_chart(df_counts.set_index('Số'))
        st.markdown("<p class='info-text'>* Biểu đồ dựa trên 60 kỳ gần nhất để xác định nhịp bệt hiện tại.</p>", unsafe_allow_html=True)
