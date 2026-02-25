import streamlit as st
import google.generativeai as genai
import re
import json
import os
from collections import Counter

# ================= CẤU HÌNH HỆ THỐNG TITAN v23.1 =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_v23_core.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None

neural_engine = setup_neural()

# ================= QUẢN LÝ DỮ LIỆU SẠCH =================
def load_memory():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try: return json.load(f)
            except: return []
    return []

def save_memory(data):
    with open(DB_FILE, "w") as f:
        json.dump(data[-2000:], f)

if "history" not in st.session_state:
    st.session_state.history = load_memory()

# ================= THUẬT TOÁN NHẬN DIỆN BỆT ẢO =================
def detect_streak_and_danger(data):
    if len(data) < 10: return False, "Đang thu thập dữ liệu"
    
    all_digits = "".join(data[-5:])
    counts = Counter(all_digits)
    
    # Kiểm tra nếu có 1 số xuất hiện quá dày (Bệt số)
    for num, freq in counts.items():
        if freq >= 4: # Một số xuất hiện 4/5 kỳ gần nhất
            return True, f"CẢNH BÁO BỆT: Số {num} đang bệt ảo. Nhà cái đang giam cầu!"
            
    return False, "Nhịp cầu ổn định"

# ================= GIAO DIỆN TITAN v23.1 =================
st.set_page_config(page_title="TITAN v23.1 - ANTI-FRAUD AI", layout="wide")
st.markdown("""
    <style>
    .stApp { background: #0b0e14; color: #e6edf3; }
    .danger-zone { background: #440000; border: 2px solid #ff0000; padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; }
    .safe-zone { background: #002200; border: 2px solid #00ff00; padding: 20px; border-radius: 10px; text-align: center; }
    .main-num { font-size: 110px; color: #00ff00; font-weight: 900; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("🧬 TITAN v23.1 - HỆ THỐNG PHÒNG THỦ & BẺ CẦU")

raw_input = st.text_area("📥 NẠP DỮ LIỆU (5 số viết liền):", height=100)

if st.button("🚀 KÍCH HOẠT PHÂN TÍCH CHỐNG BỆT"):
    clean_data = re.findall(r"\d{5}", raw_input)
    if clean_data:
        st.session_state.history.extend(clean_data)
        save_memory(st.session_state.history)
        
        is_danger, msg = detect_streak_and_danger(st.session_state.history)
        
        # PROMPT v23.1 - YÊU CẦU KHẮT KHE
        prompt = f"""
        Hệ thống: TITAN v23.1 PRO. 
        Lịch sử: {st.session_state.history[-50:]}.
        Tình trạng: {msg}.
        Nhiệm vụ: 
        1. Nếu 'is_danger' là True, TUYỆT ĐỐI không cho số, trả về warning: true.
        2. Nếu an toàn, phân tích Bóng số và Ma trận vị trí để chọn 3 số chủ lực.
        3. Phân biệt rõ cầu Bệt và cầu Nhảy. Không đánh theo cầu đã bệt quá 4 kỳ.
        TRẢ VỀ JSON: {{"main_3": "ABC", "support_4": "DEFG", "warning": {str(is_danger).lower()}, "logic": "{msg}", "confidence": 95}}
        """
        
        try:
            response = neural_engine.generate_content(prompt)
            st.session_state.v23_1_res = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
        except:
            st.session_state.v23_1_res = {"warning": True, "logic": "Lỗi kết nối hoặc cầu quá xấu."}
        st.rerun()

# ================= HIỂN THỊ KẾT QUẢ =================
if "v23_1_res" in st.session_state:
    res = st.session_state.v23_1_res
    
    if res.get('warning'):
        st.markdown(f"<div class='danger-zone'>🚫 KHÔNG ĐÁNH KỲ NÀY<br>{res['logic']}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='safe-zone'>", unsafe_allow_html=True)
        st.write(f"✅ NHỊP CẦU AN TOÀN - CHIẾN THUẬT: {res['logic']}")
        st.markdown(f"<div class='main-num'>{res['main_3']}</div>", unsafe_allow_html=True)
        st.write(f"Dàn lót: {res['support_4']} | Độ tin cậy: {res['confidence']}%")
        st.markdown("</div>", unsafe_allow_html=True)

st.sidebar.write(f"Dữ liệu tích lũy: {len(st.session_state.history)} kỳ")
