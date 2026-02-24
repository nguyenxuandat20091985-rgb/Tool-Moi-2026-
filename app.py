import streamlit as st
import google.generativeai as genai
import re
import json
import os
from collections import Counter

# ================= CẤU HÌNH TỐI CAO =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
genai.configure(api_key=API_KEY)

st.set_page_config(page_title="TITAN v23.0 ULTIMATE", layout="wide")

# CSS Chuyên dụng cho chế độ "Gỡ Vốn"
st.markdown("""
    <style>
    .stApp { background: #0a0a0a; color: #00ff41; font-family: 'Courier New', monospace; }
    .main-card { border: 2px solid #00ff41; padding: 20px; border-radius: 10px; background: #000; box-shadow: 0 0 20px #00ff41; }
    .hot-num { color: #ff0000; font-size: 70px; font-weight: bold; text-shadow: 0 0 10px #ff0000; }
    .logic-text { color: #888; font-style: italic; border-left: 3px solid #444; padding-left: 10px; }
    </style>
""", unsafe_allow_html=True)

# ================= THUẬT TOÁN ĐỐI KHÁNG AI NHÀ CÁI =================
def analyze_kubet_logic(history):
    if not history: return {}
    
    # 1. Phân tích tần suất sâu (Digit Frequency)
    all_str = "".join(history)
    freq = Counter(all_str)
    
    # 2. Phân tích "Cặp bài trùng" (Co-occurrence)
    # Tìm xem nếu số A ra thì số B nào hay ra cùng
    pairs = []
    for s in history:
        unique_nums = sorted(list(set(s)))
        for i in range(len(unique_nums)):
            for j in range(i+1, len(unique_nums)):
                pairs.append(unique_nums[i] + unique_nums[j])
    common_pairs = Counter(pairs).most_common(5)
    
    return {"freq": freq, "pairs": common_pairs}

# ================= PROMPT CHIẾN ĐẤU (ULTIMATE) =================
def get_ai_prediction(history):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Lấy 30 kỳ gần nhất để AI không bị loãng
    recent_data = history[-30:]
    
    prompt = f"""
    Hệ thống: TITAN v23.0 - Chuyên gia đối kháng AI Kubet 5D.
    Dữ liệu 30 kỳ: {recent_data}.
    Quy luật phát hiện: {analyze_kubet_logic(recent_data)}.
    
    Yêu cầu khắt khe:
    1. Phân tích kèo "3 số 5 tinh" (Chỉ cần 3 số dự đoán xuất hiện trong 5 vị trí kết quả là thắng).
    2. Phát hiện "Cầu bệt" và "Cầu nhảy". Ví dụ số 4 đang ra cực dày thì phải tận dụng.
    3. Chọn ra 3 số CHỦ LỰC (Dàn 3 số 5 tinh).
    4. Nếu xác suất thắng dưới 80%, đặt 'action': 'WAIT'.
    
    TRẢ VỀ JSON:
    {{
        "top_3": ["x", "y", "z"],
        "support": ["a", "b"],
        "logic": "Giải mã ngắn gọn cầu đang chạy",
        "action": "BET" hoặc "WAIT",
        "confidence": 95
    }}
    """
    try:
        response = model.generate_content(prompt)
        return json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
    except:
        return None

# ================= GIAO DIỆN CHÍNH =================
st.title("⚡ TITAN v23.0 ULTIMATE: ANTI-AI KUBET")

with st.sidebar:
    st.header("📥 DỮ LIỆU ĐẦU VÀO")
    raw_data = st.text_area("Dán danh sách kết quả (5 số):", height=300)
    if st.button("🔥 GIẢI MÃ NGAY"):
        clean = re.findall(r"\d{5}", raw_data)
        st.session_state.history = clean
        st.session_state.prediction = get_ai_prediction(clean)

if "history" in st.session_state:
    st.write(f"📊 Đã nạp: **{len(st.session_state.history)}** kỳ.")
    
    if "prediction" in st.session_state and st.session_state.prediction:
        res = st.session_state.prediction
        
        # HIỂN THỊ CẢNH BÁO
        if res['action'] == 'WAIT':
            st.warning("⚠️ AI NHÀ CÁI ĐANG ĐẢO CẦU - LỆNH: CHỜ (KHÔNG VÀO TIỀN)")
        else:
            st.success("✅ TÍN HIỆU ĐẸP - LỆNH: VÀO TIỀN")

        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("🎯 **3 SỐ CHỦ LỰC (Kèo 3 số 5 tinh):**")
            st.markdown(f"<div class='hot-num'>{' - '.join(res['top_3'])}</div>", unsafe_allow_html=True)
            st.write(f"💡 **Logic AI:** {res['logic']}")
        
        with col2:
            st.metric("Độ tự tin", f"{res['confidence']}%")
            st.write("**Số lót an toàn:**")
            st.write(f"👉 {', '.join(res['support'])}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Phân tích thực tế từ dữ liệu anh gửi
        st.divider()
        st.subheader("📈 Phân tích nhịp cầu thực tế")
        logic_data = analyze_kubet_logic(st.session_state.history)
        st.write(f"Số xuất hiện nhiều nhất: **{logic_data['freq'].most_common(1)[0][0]}**")
        st.write(f"Cặp số hay đi cùng nhau: **{', '.join([p[0] for p in logic_data['pairs']])}**")

else:
    st.info("Hãy dán kết quả vào cột bên trái để AI bắt đầu quét chu kỳ.")
