# ================= IMPORT THƯ VIỆN =================
import streamlit as st
import google.generativeai as genai
import re
import json
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime

# ================= CẤU HÌNH =================
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    st.error("⚠️ Chưa cấu hình API Key!")
    st.stop()

st.set_page_config(page_title="TITAN v27.0 - ANTI-SCAM", layout="wide", page_icon="🛡️")

@st.cache_resource
def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash-latest')
    except: 
        return None

neural_engine = setup_neural()

# ================= QUẢN LÝ DỮ LIỆU =================
def load_data_from_json(uploaded_file):
    if uploaded_file is not None:
        try:
            return json.load(uploaded_file)
        except:
            return []
    return []

def convert_df_to_json(data):
    return json.dumps(data, ensure_ascii=False).encode('utf-8')

if "history" not in st.session_state:
    st.session_state.history = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ================= THUẬT TOÁN 1: PHÁT HIỆN CẦU LỪA =================
def detect_scam_patterns(history):
    """
    Phát hiện các dấu hiệu nhà cái đang "lừa":
    1. Pattern quá đẹp/quá hoàn hảo
    2. Thay đổi đột ngột sau khi pattern ổn định
    3. Ra số trùng lặp bất thường
    """
    if len(history) < 10:
        return {"scam_level": "UNKNOWN", "warnings": [], "risk_score": 0}
    
    warnings = []
    risk_score = 0
    
    recent = history[-20:]
    
    # 1. Kiểm tra pattern lặp lại QUÁ NHIỀU
    all_nums = "".join(recent)
    digit_freq = Counter(all_nums)
    most_common_count = max(digit_freq.values()) if digit_freq else 0
    
    if most_common_count > 15:  # Một số xuất hiện >15 lần trong 20 kỳ
        warnings.append(f"⚠️ Số {digit_freq.most_common(1)[0][0]} ra QUÁ NHIỀU ({most_common_count} lần)")
        risk_score += 30
    
    # 2. Kiểm tra cầu bệt bất thường
    for pos in range(5):
        pos_seq = [int(num[pos]) if len(num) > pos else 0 for num in recent]
        
        # Đếm streak (chuỗi liên tiếp)
        max_streak = 1
        current_streak = 1
        for i in range(1, len(pos_seq)):
            if pos_seq[i] == pos_seq[i-1]:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
        
        if max_streak >= 4:  # Bệt 4 kỳ trở lên
            warnings.append(f"🎭 Vị trí {pos} bệt {max_streak} kỳ → DẤU HIỆU LỪA")
            risk_score += 25
    
    # 3. Kiểm tra sự thay đổi ĐỘT NGỘT
    if len(history) >= 30:
        old_recent = history[-30:-10]
        new_recent = history[-10:]
        
        old_avg = sum(int(d) for num in old_recent for d in num) / len(old_recent) / 5
        new_avg = sum(int(d) for num in new_recent for d in num) / len(new_recent) / 5
        
        if abs(new_avg - old_avg) > 2:  # Thay đổi trung bình >2
            warnings.append(f"📉 Thay đổi đột ngột: TB cũ {old_avg:.1f} → TB mới {new_avg:.1f}")
            risk_score += 20
    
    # 4. Kiểm tra tổng các số
    totals = [sum(int(d) for d in num) for num in recent]
    total_std = np.std(totals)
    
    if total_std < 3:  # Độ lệch chuẩn quá thấp → quá ổn định
        warnings.append(f"⚡ Tổng số QUÁ ỔN ĐỊNH (std={total_std:.2f}) → DẤU HIỆU GIẢ")
        risk_score += 25
    
    # 5. Kiểm tra số trùng lặp HOÀN TOÀN
    unique_nums = set(history[-20:])
    if len(unique_nums) < 15:  # Ít hơn 15 số khác nhau trong 20 kỳ
        warnings.append(f"🔄 Quá ít số độc nhất ({len(unique_unique)}/20) → NHÀ CÁI ĐIỀU KHIỂN")
        risk_score += 30
    
    # Đánh giá mức độ rủi ro
    if risk_score >= 60:
        scam_level = "HIGH - NÊN DỪNG"
    elif risk_score >= 40:
        scam_level = "MEDIUM - CẨN THẬN"
    elif risk_score >= 20:
        scam_level = "LOW - THEO DÕI"
    else:
        scam_level = "NORMAL - CÓ THỂ CHƠI"
    
    return {
        "scam_level": scam_level,
        "warnings": warnings,
        "risk_score": risk_score,
        "recommendation": "DỪNG NGAY" if risk_score >= 60 else "CHỜ" if risk_score >= 40 else "CÂN NHẮC"
    }

# ================= THUẬT TOÁN 2: PHÁT HIỆN BẺ CẦU =================
def detect_bridge_break(history):
    """
    Phát hiện khi nhà cái đang "bẻ cầu":
    - Pattern đẹp đột ngột biến mất
    - Số ra ngẫu nhiên bất thường
    """
    if len(history) < 15:
        return {"breaking": False, "signs": []}
    
    signs = []
    recent = history[-15:]
    
    # 1. Kiểm tra sự gián đoạn pattern
    for pos in range(5):
        pos_seq = [int(num[pos]) if len(num) > pos else 0 for num in recent]
        
        # Tính autocorrelation (tương quan tự động)
        if len(pos_seq) > 5:
            correlation = np.corrcoef(pos_seq[:-1], pos_seq[1:])[0, 1]
            
            if abs(correlation) < 0.2:  # Tương quan rất thấp → ngẫu nhiên
                signs.append(f"Vị trí {pos}: Tương quan thấp ({correlation:.2f}) → BẺ CẦU")
    
    # 2. Kiểm tra độ phức tạp (entropy)
    all_digits = "".join(recent)
    freq = Counter(all_digits)
    total = len(all_digits)
    entropy = -sum((count/total) * np.log2(count/total) for count in freq.values())
    
    if entropy > 3.2:  # Entropy cao → quá ngẫu nhiên
        signs.append(f"🎲 Entropy cao ({entropy:.2f}) → NGẪU NHIÊN BẤT THƯỜNG")
    
    return {
        "breaking": len(signs) > 0,
        "signs": signs,
        "entropy": entropy
    }

# ================= THUẬT TOÁN 3: PHÂN TÍCH NHỊP NHÀ CÁI =================
def analyze_house_rhythm(history):
    """
    Phân tích nhịp độ nhà cái thay đổi:
    - Chu kỳ thay đổi pattern
    - Thời điểm "an toàn" vs "nguy hiểm"
    """
    if len(history) < 30:
        return {"cycle": "UNKNOWN", "safe_period": False}
    
    # Phân tích theo chu kỳ 10 kỳ
    cycles = []
    for i in range(0, len(history)-10, 10):
        chunk = history[i:i+10]
        unique = len(set(chunk))
        cycles.append(unique)
    
    # Nếu chu kỳ thay đổi đều đặn → có pattern
    if len(cycles) >= 3:
        cycle_std = np.std(cycles)
        if cycle_std < 1.5:
            return {
                "cycle": f"ỔN ĐỊNH ({np.mean(cycles):.1f} số độc nhất/chu kỳ)",
                "safe_period": True,
                "warning": "Nhà cái đang theo chu kỳ → Có thể dự đoán"
            }
        else:
            return {
                "cycle": "BẤT ỔN",
                "safe_period": False,
                "warning": "Nhà cái thay đổi liên tục → RỦI RO CAO"
            }
    
    return {"cycle": "UNKNOWN", "safe_period": False}

# ================= GIAO DIỆN =================
st.markdown("""
    <style>
    .stApp { background: #0d1117; color: #c9d1d9; }
    .main-card { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 20px; margin: 10px 0; }
    .danger-box { background: #7c2d12; border-left: 5px solid #fbbf24; padding: 15px; border-radius: 8px; margin: 10px 0; }
    .warning-box { background: #451a03; border-left: 5px solid #f59e0b; padding: 15px; border-radius: 8px; margin: 10px 0; }
    .safe-box { background: #064e3b; border-left: 5px solid #10b981; padding: 15px; border-radius: 8px; margin: 10px 0; }
    .big-number { font-size: 60px; font-weight: 800; color: #ff7b72; text-align: center; letter-spacing: 8px; }
    .risk-high { color: #ef4444; font-size: 24px; font-weight: bold; }
    .risk-med { color: #f59e0b; font-size: 24px; font-weight: bold; }
    .risk-low { color: #10b981; font-size: 24px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ TITAN v27.0 - PHÁT HIỆN CẦU LỪA")
st.markdown("### 🎭 Nhận diện thủ thuật nhà cái trước khi mất tiền")
st.markdown("---")

# ================= SIDEBAR =================
with st.sidebar:
    st.header("💾 Database")
    
    uploaded_db = st.file_uploader("📂 Nạp DB (JSON)", type="json")
    if uploaded_db:
        st.session_state.history = load_data_from_json(uploaded_db)
        st.success(f"Đã nạp {len(st.session_state.history)} kỳ!")
        st.rerun()
    
    st.divider()
    
    if st.session_state.history:
        json_data = convert_df_to_json(st.session_state.history)
        st.download_button(
            label="💾 Tải DB về",
            data=json_data,
            file_name=f"titan_db_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
    
    st.divider()
    st.write(f"📊 **Tổng kỳ:** {len(st.session_state.history)}")

# ================= NHẬP LIỆU =================
col1, col2 = st.columns([3, 1])
with col1:
    raw_input = st.text_area("📡 Dán kết quả (Mỗi dòng 5 số)", height=150, placeholder="32880\n21808...")
with col2:
    st.metric("Kỳ gần nhất", len(st.session_state.history))
    if st.button("🚀 LƯU DỮ LIỆU", type="primary", use_container_width=True):
        if raw_input:
            clean = re.findall(r"\d{5}", raw_input)
            if clean:
                new_data = list(dict.fromkeys(clean))
                st.session_state.history.extend(new_data)
                st.session_state.history = st.session_state.history[-1000:]
                st.success(f"✅ Đã lưu {len(new_data)} kỳ!")
                st.rerun()

# ================= PHÂN TÍCH CẦU LỪA =================
st.markdown("---")
st.subheader("🎭 PHÂN TÍCH RỦI RO NHÀ CÁI")

if st.session_state.history and len(st.session_state.history) >= 15:
    if st.button("🔍 QUÉT CẦU LỪA", type="secondary", use_container_width=True):
        with st.spinner("🔬 Đang phân tích pattern nhà cái..."):
            
            # Chạy 3 thuật toán
            scam_detect = detect_scam_patterns(st.session_state.history)
            bridge_break = detect_bridge_break(st.session_state.history)
            rhythm = analyze_house_rhythm(st.session_state.history)
            
            # Lưu kết quả
            st.session_state.last_prediction = {
                "scam": scam_detect,
                "bridge_break": bridge_break,
                "rhythm": rhythm
            }
            st.rerun()

elif st.session_state.history:
    st.warning(f"⚠️ Cần ít nhất 15 kỳ để phân tích (hiện có {len(st.session_state.history)})")

# ================= HIỂN THỊ CẢNH BÁO =================
if st.session_state.last_prediction and "scam" in st.session_state.last_prediction:
    data = st.session_state.last_prediction
    
    st.markdown("---")
    
    # Hiển thị mức độ rủi ro
    risk_score = data['scam']['risk_score']
    
    if risk_score >= 60:
        st.markdown(f"""
            <div class='danger-box'>
                <h2 style='color: #fbbf24'>🚨 CẢNH BÁO ĐỎ - RỦI RO: {risk_score}/100</h2>
                <p style='font-size: 18px'><strong>{data['scam']['scam_level']}</strong></p>
                <p>📌 Khuyến nghị: <strong>{data['scam']['recommendation']}</strong></p>
            </div>
        """, unsafe_allow_html=True)
    elif risk_score >= 40:
        st.markdown(f"""
            <div class='warning-box'>
                <h3 style='color: #f59e0b'>⚠️ CẢNH BÁO VÀNG - RỦI RO: {risk_score}/100</h3>
                <p><strong>{data['scam']['scam_level']}</strong></p>
                <p>📌 Khuyến nghị: <strong>{data['scam']['recommendation']}</strong></p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class='safe-box'>
                <h3 style='color: #10b981'>✅ TƯƠNG ĐỐI AN TOÀN - RỦI RO: {risk_score}/100</h3>
                <p><strong>{data['scam']['scam_level']}</strong></p>
            </div>
        """, unsafe_allow_html=True)
    
    # Hiển thị các dấu hiệu cảnh báo
    if data['scam']['warnings']:
        st.markdown("### 🚩 Các dấu hiệu phát hiện:")
        for warning in data['scam']['warnings']:
            st.write(f"- {warning}")
    
    # Hiển thị dấu hiệu bẻ cầu
    if data['bridge_break']['breaking']:
        st.markdown("### 🔨 Dấu hiệu bẻ cầu:")
        for sign in data['bridge_break']['signs']:
            st.write(f"- {sign}")
    
    # Hiển thị nhịp nhà cái
    st.markdown("### 📊 Phân tích nhịp nhà cái:")
    st.write(f"**Chu kỳ:** {data['rhythm']['cycle']}")
    if not data['rhythm']['safe_period']:
        st.warning(data['rhythm']['warning'])
    else:
        st.success(data['rhythm']['warning'])
    
    st.divider()
    
    # Khuyến nghị cụ thể
    st.markdown("### 💡 KHUYẾN NGHỊ CHIẾN LƯỢC:")
    
    if risk_score >= 60:
        st.error("""
        **DỪNG CHƠI NGAY!**
        
        - Nhà cái đang điều khiển kết quả rõ ràng
        - Pattern quá bất thường
        - Chờ ít nhất 10-15 kỳ nữa để quan sát
        - Không vào tiền lúc này!
        """)
    elif risk_score >= 40:
        st.warning("""
        **CHỜ VÀ QUAN SÁT**
        
        - Có dấu hiệu nhà cái đang test pattern
        - Nếu muốn chơi: Đánh nhỏ để thăm dò
        - Theo dõi thêm 5-10 kỳ
        - Không all-in!
        """)
    else:
        st.success("""
        **CÓ THỂ THAM GIA**
        
        - Pattern tương đối ổn định
        - Vẫn nên đánh nhỏ, quản lý vốn chặt
        - Theo dõi sát sao từng kỳ
        - Sẵn sàng dừng nếu có dấu hiệu lạ
        """)

# ================= BIỂU ĐỒ =================
st.markdown("---")
with st.expander("📈 Biểu đồ phân tích"):
    if st.session_state.history:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("##### Tần suất số (20 kỳ gần)")
            all_d = "".join(st.session_state.history[-20:])
            df_freq = pd.Series(Counter(all_d)).sort_index()
            st.bar_chart(df_freq, color="#f59e0b")
        
        with col2:
            st.write("##### Tổng các kỳ")
            totals = [sum(int(d) for d in num) for num in st.session_state.history[-20:]]
            st.line_chart(pd.Series(totals))

# ================= FOOTER =================
st.markdown("---")
st.caption("""
⚠️ **LƯU Ý QUAN TRỌNG:** 
- Tool này giúp NHẬN DIỆN RỦI RO, không đảm bảo thắng
- Nhà cái online CÓ THỂ điều khiển kết quả
- Chỉ chơi với số tiền có thể mất
- DỪNG ĐÚNG LÚC quan trọng hơn thắng
""")
st.caption(f"🕐 Cập nhật: {datetime.now().strftime('%d/%m/%Y %H:%M')}")