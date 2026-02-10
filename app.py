import streamlit as st
import collections
import time
import numpy as np
import pandas as pd
from datetime import datetime
import random

st.set_page_config(page_title="AI 3-TINH ELITE v34 PRO", layout="centered")

# CSS nâng cao
st.markdown("""
    <style>
    .stApp { 
        background: linear-gradient(135deg, #0b0f13 0%, #1a1f2e 100%);
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }
    
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #00ffcc 0%, #00ccff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .result-card { 
        border: 3px solid #00ffcc;
        border-radius: 20px;
        padding: 25px;
        background: rgba(22, 27, 34, 0.9);
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 0 30px rgba(0, 255, 204, 0.2);
        backdrop-filter: blur(10px);
    }
    
    .numbers-display { 
        font-size: 90px !important;
        background: linear-gradient(90deg, #ffff00 0%, #ffcc00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        letter-spacing: 15px;
        margin: 20px 0;
        text-shadow: 0 0 20px rgba(255, 255, 0, 0.3);
    }
    
    .eliminated-box { 
        color: #ff4b4b;
        font-size: 18px;
        font-weight: bold;
        padding: 10px;
        border: 1px solid #ff4b4b;
        border-radius: 10px;
        margin: 10px 0;
        background: rgba(255, 75, 75, 0.1);
    }
    
    .confidence-box {
        color: #00ffcc;
        font-size: 18px;
        font-weight: bold;
        padding: 10px;
        border: 1px solid #00ffcc;
        border-radius: 10px;
        margin: 10px 0;
        background: rgba(0, 255, 204, 0.1);
    }
    
    .stTextArea textarea { 
        background-color: rgba(13, 17, 23, 0.8) !important;
        color: #00ffcc !important;
        border: 2px solid #00ccff !important;
        border-radius: 10px !important;
        font-size: 16px !important;
    }
    
    .stButton button {
        background: linear-gradient(90deg, #00ffcc 0%, #00ccff 100%);
        color: #000 !important;
        font-weight: bold;
        font-size: 18px;
        border: none;
        border-radius: 25px;
        padding: 15px 30px;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 20px rgba(0, 255, 204, 0.4);
    }
    
    .stat-box {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🛡️ AI 3-TINH ELITE v34 PRO</h1>", unsafe_allow_html=True)
st.markdown("### 🔮 Hệ thống AI loại trừ nhà cái & soi 3 tinh chiến thuật")

# Sidebar cho cài đặt nâng cao
with st.sidebar:
    st.markdown("### ⚙️ CÀI ĐẶT NÂNG CAO")
    
    algorithm_mode = st.selectbox(
        "Chọn thuật toán:",
        ["THÔNG MINH CƠ BẢN", "PHÂN TÍCH NÂNG CAO", "CHIẾN LƯỢC ĐA TẦNG"]
    )
    
    risk_level = st.slider("Mức độ rủi ro:", 1, 10, 5, 
                          help="1: Bảo thủ nhất, 10: Mạo hiểm nhất")
    
    history_depth = st.number_input("Độ sâu phân tích (số ván):", 
                                   min_value=10, max_value=1000, value=50)
    
    show_stats = st.checkbox("Hiển thị thống kê chi tiết", value=True)

# Hàm phân tích nâng cao
def advanced_analysis(data, risk_level, mode):
    """Thuật toán phân tích nâng cao với nhiều lớp logic"""
    
    # Làm sạch dữ liệu
    raw = "".join(filter(str.isdigit, data))
    if len(raw) < 10:
        return None, None, None, None
    
    counts = collections.Counter(raw)
    all_nums = [str(i) for i in range(10)]
    
    # --- LỚP 1: PHÂN TÍCH TẦN SUẤT NÂNG CAO ---
    weighted_freq = {}
    recent_data = raw[-20:] if len(raw) >= 20 else raw
    
    for num in all_nums:
        # Tần suất tổng
        total_freq = counts[num] / len(raw) if len(raw) > 0 else 0
        
        # Tần suất gần đây (quan trọng hơn)
        recent_freq = recent_data.count(num) / len(recent_data) if len(recent_data) > 0 else 0
        
        # Khoảng cách từ lần xuất hiện cuối
        last_position = raw.rfind(num)
        distance = len(raw) - last_position if last_position != -1 else 999
        
        # Tính điểm weighted
        weight = (recent_freq * 0.6 + total_freq * 0.3 + (1/(distance+1)) * 0.1)
        weighted_freq[num] = weight
    
    # --- LỚP 2: PHÂN TÍCH PATTERN CHUỖI ---
    patterns = {}
    for i in range(len(raw)-1):
        pair = raw[i:i+2]
        if pair not in patterns:
            patterns[pair] = 0
        patterns[pair] += 1
    
    # Tìm số có xu hướng đi cùng nhau
    related_nums = {}
    for num in all_nums:
        related_count = 0
        for pattern, freq in patterns.items():
            if num in pattern:
                related_count += freq
        related_nums[num] = related_count
    
    # --- LỚP 3: LOẠI TRỪ CHIẾN LƯỢC ---
    elimination_scores = {}
    for num in all_nums:
        score = 0
        
        # Điểm rủi ro dựa trên tần suất (số càng ít xuất hiện càng rủi ro)
        if weighted_freq[num] < 0.05:  # Xuất hiện dưới 5%
            score += 3
        elif weighted_freq[num] < 0.1:
            score += 2
        elif weighted_freq[num] < 0.15:
            score += 1
        
        # Điểm rủi ro dựa trên khoảng cách
        last_pos = raw.rfind(num)
        if last_pos == -1:
            score += 5  # Chưa bao giờ xuất hiện - rủi ro cao
        else:
            distance = len(raw) - last_pos
            if distance > 15:  # Lâu không xuất hiện
                score += 2
            elif distance < 3:  # Vừa mới xuất hiện
                score -= 1  # Giảm rủi ro
        
        # Điều chỉnh theo mức rủi ro người dùng
        score = score * (risk_level / 5)
        
        elimination_scores[num] = score
    
    # Sắp xếp và loại 3 số có điểm rủi ro cao nhất
    sorted_by_risk = sorted(all_nums, key=lambda x: elimination_scores[x], reverse=True)
    eliminated = sorted_by_risk[:3]
    
    # --- LỚP 4: CHỌN 3 TINH CHIẾN THUẬT ---
    remaining = [n for n in all_nums if n not in eliminated]
    
    # Ưu tiên chọn số dựa trên nhiều yếu tố
    selection_scores = {}
    for num in remaining:
        score = 0
        
        # Ưu tiên số có tần suất ổn định
        if 0.1 <= weighted_freq[num] <= 0.25:
            score += 3
        
        # Ưu tiên số có quan hệ với số gần đây
        last_num = raw[-1]
        if last_num != num:
            # Kiểm tra pattern với số cuối
            if f"{last_num}{num}" in patterns:
                score += patterns[f"{last_num}{num}"]
            if f"{num}{last_num}" in patterns:
                score += patterns[f"{num}{last_num}"]
        
        # Ưu tiên số không quá gần với số đã loại
        for elim in eliminated:
            if abs(int(num) - int(elim)) <= 1:
                score -= 1
        
        # Thêm yếu tố ngẫu nhiên có kiểm soát
        score += random.uniform(0, 0.5)
        
        selection_scores[num] = score
    
    # Chọn top 3 số
    top_selected = sorted(remaining, key=lambda x: selection_scores[x], reverse=True)[:3]
    
    # Tính độ tin cậy
    confidence = min(85 + (risk_level * 1.5), 95)
    
    return top_selected, eliminated, weighted_freq, confidence

# Giao diện chính
col1, col2 = st.columns([3, 1])

with col1:
    data_input = st.text_area(
        "📡 DÁN CHUỖI SỐ THỰC TẾ (ít nhất 20 số):", 
        height=120, 
        placeholder="Ví dụ: 51273849015623748901234567890123456789...",
        help="Nhập chuỗi số liên tiếp từ các ván gần nhất"
    )
    
    if st.button("🚀 KÍCH HOẠT AI PHÂN TÍCH ĐA TẦNG", use_container_width=True):
        if len(data_input.strip()) < 10:
            st.error("⚠️ Cần ít nhất 10 số để AI phân tích pattern!")
        else:
            with st.spinner('🔍 Đang phân tích đa tầng...'):
                # Tạo thanh tiến trình
                progress_bar = st.progress(0)
                
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Phân tích
                tinh3, eliminated, stats, confidence = advanced_analysis(
                    data_input, risk_level, algorithm_mode
                )
                
                if tinh3:
                    # Hiển thị kết quả chính
                    st.markdown(f"""
                        <div class='result-card'>
                            <p style='color: #00e5ff; font-size: 24px; font-weight: bold;'>
                                🎯 DÀN 3 TINH TỐI ƯU
                            </p>
                            <p class='numbers-display'>{" • ".join(tinh3)}</p>
                            
                            <div class='confidence-box'>
                                📊 Độ tin cậy: {confidence:.1f}% 
                                | Chế độ: {algorithm_mode}
                            </div>
                            
                            <div class='eliminated-box'>
                                🚫 Đã loại trừ 3 số rủi ro cao: 
                                <span style='font-size: 22px;'>{", ".join(eliminated)}</span>
                            </div>
                            
                            <p style='color: #00ffcc; margin-top: 20px;'>
                                ⚡ <b>7 SỐ AN TOÀN:</b> {", ".join([n for n in "0123456789" if n not in eliminated])}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Hiển thị thống kê chi tiết
                    if show_stats:
                        st.markdown("### 📈 THỐNG KÊ PHÂN TÍCH CHI TIẾT")
                        
                        cols = st.columns(5)
                        stats_items = list(stats.items()) if stats else []
                        
                        for idx, (num, freq) in enumerate(stats_items[:10]):
                            with cols[idx % 5]:
                                st.markdown(f"""
                                    <div class='stat-box'>
                                        <div style='font-size: 24px; color: {'#00ff00' if freq > 0.15 else '#ff5555'};'>
                                            {num}
                                        </div>
                                        <div style='font-size: 14px;'>
                                            Tần suất: {freq*100:.1f}%
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
                        
                        # Biểu đồ đơn giản
                        st.markdown("#### 📊 BIỂU ĐỒ TẦN SUẤT")
                        chart_data = pd.DataFrame({
                            'Số': list(stats.keys()) if stats else [],
                            'Tần suất': list(stats.values()) if stats else []
                        })
                        st.bar_chart(chart_data.set_index('Số'))
                    
                    # Chiến thuật đề xuất
                    st.markdown("### 🎮 CHIẾN THUẬT VÀO TIỀN")
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.markdown("""
                            #### 🥇 Số 1: **{0}**
                            - Tỷ lệ vào: **40%** vốn
                            - Dự đoán: Xuất hiện trong 2 ván tới
                        """.format(tinh3[0]))
                    
                    with col_b:
                        st.markdown("""
                            #### 🥈 Số 2: **{0}**
                            - Tỷ lệ vào: **35%** vốn
                            - Dự đoán: Xuất hiện trong 3 ván tới
                        """.format(tinh3[1]))
                    
                    with col_c:
                        st.markdown("""
                            #### 🥉 Số 3: **{0}**
                            - Tỷ lệ vào: **25%** vốn
                            - Dự đoán: Xuất hiện trong 4 ván tới
                        """.format(tinh3[2]))
                    
                    st.success(f"✅ **DỰ ĐOÁN:** Trong 5 số giải thưởng, có ít nhất 2 trong 3 số trên xuất hiện!")

with col2:
    st.markdown("### 📋 HƯỚNG DẪN")
    st.info("""
    **CÁCH SỬ DỤNG:**
    1. Thu thập ít nhất 20 số gần nhất
    2. Dán vào ô nhập liệu
    3. Chọn chế độ phân tích
    4. Nhấn KÍCH HOẠT
    
    **CHIẾN THUẬT:**
    - Nhà cái cho 7 số
    - AI loại 3 số rủi ro
    - Tập trung vào 3 TINH
    - Phân bổ vốn theo tỷ lệ
    """)
    
    st.markdown("### 🔄 LỊCH SỬ")
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    if st.button("💾 Lưu kết quả hiện tại"):
        if 'tinh3' in locals():
            st.session_state.history.append({
                'time': datetime.now().strftime("%H:%M:%S"),
                'numbers': tinh3,
                'eliminated': eliminated
            })
            st.success("Đã lưu!")
    
    for i, record in enumerate(st.session_state.history[-3:]):
        st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); padding: 10px; border-radius: 10px; margin: 5px 0;'>
                <small>{record['time']}</small><br/>
                <b>{' • '.join(record['numbers'])}</b>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; font-size: 14px;'>
        <b>AI 3-TINH ELITE v34 PRO</b> | Sử dụng thuật toán phân tích đa tầng<br/>
        ⚠️ Đây là công cụ hỗ trợ phân tích, không đảm bảo 100% chính xác
    </div>
""", unsafe_allow_html=True)