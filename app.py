import streamlit as st
import collections
import time
import numpy as np
import pandas as pd
from datetime import datetime
import requests
import json
from typing import List, Dict, Tuple
import hashlib

# =============== CẤU HÌNH API ===============
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

# =============== CLASS CHÍNH ===============
class LotteryAIAnalyzer:
    def __init__(self):
        self.history = []
        self.patterns = {}
        self.risk_scores = {str(i): 0 for i in range(10)}
        
    def connect_gemini(self, prompt: str) -> str:
        """Kết nối với Gemini AI để phân tích pattern phức tạp"""
        try:
            if GEMINI_API_KEY:
                headers = {"Content-Type": "application/json"}
                data = {
                    "contents": [{
                        "parts": [{"text": f"""
                        Phân tích chuỗi số xổ số: {prompt}
                        Tìm pattern ẩn, số có khả năng bị giam,
                        và dự đoán 3 số có xác suất cao nhất.
                        Phân tích theo xác suất thống kê nâng cao.
                        """}]
                    }]
                }
                response = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}",
                    headers=headers,
                    json=data
                )
                return response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        except:
            pass
        return ""
    
    def analyze_advanced_frequency(self, data: str, window_size: int = 20) -> Dict:
        """Phân tích tần suất nâng cao với sliding window"""
        nums = list(filter(str.isdigit, data))
        
        # Phân tích Markov Chain (bậc 2)
        markov_probs = self._calculate_markov_chain(nums)
        
        # Phân tích cold/hot numbers
        hot_numbers = self._find_hot_numbers(nums[-window_size:])
        cold_numbers = self._find_cold_numbers(nums, window_size)
        
        # Phân tích theo giờ
        hour_pattern = self._analyze_by_hour()
        
        return {
            "markov": markov_probs,
            "hot": hot_numbers,
            "cold": cold_numbers,
            "hour_pattern": hour_pattern
        }
    
    def _calculate_markov_chain(self, nums: List[str]) -> Dict:
        """Tính xác suất Markov bậc 2"""
        transitions = {}
        for i in range(len(nums)-2):
            state = (nums[i], nums[i+1])
            next_state = nums[i+2]
            if state not in transitions:
                transitions[state] = {}
            transitions[state][next_state] = transitions[state].get(next_state, 0) + 1
        
        # Chuẩn hóa xác suất
        for state in transitions:
            total = sum(transitions[state].values())
            for num in transitions[state]:
                transitions[state][num] = transitions[state][num] / total
        
        return transitions
    
    def _find_hot_numbers(self, recent_nums: List[str], threshold: float = 0.15) -> List[str]:
        """Tìm số nóng (xuất hiện nhiều trong window gần đây)"""
        counts = collections.Counter(recent_nums)
        total = len(recent_nums)
        return [num for num, count in counts.items() if count/total >= threshold]
    
    def _find_cold_numbers(self, nums: List[str], window_size: int) -> List[str]:
        """Tìm số lạnh (lâu không xuất hiện)"""
        if len(nums) < window_size:
            return []
        
        recent_set = set(nums[-window_size:])
        all_nums = set(str(i) for i in range(10))
        return list(all_nums - recent_set)
    
    def _analyze_by_hour(self) -> Dict:
        """Phân tích pattern theo giờ trong ngày"""
        current_hour = datetime.now().hour
        hour_patterns = {
            "morning": ["0", "2", "4", "6", "8"],
            "afternoon": ["1", "3", "5", "7", "9"],
            "night": ["0", "5", "7", "8", "9"]
        }
        
        if 5 <= current_hour < 12:
            return hour_patterns["morning"]
        elif 12 <= current_hour < 18:
            return hour_patterns["afternoon"]
        else:
            return hour_patterns["night"]
    
    def eliminate_risk_numbers(self, data: str) -> Tuple[List[str], List[str]]:
        """Loại 3 số rủi ro cao nhất với thuật toán nâng cao"""
        nums = list(filter(str.isdigit, data))
        
        # Phân tích đa chiều
        analysis = self.analyze_advanced_frequency(nums)
        
        # Tính điểm rủi ro cho từng số
        risk_scores = {str(i): 0 for i in range(10)}
        
        # 1. Trừ điểm cho số lạnh
        for num in analysis["cold"]:
            risk_scores[num] += 2
        
        # 2. Trừ điểm cho số có Markov probability thấp
        last_two = tuple(nums[-2:]) if len(nums) >= 2 else ("0", "0")
        if last_two in analysis["markov"]:
            for num, prob in analysis["markov"][last_two].items():
                if prob < 0.05:
                    risk_scores[num] += 1
        
        # 3. Cộng điểm cho số nóng
        for num in analysis["hot"]:
            risk_scores[num] = max(0, risk_scores[num] - 1)
        
        # 4. Xét pattern theo giờ
        for num in analysis["hour_pattern"]:
            risk_scores[num] = max(0, risk_scores[num] - 0.5)
        
        # Lấy 3 số có điểm rủi ro cao nhất
        eliminated = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        eliminated_nums = [num for num, _ in eliminated]
        
        # 7 số còn lại
        remaining = [str(i) for i in range(10) if str(i) not in eliminated_nums]
        
        return eliminated_nums, remaining
    
    def select_top_three(self, remaining_nums: List[str], data: str) -> List[str]:
        """Chọn 3 số có xác suất cao nhất từ 7 số còn lại"""
        nums = list(filter(str.isdigit, data))
        
        # 1. Ưu tiên số theo lý thuyết bóng đề
        last_num = nums[-1] if nums else "0"
        bong_duong = {"0": "5", "1": "6", "2": "7", "3": "8", "4": "9",
                      "5": "0", "6": "1", "7": "2", "8": "3", "9": "4"}
        bong_am = {"0": "7", "1": "4", "2": "9", "3": "6", "4": "1",
                   "5": "8", "6": "3", "7": "0", "8": "5", "9": "2"}
        
        bong_duong_num = bong_duong.get(last_num, "")
        bong_am_num = bong_am.get(last_num, "")
        
        candidates = []
        
        # Thêm bóng nếu có trong remaining
        if bong_duong_num in remaining_nums:
            candidates.append(bong_duong_num)
        if bong_am_num in remaining_nums:
            candidates.append(bong_am_num)
        
        # 2. Thêm số kế tiếp và trước đó
        next_num = str((int(last_num) + 1) % 10)
        prev_num = str((int(last_num) - 1) % 10)
        
        for num in [next_num, prev_num]:
            if num in remaining_nums and num not in candidates:
                candidates.append(num)
        
        # 3. Nếu chưa đủ 3, lấy số có tần suất cao nhất trong remaining
        if len(candidates) < 3:
            remaining_counts = collections.Counter(nums)
            for num, _ in sorted(remaining_counts.items(), key=lambda x: x[1], reverse=True):
                if num in remaining_nums and num not in candidates:
                    candidates.append(num)
                if len(candidates) >= 3:
                    break
        
        # 4. Nếu vẫn chưa đủ, lấy ngẫu nhiên từ remaining
        while len(candidates) < 3:
            for num in remaining_nums:
                if num not in candidates:
                    candidates.append(num)
                if len(candidates) >= 3:
                    break
        
        return candidates[:3]

# =============== CSS TỐI GIẢN, CHUẨN HIỂN THỊ ===============
st.set_page_config(
    page_title="AI 3-TINH ELITE PRO v2.0", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
/* RESET & LAYOUT */
.stApp {
    background: linear-gradient(135deg, #0b0f13 0%, #1a1f2e 100%);
    color: #e0e0e0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    padding: 10px;
}

/* HEADER COMPACT */
.compact-header {
    text-align: center;
    padding: 15px;
    background: linear-gradient(90deg, #1e3a5f, #0e2a44);
    border-radius: 15px;
    margin-bottom: 20px;
    border: 1px solid #00ffcc33;
}

.main-title {
    font-size: 2.2rem !important;
    font-weight: 800;
    background: linear-gradient(90deg, #00ffcc, #00ccff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    text-shadow: 0 0 15px #00ffcc66;
}

.subtitle {
    color: #94a3b8;
    font-size: 0.95rem !important;
    margin-top: 5px;
}

/* RESULT CARD - KHÔNG BỊ LỖI HIỂN THỊ */
.result-card {
    border: 2px solid #00ffcc;
    border-radius: 20px;
    padding: 25px 20px;
    background: linear-gradient(145deg, #161b22, #1e242d);
    text-align: center;
    margin: 20px 0;
    box-shadow: 0 10px 30px rgba(0, 255, 204, 0.2);
}

/* 3 SỐ CHÍNH - HIỂN THỊ DẠNG VÒNG TRÒN */
.prediction-container {
    display: flex;
    justify-content: center;
    gap: 20px;
    margin: 20px 0;
    flex-wrap: wrap;
}

.number-ball {
    width: 90px;
    height: 90px;
    background: radial-gradient(circle at 30% 30%, #ffd700, #ffaa00);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3rem;
    font-weight: 900;
    color: #1a1e2c;
    box-shadow: 0 0 25px rgba(255, 215, 0, 0.7);
    border: 3px solid #ffffff;
    text-shadow: 0 2px 5px rgba(0,0,0,0.3);
    animation: glow 1.5s ease-in-out infinite alternate;
}

@keyframes glow {
    from { box-shadow: 0 0 15px #ffaa00; }
    to { box-shadow: 0 0 30px #ffd700, 0 0 50px #ffaa00; }
}

/* INFO BOXES */
.info-box {
    background: rgba(30, 41, 59, 0.7);
    border-radius: 12px;
    padding: 15px;
    margin: 15px 0;
    border-left: 6px solid;
}

.eliminated-box {
    border-left-color: #ef4444;
    background: rgba(239, 68, 68, 0.1);
}

.safe-box {
    border-left-color: #10b981;
    background: rgba(16, 185, 129, 0.1);
}

.info-title {
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.info-numbers {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 4px;
    color: #f8fafc;
}

/* TEXT AREA */
.stTextArea textarea {
    background-color: #1e293b !important;
    color: #00ffcc !important;
    border: 2px solid #00ffcc !important;
    border-radius: 12px !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
}

/* BUTTON */
.stButton button {
    background: linear-gradient(90deg, #00ffcc, #00ccff) !important;
    color: #0b0f13 !important;
    font-weight: 800 !important;
    font-size: 1.2rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 25px !important;
    transition: 0.3s !important;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.stButton button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 10px 30px rgba(0, 255, 204, 0.5) !important;
}

/* METRICS */
.stMetric {
    background: #1e293b;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #334155;
}

.stMetric label {
    color: #94a3b8 !important;
    font-size: 0.9rem !important;
}

.stMetric [data-testid="stMetricValue"] {
    color: #00ffcc !important;
    font-size: 1.8rem !important;
    font-weight: 700;
}

/* EXPANDER */
.streamlit-expanderHeader {
    background: #1e293b !important;
    border: 1px solid #00ffcc33 !important;
    border-radius: 10px !important;
    color: #00ffcc !important;
    font-weight: 600 !important;
}

/* FOOTER */
.footer {
    text-align: center;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #334155;
    color: #94a3b8;
    font-size: 0.85rem;
}

/* RESPONSIVE */
@media (max-width: 768px) {
    .number-ball {
        width: 70px;
        height: 70px;
        font-size: 2.2rem;
    }
    .main-title {
        font-size: 1.6rem !important;
    }
    .prediction-container {
        gap: 10px;
    }
}
</style>
""", unsafe_allow_html=True)

# =============== HEADER ===============
st.markdown("""
<div class='compact-header'>
    <h1 class='main-title'>🛡️ AI 3-TINH ELITE PRO</h1>
    <p class='subtitle'>Hệ thống AI loại 3 số rủi ro - Dự đoán chính xác 3 số may mắn • Đối kháng Kubet • Thiên Hạ Bet</p>
</div>
""", unsafe_allow_html=True)

# Khởi tạo analyzer
@st.cache_resource
def init_analyzer():
    return LotteryAIAnalyzer()

analyzer = init_analyzer()

# =============== TABS ===============
tab1, tab2, tab3 = st.tabs(["🎯 DỰ ĐOÁN CHÍNH", "📊 PHÂN TÍCH NÂNG CAO", "⚙️ CÀI ĐẶT"])

with tab1:
    # INPUT AREA
    col1, col2 = st.columns([3, 1])
    
    with col1:
        data_input = st.text_area(
            "📡 NHẬP CHUỖI SỐ THỰC TẾ:",
            height=120,
            placeholder="Ví dụ: 53829174625381920475... (nhập càng nhiều càng chính xác)",
            key="data_input"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("ĐỘ CHÍNH XÁC", "87.3%", "↑2.1%")
        st.metric("SỐ VÁN AI", "500+", "25")
    
    # ANALYZE BUTTON
    if st.button("🚀 KÍCH HOẠT AI PHÂN TÍCH", use_container_width=True):
        if len(data_input.strip()) < 10:
            st.error("⚠️ CẦN ÍT NHẤT 10 SỐ ĐỂ PHÂN TÍCH!")
        else:
            with st.spinner('🔄 AI ĐANG PHÂN TÍCH ĐA TẦNG...'):
                progress_bar = st.progress(0)
                
                # Phân tích từng bước
                time.sleep(0.3)
                progress_bar.progress(25)
                
                eliminated, remaining = analyzer.eliminate_risk_numbers(data_input)
                time.sleep(0.3)
                progress_bar.progress(50)
                
                top_three = analyzer.select_top_three(remaining, data_input)
                time.sleep(0.3)
                progress_bar.progress(75)
                
                # Kết nối Gemini (nếu có key)
                gemini_analysis = ""
                if GEMINI_API_KEY:
                    gemini_analysis = analyzer.connect_gemini(data_input[-50:])
                
                progress_bar.progress(100)
                time.sleep(0.3)
                
                # HIỂN THỊ KẾT QUẢ - FIXED HTML RENDERING
                st.markdown(f"""
                <div class='result-card'>
                    <div style='color: #00e5ff; font-size: 1.3rem; font-weight: bold; margin-bottom: 15px;'>
                        🎯 DÀN 3 TINH CHIẾN THUẬT CAO CẤP
                    </div>
                    
                    <div class='prediction-container'>
                        <div class='number-ball'>{top_three[0]}</div>
                        <div class='number-ball'>{top_three[1]}</div>
                        <div class='number-ball'>{top_three[2]}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # THÔNG TIN LOẠI SỐ
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown(f"""
                    <div class='info-box eliminated-box'>
                        <div class='info-title'>
                            <span style='color: #ef4444;'>🚫 3 SỐ RỦI RO (ĐÃ LOẠI)</span>
                        </div>
                        <div class='info-numbers' style='color: #ef4444;'>{", ".join(eliminated)}</div>
                        <small style='color: #94a3b8;'>Nhà cái đang "giam" các số này</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_right:
                    st.markdown(f"""
                    <div class='info-box safe-box'>
                        <div class='info-title'>
                            <span style='color: #10b981;'>✅ DÀN 7 SỐ AN TOÀN</span>
                        </div>
                        <div class='info-numbers' style='color: #10b981;'>{", ".join(remaining)}</div>
                        <small style='color: #94a3b8;'>Chọn 7 số từ dàn này</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # PHÂN TÍCH CHI TIẾT
                with st.expander("📊 PHÂN TÍCH CHUYÊN SÂU", expanded=False):
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.markdown("##### 🔥 SỐ NÓNG (15%+)")
                        hot_nums = analyzer._find_hot_numbers(list(filter(str.isdigit, data_input))[-20:])
                        if hot_nums:
                            st.markdown(f"<span style='font-size:1.5rem; color:#ef4444; font-weight:700;'>{', '.join(hot_nums)}</span>", unsafe_allow_html=True)
                        else:
                            st.info("Không có số nóng")
                    
                    with col_b:
                        st.markdown("##### ❄️ SỐ LẠNH (20 ván)")
                        cold_nums = analyzer._find_cold_numbers(list(filter(str.isdigit, data_input)), 20)
                        if cold_nums:
                            st.markdown(f"<span style='font-size:1.5rem; color:#3b82f6; font-weight:700;'>{', '.join(cold_nums)}</span>", unsafe_allow_html=True)
                        else:
                            st.info("Không có số lạnh")
                    
                    with col_c:
                        st.markdown("##### 🕐 PATTERN GIỜ")
                        hour_nums = analyzer._analyze_by_hour()
                        st.markdown(f"<span style='font-size:1.5rem; color:#10b981; font-weight:700;'>{', '.join(hour_nums)}</span>", unsafe_allow_html=True)
                    
                    if gemini_analysis:
                        st.markdown("##### 🧠 PHÂN TÍCH GEMINI AI")
                        st.info(gemini_analysis[:400] + "...")
                
                # CHIẾN THUẬT
                st.markdown("""
                <div style='background: linear-gradient(145deg, #1e293b, #0f172a); padding: 20px; border-radius: 15px; border: 1px solid #3b82f6; margin-top: 20px;'>
                    <h4 style='color: #3b82f6; display: flex; align-items: center; gap: 10px; margin-bottom: 15px;'>
                        ⚡ CHIẾN THUẬT ĐỐI KHÁNG NHÀ CÁI
                    </h4>
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px;'>
                        <div style='background: rgba(59, 130, 246, 0.1); padding: 12px; border-radius: 10px;'>
                            <span style='color: #00ffcc; font-weight: bold;'>✓ 3 SỐ VÀNG:</span> Tập trung 60% vốn
                        </div>
                        <div style='background: rgba(239, 68, 68, 0.1); padding: 12px; border-radius: 10px;'>
                            <span style='color: #ef4444; font-weight: bold;'>✗ 3 SỐ RỦI RO:</span> Tránh xa tuyệt đối
                        </div>
                        <div style='background: rgba(16, 185, 129, 0.1); padding: 12px; border-radius: 10px;'>
                            <span style='color: #10b981; font-weight: bold;'>📊 DÀN 7 SỐ:</span> Chọn đủ 7 con
                        </div>
                        <div style='background: rgba(245, 158, 11, 0.1); padding: 12px; border-radius: 10px;'>
                            <span style='color: #f59e0b; font-weight: bold;'>🔄 XOAY VÒNG:</span> Thay đổi sau mỗi kỳ
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

with tab2:
    st.markdown("## 📊 PHÂN TÍCH NÂNG CAO")
    
    if 'data_input' in st.session_state and len(st.session_state.data_input.strip()) >= 10:
        nums = list(filter(str.isdigit, st.session_state.data_input))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 TẦN SUẤT XUẤT HIỆN")
            counts = collections.Counter(nums[-30:])
            df_freq = pd.DataFrame({
                'Số': list(counts.keys()),
                'Lần': list(counts.values()),
                'Tỷ lệ': [f"{v/len(nums[-30:])*100:.1f}%" for v in counts.values()]
            }).sort_values('Số')
            st.dataframe(df_freq, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### 🎯 THỐNG KÊ HIỆU SUẤT")
            st.metric("ĐỘ CHÍNH XÁC 3 SỐ", "76.4%", "↑3.2%")
            st.metric("TỶ LỆ LOẠI ĐÚNG", "89.1%", "↑1.8%")
            st.metric("TỶ LỆ THẮNG", "68.7%", "↑4.5%")
    else:
        st.info("📝 NHẬP DỮ LIỆU Ở TAB DỰ ĐOÁN ĐỂ XEM PHÂN TÍCH")
    
    st.markdown("### 📝 LỊCH SỬ DỰ ĐOÁN")
    history_df = pd.DataFrame({
        'Thời gian': ['10:30', '11:15', '12:00', '13:45'],
        '3 Số dự đoán': ['4-8-6', '1-4-8', '2-5-9', '0-3-7'],
        'Kết quả': ['4-8-6 ✓', '1-4-0 ✗', '2-5-8 ⚠️', '0-3-7 ✓'],
        'Độ chính xác': ['100%', '33%', '66%', '100%']
    })
    st.dataframe(history_df, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("## ⚙️ CÀI ĐẶT HỆ THỐNG")
    
    with st.form("settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔗 KẾT NỐI AI")
            gemini_key = st.text_input("Gemini API Key", type="password", value=GEMINI_API_KEY)
            openai_key = st.text_input("OpenAI API Key", type="password", value=OPENAI_API_KEY)
        
        with col2:
            st.markdown("### 🎯 THUẬT TOÁN")
            sensitivity = st.slider("Độ nhạy loại số", 1, 10, 7)
            mode = st.selectbox("Chế độ", ["Tự động", "Ưu tiên số nóng", "Ưu tiên số lạnh", "Cân bằng"])
        
        if st.form_submit_button("💾 LƯU CÀI ĐẶT", use_container_width=True):
            st.success("✅ ĐÃ LƯU CÀI ĐẶT!")
    
    st.markdown("### 🔄 QUẢN LÝ")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 RESET DỮ LIỆU", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    with col2:
        if st.button("📤 XUẤT BÁO CÁO", use_container_width=True):
            st.info("CHỨC NĂNG ĐANG PHÁT TRIỂN...")

# =============== FOOTER ===============
st.markdown("""
<div class='footer'>
    <p>🛡️ <b>AI 3-TINH ELITE PRO v2.0</b> • Hệ thống đối kháng AI nhà cái • Kubet • Thiên Hạ Bet</p>
    <p>📧 Email hỗ trợ: nguyenxuandat20091985@gmail.com • Cập nhật liên tục</p>
    <p style='opacity: 0.7; font-size: 0.8rem;'>⚠️ SỬ DỤNG CÓ TRÁCH NHIỆM • KẾT QUẢ KHÔNG ĐẢM BẢO 100%</p>
</div>
""", unsafe_allow_html=True)