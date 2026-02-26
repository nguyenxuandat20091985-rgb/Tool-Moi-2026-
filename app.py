import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import random

# ================= CẤU HÌNH =================
API_KEY = "AIzaSyB5PRp04XlMHKl3oGfCRbsKXjlTA-CZifc"
DB_FILE = "titan_digit_analysis.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except:
        return None

neural_engine = setup_neural()

def load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try:
                data = json.load(f)
                return data if isinstance(data, list) else []
            except:
                return []
    return []

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data[-5000:], f)

# Khởi tạo session
if "history" not in st.session_state:
    st.session_state.history = load_db()
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "accuracy" not in st.session_state:
    st.session_state.accuracy = {"total": 0, "correct": 0, "history": []}

# ================= DIGIT ANALYSIS ENGINE =================

class DigitAnalysisEngine:
    def __init__(self, history):
        self.history = history
        self.digits = [str(i) for i in range(10)]
        self.total_ky = len(history)
        
    def get_frequency_score(self, digit, last_n):
        """Tính tần suất xuất hiện"""
        if self.total_ky < last_n:
            return 0
        
        recent = ''.join(self.history[-last_n:])
        count = recent.count(digit)
        return count / last_n  # Tỷ lệ 0-1
    
    def get_gan_score(self, digit):
        """Tính điểm gan - số ngày không về"""
        if self.total_ky == 0:
            return 1.0
        
        # Tìm kỳ gần nhất xuất hiện
        for i, num in enumerate(reversed(self.history)):
            if digit in num:
                days_absent = i
                # Normalize 0-1 (càng lâu không về điểm càng cao)
                return min(1.0, days_absent / 50)  # Max 50 kỳ
        
        return 1.0  # Chưa bao giờ về
    
    def get_momentum_score(self, digit):
        """Đà xu hướng - xu hướng gần đây"""
        if self.total_ky < 20:
            return 0.5
        
        # So sánh 10 kỳ gần vs 10 kỳ trước đó
        last_10 = ''.join(self.history[-10:]).count(digit)
        prev_10 = ''.join(self.history[-20:-10]).count(digit)
        
        if prev_10 == 0:
            return 1.0 if last_10 > 0 else 0.5
        
        momentum = (last_10 - prev_10) / prev_10
        # Normalize về 0-1
        return max(0, min(1, 0.5 + momentum/2))
    
    def get_pattern_score(self, digit):
        """Điểm pattern - phát hiện các quy luật"""
        if self.total_ky < 10:
            return 0.5
        
        score = 0
        total_patterns = 0
        
        # 1. Bet pattern - số về liên tiếp
        last_5 = self.history[-5:]
        bet_count = sum(1 for num in last_5 if digit in num)
        score += bet_count / 5
        total_patterns += 1
        
        # 2. Jump pattern - số nhảy cách quãng
        if self.total_ky >= 10:
            jump_count = 0
            for i in range(1, 6):
                if i < len(self.history) and digit in self.history[-i] and digit in self.history[-i-2]:
                    jump_count += 1
            score += jump_count / 5
            total_patterns += 1
        
        # 3. Mirror pattern - số đối xứng
        mirror_digit = str(9 - int(digit))
        mirror_count = ''.join(self.history[-10:]).count(mirror_digit)
        score += mirror_count / 10
        total_patterns += 1
        
        # 4. Repeat pattern - số lặp theo chu kỳ
        repeat_score = 0
        for period in [3, 5, 7]:
            if self.total_ky > period:
                pattern_found = True
                for i in range(1, 4):
                    if not (digit in self.history[-i] and digit in self.history[-i-period]):
                        pattern_found = False
                        break
                if pattern_found:
                    repeat_score += 0.25
        score += repeat_score
        total_patterns += 1
        
        return score / total_patterns if total_patterns > 0 else 0.5
    
    def get_entropy_score(self, digit):
        """Điểm entropy - độ hỗn loạn"""
        if self.total_ky < 30:
            return 0.5
        
        all_digits = ''.join(self.history[-30:])
        total = len(all_digits)
        if total == 0:
            return 0.5
        
        # Tần suất thực tế
        p_real = all_digits.count(digit) / total
        
        # Tần suất kỳ vọng (ngẫu nhiên đều = 0.1)
        p_expected = 0.1
        
        # Entropy càng cao càng hỗn loạn
        if p_real == 0:
            return 0.8  # Số hiếm -> dễ xuất hiện
        
        entropy_ratio = p_real / p_expected
        # Normalize: càng gần 1 càng tốt
        return max(0, min(1, 1 - abs(1 - entropy_ratio) / 2))
    
    def get_volatility_score(self, digit):
        """Điểm biến động - tốc độ thay đổi"""
        if self.total_ky < 20:
            return 0.5
        
        # Xem xét 20 kỳ gần
        appearances = []
        for i, num in enumerate(self.history[-20:]):
            appearances.append(1 if digit in num else 0)
        
        if len(appearances) < 2:
            return 0.5
        
        # Tính số lần thay đổi (0->1 hoặc 1->0)
        changes = sum(1 for i in range(1, len(appearances)) if appearances[i] != appearances[i-1])
        
        # Biến động càng cao điểm càng thấp (khó đoán)
        volatility = changes / (len(appearances) - 1)
        return 1 - volatility  # Ổn định cao = điểm cao
    
    def get_markov_score(self, digit):
        """Điểm Markov - xác suất chuyển tiếp"""
        if self.total_ky < 10:
            return 0.5
        
        # Tạo chuỗi các số cuối cùng của mỗi kỳ
        last_digits = [num[-1] for num in self.history[-50:]]
        
        # Tìm các lần digit xuất hiện
        markov_prob = 0
        count = 0
        
        for i in range(len(last_digits) - 1):
            if last_digits[i] == digit:
                count += 1
                if last_digits[i + 1] == digit:
                    markov_prob += 1
        
        if count > 0:
            return markov_prob / count
        return 0.5
    
    def get_bayesian_score(self, digit):
        """Điểm Bayes - xác suất hậu nghiệm"""
        if self.total_ky < 20:
            return 0.5
        
        # Prior: tần suất tổng thể
        all_digits = ''.join(self.history)
        prior = all_digits.count(digit) / len(all_digits) if all_digits else 0.1
        
        # Likelihood: tần suất gần đây
        recent = ''.join(self.history[-20:])
        likelihood = recent.count(digit) / len(recent) if recent else 0.1
        
        # Posterior đơn giản
        posterior = (prior + likelihood) / 2
        return posterior
    
    def get_montecarlo_score(self, digit, simulations=10000):
        """Điểm Monte Carlo - mô phỏng 10000 lần"""
        if self.total_ky < 20:
            return 0.5
        
        # Lấy phân phối từ lịch sử
        all_digits = ''.join(self.history[-100:])
        digit_freq = [all_digits.count(str(d)) for d in range(10)]
        total = sum(digit_freq)
        
        if total == 0:
            return 0.5
        
        probs = [f/total for f in digit_freq]
        
        # Mô phỏng
        success = 0
        for _ in range(simulations):
            if random.random() < probs[int(digit)]:
                success += 1
        
        return success / simulations
    
    def get_neural_score(self, digit):
        """Điểm từ AI - sử dụng Gemini"""
        try:
            if neural_engine and self.total_ky > 20:
                prompt = f"""
                Dựa trên lịch sử 100 kỳ gần: {self.history[-100:]}
                Hãy dự đoán khả năng xuất hiện của số {digit} trong kỳ tới (0-1):
                Chỉ trả về số thập phân, không giải thích.
                """
                response = neural_engine.generate_content(prompt)
                try:
                    score = float(response.text.strip())
                    return max(0, min(1, score))
                except:
                    return 0.5
        except:
            pass
        return 0.5
    
    def calculate_digit_power_index(self, digit):
        """Tính DIGIT POWER INDEX theo công thức"""
        
        # Lấy các điểm thành phần
        frequency_score = (
            self.get_frequency_score(digit, 10) * 0.4 +
            self.get_frequency_score(digit, 30) * 0.3 +
            self.get_frequency_score(digit, 100) * 0.3
        )
        
        gan_score = self.get_gan_score(digit)
        momentum_score = self.get_momentum_score(digit)
        pattern_score = self.get_pattern_score(digit)
        entropy_score = self.get_entropy_score(digit)
        volatility_score = self.get_volatility_score(digit)
        markov_score = self.get_markov_score(digit)
        bayesian_score = self.get_bayesian_score(digit)
        montecarlo_score = self.get_montecarlo_score(digit)
        neural_score = self.get_neural_score(digit)
        
        # Tổng hợp theo công thức
        dpi = (
            0.15 * frequency_score +
            0.10 * gan_score +
            0.10 * momentum_score +
            0.15 * pattern_score +
            0.10 * entropy_score +
            0.10 * volatility_score +
            0.10 * markov_score +
            0.05 * bayesian_score +
            0.10 * montecarlo_score +
            0.05 * neural_score
        )
        
        return dpi
    
    def get_top_digits(self, n=7):
        """Lấy top n digits có DPI cao nhất"""
        dpis = []
        for digit in self.digits:
            dpi = self.calculate_digit_power_index(digit)
            dpis.append((digit, dpi))
        
        # Sắp xếp theo DPI giảm dần
        dpis.sort(key=lambda x: x[1], reverse=True)
        return [d[0] for d in dpis[:n]]

# ================= GIAO DIỆN =================
st.set_page_config(page_title="DIGIT ANALYSIS ENGINE", layout="wide")

st.markdown("""
<style>
    .stApp { background: #0a0f1f; }
    .main-title {
        text-align: center;
        color: #00ff88;
        font-size: 50px;
        font-weight: 900;
        text-shadow: 0 0 20px #00ff88;
        margin-bottom: 10px;
    }
    .sub-title {
        text-align: center;
        color: #8899bb;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .pred-card {
        background: linear-gradient(145deg, #1a1f35, #0f1425);
        border: 2px solid #00ff88;
        border-radius: 30px;
        padding: 40px;
        margin: 20px 0;
        box-shadow: 0 20px 40px rgba(0,255,136,0.2);
    }
    .main-number {
        font-size: 120px;
        font-weight: 900;
        color: #ffaa00;
        text-align: center;
        letter-spacing: 20px;
        text-shadow: 0 0 30px #ffaa00;
        background: #1a1f35;
        padding: 20px;
        border-radius: 20px;
        border: 1px solid #ffaa00;
    }
    .support-number {
        font-size: 80px;
        font-weight: 700;
        color: #00aaff;
        text-align: center;
        letter-spacing: 15px;
        text-shadow: 0 0 20px #00aaff;
    }
    .status-bar {
        background: #1f2a44;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        border-left: 5px solid #00ff88;
        margin: 20px 0;
    }
    .dpi-meter {
        background: #161b28;
        padding: 15px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .dpi-value {
        float: right;
        color: #00ff88;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🎯 DIGIT ANALYSIS ENGINE</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>10 thuật toán - 10000 mô phỏng - Độ chính xác tối đa</div>", unsafe_allow_html=True)

# Layout
col_input, col_info = st.columns([2, 1])

with col_input:
    raw_input = st.text_area("📊 NHẬP KẾT QUẢ MỚI:", height=100,
                            placeholder="Nhập số 5D (VD: 12345 67890)...")
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        analyze_btn = st.button("🔮 PHÂN TÍCH DPI", use_container_width=True)
    with col_btn2:
        reset_btn = st.button("🔄 RESET", use_container_width=True)
    with col_btn3:
        if st.session_state.last_prediction:
            if st.button("✅ XÁC NHẬN ĐÚNG", use_container_width=True):
                st.session_state.accuracy["total"] += 1
                st.session_state.accuracy["correct"] += 1
                st.rerun()

with col_info:
    st.metric("📈 TỔNG SỐ KỲ", len(st.session_state.history))
    
    if st.session_state.accuracy["total"] > 0:
        acc = (st.session_state.accuracy["correct"] / st.session_state.accuracy["total"]) * 100
        st.metric("🎯 ĐỘ CHÍNH XÁC", f"{acc:.1f}%")
    
    if st.session_state.history:
        st.write("**🔢 5 KỲ GẦN NHẤT:**")
        for i, num in enumerate(st.session_state.history[-5:], 1):
            st.code(f"Kỳ {i}: {num}", language="text")

# Xử lý reset
if reset_btn:
    st.session_state.history = []
    st.session_state.last_prediction = None
    st.session_state.accuracy = {"total": 0, "correct": 0, "history": []}
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    st.success("✅ Đã reset toàn bộ dữ liệu")
    st.rerun()

# Xử lý phân tích
if analyze_btn and raw_input:
    numbers = re.findall(r'\b\d{5}\b', raw_input)
    
    if numbers:
        # Thêm vào lịch sử
        for num in numbers:
            if num not in st.session_state.history:
                st.session_state.history.append(num)
        
        save_db(st.session_state.history)
        
        # Khởi tạo engine
        engine = DigitAnalysisEngine(st.session_state.history)
        
        # Lấy top digits
        top_digits = engine.get_top_digits(7)
        
        # Tạo dự đoán
        main = ''.join(top_digits[:3])
        support = ''.join(top_digits[3:7])
        
        # Tính điểm trung bình
        avg_dpi = np.mean([engine.calculate_digit_power_index(d) for d in top_digits])
        
        if avg_dpi > 0.7:
            status = "🚀 CẦU MẠNH - TỶ LỆ CAO"
        elif avg_dpi > 0.5:
            status = "📊 CẦU ỔN ĐỊNH"
        else:
            status = "⚠️ CẦU YẾU - THẬN TRỌNG"
        
        # Lưu dự đoán
        st.session_state.last_prediction = {
            'main': main,
            'support': support,
            'status': status,
            'time': datetime.now().strftime("%H:%M:%S"),
            'top_digits': top_digits,
            'dpi_values': {d: engine.calculate_digit_power_index(d) for d in top_digits}
        }
        
        st.rerun()

# Hiển thị kết quả
if st.session_state.last_prediction:
    pred = st.session_state.last_prediction
    
    st.markdown(f"<div class='status-bar'>{pred['status']}</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='pred-card'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("### 🔴 3 SỐ CHỦ LỰC")
        st.markdown(f"<div class='main-number'>{pred['main']}</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🔵 4 SỐ LÓT")
        st.markdown(f"<div class='support-number'>{pred['support']}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Hiển thị DPI của từng số
    st.markdown("### 📊 DIGIT POWER INDEX")
    for digit in pred['top_digits']:
        dpi = pred['dpi_values'][digit]
        color = "#00ff88" if dpi > 0.7 else "#ffaa00" if dpi > 0.5 else "#ff5555"
        st.markdown(f"""
            <div class='dpi-meter'>
                <span style='font-size:20px;'>Số {digit}</span>
                <span class='dpi-value' style='color:{color};'>{dpi:.3f}</span>
                <div style='background:#2a2f45; height:5px; border-radius:5px; margin-top:5px;'>
                    <div style='background:{color}; width:{dpi*100}%; height:5px; border-radius:5px;'></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"<p style='text-align:right; color:#666; margin-top:20px;'>⏰ {pred['time']}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Công thức
    with st.expander("📐 XEM CÔNG THỨC DPI"):
        st.markdown("""
        ```
        DIGIT POWER INDEX =
        0.15*frequency_score +
        0.10*gan_score +
        0.10*momentum_score +
        0.15*pattern_score +
        0.10*entropy_score +
        0.10*volatility_score +
        0.10*markov_score +
        0.05*bayesian_score +
        0.10*montecarlo_score +
        0.05*neural_score
        ```
        """)

# Hiển thị phân tích
if st.session_state.history:
    with st.expander("📈 PHÂN TÍCH CHUYÊN SÂU"):
        engine = DigitAnalysisEngine(st.session_state.history)
        
        # Tạo bảng DPI cho tất cả số
        dpi_data = []
        for d in range(10):
            dpi = engine.calculate_digit_power_index(str(d))
            dpi_data.append({
                'Số': d,
                'DPI': f"{dpi:.3f}",
                'Tần suất 10': f"{engine.get_frequency_score(str(d), 10):.2f}",
                'Gan': f"{engine.get_gan_score(str(d)):.2f}",
                'Pattern': f"{engine.get_pattern_score(str(d)):.2f}"
            })
        
        df = pd.DataFrame(dpi_data)
        st.dataframe(df, use_container_width=True)