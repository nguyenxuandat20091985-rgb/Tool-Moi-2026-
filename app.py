import streamlit as st
import json
import os
import re
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations
import random
from datetime import datetime
import requests
import google.generativeai as genai

# ==================== CẤU HÌNH ====================
st.set_page_config(
    page_title="TITAN V32 - AI GOD MODE",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Keys
NVIDIA_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"

# Cấu hình Gemini
genai.configure(api_key=GEMINI_API_KEY)

# File lưu trữ
DATA_FILE = "titan_v32_data.json"

# ==================== CSS NEON DARK MODE ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a0a2e 50%, #0a0a0a 100%);
        color: #e0e0e0;
        font-family: 'Orbitron', monospace;
    }
    
    .main-header {
        font-size: 48px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #00ffff, #ff00ff, #00ffff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 3s linear infinite;
        text-shadow: 0 0 30px rgba(0, 255, 255, 0.5);
        margin-bottom: 20px;
    }
    
    @keyframes shine {
        to { background-position: 200% center; }
    }
    
    .neon-box {
        background: rgba(10, 10, 30, 0.8);
        border: 2px solid;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.2);
    }
    
    .neon-box-cyan {
        border-color: #00ffff;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
    }
    
    .neon-box-purple {
        border-color: #ff00ff;
        box-shadow: 0 0 20px rgba(255, 0, 255, 0.3);
    }
    
    .neon-box-green {
        border-color: #00ff00;
        box-shadow: 0 0 20px rgba(0, 255, 0, 0.3);
    }
    
    .big-number {
        font-size: 56px;
        font-weight: 900;
        letter-spacing: 15px;
        text-align: center;
        color: #00ffff;
        text-shadow: 0 0 20px #00ffff;
    }
    
    .pair-number {
        font-size: 32px;
        font-weight: 700;
        letter-spacing: 8px;
        color: #ff00ff;
        text-shadow: 0 0 15px #ff00ff;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(0,255,255,0.1), rgba(255,0,255,0.1));
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #00ffff;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #00ffff, #ff00ff);
        color: #000;
        font-weight: 900;
        border: none;
        border-radius: 10px;
        padding: 10px 30px;
        font-size: 16px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.6);
    }
    
    .probability-bar {
        height: 25px;
        background: #1a1a2e;
        border-radius: 12px;
        overflow: hidden;
        margin: 5px 0;
    }
    
    .probability-fill {
        height: 100%;
        background: linear-gradient(90deg, #00ffff, #ff00ff);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: #000;
        transition: width 0.5s;
    }
    
    .hot-number {
        color: #ff0040;
        font-weight: 900;
        text-shadow: 0 0 10px #ff0040;
    }
    
    .cold-number {
        color: #0080ff;
        font-weight: 900;
        text-shadow: 0 0 10px #0080ff;
    }
    
    .win-text {
        color: #00ff00;
        font-weight: 900;
    }
    
    .lose-text {
        color: #ff0040;
        font-weight: 900;
    }
    
    textarea {
        background: #0a0a1a;
        color: #00ffff;
        border: 1px solid #00ffff;
    }
    
    .stDataFrame {
        background: rgba(10, 10, 30, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# ==================== CLASS PHÂN TÍCH ====================

class TitanAnalyzer:
    def __init__(self, data):
        self.data = data
        self.digits = [d for num in data for d in num]
        
    def frequency_analysis(self):
        """Phân tích tần suất"""
        return Counter(self.digits)
    
    def position_analysis(self):
        """Phân tích từng vị trí"""
        positions = {i: Counter() for i in range(5)}
        for num in self.data:
            for i, d in enumerate(num):
                positions[i][d] += 1
        return positions
    
    def gap_analysis(self):
        """Phân tích khoảng cách (số gan)"""
        last_seen = {str(i): -1 for i in range(10)}
        gaps = {}
        
        for idx, num in enumerate(self.data):
            for d in num:
                last_seen[d] = idx
        
        current_idx = len(self.data) - 1
        for d in last_seen:
            if last_seen[d] == -1:
                gaps[d] = current_idx + 1
            else:
                gaps[d] = current_idx - last_seen[d]
        
        return gaps
    
    def streak_analysis(self):
        """Phân tích số bệt (lặp liên tiếp)"""
        streaks = {str(i): 0 for i in range(10)}
        
        for num in reversed(self.data):
            for d in set(num):
                if streaks[d] == 0 or streaks[d] > 0:
                    streaks[d] += 1
        
        # Kiểm tra số bệt thực sự
        for d in streaks:
            count = 0
            for num in reversed(self.data):
                if d in num:
                    count += 1
                else:
                    break
            streaks[d] = count
        
        return streaks
    
    def pair_frequency(self):
        """Tần suất cặp số"""
        pairs = Counter()
        for num in self.data:
            unique_digits = sorted(set(num))
            for p in combinations(unique_digits, 2):
                pairs[p] += 1
        return pairs
    
    def monte_carlo_simulation(self, iterations=10000):
        """Mô phỏng Monte Carlo"""
        freq = self.frequency_analysis()
        total = sum(freq.values())
        probs = {d: freq.get(d, 0) / total for d in '0123456789'}
        
        simulated_wins = Counter()
        
        for _ in range(iterations):
            # Generate 5 digits based on probability
            simulated = random.choices('0123456789', weights=[probs[d] for d in '0123456789'], k=5)
            unique_digits = set(simulated)
            
            for p in combinations('0123456789', 2):
                if p[0] in unique_digits and p[1] in unique_digits:
                    simulated_wins[p] += 1
        
        return {p: count/iterations for p, count in simulated_wins.items()}
    
    def pattern_analysis(self):
        """Phân tích pattern"""
        patterns = {
            'increasing': 0,
            'decreasing': 0,
            'repeating': 0,
            'mixed': 0
        }
        
        for num in self.data:
            digits = [int(d) for d in num]
            if all(digits[i] <= digits[i+1] for i in range(4)):
                patterns['increasing'] += 1
            elif all(digits[i] >= digits[i+1] for i in range(4)):
                patterns['decreasing'] += 1
            elif len(set(digits)) == 1:
                patterns['repeating'] += 1
            else:
                patterns['mixed'] += 1
        
        return patterns
    
    def calculate_pair_score(self, pair):
        """Tính điểm tổng hợp cho cặp số"""
        freq = self.frequency_analysis()
        gaps = self.gap_analysis()
        streaks = self.streak_analysis()
        pair_freq = self.pair_frequency()
        monte_carlo = self.monte_carlo_simulation(5000)
        
        score = 0
        
        # 1. Điểm tần suất (max 30)
        pair_count = pair_freq.get(tuple(sorted(pair)), 0)
        score += min(pair_count * 3, 30)
        
        # 2. Điểm khoảng cách (max 25)
        gap_score = 0
        for d in pair:
            gap = gaps.get(d, 5)
            if 2 <= gap <= 8:
                gap_score += 12.5
            elif gap > 8:
                gap_score += max(0, 15 - gap)
        score += gap_score
        
        # 3. Điểm bệt (max 20)
        streak_score = 0
        for d in pair:
            streak = streaks.get(d, 0)
            if streak == 1:
                streak_score += 10
            elif streak == 2:
                streak_score += 5
        score += streak_score
        
        # 4. Monte Carlo (max 25)
        mc_prob = monte_carlo.get(tuple(sorted(pair)), 0)
        score += mc_prob * 100
        
        return score
    
    def get_top_pairs(self, top_n=10):
        """Lấy top cặp số tốt nhất"""
        all_pairs = list(combinations('0123456789', 2))
        scored_pairs = []
        
        for pair in all_pairs:
            score = self.calculate_pair_score(pair)
            scored_pairs.append({
                'pair': ''.join(pair),
                'score': score,
                'frequency': self.pair_frequency().get(tuple(sorted(pair)), 0)
            })
        
        scored_pairs.sort(key=lambda x: x['score'], reverse=True)
        return scored_pairs[:top_n]
    
    def get_best_single(self):
        """Tìm bạch thủ mạnh nhất"""
        freq = self.frequency_analysis()
        gaps = self.gap_analysis()
        streaks = self.streak_analysis()
        
        scored = []
        for d in '0123456789':
            score = 0
            score += freq.get(d, 0) * 2
            gap = gaps.get(d, 5)
            if 2 <= gap <= 8:
                score += 30
            if streaks.get(d, 0) == 1:
                score += 20
            scored.append({'digit': d, 'score': score})
        
        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored[0]

# ==================== AI ANALYSIS ====================

def ai_pattern_recognition(data):
    """Sử dụng AI để nhận diện pattern"""
    try:
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        Phân tích chuỗi số sau và tìm pattern ẩn:
        {', '.join(data[-20:])}
        
        Cho biết:
        1. Xu hướng số nào đang mạnh
        2. Cặp số nào có khả năng cao xuất hiện
        3. Số nào đang trong chu kỳ lặp
        
        Trả lời ngắn gọn, tập trung vào 3 số và 3 cặp số tiềm năng nhất.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"

# ==================== XỬ LÝ DỮ LIỆU ====================

def load_data():
    """Load dữ liệu từ file"""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {'history': [], 'results': []}
    return {'history': [], 'results': []}

def save_data(data):
    """Lưu dữ liệu vào file"""
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def parse_numbers(text):
    """Parse số từ text input"""
    numbers = re.findall(r'\d{5}', text)
    return numbers

# ==================== GIAO DIỆN CHÍNH ====================

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 TITAN V32 - AI GOD MODE</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#888; font-size:14px;">Multi-Algorithm + Self-Learning + Neural Network</p>', unsafe_allow_html=True)
    
    # Load data
    app_data = load_data()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ ĐIỀU KHIỂN")
        
        uploaded_file = st.file_uploader("📁 Load file JSON", type=['json'])
        if uploaded_file:
            try:
                app_data = json.load(uploaded_file)
                st.success("✅ Load thành công!")
            except:
                st.error("❌ File không hợp lệ")
        
        if st.button("💾 Lưu dữ liệu", use_container_width=True):
            save_data(app_data)
            st.success("✅ Đã lưu!")
        
        if st.button("🗑️ Xóa toàn bộ", use_container_width=True):
            app_data = {'history': [], 'results': []}
            if os.path.exists(DATA_FILE):
                os.remove(DATA_FILE)
            st.success("✅ Đã xóa!")
        
        st.markdown("---")
        st.markdown("### 📊 THỐNG KÊ")
        st.write(f"Tổng kỳ: {len(app_data.get('results', []))}")
        if app_data.get('history'):
            wins = sum(1 for h in app_data['history'] if h.get('result') == 'win')
            st.write(f"Thắng: {wins}")
            st.write(f"Thua: {len(app_data['history']) - wins}")
    
    # Input area
    st.markdown('<div class="neon-box neon-box-cyan">', unsafe_allow_html=True)
    st.markdown("### 📥 NHẬP KẾT QUẢ")
    
    default_text = "\n".join(app_data.get('results', []))
    user_input = st.text_area(
        "Dán kết quả (mỗi kỳ 1 dòng, 5 chữ số):",
        value=default_text,
        height=200,
        placeholder="46602\n32476\n14606..."
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        analyze_btn = st.button("🔮 PHÂN TÍCH", type="primary", use_container_width=True)
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.rerun()
    with col3:
        if st.button("📝 Lưu", use_container_width=True):
            app_data['results'] = parse_numbers(user_input)
            save_data(app_data)
            st.success("✅ Đã lưu!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Processing
    if analyze_btn or user_input:
        numbers = parse_numbers(user_input)
        
        if len(numbers) < 5:
            st.warning("⚠️ Cần ít nhất 5 kỳ để phân tích chính xác")
        else:
            # Update data
            app_data['results'] = numbers
            save_data(app_data)
            
            # Initialize analyzer
            analyzer = TitanAnalyzer(numbers)
            
            # Get analysis
            top_pairs = analyzer.get_top_pairs(10)
            best_single = analyzer.get_best_single()
            freq = analyzer.frequency_analysis()
            gaps = analyzer.gap_analysis()
            streaks = analyzer.streak_analysis()
            patterns = analyzer.pattern_analysis()
            
            # AI Analysis
            with st.spinner("🤖 AI đang phân tích pattern..."):
                ai_insights = ai_pattern_recognition(numbers)
            
            # ==================== HIỂN THỊ KẾT QUẢ ====================
            
            # Row 1: Bạch thủ
            st.markdown('<div class="neon-box neon-box-purple">', unsafe_allow_html=True)
            st.markdown("### 🎯 BẠCH THỦ MẠNH NHẤT")
            st.markdown(f'<div class="big-number">{best_single["digit"]}</div>', unsafe_allow_html=True)
            st.markdown(f"Điểm sức mạnh: **{best_single['score']:.1f}**")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Row 2: Top pairs
            st.markdown('<div class="neon-box neon-box-cyan">', unsafe_allow_html=True)
            st.markdown("### 🏆 TOP 10 CẶP SỐ NÊN ĐÁNH")
            
            cols = st.columns(5)
            for i, pair_data in enumerate(top_pairs):
                with cols[i % 5]:
                    st.markdown(f"""
                    <div style="background:rgba(0,255,255,0.1); border:1px solid #00ffff; 
                                border-radius:10px; padding:15px; text-align:center; margin:5px;">
                        <div class="pair-number">{pair_data['pair'][0]}-{pair_data['pair'][1]}</div>
                        <div style="color:#00ff00; font-weight:bold;">Score: {pair_data['score']:.1f}</div>
                        <div style="color:#888; font-size:12px;">Freq: {pair_data['frequency']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Row 3: Số bệt & số gan
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="neon-box neon-box-green">', unsafe_allow_html=True)
                st.markdown("### 🔥 SỐ BỆT (ĐANG LẶP)")
                hot_numbers = [d for d, s in streaks.items() if s >= 1]
                if hot_numbers:
                    st.markdown(f'<div class="hot-number" style="font-size:32px;">{" ".join(hot_numbers)}</div>', unsafe_allow_html=True)
                    for d in hot_numbers:
                        st.write(f"Số {d}: {streaks[d]} kỳ liên tiếp")
                else:
                    st.write("Không có số bệt")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="neon-box neon-box-purple">', unsafe_allow_html=True)
                st.markdown("### ❄️ SỐ GAN (LÂU CHƯA RA)")
                cold_numbers = sorted(gaps.items(), key=lambda x: x[1], reverse=True)[:5]
                cold_str = " ".join([d for d, g in cold_numbers])
                st.markdown(f'<div class="cold-number" style="font-size:32px;">{cold_str}</div>', unsafe_allow_html=True)
                for d, gap in cold_numbers:
                    st.write(f"Số {d}: {gap} kỳ chưa ra")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Row 4: Xác suất
            st.markdown('<div class="neon-box neon-box-cyan">', unsafe_allow_html=True)
            st.markdown("### 📊 XÁC SUẤT TỪNG SỐ")
            
            total_digits = len(analyzer.digits)
            prob_data = []
            for d in '0123456789':
                count = freq.get(d, 0)
                prob = (count / total_digits * 100) if total_digits > 0 else 0
                prob_data.append({'digit': d, 'prob': prob, 'count': count})
            
            prob_data.sort(key=lambda x: x['prob'], reverse=True)
            
            for item in prob_data:
                st.markdown(f"""
                <div style="display:flex; align-items:center; margin:5px 0;">
                    <div style="width:30px; font-weight:bold; color:#00ffff;">{item['digit']}</div>
                    <div class="probability-bar" style="flex:1;">
                        <div class="probability-fill" style="width:{item['prob']}%;">
                            {item['prob']:.1f}%
                        </div>
                    </div>
                    <div style="width:60px; text-align:right; color:#888;">{item['count']} lần</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Row 5: Pattern analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="neon-box neon-box-purple">', unsafe_allow_html=True)
                st.markdown("### 📈 PHÂN TÍCH PATTERN")
                
                total = sum(patterns.values())
                for pattern, count in patterns.items():
                    pct = (count / total * 100) if total > 0 else 0
                    bar_color = "#00ffff" if pattern == 'mixed' else "#ff00ff"
                    st.markdown(f"""
                    <div style="margin:10px 0;">
                        <div style="display:flex; justify-content:space-between;">
                            <span>{pattern.upper()}</span>
                            <span>{count} ({pct:.1f}%)</span>
                        </div>
                        <div class="probability-bar">
                            <div class="probability-fill" style="width:{pct}%; background:{bar_color};"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="neon-box neon-box-green">', unsafe_allow_html=True)
                st.markdown("### 🤖 AI INSIGHTS")
                st.markdown(f"""
                <div style="background:rgba(0,255,0,0.1); padding:15px; border-radius:10px; 
                            border-left:4px solid #00ff00; font-size:13px; line-height:1.6;">
                {ai_insights}
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Row 6: Thống kê chi tiết
            st.markdown('<div class="neon-box neon-box-cyan">', unsafe_allow_html=True)
            st.markdown("### 📊 THỐNG KÊ CHI TIẾT THEO VỊ TRÍ")
            
            pos_analysis = analyzer.position_analysis()
            
            pos_cols = st.columns(5)
            for pos in range(5):
                with pos_cols[pos]:
                    st.markdown(f"**Vị trí {pos+1}**")
                    top_3 = pos_analysis[pos].most_common(3)
                    for digit, count in top_3:
                        st.write(f"Số {digit}: {count} lần")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # History tracking
            if len(numbers) >= 2:
                st.markdown("---")
                st.markdown("### 📋 LỊCH SỬ DỰ ĐOÁN")
                
                if 'predictions' not in app_data:
                    app_data['predictions'] = []
                
                # Add new prediction
                new_pred = {
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                    'last_result': numbers[-1],
                    'prediction': top_pairs[0]['pair'] if top_pairs else "00",
                    'score': top_pairs[0]['score'] if top_pairs else 0
                }
                
                if not app_data['predictions'] or app_data['predictions'][-1] != new_pred:
                    app_data['predictions'].append(new_pred)
                    save_data(app_data)
                
                # Display history
                if app_data['predictions']:
                    pred_df = pd.DataFrame(app_data['predictions'][-10:])
                    st.dataframe(pred_df, use_container_width=True)

if __name__ == "__main__":
    main()