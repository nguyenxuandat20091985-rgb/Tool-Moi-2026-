import streamlit as st
import json
import os
import re
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import google.generativeai as genai
import requests
import math

# === CẤU HÌNH API ===
NVIDIA_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
DB_FILE = "titan_v32_data.json"

# Cấu hình Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

# === CSS NEON ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .stApp {
        background: linear-gradient(180deg, #0a0a0a 0%, #1a0a2e 50%, #0a0a0a 100%);
        color: #00ffff;
        font-family: 'Orbitron', monospace;
    }
    
    .main-header {
        font-size: 32px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #00ffff, #ff00ff, #00ffff);
        background-size: 200% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 3s ease infinite;
        text-shadow: 0 0 30px rgba(0, 255, 255, 0.5);
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .neon-box {
        background: rgba(0, 20, 40, 0.8);
        border: 2px solid #00ffff;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.3), inset 0 0 20px rgba(0, 255, 255, 0.1);
    }
    
    .neon-box-purple {
        border-color: #ff00ff;
        box-shadow: 0 0 20px rgba(255, 0, 255, 0.3), inset 0 0 20px rgba(255, 0, 255, 0.1);
    }
    
    .big-number {
        font-size: 48px;
        font-weight: 900;
        letter-spacing: 12px;
        text-align: center;
        color: #00ffff;
        text-shadow: 0 0 20px #00ffff, 0 0 40px #00ffff;
    }
    
    .metric-row {
        display: flex;
        justify-content: space-around;
        margin: 15px 0;
    }
    
    .metric-cell {
        background: rgba(0, 20, 40, 0.6);
        border: 1px solid #00ffff;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        flex: 1;
        margin: 0 5px;
    }
    
    .tag {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 10px;
        font-weight: bold;
        margin: 3px;
        background: linear-gradient(90deg, #00ffff, #ff00ff);
        color: #000;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, rgba(0,20,40,0.9), rgba(20,0,40,0.9));
        border: 2px solid #ff00ff;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 0 30px rgba(255, 0, 255, 0.4);
    }
    
    .history-win { color: #00ff00; font-weight: 900; text-shadow: 0 0 10px #00ff00; }
    .history-lose { color: #ff0040; font-weight: 900; text-shadow: 0 0 10px #ff0040; }
    
    .stTextInput > div > div > input {
        background: rgba(0, 20, 40, 0.8);
        color: #00ffff;
        border: 1px solid #00ffff;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #00ffff, #ff00ff);
        color: #000;
        font-weight: 900;
        border: none;
        border-radius: 10px;
        padding: 10px 30px;
    }
    
    .progress-bar {
        height: 20px;
        background: rgba(0, 20, 40, 0.6);
        border-radius: 10px;
        overflow: hidden;
        margin: 10px 0;
        border: 1px solid #00ffff;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #00ffff, #ff00ff);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: bold;
        color: #000;
    }
    
    .warning-box {
        background: rgba(40, 20, 0, 0.8);
        border: 2px solid #ffaa00;
        color: #ffaa00;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        animation: pulse-warning 2s infinite;
    }
    
    @keyframes pulse-warning {
        0%, 100% { box-shadow: 0 0 10px rgba(255, 170, 0, 0.5); }
        50% { box-shadow: 0 0 30px rgba(255, 170, 0, 0.8); }
    }
</style>
""", unsafe_allow_html=True)

# === QUẢN LÝ DỮ LIỆU ===

def load_data():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"results": [], "predictions": [], "model_weights": {}}

def save_data(data):
    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_nums(text):
    clean_text = re.sub(r'\s+', '', text)
    return [n for n in re.findall(r"\d{5}", clean_text) if n]

# === THUẬT TOÁN NÂNG CAO ===

class AdvancedAnalyzer:
    def __init__(self, db, model_weights=None):
        self.db = db
        self.weights = model_weights or self._default_weights()
        
    def _default_weights(self):
        return {
            'frequency': 1.0,
            'markov': 1.0,
            'cycle': 1.0,
            'pattern': 1.0,
            'gap': 0.8
        }
    
    def calculate_frequency_score(self, pair):
        """Tính điểm tần suất với trọng số thời gian"""
        if not self.db:
            return 0
        
        score = 0
        for idx, num in enumerate(reversed(self.db[-50:])):
            if set(pair).issubset(set(num)):
                weight = math.exp(-idx / 20)  # Số gần đây quan trọng hơn
                score += weight * 10
        return score
    
    def markov_chain_predict(self):
        """Markov Chain dự đoán chuyển tiếp"""
        if len(self.db) < 10:
            return {}
        
        transitions = defaultdict(lambda: defaultdict(int))
        
        for i in range(len(self.db) - 1):
            current = self.db[i]
            next_num = self.db[i + 1]
            
            for d in current:
                for nd in next_num:
                    transitions[d][nd] += 1
        
        # Tính xác suất chuyển tiếp
        probs = {}
        for digit in transitions:
            total = sum(transitions[digit].values())
            if total > 0:
                probs[digit] = {
                    next_d: count/total 
                    for next_d, count in transitions[digit].items()
                }
        
        return probs
    
    def detect_cycles(self):
        """Phát hiện chu kỳ lặp"""
        cycles = {}
        if len(self.db) < 20:
            return cycles
        
        for digit in '0123456789':
            positions = [i for i, num in enumerate(self.db) if digit in num]
            if len(positions) >= 3:
                gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                if gaps:
                    avg_gap = sum(gaps) / len(gaps)
                    cycles[digit] = {
                        'avg_gap': avg_gap,
                        'last_pos': len(self.db) - positions[-1] - 1,
                        'regularity': 1 - (np.std(gaps) / avg_gap if avg_gap > 0 else 1)
                    }
        
        return cycles
    
    def calculate_gap_score(self, pair):
        """Tính điểm dựa trên độ gan"""
        score = 0
        for d in pair:
            last_seen = -1
            for i in range(len(self.db) - 1, -1, -1):
                if d in self.db[i]:
                    last_seen = len(self.db) - 1 - i
                    break
            
            if last_seen == -1:
                score += 50  # Chưa bao giờ ra
            elif 5 <= last_seen <= 15:
                score += 30  # Gan vừa phải
            elif last_seen > 20:
                score += 40  # Gan lâu
        
        return score
    
    def detect_patterns(self):
        """Phát hiện pattern phức tạp"""
        patterns = {
            'consecutive': [],
            'mirror': [],
            'sum_pattern': []
        }
        
        if len(self.db) < 5:
            return patterns
        
        # Pattern số liên tiếp
        for num in self.db[-20:]:
            digits = sorted([int(d) for d in num])
            for i in range(len(digits) - 1):
                if digits[i+1] - digits[i] == 1:
                    patterns['consecutive'].append((str(digits[i]), str(digits[i+1])))
        
        # Pattern tổng
        sums = [sum(int(d) for d in num) for num in self.db[-20:]]
        if len(sums) >= 3:
            patterns['sum_trend'] = sums[-1] - sums[-2]
        
        return patterns
    
    def calculate_pair_score(self, pair):
        """Tính điểm tổng hợp cho cặp số"""
        scores = {}
        
        # 1. Frequency score
        scores['frequency'] = self.calculate_frequency_score(pair)
        
        # 2. Markov score
        markov_probs = self.markov_chain_predict()
        markov_score = 0
        for d in pair:
            if d in markov_probs:
                for other in pair:
                    if other != d and other in markov_probs[d]:
                        markov_score += markov_probs[d][other] * 50
        scores['markov'] = markov_score
        
        # 3. Cycle score
        cycles = self.detect_cycles()
        cycle_score = 0
        for d in pair:
            if d in cycles:
                cycle_data = cycles[d]
                if cycle_data['last_pos'] >= cycle_data['avg_gap'] * 0.8:
                    cycle_score += 30 * cycle_data['regularity']
        scores['cycle'] = cycle_score
        
        # 4. Gap score
        scores['gap'] = self.calculate_gap_score(pair)
        
        # 5. Pattern score
        patterns = self.detect_patterns()
        pattern_score = 0
        if (pair[0], pair[1]) in patterns['consecutive'] or (pair[1], pair[0]) in patterns['consecutive']:
            pattern_score += 40
        scores['pattern'] = pattern_score
        
        # Tổng hợp với trọng số
        total_score = sum(scores[k] * self.weights.get(k, 1.0) for k in scores)
        
        return total_score, scores
    
    def get_top_pairs(self, n=5):
        """Lấy top N cặp số tốt nhất"""
        all_pairs = []
        for p in combinations('0123456789', 2):
            pair = ''.join(p)
            score, details = self.calculate_pair_score(pair)
            all_pairs.append({
                'pair': pair,
                'score': score,
                'details': details
            })
        
        all_pairs.sort(key=lambda x: x['score'], reverse=True)
        return all_pairs[:n]

# === AI ANALYSIS ===

def gemini_analysis(db, top_pairs):
    """Dùng Gemini phân tích pattern ẩn"""
    if len(db) < 10:
        return None
    
    try:
        recent_str = "\n".join(db[-20:])
        top_str = ", ".join([p['pair'] for p in top_pairs[:3]])
        
        prompt = f"""
        Phân tích dãy số 5 chữ số sau và dự đoán 2 số có khả năng xuất hiện cao nhất:
        
        {recent_str}
        
        Top pairs thống kê: {top_str}
        
        Cho biết:
        1. Pattern đặc biệt nào bạn thấy?
        2. 2 số nào nên đánh?
        3. Lý do cụ thể?
        
        Trả lời ngắn gọn, tập trung vào kết quả.
        """
        
        response = gemini_model.generate_content(prompt)
        return response.text
    except:
        return None

def nvidia_inference(db, features):
    """Gửi đến NVIDIA API để tối ưu dự đoán"""
    # Placeholder cho NVIDIA API call
    # Thực tế sẽ gọi API endpoint của NVIDIA
    return {"optimized_score": features.get('score', 0) * 1.1}

# === GIAO DIỆN ===

def main():
    st.markdown('<h1 class="main-header">🧬 TITAN V32 - NEURAL PREDICTOR</h1>', unsafe_allow_html=True)
    
    data = load_data()
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📥 NHẬP KẾT QUẢ",
        "🎯 BẠCH THỦ",
        "📊 BỆT & GAN",
        "🤖 AI ANALYSIS",
        "📋 LỊCH SỬ"
    ])
    
    with tab1:
        st.markdown('<div class="neon-box">', unsafe_allow_html=True)
        st.markdown("### Nhập kết quả mới (5 chữ số)")
        
        input_text = st.text_area(
            "Dán kết quả (mỗi dòng 1 số, kỳ mới nhất ở dưới):",
            height=150,
            placeholder="12345\n67890\n..."
        )
        
        if st.button("💾 LƯU & PHÂN TÍCH", type="primary"):
            nums = get_nums(input_text)
            if nums:
                data["results"].extend(nums)
                save_data(data)
                st.success(f"✅ Đã lưu {len(nums)} kết quả!")
                st.rerun()
            else:
                st.error("❌ Không tìm thấy số hợp lệ!")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Hiển thị kết quả gần nhất
        if data["results"]:
            st.markdown('<div class="neon-box neon-box-purple">', unsafe_allow_html=True)
            st.markdown("### 📊 Kết quả gần nhất")
            recent = data["results"][-10:][::-1]
            for i, num in enumerate(recent):
                st.markdown(f"**Kỳ {len(data['results'])-i}:** `{num}`")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        if len(data["results"]) < 15:
            st.warning("⚠️ Cần ít nhất 15 kỳ để phân tích chính xác!")
        else:
            analyzer = AdvancedAnalyzer(data["results"], data.get("model_weights"))
            top_pairs = analyzer.get_top_pairs(5)
            
            st.markdown('<div class="neon-box">', unsafe_allow_html=True)
            st.markdown("### 🎯 BẠCH THỦ 2 SỐ")
            
            # Top 1
            best = top_pairs[0]
            st.markdown(f"""
            <div class="prediction-card">
                <div style="font-size:14px; color:#888;">CẶP VIP #1</div>
                <div class="big-number">{best['pair'][0]} - {best['pair'][1]}</div>
                <div style="margin-top:10px; font-size:18px; color:#00ff00;">
                    Điểm: {best['score']:.1f}
                </div>
                <div style="margin-top:10px; font-size:12px; color:#888;">
                    Frequency: {best['details']['frequency']:.1f} | 
                    Markov: {best['details']['markov']:.1f} | 
                    Cycle: {best['details']['cycle']:.1f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Top 2-5
            st.markdown("### 📊 Top 5 Cặp Số")
            for i, p in enumerate(top_pairs[1:5], 2):
                st.markdown(f"""
                <div class="neon-box" style="padding:15px; margin:5px 0;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:24px; font-weight:900; color:#ff00ff;">
                            #{i} {p['pair'][0]} - {p['pair'][1]}
                        </span>
                        <span style="font-size:18px; color:#00ffff;">
                            {p['score']:.1f} pts
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        if len(data["results"]) < 10:
            st.warning("⚠️ Cần thêm dữ liệu!")
        else:
            st.markdown('<div class="neon-box">', unsafe_allow_html=True)
            st.markdown("### 🔥 SỐ BỆT (Đang lặp)")
            
            # Tính streak
            streaks = {}
            for digit in '0123456789':
                count = 0
                for num in reversed(data["results"]):
                    if digit in num:
                        count += 1
                    else:
                        break
                if count > 0:
                    streaks[digit] = count
            
            if streaks:
                max_streak = max(streaks.items(), key=lambda x: x[1])
                st.markdown(f"""
                <div class="prediction-card">
                    <div style="font-size:20px;">Số bệt mạnh nhất:</div>
                    <div class="big-number" style="color:#ff00ff;">{max_streak[0]}</div>
                    <div style="font-size:24px; color:#00ff00;">
                        {max_streak[1]} kỳ liên tiếp
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### ⏳ SỐ GAN (Lâu chưa ra)")
            
            gaps = {}
            for digit in '0123456789':
                last_seen = -1
                for i in range(len(data["results"]) - 1, -1, -1):
                    if digit in data["results"][i]:
                        last_seen = len(data["results"]) - 1 - i
                        break
                if last_seen != -1:
                    gaps[digit] = last_seen
            
            if gaps:
                sorted_gaps = sorted(gaps.items(), key=lambda x: x[1], reverse=True)
                st.markdown(f"""
                <div class="neon-box neon-box-purple">
                    <div style="font-size:20px;">Số gan cao nhất:</div>
                    <div class="big-number">{sorted_gaps[0][0]}</div>
                    <div style="font-size:24px; color:#ffaa00;">
                        {sorted_gaps[0][1]} kỳ chưa về
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Top 5 gan
                st.markdown("**Top 5 số gan:**")
                for digit, gap in sorted_gaps[:5]:
                    bar_width = min(gap * 5, 100)
                    st.markdown(f"""
                    <div style="margin:5px 0;">
                        <div style="display:flex; justify-content:space-between;">
                            <span>Số {digit}</span>
                            <span>{gap} kỳ</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width:{bar_width}%; background:linear-gradient(90deg, #ff0000, #ffff00);">
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        if len(data["results"]) < 15:
            st.warning("⚠️ Cần ít nhất 15 kỳ!")
        else:
            st.markdown('<div class="neon-box">', unsafe_allow_html=True)
            st.markdown("### 🤖 AI PHÂN TÍCH CHUYÊN SÂU")
            
            analyzer = AdvancedAnalyzer(data["results"])
            top_pairs = analyzer.get_top_pairs(3)
            
            if st.button(" PHÂN TÍCH VỚI AI", type="primary"):
                with st.spinner("Đang phân tích pattern ẩn..."):
                    ai_result = gemini_analysis(data["results"], top_pairs)
                    
                    if ai_result:
                        st.markdown(f"""
                        <div class="neon-box neon-box-purple" style="background:rgba(40,0,40,0.8); border-color:#ff00ff;">
                            <h3>💡 Insight từ AI:</h3>
                            <p style="white-space:pre-wrap; line-height:1.6;">{ai_result}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("❌ Không thể kết nối AI!")
            
            # Hiển thị phân tích thống kê
            st.markdown("### 📊 Phân Tích Thống Kê")
            
            cycles = analyzer.detect_cycles()
            if cycles:
                st.markdown("**Chu kỳ các số:**")
                for digit, info in list(cycles.items())[:5]:
                    st.markdown(f"- Số {digit}: Chu kỳ {info['avg_gap']:.1f} kỳ, Độ đều: {info['regularity']:.2f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        st.markdown('<div class="neon-box">', unsafe_allow_html=True)
        st.markdown("### 📋 LỊCH SỬ DỰ ĐOÁN")
        
        if data["predictions"]:
            df = pd.DataFrame(data["predictions"][-20:][::-1])
            
            def color_result(val):
                if val == "WIN":
                    return "color: #00ff00; font-weight: 900; text-shadow: 0 0 10px #00ff00;"
                return "color: #ff0040; font-weight: 900; text-shadow: 0 0 10px #ff0040;"
            
            st.dataframe(
                df.style.applymap(color_result, subset=['Kết Quả']),
                use_container_width=True,
                hide_index=True
            )
            
            # Tính win rate
            wins = sum(1 for p in data["predictions"] if p.get("Kết Quả") == "WIN")
            total = len(data["predictions"])
            rate = (wins / total * 100) if total > 0 else 0
            
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-cell">
                    <div style="font-size:12px; color:#888;">TỔNG SỐ</div>
                    <div style="font-size:24px; font-weight:900; color:#00ffff;">{total}</div>
                </div>
                <div class="metric-cell">
                    <div style="font-size:12px; color:#888;">THẮNG</div>
                    <div style="font-size:24px; font-weight:900; color:#00ff00;">{wins}</div>
                </div>
                <div class="metric-cell">
                    <div style="font-size:12px; color:#888;">TỶ LỆ</div>
                    <div style="font-size:24px; font-weight:900; color:{'#00ff00' if rate >= 40 else '#ff0040'};">
                        {rate:.1f}%
                    </div>
                </div>
            </div>
            
            <div class="progress-bar">
                <div class="progress-fill" style="width:{rate}%; background:linear-gradient(90deg, {'#00ff00' if rate >= 40 else '#ff0040'}, #ffff00);">
                    {rate:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("📭 Chưa có lịch sử dự đoán")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Auto-predict sidebar
    if len(data["results"]) >= 15:
        st.sidebar.markdown('<div class="neon-box">', unsafe_allow_html=True)
        st.sidebar.markdown("### ⚡ DỰ ĐOÁN TỰ ĐỘNG")
        
        analyzer = AdvancedAnalyzer(data["results"])
        top_pairs = analyzer.get_top_pairs(3)
        
        st.sidebar.markdown(f"""
        <div style="text-align:center; padding:15px; background:rgba(0,40,40,0.6); border-radius:10px; border:1px solid #00ffff;">
            <div style="font-size:12px; color:#888;">CẶP MẠNH NHẤT</div>
            <div style="font-size:36px; font-weight:900; color:#00ffff; margin:10px 0;">
                {top_pairs[0]['pair'][0]} - {top_pairs[0]['pair'][1]}
            </div>
            <div style="font-size:18px; color:#00ff00;">
                Score: {top_pairs[0]['score']:.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence meter
        confidence = min(95, max(30, top_pairs[0]['score'] / 2))
        st.sidebar.markdown(f"""
        <div style="margin-top:15px;">
            <div style="font-size:12px; color:#888; text-align:center;">ĐỘ TIN CẬY</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width:{confidence}%; background:linear-gradient(90deg, #ff0000, #ffff00, #00ff00);">
                    {confidence:.0f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.sidebar.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()