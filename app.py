import streamlit as st
import json
import os
import re
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import random

# === CẤU HÌNH ===
NVIDIA_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
DB_FILE = "titan_v52_master_data.json"
LUCKY_OX = [0, 2, 5, 6, 7, 8]

# === CSS CAO CẤP ===
st.set_page_config(page_title="TITAN V52 - MASTER ANALYST", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0a0a0a 100%);
        color: #FFD700;
        font-family: 'Orbitron', monospace;
    }
    
    .main-header {
        font-size: 42px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #FFD700, #FF00FF, #00FFFF, #FFD700);
        background-size: 300% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 3s ease infinite;
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .metric-box {
        background: linear-gradient(135deg, #16213e, #0f3460);
        border: 2px solid #00fff5;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 0 20px rgba(0, 255, 245, 0.3);
        margin: 10px 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 2px solid;
        box-shadow: 0 5px 20px rgba(0,0,0,0.5);
    }
    
    .big-number {
        font-size: 48px;
        font-weight: 900;
        letter-spacing: 10px;
        text-align: center;
    }
    
    .tag {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 10px;
        font-weight: bold;
        margin: 3px;
    }
    
    .tag-hot { background: #ff0040; color: white; }
    .tag-cold { background: #0066ff; color: white; }
    .tag-gan { background: #9900ff; color: white; }
    .tag-gold { background: #FFD700; color: black; }
    .tag-safe { background: #00ff40; color: black; }
    
    .history-win { color: #00ff40; font-weight: 900; }
    .history-lose { color: #ff0040; font-weight: 900; }
    
    .progress-bar {
        height: 20px;
        background: #1a1a1a;
        border-radius: 10px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff0000, #ffff00, #00ff00);
        transition: width 0.5s;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: bold;
        color: #000;
    }
    
    .method-box {
        background: #0f0f1a;
        border-left: 4px solid #FFD700;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 10px 10px 0;
    }
    
    .final-pick {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: #000;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        font-size: 56px;
        font-weight: 900;
        letter-spacing: 15px;
        margin: 20px 0;
        box-shadow: 0 0 40px rgba(255, 215, 0, 0.6);
        animation: pulse-gold 2s infinite;
    }
    
    @keyframes pulse-gold {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
</style>
""", unsafe_allow_html=True)

# === DATABASE MANAGER ===

class DatabaseManager:
    def __init__(self, filename):
        self.filename = filename
        self.data = self.load()
    
    def load(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"history": [], "predictions": [], "stats": {}}
    
    def save(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def add_prediction(self, prediction):
        self.data["predictions"].append({
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction
        })
        self.save()
    
    def add_result(self, actual_number, predicted_pair, won):
        self.data["history"].append({
            "timestamp": datetime.now().isoformat(),
            "actual": actual_number,
            "predicted": predicted_pair,
            "won": won
        })
        self.save()

# === MULTI-ALGORITHM ENGINE ===

class MultiAlgorithmEngine:
    def __init__(self, db_manager):
        self.db = db_manager
    
    def analyze_frequency(self, nums, window=30):
        """Phân tích tần suất"""
        recent = "".join(nums[-window:])
        counter = Counter(recent)
        return {d: counter.get(d, 0) for d in "0123456789"}
    
    def analyze_gaps(self, nums):
        """Phân tích số gan (khoảng cách)"""
        gaps = {}
        for d in "0123456789":
            gap = 0
            for num in reversed(nums):
                if d in num:
                    break
                gap += 1
            gaps[d] = gap
        return gaps
    
    def analyze_positions(self, nums):
        """Phân tích vị trí"""
        positions = {i: Counter() for i in range(5)}
        for num in nums[-30:]:
            for i, d in enumerate(num):
                positions[i][d] += 1
        return positions
    
    def markov_chain(self, nums, order=2):
        """Markov Chain prediction"""
        if len(nums) < 20:
            return {}
        
        transitions = defaultdict(lambda: defaultdict(int))
        
        for i in range(len(nums) - order):
            state = "".join(nums[i:i+order])
            next_num = nums[i+order]
            for d in next_num:
                transitions[state][d] += 1
        
        # Predict from last state
        if len(nums) >= order:
            last_state = "".join(nums[-order:])
            if last_state in transitions:
                total = sum(transitions[last_state].values())
                return {d: c/total for d, c in transitions[last_state].items()}
        
        return {}
    
    def detect_patterns(self, nums):
        """Phát hiện pattern"""
        patterns = {
            'consecutive': [],
            'symmetry': [],
            'fibonacci': []
        }
        
        # Check consecutive patterns
        for i in range(len(nums) - 2):
            if nums[i][0:2] == nums[i+1][0:2]:
                patterns['consecutive'].append(nums[i+1])
        
        # Check symmetry (abccba pattern)
        for num in nums[-10:]:
            if num == num[::-1]:
                patterns['symmetry'].append(num)
        
        # Fibonacci digits
        fib_digits = {'0', '1', '2', '3', '5', '8'}
        for num in nums[-10:]:
            if all(d in fib_digits for d in num):
                patterns['fibonacci'].append(num)
        
        return patterns
    
    def analyze_hot_cold(self, nums, window=20):
        """Phân tích số nóng/lạnh"""
        freq = self.analyze_frequency(nums, window)
        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        
        hot = [d for d, _ in sorted_freq[:3]]
        cold = [d for d, _ in sorted_freq[-3:]]
        
        return hot, cold
    
    def calculate_pair_scores(self, nums):
        """Tính điểm cho tất cả cặp số"""
        pairs = {}
        
        freq = self.analyze_frequency(nums)
        gaps = self.analyze_gaps(nums)
        positions = self.analyze_positions(nums)
        markov = self.markov_chain(nums)
        hot, cold = self.analyze_hot_cold(nums)
        patterns = self.detect_patterns(nums)
        
        for p in combinations("0123456789", 2):
            pair = "".join(p)
            score = 0
            reasons = []
            
            # 1. Frequency score
            pair_freq = freq.get(p[0], 0) + freq.get(p[1], 0)
            if 15 <= pair_freq <= 35:
                score += 30
                reasons.append("FREQ_OPTIMAL")
            elif pair_freq > 40:
                score -= 20
                reasons.append("FREQ_HIGH")
            
            # 2. Gap score (số gan)
            pair_gap = gaps.get(p[0], 0) + gaps.get(p[1], 0)
            if 5 <= pair_gap <= 15:
                score += 40
                reasons.append("GAP_GOLDEN")
            elif pair_gap > 20:
                score += 20
                reasons.append("GAP_LONG")
            
            # 3. Position score
            pos_score = 0
            for pos in positions:
                if positions[pos].get(p[0], 0) > 0 or positions[pos].get(p[1], 0) > 0:
                    pos_score += 5
            score += pos_score
            if pos_score > 15:
                reasons.append("POS_STRONG")
            
            # 4. Markov score
            markov_score = 0
            for d in p:
                if d in markov:
                    markov_score += markov[d] * 50
            score += markov_score
            if markov_score > 20:
                reasons.append("MARKOV_STRONG")
            
            # 5. Hot/Cold balance
            hot_count = sum(1 for d in p if d in hot)
            cold_count = sum(1 for d in p if d in cold)
            if hot_count == 1 and cold_count == 1:
                score += 25
                reasons.append("BALANCED")
            elif cold_count == 2:
                score += 15
                reasons.append("COLD_DUE")
            
            # 6. Lucky ox
            if any(int(d) in LUCKY_OX for d in p):
                score += 10
                reasons.append("LUCKY_OX")
            
            # 7. Avoid obvious (anti-crowd)
            if p in ['01', '12', '23', '34', '45', '56', '67', '78', '89']:
                score -= 15
                reasons.append("AVOID_OBVIOUS")
            
            pairs[pair] = {
                'score': score,
                'reasons': reasons,
                'freq': pair_freq,
                'gap': pair_gap,
                'markov': markov_score
            }
        
        return pairs
    
    def generate_methods(self, nums):
        """Tạo 10+ phương pháp dự đoán"""
        methods = []
        pairs = self.calculate_pair_scores(nums)
        freq = self.analyze_frequency(nums)
        gaps = self.analyze_gaps(nums)
        hot, cold = self.analyze_hot_cold(nums)
        
        # Method 1: Highest frequency
        m1 = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:2]
        methods.append({
            'name': 'TẦN SUẤT CAO',
            'pair': "".join([d for d, _ in m1]),
            'score': sum([d[1] for d in m1]),
            'logic': '2 số xuất hiện nhiều nhất'
        })
        
        # Method 2: Longest gap
        m2 = sorted(gaps.items(), key=lambda x: x[1], reverse=True)[:2]
        methods.append({
            'name': 'SỐ GAN LÂU',
            'pair': "".join([d for d, _ in m2]),
            'score': sum([d[1] for d in m2]),
            'logic': '2 số lâu chưa về nhất'
        })
        
        # Method 3: Hot + Cold
        if hot and cold:
            methods.append({
                'name': 'NÓNG + LẠNH',
                'pair': hot[0] + cold[0],
                'score': 50,
                'logic': 'Cân bằng số nóng và lạnh'
            })
        
        # Method 4: Best composite score
        best_pair = max(pairs.items(), key=lambda x: x[1]['score'])
        methods.append({
            'name': 'TỔNG HỢP TỐT NHẤT',
            'pair': best_pair[0],
            'score': best_pair[1]['score'],
            'logic': f"Điểm tổng hợp: {best_pair[1]['reasons']}"
        })
        
        # Method 5: Position based
        positions = self.analyze_positions(nums)
        pos_hot = []
        for pos in positions:
            if positions[pos]:
                pos_hot.append(positions[pos].most_common(1)[0][0])
        if len(pos_hot) >= 2:
            methods.append({
                'name': 'VỊ TRÍ MẠNH',
                'pair': pos_hot[0] + pos_hot[1],
                'score': 45,
                'logic': 'Số mạnh theo vị trí'
            })
        
        # Method 6: Markov prediction
        markov = self.markov_chain(nums)
        if markov:
            m6 = sorted(markov.items(), key=lambda x: x[1], reverse=True)[:2]
            methods.append({
                'name': 'MARKOV CHAIN',
                'pair': "".join([d for d, _ in m6]),
                'score': sum([d[1] for d in m6]) * 100,
                'logic': 'Xác suất chuyển tiếp'
            })
        
        # Method 7: Fibonacci pattern
        fib_nums = ['0', '1', '2', '3', '5', '8']
        fib_freq = {d: freq.get(d, 0) for d in fib_nums}
        m7 = sorted(fib_freq.items(), key=lambda x: x[1], reverse=True)[:2]
        methods.append({
            'name': 'FIBONACCI',
            'pair': "".join([d for d, _ in m7]),
            'score': sum([d[1] for d in m7]),
            'logic': 'Số Fibonacci'
        })
        
        # Method 8: Lucky Ox special
        ox_freq = {d: freq.get(d, 0) for d in [str(d) for d in LUCKY_OX]}
        m8 = sorted(ox_freq.items(), key=lambda x: x[1], reverse=True)[:2]
        methods.append({
            'name': 'TUỔI SỬU 1985',
            'pair': "".join([d for d, _ in m8]),
            'score': sum([d[1] for d in m8]) + 20,
            'logic': 'Số may mắn tuổi Sửu'
        })
        
        # Method 9: Anti-pattern (avoid crowd)
        sorted_pairs = sorted(pairs.items(), key=lambda x: x[1]['score'], reverse=True)
        for pair, data in sorted_pairs:
            if 'AVOID_OBVIOUS' not in data['reasons']:
                methods.append({
                    'name': 'TRÁNH ĐÁM ĐÔNG',
                    'pair': pair,
                    'score': data['score'] + 15,
                    'logic': 'Số không obvious'
                })
                break
        
        # Method 10: Gap golden zone
        golden_pairs = []
        for pair, data in pairs.items():
            if 'GAP_GOLDEN' in data['reasons']:
                golden_pairs.append((pair, data['score']))
        if golden_pairs:
            golden_pairs.sort(key=lambda x: x[1], reverse=True)
            methods.append({
                'name': 'VÙNG GAN VÀNG',
                'pair': golden_pairs[0][0],
                'score': golden_pairs[0][1],
                'logic': 'Gan 5-15 kỳ (vàng)'
            })
        
        return methods
    
    def three_round_analysis(self, nums):
        """Phân tích 3 vòng"""
        results = {
            'round1': {},
            'round2': {},
            'round3': {}
        }
        
        # Round 1: Initial analysis
        methods = self.generate_methods(nums)
        results['round1']['methods'] = methods
        results['round1']['top3'] = sorted(methods, key=lambda x: x['score'], reverse=True)[:3]
        
        # Round 2: Find weaknesses
        weaknesses = []
        for method in methods:
            # Check if this pair lost recently
            recent_losses = 0
            for hist in self.db.data['history'][-10:]:
                if hist['predicted'] == method['pair'] and not hist['won']:
                    recent_losses += 1
            if recent_losses >= 2:
                weaknesses.append({
                    'method': method['name'],
                    'pair': method['pair'],
                    'reason': f'Thua {recent_losses} kỳ gần'
                })
                method['score'] -= 30  # Penalty
        
        results['round2']['weaknesses'] = weaknesses
        results['round2']['adjusted_methods'] = methods
        
        # Round 3: Final optimization
        final_methods = [m for m in methods if m['score'] > 0]
        final_methods.sort(key=lambda x: x['score'], reverse=True)
        
        # Add diversity check
        top_numbers = Counter()
        for m in final_methods[:5]:
            for d in m['pair']:
                top_numbers[d] += 1
        
        results['round3']['final_methods'] = final_methods
        results['round3']['top3'] = final_methods[:3]
        results['round3']['final_pick'] = final_methods[0] if final_methods else None
        results['round3']['number_distribution'] = dict(top_numbers)
        
        return results

# === XỬ LÝ DỮ LIỆU ===

def get_nums(text):
    clean_text = re.sub(r'\s+', '', text)
    nums = [n for n in re.findall(r"\d{5}", clean_text) if n]
    seen = set()
    return [n for n in nums if not (n in seen or seen.add(n))]

# === GIAO DIỆN ===

st.markdown('<h1 class="main-header">🎯 TITAN V52 - MASTER ANALYST</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#888;">10+ Thuật Toán | 3 Vòng Phân Tích | Tự Học & Cải Tiến</p>', unsafe_allow_html=True)

# Initialize
db_manager = DatabaseManager(DB_FILE)
engine = MultiAlgorithmEngine(db_manager)

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_area("📥 Dữ liệu (kỳ mới nhất ở dưới):", height=150, 
                          placeholder="46602\n32476\n14606\n...")

col1, col2 = st.columns(2)
with col1:
    if st.button("🔍 PHÂN TÍCH ĐA THUẬT TOÁN", type="primary", use_container_width=True):
        nums = get_nums(user_input)
        if len(nums) >= 15:
            # Check previous prediction
            if "last_analysis" in st.session_state and nums:
                last = nums[-1]
                last_pred = st.session_state.last_analysis
                if last_pred and 'final_pick' in last_pred and last_pred['final_pick']:
                    pred_pair = last_pred['final_pick']['pair']
                    won = all(d in last for d in pred_pair)
                    st.session_state.history.insert(0, {
                        'Kỳ': last,
                        'Dự đoán': pred_pair,
                        'KQ': '🔥' if won else '❌'
                    })
                    db_manager.add_result(last, pred_pair, won)
            
            # Run 3-round analysis
            analysis = engine.three_round_analysis(nums)
            st.session_state.last_analysis = analysis['round3']
            st.session_state.current_nums = nums
            st.rerun()
        else:
            st.warning(f"Cần 15+ kỳ (có {len(nums)})")

with col2:
    if st.button("🗑️ Reset", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# === HIỂN THỊ KẾT QUẢ ===

if "last_analysis" in st.session_state:
    analysis = st.session_state.last_analysis
    nums = st.session_state.current_nums
    
    # Statistics
    st.markdown("### 📊 THỐNG KÊ CƠ BẢN")
    col1, col2, col3, col4 = st.columns(4)
    
    freq = engine.analyze_frequency(nums)
    gaps = engine.analyze_gaps(nums)
    hot, cold = engine.analyze_hot_cold(nums)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div style="font-size:12px; color:#888;">SỐ NÓNG</div>
            <div style="font-size:24px; font-weight:900; color:#ff0040;">{', '.join(hot)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div style="font-size:12px; color:#888;">SỐ LẠNH</div>
            <div style="font-size:24px; font-weight:900; color:#0066ff;">{', '.join(cold)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        max_gap = max(gaps.items(), key=lambda x: x[1])
        st.markdown(f"""
        <div class="metric-box">
            <div style="font-size:12px; color:#888;">GAN NHẤT</div>
            <div style="font-size:24px; font-weight:900; color:#9900ff;">{max_gap[0]} ({max_gap[1]} kỳ)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_pairs = len(engine.calculate_pair_scores(nums))
        st.markdown(f"""
        <div class="metric-box">
            <div style="font-size:12px; color:#888;">PHƯƠNG ÁN</div>
            <div style="font-size:24px; font-weight:900; color:#00ff40;">{total_pairs}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 10 Methods
    st.markdown("### 🎲 10+ PHƯƠNG PHÁP DỰ ĐOÁN")
    
    if 'final_methods' in analysis:
        for i, method in enumerate(analysis['final_methods'][:10], 1):
            tags = ""
            if i <= 3:
                tags += '<span class="tag tag-gold">TOP ' + str(i) + '</span>'
            if 'GAP_GOLDEN' in method.get('logic', ''):
                tags += '<span class="tag tag-gan">GAN VÀNG</span>'
            if 'MARKOV' in method.get('name', ''):
                tags += '<span class="tag tag-safe">MARKOV</span>'
            
            st.markdown(f"""
            <div class="method-box">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="color:#888; font-size:12px;">{i}. {method['name']}</span><br>
                        <span style="font-size:20px; font-weight:900; color:#FFD700;">{method['pair'][0]} - {method['pair'][1]}</span>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:24px; font-weight:900; color:{'#00ff40' if method['score'] > 50 else '#ffff00'};">
                            {method['score']:.0f}
                        </div>
                        <div style="font-size:10px; color:#888;">{tags}</div>
                    </div>
                </div>
                <div style="font-size:11px; color:#666; margin-top:5px;">{method['logic']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # TOP 3
    st.markdown("### 🏆 TOP 3 PHƯƠNG ÁN MẠNH NHẤT")
    
    if 'top3' in analysis:
        cols = st.columns(3)
        for i, method in enumerate(analysis['top3']):
            with cols[i]:
                st.markdown(f"""
                <div class="prediction-card" style="border-color: {'#FFD700' if i == 0 else '#C0C0C0' if i == 1 else '#CD7F32'};">
                    <div style="font-size:14px; color:#888;">#{i+1} {method['name']}</div>
                    <div class="big-number" style="color:{'#FFD700' if i == 0 else '#C0C0C0' if i == 1 else '#CD7F32'}; font-size:36px;">
                        {method['pair'][0]} - {method['pair'][1]}
                    </div>
                    <div style="font-size:18px; font-weight:900; color:#00ff40; margin-top:10px;">
                        Score: {method['score']:.0f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # FINAL PICK
    st.markdown("### 🎯 BẠCH THỦ CUỐI CÙNG")
    
    if 'final_pick' in analysis and analysis['final_pick']:
        pick = analysis['final_pick']
        st.markdown(f"""
        <div class="final-pick">
            {pick['pair'][0]} - {pick['pair'][1]}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-box" style="border-color:#00ff40;">
            <div style="font-size:16px; font-weight:700;">PHÂN TÍCH CHI TIẾT</div>
            <div style="text-align:left; margin-top:10px; font-size:13px; line-height:1.8;">
                • <b>Phương pháp:</b> {pick['name']}<br>
                • <b>Logic:</b> {pick['logic']}<br>
                • <b>Điểm số:</b> {pick['score']:.0f}<br>
                • <b>Tần suất:</b> {freq.get(pick['pair'][0], 0) + freq.get(pick['pair'][1], 0)} lần<br>
                • <b>Khoảng cách:</b> {gaps.get(pick['pair'][0], 0) + gaps.get(pick['pair'][1], 0)} kỳ<br>
                • <b>Phân bố:</b> {analysis.get('number_distribution', {})}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # History
    if st.session_state.history:
        st.markdown("### 📋 LỊCH SỬ ĐỐI SOÁT")
        
        df = pd.DataFrame(st.session_state.history[:15])
        
        def color_kq(val):
            return 'color: #00ff40; font-weight: 900' if '🔥' in val else 'color: #ff0040; font-weight: 900'
        
        st.dataframe(df.style.applymap(color_kq, subset=['KQ']), 
                     use_container_width=True, hide_index=True)
        
        # Win rate
        wins = sum(1 for h in st.session_state.history if '🔥' in h['KQ'])
        total = len(st.session_state.history)
        rate = (wins/total*100) if total > 0 else 0
        
        color_rate = "#00ff40" if rate >= 40 else ("#ffff00" if rate >= 30 else "#ff0040")
        
        st.markdown(f"""
        <div class="metric-box" style="border-color:{color_rate}; margin-top:15px;">
            <div style="font-size:16px;">TỶ LỆ THẮNG</div>
            <div style="font-size:36px; font-weight:900; color:{color_rate};">
                {rate:.1f}% ({wins}/{total})
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width:{rate}%; background:{color_rate};">
                    {rate:.0f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# === FOOTER ===
st.markdown("""
<div style="text-align:center; color:#444; font-size:10px; margin-top:30px; padding-top:15px; border-top:1px solid #333;">
    🎯 TITAN V52 - MASTER ANALYST | 10+ Algorithms | 3-Round Analysis | Self-Learning<br>
    <i>Lưu ý: Tool hỗ trợ phân tích xác suất - Không đảm bảo 100% - Quản lý vốn thông minh</i>
</div>
""", unsafe_allow_html=True)