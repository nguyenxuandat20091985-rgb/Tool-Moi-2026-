import streamlit as st
import google.generativeai as genai
import re
import json
import os
from collections import Counter 
from datetime import datetime
import numpy as np
from typing import List, Dict, Tuple
import time

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_memory_v21.json"
PREDICTIONS_FILE = "titan_predictions_v21.json"
STATS_FILE = "titan_stats_v21.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None 

neural_engine = setup_neural() 

# ================= HỆ THỐNG GHI NHỚ =================
def load_memory():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f: 
            try: return json.load(f)
            except: return []
    return [] 

def save_memory(data):
    with open(DB_FILE, "w") as f: 
        json.dump(data[-1000:], f)

def load_predictions():
    if os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE, "r") as f:
            try: return json.load(f)
            except: return []
    return []

def save_prediction(prediction_data):
    predictions = load_predictions()
    predictions.append(prediction_data)
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(predictions[-500:], f)

def load_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, "r") as f:
            try: return json.load(f)
            except: return {"accuracy_history": [], "total_predictions": 0, "correct_predictions": 0}
    return {"accuracy_history": [], "total_predictions": 0, "correct_predictions": 0}

def save_stats(stats):
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f)

if "history" not in st.session_state:
    st.session_state.history = load_memory()
if "predictions" not in st.session_state:
    st.session_state.predictions = load_predictions()
if "stats" not in st.session_state:
    st.session_state.stats = load_stats()
if "last_actual" not in st.session_state:
    st.session_state.last_actual = None

# ================= THUẬT TOÁN DỰ ĐOÁN CAO CẤP =================
class PredictionEngine:
    def __init__(self, history):
        self.history = history[-200:] if len(history) > 200 else history
        self.last_50 = history[-50:] if len(history) >= 50 else history
        self.last_30 = history[-30:] if len(history) >= 30 else history
        self.last_20 = history[-20:] if len(history) >= 20 else history
        self.last_10 = history[-10:] if len(history) >= 10 else history
        
    def analyze_patterns(self):
        """Phân tích pattern chuyên sâu"""
        patterns = {
            'repeating': self.find_repeating_patterns(),
            'trending': self.find_trends(),
            'gap_analysis': self.analyze_gaps(),
            'hot_cold': self.analyze_hot_cold(),
            'position_patterns': self.analyze_positions_deep(),
            'cross_correlation': self.cross_position_correlation()
        }
        return patterns
    
    def find_repeating_patterns(self):
        """Tìm pattern lặp lại trong lịch sử"""
        if len(self.history) < 20:
            return []
        
        patterns = []
        history_str = "".join(self.history)
        
        # Tìm pattern 2 số lặp lại
        for length in [2, 3, 4]:
            last_pattern = history_str[-length:]
            count = history_str.count(last_pattern)
            if count >= 2:
                patterns.append({
                    'type': f'pattern_{length}_so',
                    'pattern': last_pattern,
                    'frequency': count,
                    'confidence': min(count / 5, 0.9)
                })
        
        return patterns
    
    def find_trends(self):
        """Phân tích xu hướng tăng/giảm"""
        if len(self.history) < 10:
            return {}
        
        trends = {}
        # Chuyển đổi số thành giá trị để phân tích trend
        for pos in range(5):
            pos_values = []
            for num_str in self.last_30:
                pos_values.append(int(num_str[pos]))
            
            # Tính xu hướng
            changes = [pos_values[i+1] - pos_values[i] for i in range(len(pos_values)-1)]
            avg_change = sum(changes) / len(changes) if changes else 0
            
            if abs(avg_change) > 0.5:
                trends[f'pos_{pos+1}'] = {
                    'direction': 'up' if avg_change > 0 else 'down',
                    'strength': abs(avg_change),
                    'next_prediction': self.predict_by_trend(pos_values)
                }
        
        return trends
    
    def predict_by_trend(self, values):
        """Dự đoán dựa trên xu hướng"""
        if len(values) < 5:
            return None
        
        # Linear regression đơn giản
        x = list(range(len(values)))
        y = values
        
        # Tính slope
        n = len(x)
        slope = (n * sum(x[i]*y[i] for i in range(n)) - sum(x)*sum(y)) / (n * sum(x[i]**2 for i in range(n)) - sum(x)**2)
        
        # Dự đoán giá trị tiếp theo
        next_value = values[-1] + slope
        # Giới hạn trong 0-9
        next_value = max(0, min(9, round(next_value)))
        
        return str(next_value)
    
    def analyze_gaps(self):
        """Phân tích khoảng cách xuất hiện của các số"""
        gaps = {}
        all_nums = "".join(self.history)
        
        for num in '0123456789':
            positions = [i for i, n in enumerate(all_nums) if n == num]
            if len(positions) > 1:
                gaps_between = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                avg_gap = sum(gaps_between) / len(gaps_between)
                last_gap = len(all_nums) - positions[-1] if positions else None
                
                gaps[num] = {
                    'avg_gap': avg_gap,
                    'last_gap': last_gap,
                    'due': last_gap and last_gap > avg_gap * 1.5  # Quá hạn xuất hiện
                }
        
        return gaps
    
    def analyze_hot_cold(self):
        """Phân tích số nóng/lạnh chi tiết"""
        all_nums = "".join(self.last_50)
        counts = Counter(all_nums)
        total = len(all_nums)
        
        hot_cold = {}
        for num in '0123456789':
            freq = counts.get(num, 0) / total if total > 0 else 0
            if freq > 0.15:  # Xuất hiện nhiều hơn 15%
                hot_cold[num] = {'status': 'hot', 'freq': freq}
            elif freq < 0.05:  # Xuất hiện ít hơn 5%
                hot_cold[num] = {'status': 'cold', 'freq': freq}
            else:
                hot_cold[num] = {'status': 'normal', 'freq': freq}
        
        return hot_cold
    
    def analyze_positions_deep(self):
        """Phân tích sâu từng vị trí"""
        positions = {i: [] for i in range(5)}
        for num_str in self.history:
            for i, digit in enumerate(num_str):
                positions[i].append(digit)
        
        pos_analysis = {}
        for pos, digits in positions.items():
            recent = digits[-30:]
            
            # Tìm pattern tại vị trí này
            patterns = []
            for length in [2, 3]:
                if len(recent) > length:
                    last_pattern = recent[-length:]
                    # Kiểm tra pattern này đã xuất hiện bao nhiêu lần
                    pattern_count = 0
                    for i in range(len(recent) - length):
                        if recent[i:i+length] == last_pattern:
                            pattern_count += 1
                    
                    if pattern_count >= 1:
                        patterns.append({
                            'length': length,
                            'pattern': last_pattern,
                            'count': pattern_count
                        })
            
            # Dự đoán cho vị trí này
            prediction = self.predict_position(recent)
            
            pos_analysis[f'pos_{pos+1}'] = {
                'patterns': patterns,
                'prediction': prediction,
                'volatility': self.calculate_volatility(recent)
            }
        
        return pos_analysis
    
    def predict_position(self, digits):
        """Dự đoán số cho 1 vị trí cụ thể"""
        if len(digits) < 5:
            return {'number': '0', 'confidence': 0.1}
        
        # Phân tích Markov chain đơn giản
        transitions = {}
        for i in range(len(digits)-1):
            current = digits[i]
            next_num = digits[i+1]
            if current not in transitions:
                transitions[current] = []
            transitions[current].append(next_num)
        
        # Dự đoán dựa trên số hiện tại
        current = digits[-1]
        if current in transitions and transitions[current]:
            next_nums = Counter(transitions[current])
            most_common = next_nums.most_common(1)[0]
            confidence = most_common[1] / len(transitions[current])
            return {
                'number': most_common[0],
                'confidence': confidence,
                'method': 'markov'
            }
        
        # Fallback: chọn số phổ biến nhất
        counts = Counter(digits[-10:])
        most_common = counts.most_common(1)[0]
        return {
            'number': most_common[0],
            'confidence': most_common[1] / 10,
            'method': 'frequency'
        }
    
    def calculate_volatility(self, digits):
        """Tính độ biến động của 1 vị trí"""
        if len(digits) < 5:
            return 0
        
        # Chuyển sang số
        nums = [int(d) for d in digits[-20:]]
        changes = [abs(nums[i+1] - nums[i]) for i in range(len(nums)-1)]
        avg_change = sum(changes) / len(changes) if changes else 0
        
        return avg_change / 9  # Chuẩn hóa về 0-1
    
    def cross_position_correlation(self):
        """Tìm tương quan chéo giữa các vị trí"""
        if len(self.history) < 20:
            return {}
        
        all_nums = [list(num_str) for num_str in self.last_50]
        correlations = {}
        
        for i in range(5):
            for j in range(i+1, 5):
                pos_i = [int(row[i]) for row in all_nums]
                pos_j = [int(row[j]) for row in all_nums]
                
                # Tính correlation đơn giản
                same = sum(1 for a, b in zip(pos_i, pos_j) if a == b)
                diff = sum(1 for a, b in zip(pos_i, pos_j) if abs(a-b) <= 2)
                
                same_ratio = same / len(pos_i)
                diff_ratio = diff / len(pos_i)
                
                if same_ratio > 0.3 or diff_ratio > 0.6:
                    correlations[f'{i+1}-{j+1}'] = {
                        'same_ratio': same_ratio,
                        'diff_ratio': diff_ratio,
                        'strength': max(same_ratio, diff_ratio)
                    }
        
        return correlations
    
    def calculate_weighted_probabilities(self):
        """Tính xác suất có trọng số"""
        if len(self.history) < 10:
            return {num: 0.1 for num in '0123456789'}
        
        probabilities = {num: 0 for num in '0123456789'}
        weights = {
            'recent': 0.35,      # 20 kỳ gần
            'medium': 0.25,      # 50 kỳ gần
            'position': 0.20,    # Phân tích vị trí
            'pattern': 0.20      # Pattern phát hiện
        }
        
        # 1. Recent frequency (20 kỳ)
        recent_nums = "".join(self.last_20)
        recent_counts = Counter(recent_nums)
        recent_total = len(recent_nums)
        
        # 2. Medium frequency (50 kỳ)
        medium_nums = "".join(self.last_50)
        medium_counts = Counter(medium_nums)
        medium_total = len(medium_nums)
        
        # 3. Position analysis
        pos_analysis = self.analyze_positions_deep()
        pos_scores = {num: 0 for num in '0123456789'}
        for pos_data in pos_analysis.values():
            if 'prediction' in pos_data and pos_data['prediction']:
                pred_num = pos_data['prediction']['number']
                pos_scores[pred_num] += pos_data['prediction']['confidence']
        
        # Chuẩn hóa position scores
        pos_total = sum(pos_scores.values())
        if pos_total > 0:
            for num in pos_scores:
                pos_scores[num] /= pos_total
        
        # 4. Pattern analysis
        patterns = self.find_repeating_patterns()
        pattern_scores = {num: 0 for num in '0123456789'}
        for pattern in patterns:
            if pattern['type'] == 'pattern_2_so':
                for digit in pattern['pattern']:
                    pattern_scores[digit] += pattern['confidence'] * 0.5
        
        # Kết hợp tất cả
        for num in '0123456789':
            recent_prob = recent_counts.get(num, 0) / recent_total if recent_total > 0 else 0.1
            medium_prob = medium_counts.get(num, 0) / medium_total if medium_total > 0 else 0.1
            
            probabilities[num] = (
                recent_prob * weights['recent'] +
                medium_prob * weights['medium'] +
                pos_scores[num] * weights['position'] +
                pattern_scores[num] * weights['pattern']
            )
        
        return probabilities
    
    def ensemble_prediction(self):
        """Kết hợp nhiều phương pháp dự đoán"""
        predictions = []
        
        # Phương pháp 1: Dựa trên xác suất có trọng số
        probs = self.calculate_weighted_probabilities()
        sorted_by_prob = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        # Phương pháp 2: Dựa trên phân tích vị trí
        pos_analysis = self.analyze_positions_deep()
        pos_predictions = []
        for pos_data in pos_analysis.values():
            if 'prediction' in pos_data:
                pos_predictions.append(pos_data['prediction']['number'])
        
        # Phương pháp 3: Dựa trên trend
        trends = self.find_trends()
        trend_predictions = [data['next_prediction'] for data in trends.values() if data.get('next_prediction')]
        
        # Kết hợp voting
        all_votes = []
        for num, prob in sorted_by_prob[:5]:  # Top 5 theo probability
            all_votes.extend([num] * int(prob * 10))
        
        all_votes.extend(pos_predictions * 2)  # Weight cho position predictions
        all_votes.extend(trend_predictions * 3)  # Weight cao cho trend
        
        vote_counts = Counter(all_votes)
        final_ranking = vote_counts.most_common()
        
        return {
            'ranked_numbers': [num for num, _ in final_ranking],
            'vote_counts': dict(final_ranking),
            'probabilities': probs
        }

# ================= UI DESIGN =================
st.set_page_config(page_title="TITAN v21.0 PRO MAX", layout="centered")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #c9d1d9; }
    .status-active { color: #238636; font-weight: bold; border-left: 3px solid #238636; padding-left: 10px; }
    .prediction-card {
        background: #0d1117; border: 2px solid #30363d;
        border-radius: 12px; padding: 25px; margin-top: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    .num-display { 
        font-size: 60px; font-weight: 900; color: #58a6ff; 
        text-align: center; letter-spacing: 10px; text-shadow: 0 0 25px #58a6ff;
    }
    .logic-box { 
        font-size: 14px; color: #8b949e; background: #161b22; 
        padding: 15px; border-radius: 8px; margin-bottom: 20px;
        border-left: 4px solid #58a6ff;
    }
    .hot-number {
        background: #238636; color: white; padding: 5px 10px;
        border-radius: 20px; font-weight: bold; display: inline-block;
        margin: 2px;
    }
    .cold-number {
        background: #6e7681; color: white; padding: 5px 10px;
        border-radius: 20px; font-weight: bold; display: inline-block;
        margin: 2px;
    }
    .accuracy-badge {
        background: #1f6feb; color: white; padding: 5px 15px;
        border-radius: 20px; font-weight: bold; display: inline-block;
    }
    </style>
""", unsafe_allow_html=True) 

# Header
st.markdown("<h2 style='text-align: center; color: #58a6ff;'>🧬 TITAN v21.0 PRO MAX</h2>", unsafe_allow_html=True)

# Hiển thị accuracy
total_preds = st.session_state.stats.get('total_predictions', 0)
correct_preds = st.session_state.stats.get('correct_predictions', 0)
accuracy = (correct_preds / total_preds * 100) if total_preds > 0 else 0

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<p class='status-active'>● KẾT NỐI: OK</p>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<p style='color: #58a6ff;'>📊 DỮ LIỆU: {len(st.session_state.history)} Kỳ</p>", unsafe_allow_html=True)
with col3:
    accuracy_color = "#238636" if accuracy >= 40 else "#f2cc60" if accuracy >= 30 else "#f85149"
    st.markdown(f"<p style='color: {accuracy_color};'>🎯 ĐỘ CHÍNH XÁC: {accuracy:.1f}%</p>", unsafe_allow_html=True)

# ================= NHẬP KẾT QUẢ THỰC TẾ =================
if st.session_state.last_actual:
    st.success(f"✅ Kết quả kỳ trước: {st.session_state.last_actual}")

actual_result = st.text_input("🎯 NHẬP KẾT QUẢ THỰC TẾ (5 số):", placeholder="ví dụ: 69962", max_chars=5)
if actual_result and len(actual_result) == 5 and actual_result.isdigit():
    st.session_state.last_actual = actual_result
    st.rerun()

# ================= NHẬP DỮ LIỆU LỊCH SỬ =================
raw_input = st.text_area("📡 NẠP DỮ LIỆU LỊCH SỬ:", height=100, placeholder="32880\n21808\n69962\n...") 

col1, col2, col3, col4 = st.columns([2,1,1,1])
with col1:
    if st.button("🚀 DỰ ĐOÁN NGAY", use_container_width=True):
        new_data = re.findall(r"\d{5}", raw_input)
        if new_data:
            # Thêm dữ liệu mới vào history
            st.session_state.history.extend(new_data)
            save_memory(st.session_state.history)
            
            # Khởi tạo prediction engine
            engine = PredictionEngine(st.session_state.history)
            
            # Lấy ensemble prediction
            ensemble = engine.ensemble_prediction()
            patterns = engine.analyze_patterns()
            hot_cold = engine.analyze_hot_cold()
            
            # Top predictions
            top_numbers = ensemble['ranked_numbers'][:10]  # Lấy top 10
            
            # Tạo prompt cho Gemini với phân tích chi tiết
            prompt = f"""
            Bạn là AI chuyên gia phân tích số 5D với độ chính xác cao.
            
            DỮ LIỆU PHÂN TÍCH CHI TIẾT:
            
            1. Lịch sử 50 kỳ gần nhất:
            {st.session_state.history[-50:]}
            
            2. Phân tích số nóng/lạnh:
            {hot_cold}
            
            3. Pattern phát hiện:
            {patterns['repeating']}
            
            4. Xu hướng các vị trí:
            {patterns['trending']}
            
            5. Xác suất có trọng số:
            {ensemble['probabilities']}
            
            6. Top 10 số tiềm năng (có voting):
            {top_numbers}
            
            YÊU CẦU: Dựa vào phân tích trên, hãy chọn:
            - 4 SỐ CHỦ LỰC (dan4): Ưu tiên số đang HOT, có xu hướng mạnh, xác suất cao
            - 3 SỐ LÓT (dan3): Số có tiềm năng nhưng cần thận trọng
            
            CHỈ TRẢ VỀ JSON:
            {{
                "dan4": ["4 số"],
                "dan3": ["3 số"],
                "logic": "phân tích ngắn gọn lý do chọn số",
                "xu_huong": "xu hướng chính hiện tại",
                "canh_bao": "cảnh báo nếu có"
            }}
            """
            
            try:
                response = neural_engine.generate_content(prompt)
                res_text = response.text
                json_match = re.search(r'\{.*\}', res_text, re.DOTALL)
                
                if json_match:
                    data = json.loads(json_match.group())
                    
                    # Đảm bảo đủ số
                    if len(data.get('dan4', [])) < 4:
                        data['dan4'] = top_numbers[:4]
                    if len(data.get('dan3', [])) < 3:
                        data['dan3'] = top_numbers[4:7]
                    
                    # Lưu dự đoán
                    prediction_record = {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "history_last": st.session_state.history[-10:],
                        "dan4": data['dan4'],
                        "dan3": data['dan3'],
                        "logic": data.get('logic', ''),
                        "xu_huong": data.get('xu_huong', ''),
                        "actual": None
                    }
                    save_prediction(prediction_record)
                    st.session_state.predictions = load_predictions()
                    
                    st.session_state.last_result = data
                    
            except Exception as e:
                # Fallback nếu Gemini lỗi
                st.session_state.last_result = {
                    "dan4": top_numbers[:4],
                    "dan3": top_numbers[4:7],
                    "logic": f"Dựa trên phân tích: Hot: {[n for n, d in hot_cold.items() if d.get('status')=='hot'][:3]}, Xu hướng: {len(patterns['trending'])} vị trí có trend",
                    "xu_huong": "bệt" if any('bệt' in str(p) for p in patterns['repeating']) else "đan xen",
                    "canh_bao": ""
                }
            
            st.rerun()

with col2:
    if st.button("🗑️ RESET", use_container_width=True):
        st.session_state.history = []
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

with col3:
    if st.button("📜 LỊCH SỬ", use_container_width=True):
        st.session_state.show_predictions = not st.session_state.get('show_predictions', False)
        st.rerun()

with col4:
    if st.button("📊 STATS", use_container_width=True):
        st.session_state.show_stats = not st.session_state.get('show_stats', False)
        st.rerun()

# ================= HIỂN THỊ STATS =================
if st.session_state.get('show_stats', False):
    with st.expander("📊 THỐNG KÊ ĐỘ CHÍNH XÁC", expanded=True):
        stats = st.session_state.stats
        total = stats.get('total_predictions', 0)
        correct = stats.get('correct_predictions', 0)
        
        if total > 0:
            acc = (correct / total) * 100
            
            st.markdown(f"""
            <div style='background: #161b22; padding: 20px; border-radius: 10px;'>
                <h3 style='color: #58a6ff; text-align: center;'>ĐỘ CHÍNH XÁC TỔNG THỂ</h3>
                <div style='font-size: 48px; text-align: center; color: {"#238636" if acc >= 40 else "#f2cc60" if acc >= 30 else "#f85149"};'>
                    {acc:.1f}%
                </div>
                <div style='text-align: center;'>Đúng: {correct} / {total} dự đoán</div>
                
                <div style='margin-top: 20px;'>
                    <h4>Lịch sử độ chính xác (20 gần nhất):</h4>
            """, unsafe_allow_html=True)
            
            history = stats.get('accuracy_history', [])
            for i, h in enumerate(history[-20:]):
                color = "#238636" if h >= 40 else "#f2cc60" if h >= 30 else "#f85149"
                st.markdown(f"""
                <div style='display: flex; align-items: center; margin: 5px 0;'>
                    <div style='width: 50px;'>Kỳ {i+1}</div>
                    <div style='flex-grow: 1; background: #0d1117; height: 20px; border-radius: 10px;'>
                        <div style='width: {h}%; background: {color}; height: 20px; border-radius: 10px;'></div>
                    </div>
                    <div style='width: 50px; text-align: right;'>{h:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Chưa có dữ liệu thống kê")

# ================= HIỂN THỊ LỊCH SỬ DỰ ĐOÁN =================
if st.session_state.get('show_predictions', False):
    with st.expander("📜 LỊCH SỬ DỰ ĐOÁN", expanded=True):
        predictions = load_predictions()
        if predictions:
            for pred in reversed(predictions[-30:]):
                # Kiểm tra xem có kết quả thực tế không
                has_actual = pred.get('actual') is not None
                border_color = "#238636" if has_actual else "#30363d"
                
                st.markdown(f"""
                <div style='background: #161b22; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid {border_color};'>
                    <small>{pred['time']}</small>
                    <div style='font-size: 24px; letter-spacing: 5px; margin: 5px 0;'>
                        <span style='color: #58a6ff;'>{''.join(pred['dan4'])}</span>
                        <span style='color: #f2cc60;'>{''.join(pred['dan3'])}</span>
                    </div>
                    <small>💡 {pred['logic'][:100]}...</small>
                    <br><small>📊 Xu hướng: {pred.get('xu_huong', 'N/A')}</small>
                    {f"<br><small>✅ Kết quả: {pred['actual']}</small>" if pred.get('actual') else ""}
                </div>
                """, unsafe_allow_html=True)
            
            # Form nhập kết quả thực tế
            st.markdown("---")
            st.markdown("### 📝 CẬP NHẬT KẾT QUẢ")
            pred_index = st.number_input("Chọn dự đoán (số thứ tự từ dưới lên):", min_value=1, max_value=len(predictions), value=1)
            actual_input = st.text_input("Kết quả thực tế (5 số):", max_chars=5, key="actual_input")
            
            if st.button("Cập nhật kết quả"):
                if actual_input and len(actual_input) == 5:
                    predictions[-pred_index]['actual'] = actual_input
                    with open(PREDICTIONS_FILE, "w") as f:
                        json.dump(predictions, f)
                    
                    # Cập nhật stats
                    stats = st.session_state.stats
                    stats['total_predictions'] = stats.get('total_predictions', 0) + 1
                    
                    # Kiểm tra dự đoán đúng
                    last_pred = predictions[-pred_index]
                    all_pred = last_pred['dan4'] + last_pred['dan3']
                    if actual_input in all_pred:
                        stats['correct_predictions'] = stats.get('correct_predictions', 0) + 1
                    
                    # Tính accuracy gần đây
                    recent_preds = predictions[-20:]
                    correct_recent = sum(1 for p in recent_preds if p.get('actual') and p['actual'] in (p['dan4'] + p['dan3']))
                    recent_acc = (correct_recent / len(recent_preds)) * 100 if recent_preds else 0
                    
                    if 'accuracy_history' not in stats:
                        stats['accuracy_history'] = []
                    stats['accuracy_history'].append(recent_acc)
                    
                    save_stats(stats)
                    st.session_state.stats = stats
                    
                    st.success("Đã cập nhật!")
                    st.rerun()
        else:
            st.info("Chưa có lịch sử dự đoán")

# ================= HIỂN THỊ KẾT QUẢ =================
if "last_result" in st.session_state:
    res = st.session_state.last_result
    
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'>
        <span style='color: #8b949e;'>🎯 KẾT QUẢ DỰ ĐOÁN MỚI NHẤT</span>
        <span class='accuracy-badge'>LIVE</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Phân tích nhanh
    engine = PredictionEngine(st.session_state.history)
    hot_cold = engine.analyze_hot_cold()
    
    hot_nums = [num for num, data in hot_cold.items() if data.get('status') == 'hot']
    cold_nums = [num for num, data in hot_cold.items() if data.get('status') == 'cold']
    
    col1, col2 = st.columns(2)
    with col1:
        if hot_nums:
            st.markdown("**🔥 Số HOT:** " + " ".join([f"<span class='hot-number'>{n}</span>" for n in hot_nums[:5]]), unsafe_allow_html=True)
    with col2:
        if cold_nums:
            st.markdown("**❄️ Số LẠNH:** " + " ".join([f"<span class='cold-number'>{n}</span>" for n in cold_nums[:5]]), unsafe_allow_html=True)
    
    # Logic phân tích
    st.markdown(f"""
    <div class='logic-box'>
        <b>🧠 PHÂN TÍCH:</b><br>
        {res['logic']}
        <br><br>
        <b>📊 Xu hướng:</b> {res.get('xu_huong', 'Đan xen')}
    </div>
    """, unsafe_allow_html=True)
    
    # Hiển thị dàn số
    st.markdown("<p style='text-align:center; font-size:16px; color:#888;'>🎯 4 SỐ CHỦ LỰC (ĐẶT CƯỢC CHÍNH)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-display'>{''.join(map(str, res['dan4']))}</div>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align:center; font-size:16px; color:#888; margin-top:20px;'>🛡️ 3 SỐ LÓT (ĐẶT THÊM)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-display' style='color:#f2cc60; text-shadow: 0 0 25px #f2cc60;'>{''.join(map(str, res['dan3']))}</div>", unsafe_allow_html=True)
    
    # Nút copy
    copy_val = "".join(map(str, res['dan4'])) + "".join(map(str, res['dan3']))
    st.text_input("📋 DÀN 7 SỐ:", copy_val, key="copy_final")
    
    # Cảnh báo nếu có
    if res.get('canh_bao'):
        st.warning(f"⚠️ {res['canh_bao']}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<br>
<div style='text-align:center; font-size:11px; color:#444; border-top: 1px solid #30363d; padding-top: 15px;'>
    🧬 TITAN v21.0 PRO MAX - Thuật toán ensemble | Phân tích đa chiều | Markov Chain | Trend Detection<br>
    ⚡ Độ chính xác đang được cải thiện dựa trên feedback thực tế
</div>
""", unsafe_allow_html=True)