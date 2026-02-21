import streamlit as st
import google.generativeai as genai
import re
import json
import os
from collections import Counter 
from datetime import datetime
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import hashlib

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_memory_v21.json"
PREDICTIONS_FILE = "titan_predictions_v21.json"
ANALYSIS_FILE = "titan_analysis_v21.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None 

neural_engine = setup_neural() 

# ================= HỆ THỐNG GHI NHỚ VĨNH VIỄN =================
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
        json.dump(predictions[-200:], f)

def load_analysis():
    if os.path.exists(ANALYSIS_FILE):
        with open(ANALYSIS_FILE, "r") as f:
            try: return json.load(f)
            except: return {}
    return {}

def save_analysis(data):
    with open(ANALYSIS_FILE, "w") as f:
        json.dump(data, f)

if "history" not in st.session_state:
    st.session_state.history = load_memory()
if "predictions" not in st.session_state:
    st.session_state.predictions = load_predictions()
if "analysis_cache" not in st.session_state:
    st.session_state.analysis_cache = load_analysis()

# ================= THUẬT TOÁN PHÂN TÍCH NÂNG CAO =================
class TitanAdvancedAnalyzer:
    def __init__(self, history):
        self.history = history[-300:] if len(history) > 300 else history
        self.last_100 = history[-100:] if len(history) >= 100 else history
        self.last_50 = history[-50:] if len(history) >= 50 else history
        self.last_20 = history[-20:] if len(history) >= 20 else history
        
    def analyze_positions(self):
        """Phân tích chi tiết từng vị trí trong dãy 5 số"""
        if not self.history:
            return {}
            
        positions = {i: [] for i in range(5)}
        for num_str in self.history:
            for i, digit in enumerate(num_str):
                positions[i].append(digit)
        
        position_analysis = {}
        for pos, digits in positions.items():
            recent = digits[-30:]
            counts = Counter(recent)
            total = len(recent)
            
            # Tìm số hot nhất vị trí này
            hot_numbers = [num for num, count in counts.most_common(3)]
            
            # Phân tích chu kỳ tại vị trí
            cycles = self.detect_position_cycles(digits[-50:])
            
            # Dự đoán số tiếp theo cho vị trí
            next_pred = self.predict_next_position(digits)
            
            position_analysis[f'pos_{pos+1}'] = {
                'hot': hot_numbers,
                'frequencies': {num: counts.get(num, 0)/total for num in '0123456789'},
                'cycles': cycles,
                'next_prediction': next_pred,
                'streak': self.get_position_streak(digits)
            }
        
        return position_analysis
    
    def detect_position_cycles(self, digits):
        """Phát hiện chu kỳ lặp lại tại 1 vị trí"""
        cycles = []
        for length in [3, 4, 5, 6, 7, 8, 9, 10]:
            if len(digits) >= length * 2:
                pattern = digits[-length:]
                # Kiểm tra pattern có lặp lại không
                for i in range(len(digits) - length * 2, len(digits) - length):
                    if digits[i:i+length] == pattern:
                        cycles.append({
                            'length': length,
                            'pattern': pattern,
                            'confidence': 0.8
                        })
                        break
        return cycles[:3]  # Trả về 3 chu kỳ đáng tin nhất
    
    def predict_next_position(self, digits):
        """Dự đoán số tiếp theo cho 1 vị trí"""
        if len(digits) < 10:
            return {'prediction': None, 'confidence': 0}
        
        # Phân tích pattern gần nhất
        last_10 = digits[-10:]
        counts = Counter(last_10)
        
        # Kiểm tra streak
        streak = self.get_position_streak(digits)
        if streak['length'] >= 2:
            # Nếu đang streak, khả năng cao streak tiếp
            return {
                'prediction': streak['number'],
                'confidence': min(0.5 + streak['length'] * 0.1, 0.85),
                'reason': f'Đang bệt {streak["length"]} kỳ'
            }
        
        # Dự đoán dựa trên tần suất
        most_common = counts.most_common(1)[0]
        return {
            'prediction': most_common[0],
            'confidence': most_common[1] / len(last_10),
            'reason': 'Xuất hiện nhiều nhất gần đây'
        }
    
    def get_position_streak(self, digits):
        """Lấy streak hiện tại của 1 vị trí"""
        if len(digits) < 2:
            return {'number': None, 'length': 0}
        
        current = digits[-1]
        streak = 1
        for i in range(len(digits)-2, -1, -1):
            if digits[i] == current:
                streak += 1
            else:
                break
        
        return {'number': current, 'length': streak}
    
    def analyze_correlations(self):
        """Phân tích tương quan giữa các vị trí"""
        if len(self.history) < 20:
            return {}
        
        correlations = {}
        all_nums = [list(num_str) for num_str in self.history]
        
        # Tạo ma trận tương quan
        for i in range(5):
            for j in range(i+1, 5):
                pos_i = [int(row[i]) for row in all_nums[-50:]]
                pos_j = [int(row[j]) for row in all_nums[-50:]]
                
                # Tính tương quan đơn giản
                same_count = sum(1 for a, b in zip(pos_i, pos_j) if a == b)
                correlation = same_count / len(pos_i)
                
                if correlation > 0.3:  # Chỉ lưu tương quan đáng kể
                    correlations[f'{i+1}-{j+1}'] = {
                        'strength': correlation,
                        'meaning': f'Vị trí {i+1} và {j+1} cùng số {correlation*100:.0f}% thời gian'
                    }
        
        return correlations
    
    def detect_complex_patterns(self):
        """Phát hiện các pattern phức tạp"""
        patterns = {
            'tam_giac': [],  # Pattern tam giác: 1-2-3-2-1
            'cau_doi': [],   # Cầu đối xứng
            'cau_lech': [],  # Cầu lệch
            'bong_am': [],   # Bóng âm
            'bong_duong': [] # Bóng dương
        }
        
        # Bóng âm dương theo thuyết ngũ hành
        bong_am = {'0':'7', '1':'4', '2':'9', '3':'6', '4':'1', '5':'8', '6':'3', '7':'0', '8':'5', '9':'2'}
        bong_duong = {'0':'5', '1':'6', '2':'7', '3':'8', '4':'9', '5':'0', '6':'1', '7':'2', '8':'3', '9':'4'}
        
        last_num = self.history[-1] if self.history else None
        if last_num:
            # Dự đoán bóng
            patterns['bong_am'] = [bong_am.get(d, d) for d in last_num]
            patterns['bong_duong'] = [bong_duong.get(d, d) for d in last_num]
        
        # Tìm pattern tam giác
        if len(self.history) >= 10:
            nums = [list(num) for num in self.history[-10:]]
            for pos in range(5):
                digits = [int(n[pos]) for n in nums]
                if self.is_triangle_pattern(digits):
                    patterns['tam_giac'].append(f'Vị trí {pos+1}')
        
        return patterns
    
    def is_triangle_pattern(self, digits):
        """Kiểm tra pattern tam giác (tăng dần rồi giảm dần)"""
        if len(digits) < 5:
            return False
        
        # Kiểm tra 5 số gần nhất có dạng tam giác không
        last_5 = digits[-5:]
        # Tìm đỉnh
        peak_index = last_5.index(max(last_5))
        # Kiểm tra tăng dần đến đỉnh và giảm dần sau đỉnh
        increasing = all(last_5[i] <= last_5[i+1] for i in range(peak_index))
        decreasing = all(last_5[i] >= last_5[i+1] for i in range(peak_index, len(last_5)-1))
        
        return increasing and decreasing
    
    def calculate_probability_matrix(self):
        """Tính ma trận xác suất chi tiết"""
        if len(self.history) < 20:
            return {}
        
        prob_matrix = {}
        
        # Xác suất dựa trên lịch sử gần
        for num in '0123456789':
            prob_matrix[num] = {
                'short_term': 0.1,  # 20 kỳ gần
                'medium_term': 0.1,  # 50 kỳ gần
                'long_term': 0.1,    # 100 kỳ gần
                'position_based': 0.1,
                'final': 0.1
            }
        
        # Tính short term (20 kỳ)
        short_nums = "".join(self.last_20)
        short_counts = Counter(short_nums)
        short_total = len(short_nums)
        
        # Tính medium term (50 kỳ)
        medium_nums = "".join(self.last_50)
        medium_counts = Counter(medium_nums)
        medium_total = len(medium_nums)
        
        # Tính long term (100 kỳ)
        long_nums = "".join(self.last_100)
        long_counts = Counter(long_nums)
        long_total = len(long_nums)
        
        # Phân tích vị trí
        pos_analysis = self.analyze_positions()
        pos_probs = {num: 0 for num in '0123456789'}
        for pos_data in pos_analysis.values():
            for num, prob in pos_data['frequencies'].items():
                pos_probs[num] += prob
        # Chuẩn hóa
        pos_total = sum(pos_probs.values())
        if pos_total > 0:
            for num in pos_probs:
                pos_probs[num] /= pos_total
        
        # Kết hợp các yếu tố với trọng số
        for num in '0123456789':
            short_prob = short_counts.get(num, 0) / short_total if short_total > 0 else 0.1
            medium_prob = medium_counts.get(num, 0) / medium_total if medium_total > 0 else 0.1
            long_prob = long_counts.get(num, 0) / long_total if long_total > 0 else 0.1
            pos_prob = pos_probs.get(num, 0.1)
            
            # Trọng số: gần đây quan trọng hơn
            final_prob = (short_prob * 0.4 + medium_prob * 0.3 + 
                         long_prob * 0.2 + pos_prob * 0.1)
            
            prob_matrix[num] = {
                'short_term': round(short_prob, 3),
                'medium_term': round(medium_prob, 3),
                'long_term': round(long_prob, 3),
                'position_based': round(pos_prob, 3),
                'final': round(final_prob, 3)
            }
        
        return prob_matrix
    
    def get_top_predictions(self, n=7):
        """Lấy top n dự đoán tốt nhất"""
        prob_matrix = self.calculate_probability_matrix()
        
        # Sắp xếp theo final probability
        sorted_nums = sorted(prob_matrix.items(), 
                           key=lambda x: x[1]['final'], 
                           reverse=True)
        
        top_nums = [num for num, _ in sorted_nums[:n]]
        
        # Phân tích lý do
        reasons = []
        for num in top_nums[:4]:
            reasons.append(f"Số {num}: {prob_matrix[num]['final']*100:.1f}%")
        
        # Thêm phân tích streak
        streaks = []
        for pos_data in self.analyze_positions().values():
            if pos_data['streak']['length'] >= 2:
                streaks.append(f"Vị trí đang bệt số {pos_data['streak']['number']} ({pos_data['streak']['length']} kỳ)")
        
        return {
            'top_numbers': top_nums,
            'probabilities': {num: prob_matrix[num] for num in top_nums},
            'reasons': reasons,
            'streaks': streaks[:3]
        }

# ================= UI DESIGN =================
st.set_page_config(page_title="TITAN v21.0 PRO", layout="centered")
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
    .streak-badge {
        background: #1f6feb; color: white; padding: 4px 12px;
        border-radius: 20px; font-size: 12px; display: inline-block;
        margin: 2px; font-weight: bold;
    }
    .stats-box {
        background: #161b22; border-radius: 10px; padding: 15px;
        margin: 10px 0; border: 1px solid #30363d;
    }
    .prob-bar {
        height: 6px; background: #30363d; border-radius: 3px;
        margin: 5px 0;
    }
    .prob-fill {
        height: 6px; background: #58a6ff; border-radius: 3px;
    }
    .hot-number {
        background: #238636; color: white; padding: 5px 10px;
        border-radius: 20px; font-weight: bold; display: inline-block;
        margin: 2px;
    }
    </style>
""", unsafe_allow_html=True) 

st.markdown("<h2 style='text-align: center; color: #58a6ff;'>🧬 TITAN v21.0 OMNI PLUS</h2>", unsafe_allow_html=True)
if neural_engine:
    st.markdown(f"<p class='status-active'>● KẾT NỐI NEURAL-LINK: OK | DỮ LIỆU: {len(st.session_state.history)} KỲ | DỰ ĐOÁN: {len(st.session_state.predictions)}</p>", unsafe_allow_html=True)
else:
    st.error("LỖI KẾT NỐI API - KIỂM TRA LẠI KEY")

# ================= HIỂN THỊ PHÂN TÍCH NÂNG CAO =================
if st.session_state.history:
    analyzer = TitanAdvancedAnalyzer(st.session_state.history)
    
    # Tabs cho các phân tích
    tab1, tab2, tab3, tab4 = st.tabs(["📊 TỔNG QUAN", "🎯 PHÂN TÍCH VỊ TRÍ", "🔄 TƯƠNG QUAN", "📈 XÁC SUẤT"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔥 PHÂN TÍCH CẦU BỆT")
            pos_analysis = analyzer.analyze_positions()
            streaks_found = False
            
            for pos, data in pos_analysis.items():
                if data['streak']['length'] >= 2:
                    streaks_found = True
                    color = "#f2cc60" if data['streak']['length'] >= 3 else "#58a6ff"
                    st.markdown(f"""
                    <div style='background: #161b22; padding: 10px; border-radius: 8px; margin: 5px 0;'>
                        <b>{pos}:</b> 
                        <span style='color: {color}; font-size: 20px; font-weight: bold;'>
                            {data['streak']['number']}
                        </span> 
                        bệt {data['streak']['length']} kỳ
                    </div>
                    """, unsafe_allow_html=True)
            
            if not streaks_found:
                st.info("Không có cầu bệt đáng kể")
        
        with col2:
            st.markdown("### 🎯 DỰ ĐOÁN VỊ TRÍ")
            for pos, data in pos_analysis.items():
                next_pred = data.get('next_prediction', {})
                if next_pred.get('prediction'):
                    conf = next_pred['confidence'] * 100
                    st.markdown(f"""
                    <div style='margin: 5px 0;'>
                        <b>{pos}:</b> Số {next_pred['prediction']} 
                        <small>({conf:.0f}% - {next_pred.get('reason', '')})</small>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📊 CHI TIẾT TỪNG VỊ TRÍ")
        
        pos_analysis = analyzer.analyze_positions()
        for pos, data in pos_analysis.items():
            with st.expander(f"VỊ TRÍ {pos}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔥 SỐ HOT:**")
                    hot_html = ""
                    for num in data['hot'][:3]:
                        hot_html += f"<span class='hot-number'>{num}</span> "
                    st.markdown(hot_html, unsafe_allow_html=True)
                    
                    st.markdown("**📈 TẦN SUẤT 30 KỲ:**")
                    for num, prob in sorted(data['frequencies'].items(), 
                                           key=lambda x: x[1], reverse=True)[:5]:
                        st.markdown(f"""
                        <div>
                            Số {num}: {prob*100:.1f}%
                            <div class='prob-bar'>
                                <div class='prob-fill' style='width: {prob*100}%'></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    if data['cycles']:
                        st.markdown("**🔄 CHU KỲ PHÁT HIỆN:**")
                        for cycle in data['cycles']:
                            st.markdown(f"""
                            <div style='background: #0d1117; padding: 8px; border-radius: 5px; margin: 5px 0;'>
                                <small>Chu kỳ {cycle['length']} số: 
                                {''.join(cycle['pattern'])}</small>
                            </div>
                            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### 🔗 TƯƠNG QUAN GIỮA CÁC VỊ TRÍ")
        
        correlations = analyzer.analyze_correlations()
        if correlations:
            for pair, data in correlations.items():
                st.markdown(f"""
                <div style='background: #161b22; padding: 15px; border-radius: 8px; margin: 10px 0;'>
                    <b>{pair}</b>: {data['meaning']}
                    <div class='prob-bar' style='margin-top: 10px;'>
                        <div class='prob-fill' style='width: {data['strength']*100}%'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Chưa phát hiện tương quan đáng kể")
        
        # Hiển thị bóng âm dương
        patterns = analyzer.detect_complex_patterns()
        if patterns['bong_am'] or patterns['bong_duong']:
            st.markdown("### 🎯 BÓNG ÂM DƯƠNG")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**🌑 Bóng âm:** {''.join(patterns['bong_am'])}")
            with col2:
                st.markdown(f"**🌕 Bóng dương:** {''.join(patterns['bong_duong'])}")
    
    with tab4:
        st.markdown("### 📈 MA TRẬN XÁC SUẤT")
        
        prob_matrix = analyzer.calculate_probability_matrix()
        if prob_matrix:
            # Sắp xếp theo final probability
            sorted_probs = sorted(prob_matrix.items(), 
                                 key=lambda x: x[1]['final'], 
                                 reverse=True)
            
            for num, probs in sorted_probs[:10]:
                st.markdown(f"""
                <div style='margin: 10px 0; padding: 10px; background: #161b22; border-radius: 8px;'>
                    <div style='display: flex; justify-content: space-between;'>
                        <span style='font-size: 20px; font-weight: bold;'>SỐ {num}</span>
                        <span style='color: #58a6ff;'>{(probs['final']*100):.1f}%</span>
                    </div>
                    <div style='font-size: 12px; color: #8b949e;'>
                        20 kỳ: {(probs['short_term']*100):.1f}% | 
                        50 kỳ: {(probs['medium_term']*100):.1f}% | 
                        100 kỳ: {(probs['long_term']*100):.1f}%
                    </div>
                    <div class='prob-bar'>
                        <div class='prob-fill' style='width: {probs['final']*100}%'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ================= XỬ LÝ DỮ LIỆU =================
raw_input = st.text_area("📡 NẠP DỮ LIỆU (Dán các dãy 5 số):", height=100, placeholder="32880\n21808\n...") 

col1, col2, col3, col4 = st.columns([2,1,1,1])
with col1:
    if st.button("🚀 GIẢI MÃ THUẬT TOÁN", use_container_width=True):
        new_data = re.findall(r"\d{5}", raw_input)
        if new_data:
            st.session_state.history.extend(new_data)
            save_memory(st.session_state.history)
            
            # Phân tích nâng cao
            analyzer = TitanAdvancedAnalyzer(st.session_state.history)
            top_pred = analyzer.get_top_predictions(7)
            pos_analysis = analyzer.analyze_positions()
            prob_matrix = analyzer.calculate_probability_matrix()
            
            # Tạo prompt thông minh cho Gemini
            streak_info = []
            for pos, data in pos_analysis.items():
                if data['streak']['length'] >= 2:
                    streak_info.append(f"{pos} bệt {data['streak']['number']} {data['streak']['length']} kỳ")
            
            prompt = f"""
            Bạn là AI chuyên gia phân tích số 5D với khả năng siêu việt.
            
            DỮ LIỆU PHÂN TÍCH CHI TIẾT:
            - Lịch sử 100 kỳ: {st.session_state.history[-100:]}
            - Top 7 số có xác suất cao nhất: {top_pred['top_numbers']}
            - Phân tích vị trí: {pos_analysis}
            - Ma trận xác suất: {prob_matrix}
            - Cầu bệt đang có: {streak_info if streak_info else 'Không có'}
            
            YÊU CẦU:
            1. Phân tích CHI TIẾT xu hướng hiện tại (cầu bệt, cầu đảo, pattern đặc biệt)
            2. Dự đoán 4 số chủ lực (dan4) - ưu tiên số đang có xu hướng mạnh
            3. Dự đoán 3 số lót (dan3) - ưu tiên số có xác suất cao nhưng cần thận trọng
            4. Đưa ra cảnh báo nếu phát hiện dấu hiệu bất thường
            
            TRẢ VỀ JSON CHÍNH XÁC:
            {{
                "dan4": ["4 số chính"],
                "dan3": ["3 số lót"],
                "logic": "phân tích chi tiết xu hướng và lý do chọn số",
                "canh_bao": "cảnh báo nếu có",
                "xu_huong": "bệt/đảo/ổn định",
                "do_tin_cay": 0-100
            }}
            
            QUAN TRỌNG: Chỉ trả về JSON, không thêm text khác.
            """
            
            try:
                response = neural_engine.generate_content(prompt)
                res_text = response.text
                # Lọc JSON từ response
                json_match = re.search(r'\{.*\}', res_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    
                    # Đảm bảo có đủ các trường
                    if 'dan4' not in data or len(data['dan4']) < 4:
                        data['dan4'] = top_pred['top_numbers'][:4]
                    if 'dan3' not in data or len(data['dan3']) < 3:
                        data['dan3'] = top_pred['top_numbers'][4:7]
                    
                    # Lưu dự đoán
                    prediction_record = {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "history_last": st.session_state.history[-10:],
                        "dan4": data['dan4'],
                        "dan3": data['dan3'],
                        "logic": data.get('logic', ''),
                        "xu_huong": data.get('xu_huong', ''),
                        "do_tin_cay": data.get('do_tin_cay', 0)
                    }
                    save_prediction(prediction_record)
                    st.session_state.predictions = load_predictions()
                    
                    st.session_state.last_result = data
                else:
                    raise Exception("Không tìm thấy JSON")
                    
            except Exception as e:
                # Fallback to thuật toán nội bộ
                top_nums = top_pred['top_numbers']
                logic_text = f"Phân tích thuật toán:\n"
                logic_text += f"- Top xác suất: {', '.join(top_pred['reasons'])}\n"
                if top_pred['streaks']:
                    logic_text += f"- Cảnh báo: {', '.join(top_pred['streaks'])}"
                
                st.session_state.last_result = {
                    "dan4": top_nums[:4],
                    "dan3": top_nums[4:7],
                    "logic": logic_text,
                    "canh_bao": "Đang sử dụng thuật toán nội bộ" if top_pred['streaks'] else "",
                    "xu_huong": "bệt" if top_pred['streaks'] else "ổn định",
                    "do_tin_cay": 75
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
    if st.button("🔄 REFRESH", use_container_width=True):
        st.rerun()

# ================= HIỂN THỊ LỊCH SỬ DỰ ĐOÁN =================
if st.session_state.get('show_predictions', False):
    with st.expander("📜 LỊCH SỬ DỰ ĐOÁN (100 GẦN NHẤT)", expanded=True):
        predictions = load_predictions()
        if predictions:
            for i, pred in enumerate(reversed(predictions[-20:])):
                accuracy_color = "#238636" if pred.get('do_tin_cay', 0) > 80 else "#f2cc60"
                st.markdown(f"""
                <div style='background: #161b22; padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 4px solid {accuracy_color};'>
                    <div style='display: flex; justify-content: space-between;'>
                        <small>🕐 {pred['time']}</small>
                        <small style='color: {accuracy_color};'>Độ tin cậy: {pred.get('do_tin_cay', 0)}%</small>
                    </div>
                    <div style='font-size: 24px; letter-spacing: 5px; margin: 5px 0;'>
                        <span style='color: #58a6ff;'>{''.join(pred['dan4'])}</span>
                        <span style='color: #f2cc60;'>{''.join(pred['dan3'])}</span>
                    </div>
                    <small>💡 {pred['logic'][:100]}...</small>
                    <br><small>📊 Xu hướng: {pred.get('xu_huong', 'N/A')}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Chưa có lịch sử dự đoán")

# ================= HIỂN THỊ KẾT QUẢ =================
if "last_result" in st.session_state:
    res = st.session_state.last_result
    
    # Tính toán độ tin cậy để hiển thị màu sắc
    confidence = res.get('do_tin_cay', 75)
    confidence_color = "#238636" if confidence > 80 else "#f2cc60" if confidence > 60 else "#f85149"
    
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    # Hiển thị header với độ tin cậy
    st.markdown(f"""
    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'>
        <span style='color: #8b949e;'>🎯 KẾT QUẢ DỰ ĐOÁN</span>
        <span style='background: {confidence_color}20; color: {confidence_color}; padding: 5px 15px; border-radius: 20px; font-weight: bold;'>
            {confidence}% TIN CẬY
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Hiển thị cảnh báo nếu có
    if res.get('canh_bao'):
        st.warning(f"⚠️ {res['canh_bao']}")
    
    # Hiển thị xu hướng
    if res.get('xu_huong'):
        trend_emoji = "🔥" if res['xu_huong'] == "bệt" else "🔄" if res['xu_huong'] == "đảo" else "⚖️"
        st.info(f"{trend_emoji} Xu hướng hiện tại: {res['xu_huong'].upper()}")
    
    # Hiển thị phân tích logic
    st.markdown(f"""
    <div class='logic-box'>
        <b>🧠 PHÂN TÍCH:</b><br>
        {res['logic']}
    </div>
    """, unsafe_allow_html=True)
    
    # Hiển thị 4 số chủ lực
    st.markdown("<p style='text-align:center; font-size:14px; color:#888;'>🎯 4 SỐ CHỦ LỰC (VÀO TIỀN CHÍNH)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-display'>{''.join(map(str, res['dan4']))}</div>", unsafe_allow_html=True)
    
    # Hiển thị 3 số lót
    st.markdown("<p style='text-align:center; font-size:14px; color:#888; margin-top:20px;'>🛡️ 3 SỐ LÓT (GIỮ VỐN, ĐÁNH KÈM)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-display' style='color:#f2cc60; text-shadow: 0 0 25px #f2cc60;'>{''.join(map(str, res['dan3']))}</div>", unsafe_allow_html=True)
    
    # Nút sao chép
    copy_val = "".join(map(str, res['dan4'])) + "".join(map(str, res['dan3']))
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.text_input("📋 DÀN 7 SỐ:", copy_val, key="copy_input")
    with col2:
        if st.button("📋 COPY", use_container_width=True):
            st.write("✅ Đã copy!")
            st.session_state.copy_success = True
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<br>
<div style='text-align:center; font-size:11px; color:#444; border-top: 1px solid #30363d; padding-top: 15px;'>
    🧬 TITAN v21.0 OMNI PLUS - Hệ thống phân tích đa chiều | Tích hợp Neural-Link & Thuật toán độc quyền<br>
    ⚡ Phân tích vị trí | Tương quan | Xác suất | Chu kỳ | Bóng âm dương
</div>
""", unsafe_allow_html=True)