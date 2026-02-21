import streamlit as st
import google.generativeai as genai
import re
import json
import os
from collections import Counter 
from datetime import datetime
import numpy as np
from typing import List, Dict, Tuple
import random

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_memory_v22.json"
PREDICTIONS_FILE = "titan_predictions_v22.json"
STATS_FILE = "titan_stats_v22.json"

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
        json.dump(data[-2000:], f)  # Lưu 2000 kỳ

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
            except: return {}
    return {}

def save_stats(data):
    with open(STATS_FILE, "w") as f:
        json.dump(data, f)

if "history" not in st.session_state:
    st.session_state.history = load_memory()
if "predictions" not in st.session_state:
    st.session_state.predictions = load_predictions()
if "stats" not in st.session_state:
    st.session_state.stats = load_stats()
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "show_predictions" not in st.session_state:
    st.session_state.show_predictions = False

# ================= THUẬT TOÁN DỰ ĐOÁN CAO CẤP =================
class TitanPredictorV22:
    def __init__(self, history):
        self.history = history
        self.stats = self.load_or_init_stats()
        
    def load_or_init_stats(self):
        """Tải thống kê hoặc khởi tạo mới"""
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, 'r') as f:
                return json.load(f)
        return {
            'position_frequency': [Counter() for _ in range(5)],
            'pair_frequency': {},
            'triple_frequency': {},
            'streak_history': [],
            'accuracy_tracking': []
        }
    
    def update_stats_with_result(self, actual_number):
        """Cập nhật thống kê với kết quả thực tế"""
        if len(actual_number) != 5:
            return
        
        # Cập nhật tần suất từng vị trí
        for i, digit in enumerate(actual_number):
            if i not in self.stats['position_frequency']:
                self.stats['position_frequency'].append(Counter())
            self.stats['position_frequency'][i][digit] += 1
        
        # Cập nhật cặp số
        for i in range(4):
            pair = actual_number[i:i+2]
            pair_key = f"{i}_{pair}"
            self.stats['pair_frequency'][pair_key] = self.stats['pair_frequency'].get(pair_key, 0) + 1
        
        # Cập nhật bộ ba
        for i in range(3):
            triple = actual_number[i:i+3]
            triple_key = f"{i}_{triple}"
            self.stats['triple_frequency'][triple_key] = self.stats['triple_frequency'].get(triple_key, 0) + 1
        
        save_stats(self.stats)
    
    def analyze_recent_patterns(self, lookback=50):
        """Phân tích patterns trong khoảng thời gian gần"""
        if len(self.history) < 20:
            return {}
        
        recent = self.history[-lookback:]
        patterns = {
            'hot_positions': [],
            'cold_positions': [],
            'repeating_digits': [],
            'trend_direction': 'unknown'
        }
        
        # Phân tích từng vị trí
        for pos in range(5):
            pos_digits = [int(num[pos]) for num in recent[-30:]]
            counter = Counter(pos_digits)
            
            # Top 3 số hot nhất
            hot = counter.most_common(3)
            patterns['hot_positions'].append({
                'position': pos + 1,
                'hot_numbers': [h[0] for h in hot],
                'frequencies': [h[1]/len(pos_digits) for h in hot]
            })
            
            # Số lạnh nhất
            cold = counter.most_common()[-1]
            patterns['cold_positions'].append({
                'position': pos + 1,
                'cold_number': cold[0],
                'frequency': cold[1]/len(pos_digits)
            })
        
        # Phát hiện xu hướng
        last_10 = [int(num[0]) for num in recent[-10:]]  # Dùng vị trí đầu để xét trend
        if len(last_10) >= 5:
            # Tính độ biến động
            volatility = np.std(last_10)
            if volatility < 2:
                patterns['trend_direction'] = 'ổn định'
            elif volatility < 3.5:
                patterns['trend_direction'] = 'dao động nhẹ'
            else:
                patterns['trend_direction'] = 'biến động mạnh'
        
        return patterns
    
    def predict_with_ml(self):
        """Dự đoán sử dụng thuật học máy đơn giản"""
        if len(self.history) < 50:
            return None
        
        predictions = []
        confidences = []
        
        for pos in range(5):
            pos_digits = [int(num[pos]) for num in self.history[-100:]]
            
            # Dùng Markov chain đơn giản
            transition_matrix = np.zeros((10, 10))
            for i in range(len(pos_digits)-1):
                transition_matrix[pos_digits[i], pos_digits[i+1]] += 1
            
            # Chuẩn hóa
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums!=0)
            
            # Dự đoán dựa trên số cuối cùng
            last_digit = pos_digits[-1]
            if row_sums[last_digit] > 0:
                probs = transition_matrix[last_digit]
                predicted = np.argmax(probs)
                confidence = probs[predicted]
            else:
                # Fallback to frequency
                counter = Counter(pos_digits[-20:])
                predicted = counter.most_common(1)[0][0]
                confidence = counter.most_common(1)[0][1] / 20
            
            predictions.append(str(predicted))
            confidences.append(confidence)
        
        return {
            'number': ''.join(predictions),
            'confidence': np.mean(confidences),
            'position_confidences': confidences
        }
    
    def detect_cycles(self):
        """Phát hiện chu kỳ lặp lại"""
        if len(self.history) < 30:
            return []
        
        cycles = []
        history_str = ''.join(self.history)
        
        for length in [3, 4, 5, 6, 7, 8, 9, 10]:
            if len(history_str) < length * 3:
                continue
            
            # Tìm pattern lặp lại
            last_pattern = history_str[-length:]
            occurrences = []
            
            for i in range(len(history_str) - length * 2, len(history_str) - length):
                if history_str[i:i+length] == last_pattern:
                    occurrences.append(i)
            
            if len(occurrences) >= 2:
                cycles.append({
                    'length': length,
                    'pattern': last_pattern,
                    'frequency': len(occurrences),
                    'confidence': min(len(occurrences) / 5, 0.9)
                })
        
        return sorted(cycles, key=lambda x: x['confidence'], reverse=True)[:3]
    
    def weighted_prediction(self):
        """Kết hợp nhiều phương pháp dự đoán"""
        predictions = []
        weights = []
        
        # Phương pháp 1: Dựa trên tần suất gần nhất (trọng số cao nhất)
        if len(self.history) >= 20:
            recent_nums = "".join(self.history[-20:])
            counter = Counter(recent_nums)
            top_7 = [num for num, _ in counter.most_common(7)]
            predictions.append(top_7)
            weights.append(0.35)
        
        # Phương pháp 2: ML prediction
        ml_result = self.predict_with_ml()
        if ml_result:
            ml_nums = list(ml_result['number'])
            predictions.append(ml_nums)
            weights.append(0.25)
        
        # Phương pháp 3: Phân tích vị trí
        patterns = self.analyze_recent_patterns()
        pos_predictions = []
        for pos_data in patterns.get('hot_positions', []):
            if pos_data['hot_numbers']:
                pos_predictions.extend(pos_data['hot_numbers'])
        if pos_predictions:
            predictions.append(pos_predictions[:7])
            weights.append(0.20)
        
        # Phương pháp 4: Cycle detection
        cycles = self.detect_cycles()
        cycle_predictions = []
        for cycle in cycles:
            cycle_predictions.extend(list(cycle['pattern']))
        if cycle_predictions:
            predictions.append(cycle_predictions[:7])
            weights.append(0.20)
        
        # Kết hợp có trọng số
        if not predictions:
            return {'dan4': [], 'dan3': [], 'confidence': 0}
        
        # Đếm phiếu có trọng số
        weighted_votes = {}
        for pred_list, weight in zip(predictions, weights):
            for num in pred_list[:7]:
                weighted_votes[num] = weighted_votes.get(num, 0) + weight
        
        # Sắp xếp theo điểm số
        sorted_votes = sorted(weighted_votes.items(), key=lambda x: x[1], reverse=True)
        top_numbers = [num for num, _ in sorted_votes[:7]]
        
        # Đảm bảo đủ 7 số
        if len(top_numbers) < 7:
            all_nums = list('0123456789')
            for num in all_nums:
                if num not in top_numbers and len(top_numbers) < 7:
                    top_numbers.append(num)
        
        return {
            'dan4': top_numbers[:4],
            'dan3': top_numbers[4:7],
            'confidence': sum([v for _, v in sorted_votes[:7]]) / 7,
            'full_predictions': predictions,
            'weights': weights
        }
    
    def generate_analysis_text(self, prediction_result, patterns, cycles):
        """Tạo text phân tích chi tiết"""
        lines = []
        
        # Xu hướng chính
        lines.append(f"📊 XU HƯỚNG: {patterns.get('trend_direction', 'chưa xác định').upper()}")
        
        # Phân tích vị trí
        lines.append("\n📍 PHÂN TÍCH VỊ TRÍ:")
        for pos_data in patterns.get('hot_positions', [])[:3]:
            pos = pos_data['position']
            hot = pos_data['hot_numbers']
            lines.append(f"  Vị trí {pos}: Số hot {', '.join(map(str, hot))}")
        
        # Chu kỳ phát hiện
        if cycles:
            lines.append(f"\n🔄 CHU KỲ PHÁT HIỆN:")
            for cycle in cycles[:2]:
                lines.append(f"  Chu kỳ {cycle['length']}: {cycle['pattern']} (độ tin cậy {cycle['confidence']*100:.0f}%)")
        
        # Đề xuất chiến thuật
        confidence = prediction_result.get('confidence', 0)
        if confidence > 0.7:
            lines.append(f"\n💪 CHIẾN THUẬT: Tự tin vào dàn 4 số chính")
        elif confidence > 0.5:
            lines.append(f"\n⚖️ CHIẾN THUẬT: Chia đều vốn cho cả 7 số")
        else:
            lines.append(f"\n⚠️ CHIẾN THUẬT: Thận trọng, chỉ đánh nhỏ lẻ")
        
        return '\n'.join(lines)

# ================= UI DESIGN TỐI ƯU =================
st.set_page_config(page_title="TITAN v22.0 PRO", layout="centered")

# CSS tối ưu
st.markdown("""
    <style>
    .stApp { background: #0a0c10; color: #e6edf3; }
    .status-active { 
        color: #2ea043; 
        font-weight: bold; 
        border-left: 4px solid #2ea043; 
        padding-left: 15px;
        background: rgba(46, 160, 67, 0.1);
        border-radius: 0 8px 8px 0;
    }
    .prediction-card {
        background: linear-gradient(145deg, #0d1117 0%, #161b22 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 30px;
        margin-top: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.8);
    }
    .num-display-main { 
        font-size: 70px; 
        font-weight: 900; 
        color: #58a6ff; 
        text-align: center; 
        letter-spacing: 15px; 
        text-shadow: 0 0 30px #58a6ff;
        line-height: 1.2;
    }
    .num-display-sub { 
        font-size: 50px; 
        font-weight: 700; 
        color: #f2cc60; 
        text-align: center; 
        letter-spacing: 10px; 
        text-shadow: 0 0 25px #f2cc60;
        line-height: 1.2;
    }
    .logic-box { 
        background: #161b22;
        border-left: 4px solid #58a6ff;
        border-radius: 8px;
        padding: 20px;
        margin: 20px 0;
        font-size: 15px;
        color: #8b949e;
        white-space: pre-line;
    }
    .confidence-badge {
        background: #1f6feb;
        color: white;
        padding: 8px 20px;
        border-radius: 30px;
        font-weight: bold;
        display: inline-block;
        font-size: 16px;
        box-shadow: 0 4px 12px rgba(31, 111, 235, 0.3);
    }
    .hot-number {
        background: rgba(46, 160, 67, 0.2);
        border: 1px solid #2ea043;
        color: #2ea043;
        padding: 5px 15px;
        border-radius: 25px;
        display: inline-block;
        margin: 3px;
        font-weight: bold;
    }
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 10px;
        margin: 15px 0;
    }
    .stat-cell {
        background: #161b22;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        border: 1px solid #30363d;
    }
    .stat-value {
        font-size: 24px;
        font-weight: bold;
        color: #58a6ff;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <h1 style='color: #58a6ff; font-size: 42px; margin-bottom: 5px;'>🧬 TITAN v22.0</h1>
        <p style='color: #8b949e; font-size: 14px;'>Hệ thống phân tích đa chiều | Độ chính xác được cải thiện</p>
    </div>
""", unsafe_allow_html=True)

# Status
if neural_engine:
    st.markdown(f"""
        <div class='status-active'>
            ● NEURAL-LINK: ONLINE | DỮ LIỆU: {len(st.session_state.history)} KỲ 
            | DỰ ĐOÁN: {len(st.session_state.predictions)}
        </div>
    """, unsafe_allow_html=True)
else:
    st.error("⚠️ LỖI KẾT NỐI API - KIỂM TRA LẠI KEY")

# ================= MAIN INTERFACE =================
raw_input = st.text_area(
    "📡 NHẬP DỮ LIỆU MỚI (mỗi dòng 1 số 5 chữ số):", 
    height=120, 
    placeholder="Ví dụ:\n32880\n21808\n69962\n...",
    key="input_area"
)

# Buttons
col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
with col1:
    analyze_btn = st.button("🚀 PHÂN TÍCH & DỰ ĐOÁN", use_container_width=True, type="primary")
with col2:
    reset_btn = st.button("🗑️ RESET", use_container_width=True)
with col3:
    history_btn = st.button("📜 LỊCH SỬ", use_container_width=True)
with col4:
    refresh_btn = st.button("🔄 LÀM MỚI", use_container_width=True)

# Xử lý buttons
if reset_btn:
    st.session_state.history = []
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    st.rerun()

if history_btn:
    st.session_state.show_predictions = not st.session_state.show_predictions
    st.rerun()

if refresh_btn:
    st.rerun()

# ================= XỬ LÝ PHÂN TÍCH CHÍNH =================
if analyze_btn:
    new_data = re.findall(r"\d{5}", raw_input)
    if new_data:
        # Thêm dữ liệu mới
        st.session_state.history.extend(new_data)
        save_memory(st.session_state.history)
        
        # Khởi tạo predictor
        predictor = TitanPredictorV22(st.session_state.history)
        
        # Phân tích patterns
        patterns = predictor.analyze_recent_patterns()
        cycles = predictor.detect_cycles()
        
        # Dự đoán kết hợp
        prediction = predictor.weighted_prediction()
        
        # Tạo text phân tích
        analysis_text = predictor.generate_analysis_text(prediction, patterns, cycles)
        
        # Gọi Gemini để tinh chỉnh
        try:
            gemini_prompt = f"""
            Bạn là chuyên gia phân tích số 5D. Dựa trên dữ liệu sau, hãy đưa ra dự đoán TỐI ƯU NHẤT:
            
            LỊCH SỬ 20 KỲ GẦN: {st.session_state.history[-20:]}
            
            PHÂN TÍCH THUẬT TOÁN:
            - Dự đoán 4 số chính: {prediction['dan4']}
            - Dự đoán 3 số lót: {prediction['dan3']}
            - Độ tin cậy: {prediction['confidence']*100:.1f}%
            
            XU HƯỚNG: {patterns.get('trend_direction', 'unknown')}
            
            YÊU CẦU: 
            1. Xác nhận hoặc điều chỉnh dàn số để tăng độ chính xác
            2. Giải thích ngắn gọn lý do
            3. Cảnh báo rủi ro nếu có
            
            TRẢ VỀ JSON:
            {{
                "dan4": ["4 số"],
                "dan3": ["3 số"],
                "phan_tich": "giải thích ngắn",
                "canh_bao": "cảnh báo hoặc 'không'",
                "xu_huong": "bệt/đảo/ổn định"
            }}
            """
            
            response = neural_engine.generate_content(gemini_prompt)
            res_text = response.text
            json_match = re.search(r'\{.*\}', res_text, re.DOTALL)
            
            if json_match:
                gemini_result = json.loads(json_match.group())
                # Kết hợp với kết quả thuật toán
                final_dan4 = gemini_result.get('dan4', prediction['dan4'])
                final_dan3 = gemini_result.get('dan3', prediction['dan3'])
                final_analysis = gemini_result.get('phan_tich', analysis_text)
                warning = gemini_result.get('canh_bao', 'không')
            else:
                final_dan4 = prediction['dan4']
                final_dan3 = prediction['dan3']
                final_analysis = analysis_text
                warning = "không"
                
        except Exception as e:
            final_dan4 = prediction['dan4']
            final_dan3 = prediction['dan3']
            final_analysis = analysis_text + f"\n\n⚠️ Gemini: {str(e)}"
            warning = "Lỗi kết nối AI, dùng thuật toán nội bộ"
        
        # Lưu kết quả
        result = {
            "dan4": final_dan4,
            "dan3": final_dan3,
            "logic": final_analysis,
            "canh_bao": warning,
            "confidence": prediction['confidence'],
            "time": datetime.now().strftime("%H:%M:%S"),
            "patterns": patterns,
            "cycles": cycles
        }
        
        st.session_state.last_result = result
        
        # Lưu vào lịch sử dự đoán
        save_prediction({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dan4": final_dan4,
            "dan3": final_dan3,
            "confidence": prediction['confidence'],
            "logic": final_analysis[:100] + "..."
        })
        
        st.rerun()

# ================= HIỂN THỊ KẾT QUẢ =================
if st.session_state.last_result:
    res = st.session_state.last_result
    confidence_pct = int(res.get('confidence', 0.5) * 100)
    
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    # Header với confidence
    st.markdown(f"""
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;'>
            <div>
                <span style='color: #8b949e;'>KẾT QUẢ PHÂN TÍCH</span>
                <span style='color: #58a6ff; margin-left: 10px;'>{res.get('time', '')}</span>
            </div>
            <div class='confidence-badge'>🔥 {confidence_pct}% TIN CẬY</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Cảnh báo nếu có
    if res.get('canh_bao') and res['canh_bao'] != 'không':
        st.warning(f"⚠️ {res['canh_bao']}")
    
    # Hiển thị 4 số chính
    st.markdown("""
        <div style='text-align: center; margin: 10px 0;'>
            <span style='background: #1f6feb20; color: #58a6ff; padding: 5px 15px; border-radius: 20px;'>
                🎯 4 SỐ CHỦ LỰC (ĐẶT CƯỢC CHÍNH)
            </span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"<div class='num-display-main'>{''.join(res['dan4'])}</div>", unsafe_allow_html=True)
    
    # Hiển thị 3 số lót
    st.markdown("""
        <div style='text-align: center; margin: 25px 0 10px 0;'>
            <span style='background: #f2cc6020; color: #f2cc60; padding: 5px 15px; border-radius: 20px;'>
                🛡️ 3 SỐ LÓT (ĐÁNH KÈM)
            </span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"<div class='num-display-sub'>{''.join(res['dan3'])}</div>", unsafe_allow_html=True)
    
    # Phân tích chi tiết
    st.markdown(f"<div class='logic-box'>{res['logic']}</div>", unsafe_allow_html=True)
    
    # Nút copy
    full_numbers = ''.join(res['dan4']) + ''.join(res['dan3'])
    st.text_input("📋 DÀN 7 SỐ:", full_numbers, key="copy_result")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ================= HIỂN THỊ LỊCH SỬ =================
if st.session_state.show_predictions:
    with st.expander("📜 LỊCH SỬ DỰ ĐOÁN (50 GẦN NHẤT)", expanded=True):
        predictions = load_predictions()
        if predictions:
            for pred in reversed(predictions[-20:]):
                conf_color = "#2ea043" if pred.get('confidence', 0) > 0.7 else "#f2cc60"
                st.markdown(f"""
                    <div style='background: #161b22; border-radius: 10px; padding: 15px; margin: 10px 0; border-left: 4px solid {conf_color};'>
                        <div style='display: flex; justify-content: space-between;'>
                            <span style='color: #8b949e;'>{pred['time']}</span>
                            <span style='color: {conf_color};'>⚡ {int(pred.get('confidence', 0)*100)}%</span>
                        </div>
                        <div style='font-size: 28px; letter-spacing: 8px; margin: 10px 0;'>
                            <span style='color: #58a6ff;'>{''.join(pred['dan4'])}</span>
                            <span style='color: #f2cc60;'>{''.join(pred['dan3'])}</span>
                        </div>
                        <div style='color: #8b949e; font-size: 13px;'>{pred['logic']}</div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Chưa có lịch sử dự đoán")

# Footer
st.markdown("""
    <div style='text-align: center; margin-top: 40px; padding: 20px; border-top: 1px solid #30363d;'>
        <p style='color: #484f58; font-size: 12px;'>
            🧬 TITAN v22.0 - Cải thiện độ chính xác bằng thuật toán đa tầng<br>
            ⚡ Phân tích vị trí | Chu kỳ | Tần suất | Machine Learning | Gemini AI
        </p>
    </div>
""", unsafe_allow_html=True)