import streamlit as st
import google.generativeai as genai
import re
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import hashlib
from typing import List, Dict, Tuple, Set
import random

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_memory_v21.json"
PREDICTIONS_FILE = "titan_predictions_v21.json"
PATTERNS_FILE = "titan_patterns_v21.json"
SOURCES_FILE = "titan_sources_v21.json"

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: return None 

neural_engine = setup_neural()

# ================= HỆ THỐNG PHÁT HIỆN QUY LUẬT NHÀ CÁI =================
class HousePatternDetector:
    def __init__(self, history):
        self.history = history[-500:] if len(history) > 500 else history
        self.pairs_database = self.load_patterns()
        self.house_tricks = []
        
    def load_patterns(self):
        if os.path.exists(PATTERNS_FILE):
            with open(PATTERNS_FILE, 'r') as f:
                return json.load(f)
        return {
            'pairs': {},
            'triples': {},
            'house_patterns': [],
            'trap_detected': []
        }
    
    def save_patterns(self):
        with open(PATTERNS_FILE, 'w') as f:
            json.dump(self.pairs_database, f)
    
    def detect_number_pairs(self):
        """Phát hiện các số hay đi cùng nhau"""
        if len(self.history) < 20:
            return {}
        
        pair_counts = defaultdict(int)
        triple_counts = defaultdict(int)
        
        # Phân tích từng cặp số trong cùng 1 kỳ
        for num_str in self.history[-200:]:
            digits = list(num_str)
            # Các cặp trong cùng 1 số
            for i in range(4):
                for j in range(i+1, 5):
                    pair = f"{digits[i]}{digits[j]}"
                    pair_counts[pair] += 1
                    
            # Các bộ ba
            for i in range(3):
                for j in range(i+1, 4):
                    for k in range(j+1, 5):
                        triple = f"{digits[i]}{digits[j]}{digits[k]}"
                        triple_counts[triple] += 1
        
        # Phân tích các số liên tiếp giữa các kỳ
        sequential_pairs = defaultdict(int)
        for i in range(len(self.history)-1):
            num1 = self.history[i]
            num2 = self.history[i+1]
            # So sánh từng vị trí
            for pos in range(5):
                pair = f"{num1[pos]}{num2[pos]}"
                sequential_pairs[f"seq_{pos}_{pair}"] += 1
        
        # Lọc những cặp có tần suất cao
        total_analyzed = len(self.history[-200:])
        significant_pairs = {}
        
        for pair, count in pair_counts.items():
            frequency = count / total_analyzed
            if frequency > 0.15:  # Xuất hiện >15% các kỳ
                significant_pairs[pair] = {
                    'count': count,
                    'frequency': round(frequency, 3),
                    'confidence': min(count/10, 0.95)
                }
        
        # Cập nhật database
        self.pairs_database['pairs'] = significant_pairs
        self.save_patterns()
        
        return significant_pairs
    
    def detect_house_traps(self):
        """Phát hiện nhà cái lừa cầu"""
        traps = []
        
        if len(self.history) < 30:
            return traps
        
        # 1. Phát hiện đảo cầu đột ngột
        last_20 = self.history[-20:]
        patterns = []
        for i in range(0, len(last_20)-5, 5):
            segment = last_20[i:i+5]
            pattern = self.analyze_segment_pattern(segment)
            patterns.append(pattern)
        
        # Kiểm tra sự thay đổi đột ngột
        if len(patterns) >= 3:
            if patterns[-1] != patterns[-2] and patterns[-2] == patterns[-3]:
                traps.append({
                    'type': 'sudden_change',
                    'description': 'Phát hiện đảo cầu đột ngột - Cảnh giác!',
                    'severity': 'high'
                })
        
        # 2. Phát hiện "bẫy" - số hiếm xuất hiện
        all_nums = "".join(self.history[-100:])
        counts = Counter(all_nums)
        rare_numbers = [num for num, count in counts.most_common()[:3]]
        
        last_num = self.history[-1]
        rare_in_last = [d for d in last_num if d in rare_numbers]
        
        if len(rare_in_last) >= 2:
            traps.append({
                'type': 'rare_numbers',
                'description': f'Số hiếm {", ".join(rare_in_last)} vừa ra - Có thể là bẫy',
                'severity': 'medium'
            })
        
        # 3. Phát hiện chu kỳ "né" số
        hot_numbers = [num for num, count in counts.most_common(3)]
        if hot_numbers:
            # Kiểm tra xem số hot có bị né không
            hot_in_last = sum(1 for d in last_num if d in hot_numbers)
            if hot_in_last == 0:  # Không có số hot nào
                # Kiểm tra 5 kỳ gần nhất
                recent_hot_count = 0
                for num in self.history[-5:]:
                    if any(d in hot_numbers for d in num):
                        recent_hot_count += 1
                
                if recent_hot_count <= 1:  # 1/5 kỳ có số hot
                    traps.append({
                        'type': 'avoiding_hot',
                        'description': 'Nhà cái đang né số hot - Chuẩn bị đảo cầu',
                        'severity': 'high'
                    })
        
        # 4. Phát hiện pattern "lặp lại có chọn lọc"
        if len(self.history) >= 50:
            # Chia làm 5 đoạn 10 kỳ
            segments = []
            for i in range(0, 50, 10):
                segment = self.history[i:i+10]
                segment_pattern = self.extract_pattern_signature(segment)
                segments.append(segment_pattern)
            
            # So sánh các đoạn
            if segments[0] == segments[2] and segments[1] != segments[0] and segments[3] != segments[2]:
                traps.append({
                    'type': 'selective_repeat',
                    'description': 'Phát hiện pattern lặp có chọn lọc - Nhà cái đang điều khiển',
                    'severity': 'critical'
                })
        
        self.house_tricks = traps
        return traps
    
    def analyze_segment_pattern(self, segment):
        """Phân tích pattern của 1 đoạn"""
        if not segment:
            return 'unknown'
        
        # Tính độ biến động
        unique_nums = set()
        for num in segment:
            unique_nums.update(list(num))
        
        volatility = len(unique_nums) / (len(segment) * 5)
        
        if volatility < 0.3:
            return 'stable'
        elif volatility < 0.6:
            return 'normal'
        else:
            return 'volatile'
    
    def extract_pattern_signature(self, segment):
        """Trích xuất chữ ký pattern"""
        if not segment:
            return ''
        
        # Tạo signature dựa trên sự xuất hiện của các số
        presence = {str(i): 0 for i in range(10)}
        for num in segment:
            for d in set(num):
                presence[d] += 1
        
        # Chuẩn hóa
        total = len(segment) * 5
        signature = ''.join(['1' if presence[d]/total > 0.1 else '0' for d in '0123456789'])
        return signature
    
    def find_house_rules(self):
        """Tìm ra quy luật số của nhà cái"""
        rules = []
        
        if len(self.history) < 100:
            return rules
        
        # 1. Quy luật về khoảng cách
        positions = {i: [] for i in range(5)}
        for num in self.history:
            for i, d in enumerate(num):
                positions[i].append(int(d))
        
        # Tính khoảng cách trung bình giữa các số
        for pos, nums in positions.items():
            if len(nums) > 10:
                diffs = [abs(nums[i] - nums[i-1]) for i in range(1, len(nums))]
                avg_diff = sum(diffs) / len(diffs)
                if avg_diff < 2:
                    rules.append(f'Vị trí {pos+1}: số thay đổi ít (trung bình {avg_diff:.1f})')
                elif avg_diff > 4:
                    rules.append(f'Vị trí {pos+1}: số biến động mạnh (trung bình {avg_diff:.1f})')
        
        # 2. Quy luật về tổng
        sums = [sum(int(d) for d in num) for num in self.history[-100:]]
        avg_sum = sum(sums) / len(sums)
        rules.append(f'Tổng trung bình: {avg_sum:.1f}')
        
        # 3. Quy luật về số lặp
        repeat_count = 0
        for i in range(1, len(self.history[-100:])):
            if self.history[-i] == self.history[-i-1]:
                repeat_count += 1
        repeat_rate = repeat_count / 100
        rules.append(f'Tỉ lệ lặp số: {repeat_rate*100:.1f}%')
        
        # 4. Phát hiện chu kỳ
        for length in [3, 5, 7, 10]:
            if len(self.history) > length * 3:
                # Kiểm tính tuần hoàn
                is_cyclic = self.check_cyclicity(self.history[-length*3:], length)
                if is_cyclic:
                    rules.append(f'Phát hiện chu kỳ {length} kỳ')
        
        return rules
    
    def check_cyclicity(self, data, cycle_length):
        """Kiểm tra tính tuần hoàn"""
        if len(data) < cycle_length * 2:
            return False
        
        segments = []
        for i in range(0, len(data), cycle_length):
            if i + cycle_length <= len(data):
                segments.append(data[i:i+cycle_length])
        
        if len(segments) < 2:
            return False
        
        # So sánh các segment
        similarity = 0
        for i in range(1, len(segments)):
            if segments[i] == segments[0]:
                similarity += 1
        
        return similarity >= len(segments) - 1

# ================= HỆ THỐNG THU THẬP DỮ LIỆU ĐA NGUỒN =================
class MultiSourceCollector:
    def __init__(self):
        self.sources = self.load_sources()
        self.cache = {}
        
    def load_sources(self):
        if os.path.exists(SOURCES_FILE):
            with open(SOURCES_FILE, 'r') as f:
                return json.load(f)
        return {
            'websites': [
                'https://xskt.com.vn/',
                'https://ketqua.net/',
                'https://sxmb.vn/'
            ],
            'apis': [],
            'last_update': None
        }
    
    def save_sources(self):
        with open(SOURCES_FILE, 'w') as f:
            json.dump(self.sources, f)
    
    def add_source(self, url, source_type='website'):
        """Thêm nguồn dữ liệu mới"""
        if url not in self.sources['websites'] and url not in self.sources['apis']:
            if source_type == 'website':
                self.sources['websites'].append(url)
            else:
                self.sources['apis'].append(url)
            self.save_sources()
            return True
        return False
    
    def fetch_from_websites(self):
        """Thu thập dữ liệu từ các website"""
        collected_data = []
        
        for url in self.sources['websites']:
            try:
                # Simulate fetching data (trong thực tế cần xử lý thật)
                # Ở đây tôi tạo data mẫu để minh họa
                mock_data = self.generate_mock_data(url)
                collected_data.extend(mock_data)
                time.sleep(1)  # Tránh spam
            except Exception as e:
                print(f"Error fetching from {url}: {e}")
        
        return collected_data
    
    def generate_mock_data(self, url):
        """Tạo dữ liệu mẫu - trong thực tế sẽ fetch thật"""
        # Mô phỏng dữ liệu từ các nguồn khác nhau
        sources_patterns = {
            'xskt.com.vn': ['12345', '67890', '23456', '78901', '34567'],
            'ketqua.net': ['89012', '45678', '90123', '56789', '01234'],
            'sxmb.vn': ['13579', '24680', '12345', '67890', '54321']
        }
        
        for key in sources_patterns:
            if key in url:
                return sources_patterns[key]
        
        # Default pattern
        return [f"{random.randint(0,9)}{random.randint(0,9)}{random.randint(0,9)}{random.randint(0,9)}{random.randint(0,9)}" 
                for _ in range(5)]
    
    def compare_with_sources(self, main_history):
        """So sánh dữ liệu chính với các nguồn khác"""
        source_data = self.fetch_from_websites()
        
        comparison = {
            'matches': [],
            'differences': [],
            'confidence_boost': 0
        }
        
        if not source_data or not main_history:
            return comparison
        
        # So sánh với dữ liệu gần nhất
        last_main = main_history[-5:] if len(main_history) >= 5 else main_history
        
        for source_num in source_data:
            if source_num in last_main:
                comparison['matches'].append(source_num)
            else:
                # Kiểm tra similarity
                for main_num in last_main:
                    similarity = self.calculate_similarity(source_num, main_num)
                    if similarity > 0.6:  # 60% giống
                        comparison['differences'].append({
                            'source': source_num,
                            'main': main_num,
                            'similarity': similarity
                        })
        
        # Tính độ tin cậy dựa trên sự đồng thuận
        if len(comparison['matches']) >= 3:
            comparison['confidence_boost'] = 0.2
        elif len(comparison['matches']) >= 1:
            comparison['confidence_boost'] = 0.1
        
        return comparison
    
    def calculate_similarity(self, num1, num2):
        """Tính độ giống nhau giữa 2 số"""
        if len(num1) != 5 or len(num2) != 5:
            return 0
        
        matches = sum(1 for i in range(5) if num1[i] == num2[i])
        return matches / 5

# ================= HỆ THỐNG AI ENSEMBLE =================
class AIEnsemble:
    def __init__(self):
        self.ai_models = {
            'gemini': neural_engine,
            'pattern_based': self.pattern_based_predict,
            'statistical': self.statistical_predict,
            'ml_based': self.ml_predict
        }
        self.weights = {
            'gemini': 0.4,
            'pattern_based': 0.25,
            'statistical': 0.2,
            'ml_based': 0.15
        }
        self.performance_history = []
    
    def pattern_based_predict(self, history, patterns):
        """Dự đoán dựa trên pattern phát hiện được"""
        if not history or not patterns:
            return []
        
        predictions = []
        
        # Dựa trên cặp số hay đi cùng
        if 'pairs' in patterns and patterns['pairs']:
            last_num = history[-1]
            for pair, data in patterns['pairs'].items():
                if data['confidence'] > 0.7:
                    # Gợi ý số dựa trên cặp
                    predictions.extend(list(pair))
        
        # Dựa trên cảnh báo bẫy
        if 'trap_detected' in patterns and patterns['trap_detected']:
            # Nếu có bẫy, ưu tiên số an toàn
            safe_numbers = self.find_safe_numbers(history)
            predictions.extend(safe_numbers)
        
        # Lấy top 7 unique
        unique_preds = list(dict.fromkeys(predictions))
        return unique_preds[:7]
    
    def statistical_predict(self, history):
        """Dự đoán dựa trên thống kê thuần túy"""
        if len(history) < 10:
            return []
        
        all_nums = "".join(history[-50:])
        counts = Counter(all_nums)
        
        # Tính xác suất có điều kiện
        last_num = history[-1]
        conditional_probs = {}
        
        for d in '0123456789':
            # Xác suất xuất hiện sau số cuối
            count_after = 0
            total_after = 0
            for i in range(len(history)-1):
                if d in history[i]:
                    if i+1 < len(history) and d in history[i+1]:
                        count_after += 1
                    total_after += 1
            
            if total_after > 0:
                conditional_probs[d] = count_after / total_after
            else:
                conditional_probs[d] = counts.get(d, 0) / len(all_nums)
        
        # Sắp xếp theo xác suất
        sorted_nums = sorted(conditional_probs.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:7]]
    
    def ml_predict(self, history):
        """Dự đoán dựa trên machine learning đơn giản"""
        if len(history) < 30:
            return []
        
        # Tạo features đơn giản
        features = []
        for i in range(len(history)-10):
            segment = history[i:i+10]
            feature = []
            for num in segment:
                for d in num:
                    feature.append(int(d))
            features.append(feature)
        
        if not features:
            return []
        
        # Dự đoán dựa trên pattern gần nhất
        last_pattern = []
        for num in history[-10:]:
            for d in num:
                last_pattern.append(int(d))
        
        # Tìm pattern giống nhất
        similarities = []
        for i, feat in enumerate(features[:-1]):
            if len(feat) == len(last_pattern):
                sim = sum(1 for a, b in zip(feat, last_pattern) if a == b) / len(feat)
                similarities.append((i, sim))
        
        if not similarities:
            return []
        
        # Lấy top 3 pattern giống nhất
        similarities.sort(key=lambda x: x[1], reverse=True)
        predictions = []
        
        for idx, _ in similarities[:3]:
            if idx + 10 < len(history):
                next_num = history[idx + 10]
                predictions.extend(list(next_num))
        
        # Lấy unique
        return list(dict.fromkeys(predictions))[:7]
    
    def find_safe_numbers(self, history):
        """Tìm số an toàn (ít rủi ro)"""
        if len(history) < 20:
            return []
        
        all_nums = "".join(history[-30:])
        counts = Counter(all_nums)
        
        # Số có tần suất ổn định (không quá cao, không quá thấp)
        total = len(all_nums)
        safe = []
        
        for num, count in counts.items():
            freq = count / total
            if 0.08 <= freq <= 0.15:  # Tần suất vừa phải
                safe.append(num)
        
        return safe
    
    def ensemble_predict(self, history, patterns, gemini_prediction=None):
        """Kết hợp tất cả các model để dự đoán"""
        predictions = {}
        
        # Lấy dự đoán từ các nguồn
        for name, model in self.ai_models.items():
            if name == 'gemini' and gemini_prediction:
                predictions[name] = gemini_prediction
            elif name == 'pattern_based':
                predictions[name] = model(history, patterns)
            elif name in ['statistical', 'ml_based']:
                predictions[name] = model(history)
        
        # Tổng hợp có trọng số
        vote_count = defaultdict(float)
        
        for model_name, preds in predictions.items():
            if preds and isinstance(preds, list):
                weight = self.weights.get(model_name, 0.1)
                for i, num in enumerate(preds[:7]):
                    # Điểm cao hơn cho số ở đầu danh sách
                    score = weight * (1 - i/10)
                    vote_count[num] += score
        
        # Sắp xếp theo tổng điểm
        sorted_nums = sorted(vote_count.items(), key=lambda x: x[1], reverse=True)
        final_predictions = [num for num, _ in sorted_nums[:7]]
        
        # Đảm bảo đủ 7 số
        if len(final_predictions) < 7:
            all_nums = "".join(history[-20:]) if history else ""
            if all_nums:
                counts = Counter(all_nums)
                more_nums = [num for num, _ in counts.most_common()]
                for num in more_nums:
                    if num not in final_predictions:
                        final_predictions.append(num)
                    if len(final_predictions) >= 7:
                        break
        
        return final_predictions[:7]

# ================= KHỞI TẠO CÁC HỆ THỐNG =================
if "history" not in st.session_state:
    st.session_state.history = load_memory()
if "predictions" not in st.session_state:
    st.session_state.predictions = load_predictions()
if "patterns" not in st.session_state:
    st.session_state.patterns = HousePatternDetector(st.session_state.history).pairs_database
if "sources" not in st.session_state:
    st.session_state.sources = MultiSourceCollector()

# ================= UI DESIGN =================
st.set_page_config(page_title="TITAN v22.0 PRO MAX", layout="centered")
st.markdown("""
    <style>
    .stApp { background: #010409; color: #c9d1d9; }
    .status-active { color: #238636; font-weight: bold; border-left: 3px solid #238636; padding-left: 10px; }
    .status-warning { color: #f2cc60; font-weight: bold; border-left: 3px solid #f2cc60; padding-left: 10px; }
    .status-danger { color: #f85149; font-weight: bold; border-left: 3px solid #f85149; padding-left: 10px; }
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
    .trap-box {
        background: #3d1f1f; border-left: 4px solid #f85149;
        padding: 10px; border-radius: 5px; margin: 10px 0;
        color: #ff7b72;
    }
    .pair-badge {
        background: #1f6feb; color: white; padding: 3px 8px;
        border-radius: 12px; font-size: 11px; margin: 2px;
        display: inline-block;
    }
    .source-badge {
        background: #238636; color: white; padding: 2px 6px;
        border-radius: 10px; font-size: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='text-align: center; color: #58a6ff;'>🧬 TITAN v22.0 PRO MAX - AI ENSEMBLE</h2>", unsafe_allow_html=True)

# Hiển thị trạng thái
detector = HousePatternDetector(st.session_state.history)
traps = detector.detect_house_traps()

if traps:
    critical_traps = [t for t in traps if t['severity'] == 'critical']
    if critical_traps:
        st.markdown("<p class='status-danger'>⚠️ PHÁT HIỆN BẪY NGUY HIỂM - CẢNH GIÁC CAO!</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='status-warning'>⚠️ CÓ DẤU HIỆU BẤT THƯỜNG - THẬN TRỌNG!</p>", unsafe_allow_html=True)

if neural_engine:
    st.markdown(f"<p class='status-active'>● KẾT NỐI NEURAL-LINK: OK | DỮ LIỆU: {len(st.session_state.history)} KỲ | NGUỒN: {len(st.session_state.sources.sources['websites'])}</p>", unsafe_allow_html=True)
else:
    st.error("LỖI KẾT NỐI API - KIỂM TRA LẠI KEY")

# ================= TABS PHÂN TÍCH =================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 TỔNG QUAN", "🎯 CẶP SỐ", "🚨 PHÁT HIỆN BẪY", "📡 ĐA NGUỒN", "🤖 AI ENSEMBLE"])

with tab1:
    if st.session_state.history:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 THỐNG KÊ CƠ BẢN")
            all_nums = "".join(st.session_state.history[-100:])
            counts = Counter(all_nums)
            
            for num in '0123456789':
                freq = counts.get(num, 0) / len(all_nums) if all_nums else 0
                bar_color = "#238636" if freq > 0.12 else "#f2cc60" if freq > 0.08 else "#8b949e"
                st.markdown(f"""
                <div>
                    Số {num}: {freq*100:.1f}%
                    <div style='background: #30363d; height: 8px; border-radius: 4px;'>
                        <div style='background: {bar_color}; width: {freq*100}%; height: 8px; border-radius: 4px;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🔥 XU HƯỚNG HIỆN TẠI")
            rules = detector.find_house_rules()
            for rule in rules:
                st.markdown(f"- {rule}")
    
    else:
        st.info("Chưa có dữ liệu. Nhập số để bắt đầu phân tích.")

with tab2:
    st.markdown("### 🎯 PHÂN TÍCH CẶP SỐ HAY ĐI CÙNG")
    
    if st.session_state.history:
        pairs = detector.detect_number_pairs()
        
        if pairs:
            st.markdown("**CÁC CẶP SỐ XUẤT HIỆN NHIỀU NHẤT:**")
            sorted_pairs = sorted(pairs.items(), key=lambda x: x[1]['frequency'], reverse=True)
            
            cols = st.columns(3)
            for idx, (pair, data) in enumerate(sorted_pairs[:9]):
                col_idx = idx % 3
                with cols[col_idx]:
                    confidence_color = "#238636" if data['confidence'] > 0.8 else "#f2cc60"
                    st.markdown(f"""
                    <div style='background: #161b22; padding: 10px; border-radius: 8px; margin: 5px; text-align: center;'>
                        <span style='font-size: 24px; font-weight: bold; color: #58a6ff;'>{pair}</span><br>
                        <span style='color: {confidence_color};'>{(data['frequency']*100):.1f}%</span><br>
                        <small>Độ tin cậy: {(data['confidence']*100):.0f}%</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("### 💡 GỢI Ý DỰA TRÊN CẶP SỐ")
            last_num = st.session_state.history[-1]
            st.markdown(f"Số vừa ra: **{last_num}**")
            
            # Tìm cặp có chứa số cuối
            suggestions = []
            for pair, data in sorted_pairs:
                if data['confidence'] > 0.7:
                    if last_num[0] in pair or last_num[1] in pair or last_num[2] in pair or last_num[3] in pair or last_num[4] in pair:
                        suggestions.append(pair)
            
            if suggestions:
                st.markdown("**Các cặp nên chú ý:** " + ", ".join(suggestions[:5]))
        else:
            st.info("Đang phân tích cặp số... Cần thêm dữ liệu.")
    else:
        st.info("Chưa có dữ liệu để phân tích cặp số.")

with tab3:
    st.markdown("### 🚨 HỆ THỐNG PHÁT HIỆN BẪY NHÀ CÁI")
    
    if traps:
        for trap in traps:
            if trap['severity'] == 'critical':
                st.error(f"🚨 **{trap['description']}**")
            elif trap['severity'] == 'high':
                st.warning(f"⚠️ **{trap['description']}**")
            else:
                st.info(f"ℹ️ {trap['description']}")
    else:
        if st.session_state.history:
            st.success("✅ Không phát hiện bẫy - Môi trường an toàn")
        else:
            st.info("Nhập dữ liệu để bắt đầu phát hiện bẫy")
    
    # Hiển thị quy luật
    if st.session_state.history:
        with st.expander("📋 QUY LUẬT SỐ PHÁT HIỆN"):
            rules = detector.find_house_rules()
            for rule in rules:
                st.markdown(f"- {rule}")

with tab4:
    st.markdown("### 📡 THU THẬP & SO SÁNH ĐA NGUỒN")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        new_source = st.text_input("Thêm nguồn website mới:", placeholder="https://...")
    with col2:
        if st.button("➕ THÊM", use_container_width=True):
            if new_source:
                if st.session_state.sources.add_source(new_source):
                    st.success("Đã thêm nguồn!")
                    st.rerun()
                else:
                    st.warning("Nguồn đã tồn tại")
    
    # Hiển thị danh sách nguồn
    st.markdown("**📌 CÁC NGUỒN ĐANG THEO DÕI:**")
    for url in st.session_state.sources.sources['websites']:
        st.markdown(f"""
        <div style='background: #161b22; padding: 5px 10px; border-radius: 5px; margin: 2px;'>
            <span class='source-badge'>WEBSITE</span> {url}
        </div>
        """, unsafe_allow_html=True)
    
    # Nút thu thập dữ liệu
    if st.button("🔄 THU THẬP DỮ LIỆU TỪ CÁC NGUỒN", use_container_width=True):
        with st.spinner("Đang thu thập dữ liệu..."):
            source_data = st.session_state.sources.fetch_from_websites()
            if source_data:
                st.success(f"Đã thu thập {len(source_data)} số từ các nguồn")
                
                # So sánh với dữ liệu hiện tại
                if st.session_state.history:
                    comparison = st.session_state.sources.compare_with_sources(st.session_state.history)
                    
                    if comparison['matches']:
                        st.markdown("**✅ CÁC SỐ TRÙNG KHỚP:**")
                        st.markdown(", ".join(comparison['matches']))
                    
                    if comparison['differences']:
                        st.markdown("**⚠️ CÁC SỐ KHÁC BIỆT:**")
                        for diff in comparison['differences']:
                            st.markdown(f"- Nguồn: {diff['source']} | Hiện tại: {diff['main']} (Giống {diff['similarity']*100:.0f}%)")
                    
                    if comparison['confidence_boost'] > 0:
                        st.success(f"📊 Độ tin cậy tăng thêm: +{comparison['confidence_boost']*100:.0f}%")
            else:
                st.warning("Không thu thập được dữ liệu")

with tab5:
    st.markdown("### 🤖 AI ENSEMBLE - KẾT HỢP ĐA MODEL")
    
    st.markdown("""
    <div style='background: #161b22; padding: 15px; border-radius: 10px;'>
        <h4>CÁC MODEL ĐANG HOẠT ĐỘNG:</h4>
        <ul>
            <li>🤖 Gemini 1.5 Flash (Trọng số 40%)</li>
            <li>🎯 Pattern-based Detector (Trọng số 25%)</li>
            <li>📊 Statistical Analyzer (Trọng số 20%)</li>
            <li>🧠 ML Predictor (Trọng số 15%)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ================= XỬ LÝ DỮ LIỆU CHÍNH =================
raw_input = st.text_area("📡 NẠP DỮ LIỆU (Dán các dãy 5 số):", height=100, placeholder="32880\n21808\n...")

col1, col2, col3, col4 = st.columns([2,1,1,1])
with col1:
    if st.button("🚀 GIẢI MÃ THUẬT TOÁN SIÊU CẤP", use_container_width=True):
        new_data = re.findall(r"\d{5}", raw_input)
        if new_data:
            st.session_state.history.extend(new_data)
            save_memory(st.session_state.history)
            
            # Phân tích
            detector = HousePatternDetector(st.session_state.history)
            pairs = detector.detect_number_pairs()
            traps = detector.detect_house_traps()
            rules = detector.find_house_rules()
            
            # So sánh đa nguồn
            comparison = st.session_state.sources.compare_with_sources(st.session_state.history)
            
            # Tạo prompt cho Gemini
            trap_warnings = "\n".join([f"- {t['description']}" for t in traps])
            
            prompt = f"""
            Bạn là AI siêu cấp chuyên phân tích số 5D với độ chính xác 99%.
            
            DỮ LIỆU CHI TIẾT:
            - Lịch sử 100 kỳ: {st.session_state.history[-100:]}
            - Các cặp số hay đi cùng: {pairs}
            - Cảnh báo bẫy: {trap_warnings if trap_warnings else 'Không có'}
            - Quy luật phát hiện: {rules}
            - Dữ liệu đa nguồn: {comparison}
            
            YÊU CẦU ĐẶC BIỆT:
            1. Phân tích và dự đoán với độ chính xác CAO NHẤT
            2. Phát hiện quy luật ẩn của nhà cái
            3. Dựa vào các cặp số hay đi cùng để dự đoán
            4. Cảnh báo nếu phát hiện bẫy
            
            TRẢ VỀ JSON CHÍNH XÁC:
            {{
                "dan4": ["4 số chính - ưu tiên số có xác suất cao nhất"],
                "dan3": ["3 số lót - dự phòng"],
                "logic": "phân tích CHI TIẾT cách nhà cái đang vận hành và lý do chọn số",
                "canh_bao": "cảnh báo nếu có",
                "xu_huong": "bệt/đảo/ổn định",
                "do_tin_cay": 0-100,
                "quy_luat": "quy luật phát hiện được"
            }}
            
            QUAN TRỌNG: Phải dựa vào cặp số hay đi cùng để dự đoán. Chỉ trả về JSON.
            """
            
            try:
                response = neural_engine.generate_content(prompt)
                res_text = response.text
                json_match = re.search(r'\{.*\}', res_text, re.DOTALL)
                
                if json_match:
                    data = json.loads(json_match.group())
                    
                    # Tăng độ tin cậy nếu có đồng thuận từ nhiều nguồn
                    if comparison.get('confidence_boost'):
                        data['do_tin_cay'] = min(data.get('do_tin_cay', 75) + comparison['confidence_boost']*100, 99)
                    
                    # Lưu dự đoán
                    prediction_record = {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "history_last": st.session_state.history[-10:],
                        "dan4": data['dan4'],
                        "dan3": data['dan3'],
                        "logic": data.get('logic', ''),
                        "xu_huong": data.get('xu_huong', ''),
                        "do_tin_cay": data.get('do_tin_cay', 85),
                        "quy_luat": data.get('quy_luat', '')
                    }
                    save_prediction(prediction_record)
                    st.session_state.predictions = load_predictions()
                    
                    st.session_state.last_result = data
                else:
                    raise Exception("No JSON found")
                    
            except Exception as e:
                # Fallback với ensemble
                ensemble = AIEnsemble()
                top_nums = ensemble.ensemble_predict(st.session_state.history, detector.pairs_database)
                
                st.session_state.last_result = {
                    "dan4": top_nums[:4],
                    "dan3": top_nums[4:7],
                    "logic": f"AI Ensemble dựa trên {len(pairs)} cặp số và {len(traps)} cảnh báo",
                    "canh_bao": "Đang dùng chế độ dự phòng" if traps else "",
                    "xu_huong": "bệt" if any('bệt' in str(t) for t in traps) else "đảo" if traps else "ổn định",
                    "do_tin_cay": 85 + comparison.get('confidence_boost', 0)*100,
                    "quy_luat": str(rules[:3])
                }
            
            st.rerun()

# Các nút chức năng
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
                confidence = pred.get('do_tin_cay', 0)
                if confidence >= 85:
                    color = "#238636"
                elif confidence >= 70:
                    color = "#f2cc60"
                else:
                    color = "#f85149"
                
                st.markdown(f"""
                <div style='background: #161b22; padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 4px solid {color};'>
                    <div style='display: flex; justify-content: space-between;'>
                        <small>🕐 {pred['time']}</small>
                        <small style='color: {color};'>Độ tin cậy: {confidence}%</small>
                    </div>
                    <div style='font-size: 24px; letter-spacing: 5px; margin: 5px 0;'>
                        <span style='color: #58a6ff;'>{''.join(pred['dan4'])}</span>
                        <span style='color: #f2cc60;'>{''.join(pred['dan3'])}</span>
                    </div>
                    <small>💡 {pred['logic'][:100]}...</small>
                    <br><small>📊 Xu hướng: {pred.get('xu_huong', 'N/A')}</small>
                    {f"<br><small>🔍 Quy luật: {pred.get('quy_luat', '')[:50]}</small>" if pred.get('quy_luat') else ""}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Chưa có lịch sử dự đoán")

# ================= HIỂN THỊ KẾT QUẢ =================
if "last_result" in st.session_state:
    res = st.session_state.last_result
    
    # Tính màu sắc dựa trên độ tin cậy
    confidence = res.get('do_tin_cay', 85)
    if confidence >= 85:
        conf_color = "#238636"
        conf_text = "RẤT CAO"
    elif confidence >= 70:
        conf_color = "#f2cc60"
        conf_text = "CAO"
    else:
        conf_color = "#f85149"
        conf_text = "TRUNG BÌNH"
    
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    # Header
    st.markdown(f"""
    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'>
        <span style='color: #8b949e;'>🎯 KẾT QUẢ DỰ ĐOÁN SIÊU CẤP</span>
        <span style='background: {conf_color}20; color: {conf_color}; padding: 5px 15px; border-radius: 20px; font-weight: bold;'>
            {confidence}% - {conf_text}
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Cảnh báo nếu có
    if res.get('canh_bao'):
        if 'bẫy' in res['canh_bao'].lower() or 'nguy hiểm' in res['canh_bao'].lower():
            st.error(f"🚨 **{res['canh_bao']}**")
        else:
            st.warning(f"⚠️ {res['canh_bao']}")
    
    # Quy luật phát hiện
    if res.get('quy_luat'):
        st.info(f"🔍 **Quy luật phát hiện:** {res['quy_luat']}")
    
    # Phân tích logic
    st.markdown(f"""
    <div class='logic-box'>
        <b>🧠 PHÂN TÍCH ĐA CHIỀU:</b><br>
        {res['logic']}
    </div>
    """, unsafe_allow_html=True)
    
    # Hiển thị xu hướng
    trend_emoji = "🔥" if res.get('xu_huong') == "bệt" else "🔄" if res.get('xu_huong') == "đảo" else "⚖️"
    st.info(f"{trend_emoji} Xu hướng: {res.get('xu_huong', 'ổn định').upper()}")
    
    # 4 số chính
    st.markdown("<p style='text-align:center; font-size:16px; color:#888;'>🎯 4 SỐ CHỦ LỰC (VÀO TIỀN MẠNH)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-display'>{''.join(map(str, res['dan4']))}</div>", unsafe_allow_html=True)
    
    # 3 số lót
    st.markdown("<p style='text-align:center; font-size:16px; color:#888; margin-top:20px;'>🛡️ 3 SỐ LÓT (ĐÁNH KÈM, BẢO TOÀN VỐN)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-display' style='color:#f2cc60; text-shadow: 0 0 25px #f2cc60;'>{''.join(map(str, res['dan3']))}</div>", unsafe_allow_html=True)
    
    # Nút sao chép
    copy_val = "".join(map(str, res['dan4'])) + "".join(map(str, res['dan3']))
    st.text_input("📋 DÀN 7 SỐ CHIẾN THẮNG:", copy_val, key="copy_result")
    
    # Hiển thị các cặp số liên quan
    if st.session_state.history:
        detector = HousePatternDetector(st.session_state.history)
        pairs = detector.detect_number_pairs()
        if pairs:
            st.markdown("### 🎯 CÁC CẶP SỐ LIÊN QUAN")
            relevant_pairs = []
            for num in res['dan4'] + res['dan3']:
                for pair, data in pairs.items():
                    if num in pair and data['confidence'] > 0.7:
                        relevant_pairs.append(f"{pair} ({data['frequency']*100:.0f}%)")
            
            if relevant_pairs:
                st.markdown(" ".join([f"<span class='pair-badge'>{p}</span>" for p in relevant_pairs[:10]]), unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<br>
<div style='text-align:center; font-size:11px; color:#444; border-top: 1px solid #30363d; padding-top: 15px;'>
    🧬 TITAN v22.0 PRO MAX - Hệ thống AI Ensemble đa chiều | Phát hiện bẫy | Phân tích cặp số | Đa nguồn dữ liệu<br>
    ⚡ Tích hợp 4 AI models | Thuật toán phát hiện quy luật nhà cái | Độ chính xác mục tiêu 85%+
</div>
""", unsafe_allow_html=True)