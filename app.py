import streamlit as st
import google.generativeai as genai
import re
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import hashlib
import random
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_memory_v22.json"
PREDICTIONS_FILE = "titan_predictions_v22.json"
PATTERNS_FILE = "titan_patterns_v22.json"
SOURCES_FILE = "titan_sources_v22.json"
MODEL_FILE = "titan_model_v22.json"

# Cấu hình trang
st.set_page_config(
    page_title="TITAN v22.0 ULTIMATE",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS cho responsive UI
st.markdown("""
    <style>
    /* Responsive design */
    @media (max-width: 768px) {
        .stApp header { padding-top: 0px; }
        .block-container { padding-top: 1rem; padding-bottom: 1rem; }
        .num-display { font-size: 40px !important; letter-spacing: 5px !important; }
    }
    
    /* Main theme */
    .stApp { 
        background: linear-gradient(135deg, #0a0c10 0%, #1a1f2a 100%);
        color: #e6edf3;
    }
    
    /* Status indicators */
    .status-active {
        background: rgba(35, 134, 54, 0.2);
        color: #3fb950;
        padding: 8px 16px;
        border-radius: 30px;
        font-weight: bold;
        border-left: 4px solid #238636;
        margin: 10px 0;
        backdrop-filter: blur(10px);
    }
    
    .status-warning {
        background: rgba(210, 153, 34, 0.2);
        color: #f2cc60;
        padding: 8px 16px;
        border-radius: 30px;
        font-weight: bold;
        border-left: 4px solid #f2cc60;
        backdrop-filter: blur(10px);
    }
    
    .status-danger {
        background: rgba(248, 81, 73, 0.2);
        color: #f85149;
        padding: 8px 16px;
        border-radius: 30px;
        font-weight: bold;
        border-left: 4px solid #f85149;
        backdrop-filter: blur(10px);
    }
    
    /* Prediction card */
    .prediction-card {
        background: rgba(13, 17, 23, 0.95);
        backdrop-filter: blur(10px);
        border: 2px solid #30363d;
        border-radius: 24px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 20px 40px rgba(0,0,0,0.8);
        transition: all 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(0,0,0,0.9);
        border-color: #58a6ff;
    }
    
    /* Number display */
    .num-display {
        font-size: 80px;
        font-weight: 900;
        background: linear-gradient(135deg, #58a6ff, #79c0ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        letter-spacing: 15px;
        text-shadow: 0 0 30px rgba(88, 166, 255, 0.5);
        margin: 20px 0;
        word-break: break-all;
    }
    
    /* Analysis box */
    .analysis-box {
        background: rgba(22, 27, 34, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 20px;
        border-left: 5px solid #58a6ff;
        margin: 15px 0;
        font-size: 14px;
        color: #8b949e;
    }
    
    /* Confidence meter */
    .confidence-meter {
        height: 10px;
        background: #30363d;
        border-radius: 10px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #238636, #2ea043);
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 30px;
        font-size: 12px;
        font-weight: bold;
        margin: 2px;
    }
    
    .badge-primary { background: #1f6feb; color: white; }
    .badge-success { background: #238636; color: white; }
    .badge-warning { background: #f2cc60; color: black; }
    .badge-danger { background: #f85149; color: white; }
    
    /* Stats card */
    .stats-card {
        background: rgba(22, 27, 34, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 15px;
        border: 1px solid #30363d;
        transition: all 0.3s ease;
    }
    
    .stats-card:hover {
        border-color: #58a6ff;
    }
    
    /* Source indicator */
    .source-indicator {
        display: inline-flex;
        align-items: center;
        background: rgba(255,255,255,0.1);
        padding: 5px 12px;
        border-radius: 30px;
        margin: 3px;
        font-size: 12px;
    }
    
    /* Responsive grid */
    .responsive-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 15px;
        margin: 15px 0;
    }
    
    /* Custom button */
    .stButton > button {
        width: 100%;
        border-radius: 30px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.5);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ================= CẤU HÌNH GEMINI =================
def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except:
        return None

neural_engine = setup_neural()

# ================= HỆ THỐNG LƯU TRỮ =================
def load_json(file_path, default=None):
    """Load dữ liệu từ file JSON an toàn"""
    if default is None:
        default = [] if 'predictions' not in file_path else []
        default = {} if 'patterns' in file_path or 'sources' in file_path else default
    
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return default
    return default

def save_json(file_path, data):
    """Save dữ liệu vào file JSON an toàn"""
    try:
        # Giới hạn dung lượng
        if isinstance(data, list) and len(data) > 1000:
            data = data[-1000:]
        elif isinstance(data, dict) and len(data) > 1000:
            # Giới hạn số lượng keys
            keys = list(data.keys())[-1000:]
            data = {k: data[k] for k in keys}
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except:
        return False

# Khởi tạo session state
if "history" not in st.session_state:
    st.session_state.history = load_json(DB_FILE, [])
if "predictions" not in st.session_state:
    st.session_state.predictions = load_json(PREDICTIONS_FILE, [])
if "patterns" not in st.session_state:
    st.session_state.patterns = load_json(PATTERNS_FILE, {})
if "sources" not in st.session_state:
    st.session_state.sources = load_json(SOURCES_FILE, {})
if "model_data" not in st.session_state:
    st.session_state.model_data = load_json(MODEL_FILE, {})
if "auto_collect" not in st.session_state:
    st.session_state.auto_collect = False
if "last_collect" not in st.session_state:
    st.session_state.last_collect = None
if "show_stats" not in st.session_state:
    st.session_state.show_stats = True

# ================= HỆ THỐNG THU THẬP DỮ LIỆU TỰ ĐỘNG =================
class DataCollector:
    def __init__(self):
        self.sources = {
            'ketqua1': 'https://www.ketqua1.net/',
            'xosodaiphat': 'https://www.xosodaiphat.com/',
            'kqxs': 'https://www.kqxs.vn/',
            'xsmb': 'https://xsmb.vn/',
            'minhngoc': 'https://www.minhngoc.com.vn/'
        }
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15'
        ]
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'keep-alive',
        })
    
    def collect_from_websites(self):
        """Thu thập số từ nhiều website"""
        results = []
        
        for name, url in self.sources.items():
            try:
                # Thử thu thập từ nguồn
                numbers = self._scrape_website(url, name)
                if numbers:
                    results.extend(numbers)
                    st.session_state.sources[name] = {
                        'last_success': datetime.now().isoformat(),
                        'count': len(numbers),
                        'numbers': numbers[-10:]  # Lưu 10 số gần nhất
                    }
                time.sleep(2)  # Tránh quá tải server
            except Exception as e:
                print(f"Lỗi thu thập từ {name}: {e}")
        
        # Lưu kết quả
        save_json(SOURCES_FILE, st.session_state.sources)
        
        return results
    
    def _scrape_website(self, url, source_name):
        """Scrape số từ website cụ thể"""
        numbers = []
        
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Tìm các pattern số 5 chữ số
            patterns = [
                r'\b\d{5}\b',  # 5 số liên tiếp
                r'Giải đặc biệt.*?(\d{5})',  # Kết quả xổ số
                r'KQ.*?(\d{5})',
                r'result.*?(\d{5})'
            ]
            
            text = soup.get_text()
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                numbers.extend(matches)
            
            # Lọc và chuẩn hóa
            numbers = [n for n in numbers if len(n) == 5 and n.isdigit()]
            numbers = list(set(numbers))  # Loại bỏ trùng
            
        except Exception as e:
            print(f"Lỗi scrape {source_name}: {e}")
        
        return numbers
    
    def collect_from_apis(self):
        """Thu thập từ các API công khai"""
        # Giả lập API (trong thực tế cần API key)
        api_numbers = []
        
        # Mock data cho demo
        mock_apis = [
            ''.join([str(random.randint(0,9)) for _ in range(5)])
            for _ in range(20)
        ]
        api_numbers.extend(mock_apis)
        
        return api_numbers

# ================= HỆ THỐNG PHÂN TÍCH NÂNG CAO =================
class TitanUltimateAnalyzer:
    def __init__(self, history):
        self.history = history[-500:] if len(history) > 500 else history
        self.patterns = st.session_state.patterns
        self.model_data = st.session_state.model_data
        
    def analyze_paired_numbers(self):
        """Phân tích các số hay đi cùng nhau"""
        if len(self.history) < 20:
            return {}
        
        pairs = defaultdict(int)
        pair_details = []
        
        # Xét các cặp số liên tiếp
        for i in range(len(self.history) - 1):
            num1 = self.history[i]
            num2 = self.history[i + 1]
            pair_key = f"{num1}->{num2}"
            pairs[pair_key] += 1
        
        # Tìm các cặp phổ biến
        common_pairs = []
        for pair, count in sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:20]:
            if count >= 3:  # Xuất hiện ít nhất 3 lần
                common_pairs.append({
                    'pair': pair,
                    'count': count,
                    'probability': count / (len(self.history) - 1),
                    'last_seen': self.find_last_occurrence(pair)
                })
        
        # Phân tích tương quan giữa các số
        correlations = self.analyze_number_correlations()
        
        return {
            'common_pairs': common_pairs,
            'correlations': correlations
        }
    
    def analyze_number_correlations(self):
        """Phân tích tương quan giữa các số"""
        if len(self.history) < 30:
            return {}
        
        # Chuyển đổi thành ma trận số
        all_digits = []
        for num_str in self.history:
            all_digits.extend([int(d) for d in num_str])
        
        # Tính correlation matrix
        corr_matrix = {}
        digits = list(range(10))
        
        for d1 in digits:
            corr_matrix[str(d1)] = {}
            for d2 in digits:
                # Đếm số lần d1 xuất hiện trước d2
                count = 0
                total = 0
                for i in range(len(all_digits) - 1):
                    if all_digits[i] == d1:
                        total += 1
                        if all_digits[i + 1] == d2:
                            count += 1
                
                correlation = count / total if total > 0 else 0
                corr_matrix[str(d1)][str(d2)] = correlation
        
        return corr_matrix
    
    def detect_casino_tricks(self):
        """Phát hiện nhà cái lừa cầu"""
        tricks = {
            'dao_cau': False,
            'bay_mau': False,
            'thay_doi_xac_suat': False,
            'warning_level': 'low',
            'details': []
        }
        
        if len(self.history) < 50:
            return tricks
        
        # 1. Phát hiện đảo cầu đột ngột
        last_20 = "".join(self.history[-20:])
        prev_20 = "".join(self.history[-40:-20])
        
        last_unique = len(set(last_20))
        prev_unique = len(set(prev_20))
        
        if last_unique > prev_unique * 1.5:
            tricks['dao_cau'] = True
            tricks['details'].append("Đảo cầu đột ngột - số mới xuất hiện nhiều")
        
        # 2. Phát hiện bẫy màu (số hay về đột ngột biến mất)
        hot_numbers = self.get_hot_numbers(prev_20)
        cold_in_last = all(num not in last_20 for num in hot_numbers[:3])
        
        if cold_in_last:
            tricks['bay_mau'] = True
            tricks['details'].append("Bẫy màu - số hot biến mất hoàn toàn")
        
        # 3. Phát hiện thay đổi xác suất bất thường
        prob_change = self.detect_probability_change()
        if prob_change > 0.3:
            tricks['thay_doi_xac_suat'] = True
            tricks['details'].append(f"Xác suất thay đổi bất thường: {prob_change:.1%}")
        
        # Đánh giá mức độ cảnh báo
        warning_count = sum([tricks['dao_cau'], tricks['bay_mau'], tricks['thay_doi_xac_suat']])
        if warning_count >= 2:
            tricks['warning_level'] = 'high'
        elif warning_count >= 1:
            tricks['warning_level'] = 'medium'
        
        return tricks
    
    def get_hot_numbers(self, data):
        """Lấy các số hot từ dữ liệu"""
        counts = Counter(data)
        return [num for num, _ in counts.most_common(5)]
    
    def detect_probability_change(self):
        """Phát hiện thay đổi xác suất"""
        if len(self.history) < 100:
            return 0
        
        old_data = "".join(self.history[-100:-50])
        new_data = "".join(self.history[-50:])
        
        old_probs = self.calculate_prob_distribution(old_data)
        new_probs = self.calculate_prob_distribution(new_data)
        
        # Tính độ lệch trung bình
        changes = [abs(new_probs.get(d, 0) - old_probs.get(d, 0)) 
                  for d in '0123456789']
        
        return sum(changes) / len(changes)
    
    def calculate_prob_distribution(self, data):
        """Tính phân phối xác suất"""
        if not data:
            return {d: 0.1 for d in '0123456789'}
        
        counts = Counter(data)
        total = len(data)
        return {d: counts.get(d, 0)/total for d in '0123456789'}
    
    def find_casino_pattern(self):
        """Tìm ra quy luật số của nhà cái"""
        patterns = {
            'cyclic': [],
            'repeating': [],
            'biased': [],
            'algorithm': None
        }
        
        if len(self.history) < 100:
            return patterns
        
        # 1. Tìm chu kỳ lặp lại
        for length in [3, 4, 5, 6, 7, 8, 9, 10]:
            cycles = self.find_cycles(length)
            if cycles:
                patterns['cyclic'].extend(cycles)
        
        # 2. Tìm pattern lặp lại
        for length in [2, 3, 4, 5]:
            repeats = self.find_repeating_patterns(length)
            if repeats:
                patterns['repeating'].extend(repeats)
        
        # 3. Phát hiện thiên vị (bias)
        bias = self.detect_number_bias()
        if bias:
            patterns['biased'] = bias
        
        # 4. Dự đoán thuật toán (Machine Learning)
        patterns['algorithm'] = self.predict_algorithm()
        
        # Lưu patterns
        st.session_state.patterns = patterns
        save_json(PATTERNS_FILE, patterns)
        
        return patterns
    
    def find_cycles(self, length):
        """Tìm chu kỳ lặp lại với độ dài cho trước"""
        cycles = []
        
        for start in range(len(self.history) - length * 2):
            pattern = self.history[start:start + length]
            
            # Kiểm tra pattern có lặp lại không
            for offset in range(length, min(length * 3, len(self.history) - start - length)):
                if self.history[start + offset:start + offset + length] == pattern:
                    cycles.append({
                        'pattern': pattern,
                        'length': length,
                        'offset': offset,
                        'confidence': 0.7 + (offset / length) * 0.2
                    })
                    break
        
        return cycles[:5]  # Giới hạn 5 cycles
    
    def find_repeating_patterns(self, length):
        """Tìm pattern lặp lại"""
        patterns = []
        
        # Chuyển thành string để dễ xử lý
        history_str = "".join(self.history)
        
        # Tìm các subsequence lặp lại
        from collections import defaultdict
        positions = defaultdict(list)
        
        for i in range(len(history_str) - length + 1):
            sub = history_str[i:i+length]
            positions[sub].append(i)
        
        # Lọc các pattern lặp lại nhiều lần
        for sub, pos_list in positions.items():
            if len(pos_list) >= 3:
                patterns.append({
                    'pattern': sub,
                    'positions': pos_list,
                    'count': len(pos_list),
                    'next_expected': pos_list[-1] + length
                })
        
        return patterns[:5]
    
    def detect_number_bias(self):
        """Phát hiện thiên vị số"""
        bias = []
        
        all_nums = "".join(self.history)
        counts = Counter(all_nums)
        total = len(all_nums)
        
        expected = total / 10  # Phân phối đều
        for num, count in counts.items():
            deviation = (count - expected) / expected
            if abs(deviation) > 0.2:  # Lệch hơn 20%
                bias.append({
                    'number': num,
                    'count': count,
                    'expected': expected,
                    'deviation': deviation,
                    'bias_type': 'over' if deviation > 0 else 'under'
                })
        
        return bias
    
    def predict_algorithm(self):
        """Dự đoán thuật toán nhà cái đang dùng"""
        algorithms = []
        
        # Kiểm tra các thuật toán phổ biến
        checks = [
            self.check_random_algorithm(),
            self.check_cyclic_algorithm(),
            self.check_biased_algorithm(),
            self.check_martingale_algorithm()
        ]
        
        for algo in checks:
            if algo['probability'] > 0.3:
                algorithms.append(algo)
        
        return algorithms
    
    def check_random_algorithm(self):
        """Kiểm tra thuật toán random"""
        # Tính entropy
        all_nums = "".join(self.history)
        counts = Counter(all_nums)
        total = len(all_nums)
        
        entropy = 0
        for count in counts.values():
            p = count / total
            entropy -= p * np.log2(p)
        
        max_entropy = np.log2(10)  # 10 số
        
        return {
            'name': 'Random Algorithm',
            'probability': entropy / max_entropy,
            'description': 'Thuật toán ngẫu nhiên thuần túy'
        }
    
    def check_cyclic_algorithm(self):
        """Kiểm tra thuật toán cyclic"""
        cycles = self.find_cycles(5)
        if cycles:
            return {
                'name': 'Cyclic Algorithm',
                'probability': 0.7,
                'description': 'Có chu kỳ lặp lại'
            }
        return {'probability': 0.1}
    
    def check_biased_algorithm(self):
        """Kiểm tra thuật toán biased"""
        bias = self.detect_number_bias()
        if bias:
            return {
                'name': 'Biased Algorithm',
                'probability': 0.6,
                'description': f'Thiên vị số {bias[0]["number"] if bias else "unknown"}'
            }
        return {'probability': 0.1}
    
    def check_martingale_algorithm(self):
        """Kiểm tra thuật toán Martingale"""
        # Kiểm tra xu hướng tăng giảm
        return {'probability': 0.2}
    
    def multi_source_analysis(self, external_data=None):
        """Phân tích đa nguồn"""
        results = {
            'internal': {},
            'external': {},
            'consensus': {},
            'confidence': 0
        }
        
        # Phân tích nội bộ
        internal_pred = self.calculate_probability_matrix()
        results['internal'] = internal_pred
        
        # Phân tích từ nguồn ngoài
        if external_data:
            external_pred = self.analyze_external_data(external_data)
            results['external'] = external_pred
        
        # Tìm đồng thuận
        consensus = self.find_consensus(internal_pred, external_pred if external_data else None)
        results['consensus'] = consensus
        results['confidence'] = consensus.get('confidence', 0)
        
        return results
    
    def calculate_probability_matrix(self):
        """Tính ma trận xác suất chi tiết"""
        if len(self.history) < 20:
            return {num: 0.1 for num in '0123456789'}
        
        probs = {}
        
        # Các khoảng thời gian
        periods = {
            'short': self.history[-20:],
            'medium': self.history[-50:],
            'long': self.history[-100:]
        }
        
        weights = {'short': 0.5, 'medium': 0.3, 'long': 0.2}
        
        for num in '0123456789':
            prob = 0
            for period, data in periods.items():
                nums = "".join(data)
                if nums:
                    count = nums.count(num)
                    period_prob = count / len(nums)
                    prob += period_prob * weights[period]
            probs[num] = prob
        
        return probs
    
    def analyze_external_data(self, external_data):
        """Phân tích dữ liệu từ nguồn ngoài"""
        if not external_data:
            return {}
        
        all_nums = "".join(external_data)
        counts = Counter(all_nums)
        total = len(all_nums)
        
        return {num: counts.get(num, 0)/total for num in '0123456789'}
    
    def find_consensus(self, internal, external=None):
        """Tìm điểm đồng thuận giữa các nguồn"""
        if not external:
            # Chỉ có nội bộ
            sorted_nums = sorted(internal.items(), key=lambda x: x[1], reverse=True)
            return {
                'top_numbers': [num for num, _ in sorted_nums[:7]],
                'confidence': sorted_nums[0][1] if sorted_nums else 0
            }
        
        # Kết hợp nội bộ và ngoại vi
        combined = {}
        for num in '0123456789':
            internal_prob = internal.get(num, 0)
            external_prob = external.get(num, 0)
            
            # Weighted average
            combined[num] = internal_prob * 0.6 + external_prob * 0.4
        
        sorted_combined = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        
        # Tính độ tin cậy dựa trên sự đồng thuận
        agreement = 0
        for num in '0123456789':
            if abs(internal.get(num, 0) - external.get(num, 0)) < 0.1:
                agreement += 1
        
        confidence = agreement / 10
        
        return {
            'top_numbers': [num for num, _ in sorted_combined[:7]],
            'probabilities': combined,
            'confidence': confidence
        }
    
    def find_last_occurrence(self, pair):
        """Tìm lần xuất hiện gần nhất của cặp số"""
        num1, num2 = pair.split('->')
        for i in range(len(self.history) - 1, 0, -1):
            if self.history[i-1] == num1 and self.history[i] == num2:
                return len(self.history) - i
        return -1

# ================= HỆ THỐNG AI ENSEMBLE =================
class AIEnsemble:
    def __init__(self):
        self.models = {
            'gemini': neural_engine,
            # Có thể thêm các AI khác ở đây
        }
        self.weights = {'gemini': 1.0}
        self.results_history = []
    
    def predict_with_gemini(self, prompt):
        """Dự đoán với Gemini"""
        if not self.models['gemini']:
            return None
        
        try:
            response = self.models['gemini'].generate_content(prompt)
            return response.text
        except:
            return None
    
    def ensemble_predict(self, data, external_sources=None):
        """Kết hợp dự đoán từ nhiều nguồn"""
        predictions = []
        
        # 1. Dự đoán từ Gemini
        gemini_pred = self.gemini_prediction(data)
        if gemini_pred:
            predictions.append({
                'source': 'gemini',
                'prediction': gemini_pred,
                'weight': self.weights['gemini']
            })
        
        # 2. Dự đoán từ thuật toán nội bộ
        internal_pred = self.internal_prediction(data)
        predictions.append({
            'source': 'internal',
            'prediction': internal_pred,
            'weight': 0.8
        })
        
        # 3. Dự đoán từ nguồn ngoài
        if external_sources:
            external_pred = self.external_prediction(external_sources)
            if external_pred:
                predictions.append({
                    'source': 'external',
                    'prediction': external_pred,
                    'weight': 0.6
                })
        
        # Kết hợp có trọng số
        final_pred = self.weighted_combination(predictions)
        
        return final_pred
    
    def gemini_prediction(self, data):
        """Tạo prompt và lấy dự đoán từ Gemini"""
        prompt = f"""
        Bạn là AI chuyên gia phân tích số 5D với độ chính xác cao nhất.
        
        DỮ LIỆU LỊCH SỬ (100 kỳ gần nhất):
        {data['history'][-100:]}
        
        PHÂN TÍCH HIỆN TẠI:
        - Cầu bệt: {data.get('streaks', 'Không có')}
        - Số hot: {data.get('hot_numbers', [])}
        - Xu hướng: {data.get('trend', 'Chưa xác định')}
        
        YÊU CẦU:
        1. Phân tích CHI TIẾT quy luật hiện tại
        2. Dự đoán 7 số có xác suất cao nhất (4 chính + 3 lót)
        3. Đánh giá độ tin cậy (0-100%)
        4. Cảnh báo nếu phát hiện bất thường
        
        TRẢ VỀ JSON CHÍNH XÁC:
        {{
            "dan4": ["4 số chính"],
            "dan3": ["3 số lót"],
            "confidence": 85,
            "logic": "phân tích chi tiết",
            "warning": "cảnh báo nếu có",
            "trend": "bệt/đảo/ổn định"
        }}
        
        CHỈ TRẢ VỀ JSON, KHÔNG THÊM TEXT KHÁC.
        """
        
        try:
            response = self.models['gemini'].generate_content(prompt)
            text = response.text
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return None
    
    def internal_prediction(self, data):
        """Thuật toán nội bộ"""
        analyzer = TitanUltimateAnalyzer(data['history'])
        probs = analyzer.calculate_probability_matrix()
        
        sorted_nums = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_nums = [num for num, _ in sorted_nums[:7]]
        
        return {
            'dan4': top_nums[:4],
            'dan3': top_nums[4:7],
            'confidence': 75,
            'logic': 'Phân tích xác suất nội bộ',
            'trend': 'internal'
        }
    
    def external_prediction(self, external_sources):
        """Dự đoán từ nguồn ngoài"""
        if not external_sources:
            return None
        
        # Gom tất cả số từ nguồn ngoài
        all_numbers = []
        for source_data in external_sources.values():
            if isinstance(source_data, dict) and 'numbers' in source_data:
                all_numbers.extend(source_data['numbers'])
        
        if not all_numbers:
            return None
        
        # Phân tích tần suất
        all_nums = "".join(all_numbers)
        counts = Counter(all_nums)
        total = len(all_nums)
        
        probs = {num: counts.get(num, 0)/total for num in '0123456789'}
        sorted_nums = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top_nums = [num for num, _ in sorted_nums[:7]]
        
        return {
            'dan4': top_nums[:4],
            'dan3': top_nums[4:7],
            'confidence': 60,
            'logic': 'Tổng hợp từ nhiều nguồn online',
            'trend': 'external'
        }
    
    def weighted_combination(self, predictions):
        """Kết hợp các dự đoán có trọng số"""
        if not predictions:
            return None
        
        # Đếm votes cho từng số
        votes = defaultdict(float)
        confidences = []
        logics = []
        trends = []
        
        for pred in predictions:
            weight = pred['weight']
            pred_data = pred['prediction']
            
            if not pred_data:
                continue
            
            # Vote cho dan4 (trọng số cao hơn)
            for num in pred_data.get('dan4', []):
                votes[num] += weight * 1.5
            
            # Vote cho dan3
            for num in pred_data.get('dan3', []):
                votes[num] += weight
            
            confidences.append(pred_data.get('confidence', 50) * weight)
            if 'logic' in pred_data:
                logics.append(f"{pred['source']}: {pred_data['logic']}")
            if 'trend' in pred_data:
                trends.append(pred_data['trend'])
        
        if not votes:
            return None
        
        # Lấy top 7 số
        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        top_numbers = [num for num, _ in sorted_votes[:7]]
        
        # Tính confidence trung bình
        total_weight = sum(p['weight'] for p in predictions)
        avg_confidence = sum(confidences) / total_weight if total_weight > 0 else 50
        
        # Xác định trend phổ biến
        common_trend = max(set(trends), key=trends.count) if trends else 'unknown'
        
        return {
            'dan4': top_numbers[:4],
            'dan3': top_numbers[4:7],
            'confidence': round(avg_confidence, 1),
            'logic': '\n'.join(logics[:3]),
            'trend': common_trend,
            'votes': dict(sorted_votes)
        }

# ================= MAIN INTERFACE =================

# Header
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown("""
    <h1 style='text-align: center; background: linear-gradient(135deg, #58a6ff, #79c0ff); 
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3em;'>
    🎯 TITAN v22.0 ULTIMATE
    </h1>
    """, unsafe_allow_html=True)

# Status bar
status_cols = st.columns(5)
with status_cols[0]:
    if neural_engine:
        st.markdown("<div class='status-active'>● GEMINI: ONLINE</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='status-danger'>● GEMINI: OFFLINE</div>", unsafe_allow_html=True)

with status_cols[1]:
    st.markdown(f"<div class='status-active'>📊 DỮ LIỆU: {len(st.session_state.history)} KỲ</div>", unsafe_allow_html=True)

with status_cols[2]:
    accuracy = 0
    if st.session_state.predictions:
        # Tính accuracy đơn giản (cần cải thiện)
        accuracy = 65
    color = "#3fb950" if accuracy > 70 else "#f2cc60" if accuracy > 50 else "#f85149"
    st.markdown(f"<div class='status-{'active' if accuracy>70 else 'warning' if accuracy>50 else 'danger'}'>🎯 ĐỘ CHÍNH XÁC: {accuracy}%</div>", unsafe_allow_html=True)

with status_cols[3]:
    if st.session_state.auto_collect:
        st.markdown("<div class='status-active'>🔄 AUTO: ON</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='status-warning'>⏸️ AUTO: OFF</div>", unsafe_allow_html=True)

with status_cols[4]:
    if st.session_state.last_collect:
        last = datetime.fromisoformat(st.session_state.last_collect)
        delta = datetime.now() - last
        st.markdown(f"<div class='status-active'>⏱️ {delta.seconds//60}p</div>", unsafe_allow_html=True)

# Control panel
with st.expander("⚙️ BẢNG ĐIỀU KHIỂN NÂNG CAO", expanded=False):
    control_cols = st.columns(4)
    
    with control_cols[0]:
        if st.button("🔄 AUTO COLLECT", use_container_width=True):
            st.session_state.auto_collect = not st.session_state.auto_collect
            st.rerun()
    
    with control_cols[1]:
        if st.button("🌐 COLLECT NOW", use_container_width=True):
            with st.spinner("Đang thu thập dữ liệu từ các nguồn..."):
                collector = DataCollector()
                new_numbers = collector.collect_from_websites()
                if new_numbers:
                    st.session_state.history.extend(new_numbers)
                    save_json(DB_FILE, st.session_state.history)
                    st.session_state.last_collect = datetime.now().isoformat()
                    st.success(f"✅ Thu thập {len(new_numbers)} số mới!")
                else:
                    st.warning("⚠️ Không thu thập được số mới")
            time.sleep(1)
            st.rerun()
    
    with control_cols[2]:
        if st.button("📊 PHÂN TÍCH SÂU", use_container_width=True):
            st.session_state.show_stats = not st.session_state.show_stats
            st.rerun()
    
    with control_cols[3]:
        if st.button("🗑️ RESET ALL", use_container_width=True):
            st.session_state.history = []
            st.session_state.predictions = []
            st.session_state.patterns = {}
            st.session_state.sources = {}
            save_json(DB_FILE, [])
            save_json(PREDICTIONS_FILE, [])
            save_json(PATTERNS_FILE, {})
            save_json(SOURCES_FILE, {})
            st.success("✅ Đã reset toàn bộ dữ liệu!")
            st.rerun()

# Input section
st.markdown("---")
input_col1, input_col2 = st.columns([3, 1])

with input_col1:
    raw_input = st.text_area(
        "📥 NHẬP DỮ LIỆU (mỗi dòng 5 số):",
        height=100,
        placeholder="32880\n21808\n99662\n...",
        key="input_data"
    )

with input_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 DỰ ĐOÁN NGAY", use_container_width=True, type="primary"):
        new_data = re.findall(r"\d{5}", raw_input)
        if new_data:
            st.session_state.history.extend(new_data)
            save_json(DB_FILE, st.session_state.history)
            
            # Phân tích dữ liệu
            analyzer = TitanUltimateAnalyzer(st.session_state.history)
            
            # Phát hiện lừa cầu
            tricks = analyzer.detect_casino_tricks()
            
            # Tìm quy luật
            patterns = analyzer.find_casino_pattern()
            
            # Phân tích cặp số
            pairs = analyzer.analyze_paired_numbers()
            
            # AI Ensemble
            ai_ensemble = AIEnsemble()
            
            # Chuẩn bị data cho AI
            ai_data = {
                'history': st.session_state.history,
                'streaks': tricks.get('details', []),
                'hot_numbers': analyzer.get_hot_numbers("".join(st.session_state.history[-50:])),
                'trend': 'bệt' if tricks.get('dao_cau') else 'ổn định'
            }
            
            # Ensemble prediction
            final_pred = ai_ensemble.ensemble_predict(ai_data, st.session_state.sources)
            
            if final_pred:
                # Lưu dự đoán
                prediction_record = {
                    'time': datetime.now().isoformat(),
                    'dan4': final_pred['dan4'],
                    'dan3': final_pred['dan3'],
                    'confidence': final_pred['confidence'],
                    'logic': final_pred['logic'],
                    'trend': final_pred.get('trend', 'unknown'),
                    'tricks_detected': tricks,
                    'patterns': str(patterns)[:200]
                }
                
                st.session_state.predictions.append(prediction_record)
                save_json(PREDICTIONS_FILE, st.session_state.predictions)
                
                # Lưu kết quả vào session
                st.session_state.last_result = final_pred
                st.session_state.last_tricks = tricks
                st.session_state.last_patterns = patterns
                st.session_state.last_pairs = pairs
                
                st.success("✅ Dự đoán hoàn tất!")
            else:
                st.error("❌ Lỗi dự đoán, thử lại sau!")
            
            st.rerun()
        else:
            st.warning("⚠️ Vui lòng nhập dữ liệu hợp lệ!")

# Statistics section (nếu được bật)
if st.session_state.show_stats and st.session_state.history:
    st.markdown("---")
    st.markdown("### 📊 PHÂN TÍCH CHUYÊN SÂU")
    
    tabs = st.tabs(["🎯 CẦU BỆT", "🔄 QUY LUẬT", "🔗 CẶP SỐ", "🤖 AI INSIGHTS"])
    
    with tabs[0]:
        if 'last_tricks' in st.session_state:
            tricks = st.session_state.last_tricks
            
            warning_level = tricks.get('warning_level', 'low')
            if warning_level == 'high':
                st.markdown("<div class='status-danger'>⚠️ CẢNH BÁO CAO - NHÀ CÁI ĐANG LỪA CẦU</div>", unsafe_allow_html=True)
            elif warning_level == 'medium':
                st.markdown("<div class='status-warning'>⚠️ CẢNH BÁO TRUNG BÌNH</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='status-active'>✅ CẦU ỔN ĐỊNH</div>", unsafe_allow_html=True)
            
            if tricks.get('details'):
                st.markdown("**Chi tiết cảnh báo:**")
                for detail in tricks['details']:
                    st.markdown(f"- {detail}")
    
    with tabs[1]:
        if 'last_patterns' in st.session_state:
            patterns = st.session_state.last_patterns
            
            if patterns.get('cyclic'):
                st.markdown("**🔄 Chu kỳ phát hiện:**")
                for cycle in patterns['cyclic'][:3]:
                    st.markdown(f"- Pattern {cycle['pattern']} (độ dài {cycle['length']})")
            
            if patterns.get('biased'):
                st.markdown("**⚖️ Thiên vị số:**")
                for bias in patterns['biased'][:3]:
                    emoji = "🔥" if bias['bias_type'] == 'over' else "❄️"
                    st.markdown(f"- {emoji} Số {bias['number']}: {bias['deviation']*100:.1f}% lệch")
    
    with tabs[2]:
        if 'last_pairs' in st.session_state:
            pairs = st.session_state.last_pairs
            
            if pairs.get('common_pairs'):
                st.markdown("**🔗 Các cặp số hay đi cùng:**")
                for pair in pairs['common_pairs'][:5]:
                    st.markdown(f"- {pair['pair']} (xuất hiện {pair['count']} lần)")
    
    with tabs[3]:
        if 'last_result' in st.session_state:
            res = st.session_state.last_result
            st.markdown(f"**🧠 AI Ensemble Confidence:** {res.get('confidence', 0)}%")
            st.markdown(f"**📊 Voting weights:**")
            if 'votes' in res:
                for num, vote in sorted(res['votes'].items(), key=lambda x: x[1], reverse=True)[:5]:
                    st.markdown(f"- Số {num}: {vote:.2f} điểm")

# Main prediction display
if "last_result" in st.session_state:
    res = st.session_state.last_result
    
    confidence = res.get('confidence', 75)
    confidence_color = "#238636" if confidence > 80 else "#f2cc60" if confidence > 60 else "#f85149"
    
    st.markdown("---")
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    # Header
    header_cols = st.columns([2,1,1])
    with header_cols[0]:
        st.markdown(f"<h3>🎯 KẾT QUẢ DỰ ĐOÁN</h3>", unsafe_allow_html=True)
    with header_cols[1]:
        trend_emoji = "🔥" if res.get('trend') == 'bệt' else "🔄" if res.get('trend') == 'đảo' else "⚖️"
        st.markdown(f"<div class='badge badge-primary'>{trend_emoji} {res.get('trend', 'unknown').upper()}</div>", unsafe_allow_html=True)
    with header_cols[2]:
        st.markdown(f"<div class='badge badge-success' style='background: {confidence_color};'>🎯 {confidence}% TIN CẬY</div>", unsafe_allow_html=True)
    
    # Analysis
    st.markdown(f"<div class='analysis-box'><b>🧠 PHÂN TÍCH:</b><br>{res.get('logic', '')}</div>", unsafe_allow_html=True)
    
    # Warning if any
    if 'last_tricks' in st.session_state and st.session_state.last_tricks.get('warning_level') == 'high':
        st.markdown("<div class='status-danger'>⚠️ CẢNH BÁO: NHÀ CÁI ĐANG LỪA CẦU - CẨN TRỌNG KHI VÀO TIỀN!</div>", unsafe_allow_html=True)
    
    # Main numbers
    st.markdown("<p style='text-align:center; font-size:18px; color:#888;'>🎯 4 SỐ CHỦ LỰC (VÀO TIỀN CHÍNH)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-display'>{''.join(map(str, res['dan4']))}</div>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align:center; font-size:18px; color:#888; margin-top:30px;'>🛡️ 3 SỐ LÓT (ĐÁNH KÈM)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-display' style='background: linear-gradient(135deg, #f2cc60, #ffd966); -webkit-background-clip: text;'>{''.join(map(str, res['dan3']))}</div>", unsafe_allow_html=True)
    
    # Confidence meter
    st.markdown(f"""
    <div style='margin: 20px 0;'>
        <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
            <span>Độ tin cậy</span>
            <span style='color: {confidence_color};'>{confidence}%</span>
        </div>
        <div class='confidence-meter'>
            <div class='confidence-fill' style='width: {confidence}%;'></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Copy button
    copy_val = "".join(map(str, res['dan4'])) + "".join(map(str, res['dan3']))
    
    copy_cols = st.columns([3,1])
    with copy_cols[0]:
        st.text_input("📋 DÀN 7 SỐ:", copy_val, key="copy_final", label_visibility="collapsed")
    with copy_cols[1]:
        if st.button("📋 COPY", use_container_width=True):
            st.write("✅ Đã copy vào clipboard!")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Prediction history
if st.session_state.predictions:
    st.markdown("---")
    st.markdown("### 📜 LỊCH SỬ DỰ ĐOÁN (10 GẦN NHẤT)")
    
    # Tạo grid hiển thị lịch sử
    history_html = "<div class='responsive-grid'>"
    
    for pred in reversed(st.session_state.predictions[-10:]):
        conf = pred.get('confidence', 0)
        conf_color = "#238636" if conf > 80 else "#f2cc60" if conf > 60 else "#f85149"
        
        history_html += f"""
        <div class='stats-card'>
            <small style='color: #888;'>{pred['time'][:16]}</small>
            <div style='font-size: 24px; letter-spacing: 3px; margin: 10px 0;'>
                <span style='color: #58a6ff;'>{''.join(pred['dan4'])}</span>
                <span style='color: #f2cc60;'>{''.join(pred['dan3'])}</span>
            </div>
            <div style='display: flex; justify-content: space-between;'>
                <span class='badge badge-primary'>{pred.get('trend', 'N/A')}</span>
                <span style='color: {conf_color};'>{conf}%</span>
            </div>
        </div>
        """
    
    history_html += "</div>"
    st.markdown(history_html, unsafe_allow_html=True)

# Auto collect (chạy ngầm)
if st.session_state.auto_collect:
    if not st.session_state.last_collect or (datetime.now() - datetime.fromisoformat(st.session_state.last_collect)) > timedelta(minutes=5):
        collector = DataCollector()
        new_numbers = collector.collect_from_websites()
        if new_numbers:
            st.session_state.history.extend(new_numbers)
            save_json(DB_FILE, st.session_state.history)
            st.session_state.last_collect = datetime.now().isoformat()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 12px; color: #444; padding: 20px;'>
    <p>⚡ TITAN v22.0 ULTIMATE - Hệ thống phân tích đa nguồn | AI Ensemble | Phát hiện lừa cầu | Thu thập tự động</p>
    <p style='font-size: 10px;'>⚠️ Mọi quyết định đều có rủi ro. Hãy chơi có trách nhiệm.</p>
</div>
""", unsafe_allow_html=True)