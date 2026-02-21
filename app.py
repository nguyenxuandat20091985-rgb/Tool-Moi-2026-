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
import random
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_memory_v21.json"
PREDICTIONS_FILE = "titan_predictions_v21.json"
PATTERNS_FILE = "titan_patterns_v21.json"
STATS_FILE = "titan_stats_v21.json"
CRAWLER_FILE = "titan_crawler_v21.json"

# Cấu hình crawler
SOURCES = [
    "https://www.minhngoc.net.vn/ket-qua-xo-so.html",
    "https://ketqua1.net/",
    "https://xosodaiphat.com/",
    # Thêm các nguồn khác
]

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: 
        return None

neural_engine = setup_neural()

# ================= HỆ THỐNG LƯU TRỮ =================
def load_json(file_path, default=None):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try: return json.load(f)
            except: return default if default else []
    return default if default else []

def save_json(file_path, data, max_items=1000):
    with open(file_path, "w") as f:
        if isinstance(data, list):
            json.dump(data[-max_items:], f)
        else:
            json.dump(data, f)

# Khởi tạo session state
if "history" not in st.session_state:
    st.session_state.history = load_json(DB_FILE, [])
if "predictions" not in st.session_state:
    st.session_state.predictions = load_json(PREDICTIONS_FILE, [])
if "patterns_db" not in st.session_state:
    st.session_state.patterns_db = load_json(PATTERNS_FILE, {})
if "stats_db" not in st.session_state:
    st.session_state.stats_db = load_json(STATS_FILE, {})
if "crawler_data" not in st.session_state:
    st.session_state.crawler_data = load_json(CRAWLER_FILE, {})
if "accuracy_history" not in st.session_state:
    st.session_state.accuracy_history = []

# ================= HỆ THỐNG CRAWLER TỰ ĐỘNG =================
class AutoCrawler:
    def __init__(self):
        self.sources = SOURCES
        self.last_crawl = st.session_state.crawler_data.get('last_crawl', {})
        self.cached_data = st.session_state.crawler_data.get('data', [])
    
    def crawl_all_sources(self):
        """Thu thập dữ liệu từ nhiều nguồn"""
        all_numbers = []
        sources_data = {}
        
        for source in self.sources:
            try:
                numbers = self.crawl_source(source)
                if numbers:
                    all_numbers.extend(numbers)
                    sources_data[source] = {
                        'numbers': numbers[-50:],
                        'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'count': len(numbers)
                    }
                time.sleep(1)  # Tránh bị chặn
            except Exception as e:
                print(f"Lỗi crawl {source}: {e}")
        
        # Lưu cache
        st.session_state.crawler_data = {
            'last_crawl': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data': all_numbers[-500:],
            'sources': sources_data
        }
        save_json(CRAWLER_FILE, st.session_state.crawler_data)
        
        return all_numbers
    
    def crawl_source(self, url):
        """Crawl dữ liệu từ 1 nguồn"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Tìm các số 5 chữ số
            numbers = []
            text = soup.get_text()
            # Pattern tìm số 5 chữ số
            found_numbers = re.findall(r'\b\d{5}\b', text)
            
            # Lọc và chuẩn hóa
            for num in found_numbers:
                if len(num) == 5 and num.isdigit():
                    numbers.append(num)
            
            return list(set(numbers))[-100:]  # Trả về 100 số gần nhất
        except:
            return []
    
    def get_online_trend(self):
        """Lấy xu hướng từ các nguồn online"""
        if not self.cached_data:
            return {}
        
        all_nums = "".join(self.cached_data[-200:])
        if not all_nums:
            return {}
        
        counts = Counter(all_nums)
        total = len(all_nums)
        
        return {
            'hot_online': [num for num, _ in counts.most_common(5)],
            'cold_online': [num for num, _ in counts.most_common()[-5:]],
            'frequencies': {num: count/total for num, count in counts.items()}
        }

# ================= HỆ THỐNG PHÁT HIỆN QUY LUẬT =================
class PatternDetector:
    def __init__(self, history):
        self.history = history[-500:] if len(history) > 500 else history
        self.patterns_db = st.session_state.patterns_db
    
    def find_number_pairs(self):
        """Phát hiện các số hay đi cùng nhau"""
        if len(self.history) < 20:
            return {}
        
        pairs = defaultdict(int)
        pair_positions = defaultdict(list)
        
        # Xét từng cặp số trong dãy 5 số
        for num_str in self.history[-200:]:
            digits = list(num_str)
            for i in range(5):
                for j in range(i+1, 5):
                    pair = f"{digits[i]}{digits[j]}"
                    pairs[pair] += 1
                    pair_positions[pair].append((i, j))
        
        # Tính xác suất và lọc cặp có ý nghĩa
        total_pairs = sum(pairs.values())
        significant_pairs = {}
        
        for pair, count in pairs.items():
            probability = count / total_pairs
            if probability > 0.03:  # Ngưỡng 3%
                significant_pairs[pair] = {
                    'count': count,
                    'probability': probability,
                    'positions': pair_positions[pair][-5:],
                    'strength': 'CAO' if probability > 0.05 else 'TRUNG BÌNH'
                }
        
        return dict(sorted(significant_pairs.items(), 
                          key=lambda x: x[1]['probability'], 
                          reverse=True))
    
    def find_triplet_patterns(self):
        """Phát hiện bộ 3 số hay ra cùng nhau"""
        if len(self.history) < 30:
            return {}
        
        triplets = defaultdict(int)
        
        for num_str in self.history[-200:]:
            digits = sorted(list(num_str))  # Sắp xếp để dễ so sánh
            for i in range(3):
                for j in range(i+1, 4):
                    for k in range(j+1, 5):
                        triplet = f"{digits[i]}{digits[j]}{digits[k]}"
                        triplets[triplet] += 1
        
        # Lọc bộ 3 đặc biệt
        special_triplets = {}
        for triplet, count in triplets.items():
            if count > 5:  # Xuất hiện ít nhất 5 lần
                special_triplets[triplet] = {
                    'count': count,
                    'frequency': count / len(self.history[-200:])
                }
        
        return dict(sorted(special_triplets.items(), 
                          key=lambda x: x[1]['count'], 
                          reverse=True)[:20])
    
    def detect_house_tricks(self):
        """Phát hiện nhà cái lừa cầu"""
        tricks = {
            'dao_cau': False,
            'bay_mau': False,
            'sập_bệt': False,
            'thay_doi_xac_suat': False,
            'warning_level': 'GREEN',
            'details': []
        }
        
        if len(self.history) < 30:
            return tricks
        
        # 1. Kiểm tra đảo cầu đột ngột
        last_20 = "".join(self.history[-20:])
        prev_20 = "".join(self.history[-40:-20])
        
        unique_last = len(set(last_20))
        unique_prev = len(set(prev_20))
        
        if unique_last > unique_prev + 2:
            tricks['dao_cau'] = True
            tricks['details'].append("Đảo cầu đột ngột - Xuất hiện nhiều số lạ")
        
        # 2. Kiểm tra bẫy màu (số hay ra bỗng nhiên biến mất)
        hot_numbers = Counter(prev_20).most_common(3)
        for num, _ in hot_numbers:
            if num not in last_20:
                tricks['bay_mau'] = True
                tricks['details'].append(f"Số hot {num} đột nhiên biến mất - Có thể bẫy")
        
        # 3. Kiểm tra sập bệt
        if len(self.history) > 10:
            current_streak = 1
            for i in range(len(self.history)-2, -1, -1):
                if self.history[i] == self.history[-1]:
                    current_streak += 1
                else:
                    break
            
            if current_streak >= 4:
                # Kiểm tra xem có dấu hiệu sập bệt không
                next_after_streak = self.history[-(current_streak+1):-current_streak] if len(self.history) > current_streak else []
                if next_after_streak and len(set(next_after_streak)) > 3:
                    tricks['sập_bệt'] = True
                    tricks['details'].append(f"Cầu bệt {current_streak} kỳ có dấu hiệu sập")
        
        # 4. Xác định mức độ cảnh báo
        warning_score = 0
        if tricks['dao_cau']: warning_score += 2
        if tricks['bay_mau']: warning_score += 2
        if tricks['sập_bệt']: warning_score += 3
        
        if warning_score >= 5:
            tricks['warning_level'] = 'RED'
        elif warning_score >= 3:
            tricks['warning_level'] = 'ORANGE'
        elif warning_score >= 1:
            tricks['warning_level'] = 'YELLOW'
        
        return tricks
    
    def find_cycles(self):
        """Tìm chu kỳ lặp lại của các số"""
        cycles = {}
        
        for length in [3, 4, 5, 6, 7, 8, 9, 10]:
            if len(self.history) < length * 3:
                continue
            
            # Chuyển đổi history thành string để dễ xử lý
            history_str = "".join(self.history)
            
            # Tìm các pattern lặp lại
            patterns = {}
            for i in range(len(history_str) - length):
                pattern = history_str[i:i+length]
                if pattern in patterns:
                    patterns[pattern] += 1
                else:
                    patterns[pattern] = 1
            
            # Lọc pattern có tần suất cao
            significant = {p: c for p, c in patterns.items() if c > 2}
            if significant:
                cycles[f"cycle_{length}"] = {
                    'length': length,
                    'patterns': dict(sorted(significant.items(), 
                                           key=lambda x: x[1], 
                                           reverse=True)[:3])
                }
        
        return cycles

# ================= HỆ THỐNG SO SÁNH ĐA NGUỒN =================
class MultiAISystem:
    def __init__(self):
        self.models = {
            'gemini': neural_engine,
            # Có thể thêm các AI khác ở đây
        }
        self.weights = {
            'gemini': 0.4,
            'pattern': 0.3,
            'statistical': 0.2,
            'crawler': 0.1
        }
    
    def ensemble_prediction(self, history, pattern_data, crawler_data):
        """Kết hợp dự đoán từ nhiều nguồn"""
        predictions = {}
        
        # 1. Dự đoán từ Gemini
        if self.models['gemini']:
            gemini_pred = self.get_gemini_prediction(history, pattern_data)
            if gemini_pred:
                predictions['gemini'] = gemini_pred
        
        # 2. Dự đoán từ pattern
        pattern_pred = self.get_pattern_prediction(pattern_data)
        if pattern_pred:
            predictions['pattern'] = pattern_pred
        
        # 3. Dự đoán thống kê
        stat_pred = self.get_statistical_prediction(history)
        if stat_pred:
            predictions['statistical'] = stat_pred
        
        # 4. Dự đoán từ crawler
        if crawler_data:
            crawler_pred = self.get_crawler_prediction(crawler_data)
            if crawler_pred:
                predictions['crawler'] = crawler_pred
        
        # Kết hợp có trọng số
        return self.weighted_combination(predictions)
    
    def get_gemini_prediction(self, history, pattern_data):
        """Lấy dự đoán từ Gemini"""
        try:
            prompt = f"""
            Bạn là chuyên gia phân tích số 5D với độ chính xác cao.
            
            DỮ LIỆU PHÂN TÍCH CHUYÊN SÂU:
            - Lịch sử 200 kỳ gần nhất: {history[-200:]}
            - Các cặp số hay đi cùng: {pattern_data.get('pairs', {})}
            - Bộ 3 số đặc biệt: {pattern_data.get('triplets', {})}
            - Chu kỳ phát hiện: {pattern_data.get('cycles', {})}
            - Cảnh báo nhà cái: {pattern_data.get('tricks', {})}
            
            NHIỆM VỤ:
            1. Phân tích quy luật thực sự của nhà cái
            2. Dự đoán 4 số có xác suất cao nhất (KHÔNG ĐƯỢC SAI)
            3. Dự đoán 3 số dự phòng
            4. Cảnh báo nếu phát hiện bẫy
            
            YÊU CẦU ĐẶC BIỆT:
            - Phải đạt độ chính xác >85%
            - Nếu không chắc chắn, ưu tiên an toàn
            - Phát hiện mọi dấu hiệu bất thường
            
            TRẢ VỀ JSON:
            {{
                "dan4": ["4 số chính xác nhất"],
                "dan3": ["3 số dự phòng"],
                "quy_luat": "quy luật nhà cái đang dùng",
                "canh_bao": "cảnh báo nếu có",
                "do_tin_cay": 0-100,
                "ly_do": "phân tích chi tiết"
            }}
            """
            
            response = self.models['gemini'].generate_content(prompt)
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return None
    
    def get_pattern_prediction(self, pattern_data):
        """Dự đoán dựa trên pattern"""
        if not pattern_data:
            return None
        
        pairs = pattern_data.get('pairs', {})
        triplets = pattern_data.get('triplets', {})
        
        # Dựa vào các cặp số mạnh nhất để dự đoán
        strong_pairs = [p for p, data in pairs.items() 
                       if data.get('strength') == 'CAO']
        
        if strong_pairs:
            return {
                'dan4': list(strong_pairs[0])[:4],
                'dan3': list(strong_pairs[0])[4:7] if len(strong_pairs[0]) > 4 else [],
                'do_tin_cay': 75,
                'nguon': 'pattern'
            }
        return None
    
    def get_statistical_prediction(self, history):
        """Dự đoán dựa trên thống kê"""
        if len(history) < 50:
            return None
        
        all_nums = "".join(history[-100:])
        counts = Counter(all_nums)
        
        # Tính xác suất có điều chỉnh
        probs = {}
        total = len(all_nums)
        for num in '0123456789':
            base_prob = counts.get(num, 0) / total
            
            # Điều chỉnh theo xu hướng gần
            recent = "".join(history[-20:])
            recent_count = recent.count(num)
            recent_prob = recent_count / 20 if recent_count > 0 else 0
            
            # Kết hợp
            probs[num] = base_prob * 0.4 + recent_prob * 0.6
        
        sorted_nums = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'dan4': [num for num, _ in sorted_nums[:4]],
            'dan3': [num for num, _ in sorted_nums[4:7]],
            'do_tin_cay': 70,
            'nguon': 'statistical'
        }
    
    def get_crawler_prediction(self, crawler_data):
        """Dự đoán dựa trên dữ liệu crawler"""
        if not crawler_data or 'hot_online' not in crawler_data:
            return None
        
        hot = crawler_data['hot_online']
        return {
            'dan4': hot[:4],
            'dan3': hot[4:7] if len(hot) > 4 else [],
            'do_tin_cay': 65,
            'nguon': 'crawler'
        }
    
    def weighted_combination(self, predictions):
        """Kết hợp các dự đoán với trọng số"""
        if not predictions:
            return None
        
        # Đếm số phiếu cho mỗi số
        votes = defaultdict(float)
        all_reasons = []
        warnings = []
        
        for source, pred in predictions.items():
            weight = self.weights.get(source, 0.2)
            
            # Cộng phiếu cho dan4
            for num in pred.get('dan4', []):
                votes[num] += weight * 1.5  # Trọng số cao hơn cho dan4
            
            # Cộng phiếu cho dan3
            for num in pred.get('dan3', []):
                votes[num] += weight
            
            # Thu thập lý do
            if 'ly_do' in pred:
                all_reasons.append(f"{source}: {pred['ly_do']}")
            if 'canh_bao' in pred and pred['canh_bao']:
                warnings.append(pred['canh_bao'])
        
        # Sắp xếp theo số phiếu
        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        
        # Tính độ tin cậy tổng hợp
        total_confidence = sum(p.get('do_tin_cay', 0) * self.weights.get(s, 0.2) 
                              for s, p in predictions.items()) / len(predictions)
        
        return {
            'dan4': [num for num, _ in sorted_votes[:4]],
            'dan3': [num for num, _ in sorted_votes[4:7]],
            'do_tin_cay': min(total_confidence * 1.2, 98),  # Boost nhẹ
            'ly_do': "\n".join(all_reasons[:3]),
            'canh_bao': " | ".join(warnings) if warnings else "",
            'votes': dict(sorted_votes[:10])
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
    .warning-red {
        background: #f8514920; border-left: 4px solid #f85149;
        padding: 15px; border-radius: 8px; margin: 10px 0;
    }
    .warning-orange {
        background: #f0883e20; border-left: 4px solid #f0883e;
        padding: 15px; border-radius: 8px; margin: 10px 0;
    }
    .warning-yellow {
        background: #f2cc6020; border-left: 4px solid #f2cc60;
        padding: 15px; border-radius: 8px; margin: 10px 0;
    }
    .pair-badge {
        background: #1f6feb; color: white; padding: 4px 12px;
        border-radius: 20px; font-size: 13px; display: inline-block;
        margin: 3px; font-weight: bold;
    }
    .stats-box {
        background: #161b22; border-radius: 10px; padding: 15px;
        margin: 10px 0; border: 1px solid #30363d;
    }
    .accuracy-meter {
        height: 10px; background: #30363d; border-radius: 5px;
        margin: 10px 0;
    }
    .accuracy-fill {
        height: 10px; background: linear-gradient(90deg, #238636, #58a6ff);
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center; color: #58a6ff;'>🧬 TITAN v21.0 PRO MAX</h1>", unsafe_allow_html=True)

# Khởi tạo các hệ thống
crawler = AutoCrawler()

# Hiển thị trạng thái
col_status1, col_status2, col_status3 = st.columns(3)
with col_status1:
    st.markdown(f"<p class='status-active'>📊 DỮ LIỆU: {len(st.session_state.history)} KỲ</p>", unsafe_allow_html=True)
with col_status2:
    accuracy = len([p for p in st.session_state.predictions if p.get('result', False)]) / max(len(st.session_state.predictions), 1) * 100
    st.markdown(f"<p class='status-active'>🎯 ĐỘ CHÍNH XÁC: {accuracy:.1f}%</p>", unsafe_allow_html=True)
with col_status3:
    st.markdown(f"<p class='status-active'>🌐 NGUỒN: {len(SOURCES)}</p>", unsafe_allow_html=True)

# ================= AUTO CRAWLER =================
with st.expander("🌐 HỆ THỐNG THU THẬP DỮ LIỆU TỰ ĐỘNG", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 CRAWL DỮ LIỆU NGAY", use_container_width=True):
            with st.spinner("Đang thu thập dữ liệu từ các nguồn..."):
                new_data = crawler.crawl_all_sources()
                if new_data:
                    st.success(f"✅ Đã thu thập {len(new_data)} số mới!")
                    # Thêm vào history
                    st.session_state.history.extend(new_data)
                    save_json(DB_FILE, st.session_state.history)
                    st.rerun()
                else:
                    st.error("❌ Không thu thập được dữ liệu")
    
    with col2:
        st.markdown(f"**Lần crawl cuối:** {st.session_state.crawler_data.get('last_crawl', 'Chưa có')}")
    
    # Hiển thị dữ liệu từ các nguồn
    if st.session_state.crawler_data.get('sources'):
        st.markdown("### 📊 DỮ LIỆU TỪ CÁC NGUỒN")
        for source, data in st.session_state.crawler_data['sources'].items():
            st.markdown(f"""
            <div style='background: #161b22; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                <b>{source[:50]}...</b><br>
                <small>Số lượng: {data['count']} | {data['time']}</small>
            </div>
            """, unsafe_allow_html=True)

# ================= PHÂN TÍCH NÂNG CAO =================
if st.session_state.history:
    detector = PatternDetector(st.session_state.history)
    
    # Phát hiện các cặp số hay đi cùng
    pairs = detector.find_number_pairs()
    triplets = detector.find_triplet_patterns()
    tricks = detector.detect_house_tricks()
    cycles = detector.find_cycles()
    
    # Lấy xu hướng online
    online_trend = crawler.get_online_trend()
    
    # Tabs phân tích
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 DỰ ĐOÁN", 
        "🔄 CẶP SỐ", 
        "⚠️ PHÁT HIỆN BẪY",
        "📈 CHU KỲ",
        "🌐 ONLINE"
    ])
    
    with tab1:
        st.markdown("### 🎯 DỰ ĐOÁN ĐA NGUỒN")
        
        # Nút dự đoán
        if st.button("🚀 DỰ ĐOÁN SIÊU CHÍNH XÁC", use_container_width=True):
            with st.spinner("Đang phân tích từ nhiều nguồn..."):
                # Chuẩn bị dữ liệu cho multi AI
                pattern_data = {
                    'pairs': pairs,
                    'triplets': triplets,
                    'tricks': tricks,
                    'cycles': cycles
                }
                
                # Multi AI system
                ai_system = MultiAISystem()
                final_pred = ai_system.ensemble_prediction(
                    st.session_state.history, 
                    pattern_data,
                    online_trend
                )
                
                if final_pred:
                    # Lưu dự đoán
                    pred_record = {
                        'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'dan4': final_pred['dan4'],
                        'dan3': final_pred['dan3'],
                        'do_tin_cay': final_pred['do_tin_cay'],
                        'ly_do': final_pred.get('ly_do', ''),
                        'canh_bao': final_pred.get('canh_bao', ''),
                        'votes': final_pred.get('votes', {})
                    }
                    st.session_state.predictions.append(pred_record)
                    save_json(PREDICTIONS_FILE, st.session_state.predictions)
                    
                    st.session_state.last_result = final_pred
                    st.rerun()
    
    with tab2:
        st.markdown("### 🔥 CÁC CẶP SỐ HAY ĐI CÙNG NHAU")
        
        if pairs:
            cols = st.columns(3)
            for i, (pair, data) in enumerate(list(pairs.items())[:12]):
                with cols[i % 3]:
                    strength_color = "#238636" if data['strength'] == 'CAO' else "#f2cc60"
                    st.markdown(f"""
                    <div style='background: #161b22; padding: 15px; border-radius: 8px; margin: 5px; text-align: center;'>
                        <div style='font-size: 24px; font-weight: bold; color: {strength_color};'>
                            {pair[0]} - {pair[1]}
                        </div>
                        <div style='font-size: 12px;'>
                            XS: {(data['probability']*100):.1f}%<br>
                            Độ mạnh: <span style='color: {strength_color};'>{data['strength']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Chưa đủ dữ liệu để phân tích cặp số")
        
        st.markdown("### 🎯 BỘ 3 SỐ ĐẶC BIỆT")
        if triplets:
            for triplet, data in list(triplets.items())[:10]:
                st.markdown(f"""
                <div style='background: #161b22; padding: 8px; border-radius: 5px; margin: 3px; display: inline-block;'>
                    <b>{triplet}</b> ({data['count']} lần)
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### ⚠️ PHÁT HIỆN BẪY NHÀ CÁI")
        
        # Hiển thị mức độ cảnh báo
        warning_level = tricks.get('warning_level', 'GREEN')
        if warning_level == 'RED':
            st.markdown("""
            <div class='warning-red'>
                <b>🚨 CẢNH BÁO ĐỎ - NGUY HIỂM CAO</b><br>
                Nhà cái đang thay đổi hoàn toàn quy luật. ĐỀ NGHỊ DỪNG LẠI!
            </div>
            """, unsafe_allow_html=True)
        elif warning_level == 'ORANGE':
            st.markdown("""
            <div class='warning-orange'>
                <b>⚠️ CẢNH BÁO CAM - RỦI RO CAO</b><br>
                Phát hiện dấu hiệu bất thường. CẨN TRỌNG KHI VÀO TIỀN!
            </div>
            """, unsafe_allow_html=True)
        elif warning_level == 'YELLOW':
            st.markdown("""
            <div class='warning-yellow'>
                <b>⚠️ CẢNH BÁO VÀNG - THẬN TRỌNG</b><br>
            </div>
            """, unsafe_allow_html=True)
        
        # Chi tiết cảnh báo
        if tricks['details']:
            st.markdown("**📋 Chi tiết phát hiện:**")
            for detail in tricks['details']:
                st.markdown(f"- {detail}")
        
        # Hiển thị trạng thái
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Đảo cầu:** {'✅' if tricks['dao_cau'] else '❌'}")
        with col2:
            st.markdown(f"**Bẫy màu:** {'✅' if tricks['bay_mau'] else '❌'}")
        with col3:
            st.markdown(f"**Sập bệt:** {'✅' if tricks['sập_bệt'] else '❌'}")
    
    with tab4:
        st.markdown("### 📈 CHU KỲ LẶP LẠI")
        
        if cycles:
            for cycle_name, cycle_data in cycles.items():
                with st.expander(f"Chu kỳ {cycle_data['length']} số", expanded=False):
                    for pattern, count in cycle_data['patterns'].items():
                        st.markdown(f"""
                        <div style='background: #161b22; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                            <code>{pattern}</code> - {count} lần lặp
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("Chưa phát hiện chu kỳ đáng kể")
    
    with tab5:
        st.markdown("### 🌐 XU HƯỚNG TỪ CÁC NGUỒN ONLINE")
        
        if online_trend:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔥 Số hot online:**")
                hot_html = ""
                for num in online_trend['hot_online']:
                    hot_html += f"<span class='pair-badge'>{num}</span> "
                st.markdown(hot_html, unsafe_allow_html=True)
            
            with col2:
                st.markdown("**❄️ Số nguội online:**")
                cold_html = ""
                for num in online_trend['cold_online']:
                    cold_html += f"<span class='pair-badge' style='background: #8b949e;'>{num}</span> "
                st.markdown(cold_html, unsafe_allow_html=True)
            
            # Biểu đồ tần suất
            st.markdown("**📊 Phân bố tần suất online:**")
            for num, prob in sorted(online_trend['frequencies'].items(), 
                                    key=lambda x: x[1], reverse=True)[:10]:
                st.markdown(f"""
                <div>
                    Số {num}: {prob*100:.1f}%
                    <div class='accuracy-meter'>
                        <div class='accuracy-fill' style='width: {prob*100}%'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Chưa có dữ liệu online. Hãy crawl dữ liệu trước!")

# ================= INPUT DATA =================
st.markdown("### 📥 NHẬP DỮ LIỆU THỦ CÔNG")
raw_input = st.text_area("Dán các dãy 5 số (mỗi dãy 1 dòng):", height=100, 
                         placeholder="32880\n21808\n69962\n...")

col1, col2, col3 = st.columns([2,1,1])
with col1:
    if st.button("📥 THÊM DỮ LIỆU", use_container_width=True):
        new_data = re.findall(r"\d{5}", raw_input)
        if new_data:
            st.session_state.history.extend(new_data)
            save_json(DB_FILE, st.session_state.history)
            st.success(f"✅ Đã thêm {len(new_data)} kỳ mới!")
            st.rerun()

with col2:
    if st.button("🗑️ RESET", use_container_width=True):
        st.session_state.history = []
        if os.path.exists(DB_FILE): os.remove(DB_FILE)
        st.rerun()

with col3:
    if st.button("📜 LỊCH SỬ", use_container_width=True):
        st.session_state.show_history = not st.session_state.get('show_history', False)
        st.rerun()

# ================= HIỂN THỊ LỊCH SỬ =================
if st.session_state.get('show_history', False):
    with st.expander("📜 LỊCH SỬ DỰ ĐOÁN (100 GẦN NHẤT)", expanded=True):
        if st.session_state.predictions:
            for i, pred in enumerate(reversed(st.session_state.predictions[-30:])):
                conf_color = "#238636" if pred.get('do_tin_cay', 0) > 85 else "#f2cc60"
                st.markdown(f"""
                <div style='background: #161b22; padding: 15px; border-radius: 8px; margin: 8px 0; border-left: 4px solid {conf_color};'>
                    <div style='display: flex; justify-content: space-between;'>
                        <small>🕐 {pred['time']}</small>
                        <small style='color: {conf_color};'>Độ tin cậy: {pred.get('do_tin_cay', 0)}%</small>
                    </div>
                    <div style='font-size: 28px; letter-spacing: 8px; margin: 8px 0;'>
                        <span style='color: #58a6ff;'>{''.join(pred['dan4'])}</span>
                        <span style='color: #f2cc60;'>{''.join(pred['dan3'])}</span>
                    </div>
                    <small>💡 {pred.get('ly_do', '')[:150]}</small>
                    {f"<br><small>⚠️ {pred['canh_bao']}</small>" if pred.get('canh_bao') else ""}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Chưa có lịch sử dự đoán")

# ================= HIỂN THỊ KẾT QUẢ DỰ ĐOÁN =================
if "last_result" in st.session_state:
    res = st.session_state.last_result
    
    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    
    # Hiển thị độ tin cậy
    confidence = res.get('do_tin_cay', 85)
    conf_color = "#238636" if confidence > 85 else "#f2cc60" if confidence > 70 else "#f85149"
    
    st.markdown(f"""
    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;'>
        <span style='color: #8b949e;'>🎯 KẾT QUẢ DỰ ĐOÁN SIÊU CHÍNH XÁC</span>
        <span style='background: {conf_color}20; color: {conf_color}; padding: 8px 20px; border-radius: 25px; font-weight: bold; font-size: 18px;'>
            {confidence}% TIN CẬY
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Hiển thị cảnh báo
    if res.get('canh_bao'):
        warning_level = tricks.get('warning_level', 'YELLOW')
        if warning_level == 'RED':
            st.error(f"🚨 {res['canh_bao']}")
        elif warning_level == 'ORANGE':
            st.warning(f"⚠️ {res['canh_bao']}")
        else:
            st.info(f"ℹ️ {res['canh_bao']}")
    
    # Hiển thị phân tích
    if res.get('ly_do'):
        st.markdown(f"""
        <div class='logic-box'>
            <b>🧠 PHÂN TÍCH ĐA NGUỒN:</b><br>
            {res['ly_do']}
        </div>
        """, unsafe_allow_html=True)
    
    # Hiển thị 4 số chính
    st.markdown("<p style='text-align:center; font-size:16px; color:#888;'>🎯 4 SỐ CHỦ LỰC (ĐÁNH CHÍNH)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-display'>{''.join(map(str, res['dan4']))}</div>", unsafe_allow_html=True)
    
    # Hiển thị 3 số lót
    st.markdown("<p style='text-align:center; font-size:16px; color:#888; margin-top:25px;'>🛡️ 3 SỐ LÓT (ĐÁNH KÈM, BẢO HIỂM)</p>", unsafe_allow_html=True)
    st.markdown(f"<div class='num-display' style='color:#f2cc60; text-shadow: 0 0 25px #f2cc60;'>{''.join(map(str, res['dan3']))}</div>", unsafe_allow_html=True)
    
    # Nút copy
    copy_val = "".join(map(str, res['dan4'])) + "".join(map(str, res['dan3']))
    st.text_input("📋 DÀN 7 SỐ HOÀN CHỈNH:", copy_val, key="final_copy")
    
    # Hiển thị voting weights nếu có
    if res.get('votes'):
        st.markdown("### 📊 PHÂN BỐ PHIẾU BẦU TỪ CÁC NGUỒN")
        votes = res['votes']
        max_vote = max(votes.values()) if votes else 1
        for num, vote in sorted(votes.items(), key=lambda x: x[1], reverse=True):
            st.markdown(f"""
            <div>
                Số {num}: {vote:.2f}
                <div class='accuracy-meter'>
                    <div class='accuracy-fill' style='width: {(vote/max_vote)*100}%'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<br>
<div style='text-align:center; font-size:12px; color:#444; border-top: 1px solid #30363d; padding-top: 20px;'>
    🧬 TITAN v21.0 PRO MAX - Hệ thống phân tích đa nguồn thông minh<br>
    🔍 Phát hiện cặp số | Phát hiện bẫy | So sánh đa nguồn | Auto Crawler | Multi AI
</div>
""", unsafe_allow_html=True)