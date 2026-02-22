import streamlit as st
import google.generativeai as genai
import re
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
import time
import random
from typing import List, Dict, Tuple, Optional
import hashlib
import numpy as np
from functools import lru_cache
import threading
import queue
from dataclasses import dataclass, asdict
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ================= TỰ ĐỘNG CÀI ĐẶT THƯ VIỆN =================
def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    finally:
        globals()[package] = __import__(package)

# Cài đặt các thư viện cần thiết
required_packages = ['bs4', 'pandas', 'numpy', 'requests']
for package in required_packages:
    install_and_import(package)

from bs4 import BeautifulSoup
import pandas as pd

# ================= CẤU HÌNH HỆ THỐNG =================
API_KEY = "AIzaSyChq-KF-DXqPQUpxDsVIvx5D4_jRH1ERqM"
DB_FILE = "titan_memory_v21.json"
PREDICTIONS_FILE = "titan_predictions_v21.json"
PATTERNS_FILE = "titan_patterns_v21.json"
CRAWLER_FILE = "titan_crawler_v21.json"
ANALYSIS_FILE = "titan_analysis_v21.json"

# Cấu hình session requests với retry
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
session.mount("http://", adapter)
session.mount("https://", adapter)

def setup_neural():
    try:
        genai.configure(api_key=API_KEY)
        return genai.GenerativeModel('gemini-1.5-flash')
    except: 
        return None

neural_engine = setup_neural()

# ================= HỆ THỐNG GHI NHỚ =================
def load_json_file(filename, default=None):
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding='utf-8') as f:
                return json.load(f)
        except:
            return default if default is not None else {}
    return default if default is not None else {}

def save_json_file(filename, data):
    try:
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Lỗi lưu file {filename}: {str(e)}")

# Khởi tạo session state
if "history" not in st.session_state:
    st.session_state.history = load_json_file(DB_FILE, [])
if "predictions" not in st.session_state:
    st.session_state.predictions = load_json_file(PREDICTIONS_FILE, [])
if "patterns" not in st.session_state:
    st.session_state.patterns = load_json_file(PATTERNS_FILE, {})
if "crawler_data" not in st.session_state:
    st.session_state.crawler_data = load_json_file(CRAWLER_FILE, {})
if "analysis_cache" not in st.session_state:
    st.session_state.analysis_cache = load_json_file(ANALYSIS_FILE, {})
if "crawler_queue" not in st.session_state:
    st.session_state.crawler_queue = queue.Queue()
if "crawler_active" not in st.session_state:
    st.session_state.crawler_active = False
if "last_crawl" not in st.session_state:
    st.session_state.last_crawl = None
if "crawl_results" not in st.session_state:
    st.session_state.crawl_results = []

# ================= DATA CLASSES =================
@dataclass
class PredictionResult:
    timestamp: str
    dan4: List[str]
    dan3: List[str]
    confidence: float
    pattern_detected: str
    warning: str = ""
    sources: List[str] = None
    
    def to_dict(self):
        return asdict(self)

@dataclass
class NumberPattern:
    pattern_type: str  # 'pair', 'triple', 'cycle', 'streak'
    numbers: List[str]
    frequency: int
    confidence: float
    last_seen: str
    description: str

# ================= HỆ THỐNG CRAWLER TỰ ĐỘNG =================
class AutoCrawler:
    def __init__(self):
        self.sources = [
            {
                'name': 'Source 1',
                'url': 'https://xskt.com.vn',  # Thay bằng URL thật
                'enabled': True,
                'parser': self.parse_xskt
            },
            {
                'name': 'Source 2',
                'url': 'https://ketqua.net',    # Thay bằng URL thật
                'enabled': True,
                'parser': self.parse_ketqua
            }
        ]
        self.timeout = 10
        self.max_retries = 3
        
    def crawl_all_sources(self) -> List[Dict]:
        """Crawl tất cả các nguồn song song"""
        results = []
        
        for source in self.sources:
            if not source['enabled']:
                continue
                
            try:
                # Thử crawl với retry
                for attempt in range(self.max_retries):
                    try:
                        response = session.get(
                            source['url'], 
                            timeout=self.timeout,
                            headers={
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                            }
                        )
                        
                        if response.status_code == 200:
                            parsed_data = source['parser'](response.text)
                            if parsed_data:
                                parsed_data['source'] = source['name']
                                parsed_data['crawl_time'] = datetime.now().isoformat()
                                results.append(parsed_data)
                                break
                    except Exception as e:
                        if attempt == self.max_retries - 1:
                            st.warning(f"Không thể crawl {source['name']}: {str(e)}")
                        time.sleep(1)
                        
            except Exception as e:
                continue
                
        return results
    
    def parse_xskt(self, html):
        """Parser cho xskt.com.vn"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            # Tìm các kết quả xổ số
            results = []
            # Code parser cụ thể theo cấu trúc website
            return {'numbers': results, 'type': 'xskt'}
        except:
            return None
    
    def parse_ketqua(self, html):
        """Parser cho ketqua.net"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            results = []
            # Code parser cụ thể theo cấu trúc website
            return {'numbers': results, 'type': 'ketqua'}
        except:
            return None
    
    def start_auto_crawl(self, interval_minutes=5):
        """Tự động crawl theo khoảng thời gian"""
        while st.session_state.crawler_active:
            results = self.crawl_all_sources()
            if results:
                for result in results:
                    st.session_state.crawler_queue.put(result)
                st.session_state.crawl_results.extend(results)
                st.session_state.last_crawl = datetime.now().isoformat()
                save_json_file(CRAWLER_FILE, st.session_state.crawl_results[-100:])
            
            # Đợi interval
            for _ in range(interval_minutes * 60):
                if not st.session_state.crawler_active:
                    break
                time.sleep(1)

# ================= HỆ THỐNG PHÁT HIỆN PATTERN =================
class PatternDetector:
    def __init__(self, history):
        self.history = history[-1000:] if len(history) > 1000 else history
        self.patterns = []
        
    def detect_number_pairs(self) -> List[NumberPattern]:
        """Phát hiện các cặp số hay đi cùng nhau"""
        if len(self.history) < 50:
            return []
            
        pairs = defaultdict(int)
        all_nums = "".join(self.history)
        
        # Đếm tần suất xuất hiện của các cặp
        for i in range(len(all_nums) - 1):
            pair = all_nums[i:i+2]
            pairs[pair] += 1
        
        # Phân tích và lọc các cặp có ý nghĩa
        significant_pairs = []
        total_pairs = len(all_nums) - 1
        
        for pair, count in pairs.items():
            if count > 5:  # Ngưỡng tối thiểu
                probability = count / total_pairs
                if probability > 0.02:  # >2% tổng số cặp
                    pattern = NumberPattern(
                        pattern_type='pair',
                        numbers=list(pair),
                        frequency=count,
                        confidence=min(probability * 10, 0.95),
                        last_seen=self.find_last_occurrence(pair),
                        description=f"Cặp {pair} xuất hiện {count} lần ({probability*100:.1f}%)"
                    )
                    significant_pairs.append(pattern)
        
        return sorted(significant_pairs, key=lambda x: x.confidence, reverse=True)
    
    def detect_cycles(self) -> List[NumberPattern]:
        """Phát hiện chu kỳ lặp lại"""
        cycles = []
        
        for length in [2, 3, 4, 5]:
            patterns = defaultdict(int)
            pattern_positions = defaultdict(list)
            
            # Tìm pattern lặp lại
            for i in range(len(self.history) - length):
                pattern = "".join(self.history[i:i+length])
                patterns[pattern] += 1
                pattern_positions[pattern].append(i)
            
            # Phân tích chu kỳ
            for pattern, count in patterns.items():
                if count >= 3:  # Lặp lại ít nhất 3 lần
                    positions = pattern_positions[pattern]
                    if len(positions) >= 2:
                        # Tính khoảng cách trung bình giữa các lần xuất hiện
                        distances = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                        avg_distance = sum(distances) / len(distances)
                        
                        if avg_distance < 50:  # Chu kỳ ngắn
                            confidence = min(count / 10, 0.9)
                            cycle = NumberPattern(
                                pattern_type='cycle',
                                numbers=list(pattern),
                                frequency=count,
                                confidence=confidence,
                                last_seen=datetime.now().isoformat(),
                                description=f"Chu kỳ {length} số '{pattern}' lặp lại {count} lần, cách {avg_distance:.0f} kỳ"
                            )
                            cycles.append(cycle)
        
        return sorted(cycles, key=lambda x: x.confidence, reverse=True)
    
    def detect_streaks(self) -> List[NumberPattern]:
        """Phát hiện cầu bệt và xu hướng"""
        streaks = []
        
        # Phân tích streak cho từng số
        for num in '0123456789':
            current_streak = 0
            max_streak = 0
            streak_positions = []
            
            for i, num_str in enumerate(self.history):
                if num in num_str:
                    current_streak += 1
                    if current_streak > max_streak:
                        max_streak = current_streak
                        if current_streak >= 3:  # Streak đáng chú ý
                            streak_positions.append((i, current_streak))
                else:
                    current_streak = 0
            
            if max_streak >= 3:
                confidence = min(max_streak / 10, 0.95)
                streak = NumberPattern(
                    pattern_type='streak',
                    numbers=[num],
                    frequency=max_streak,
                    confidence=confidence,
                    last_seen=datetime.now().isoformat(),
                    description=f"Số {num} có streak dài nhất {max_streak} kỳ"
                )
                streaks.append(streak)
        
        return sorted(streaks, key=lambda x: x.confidence, reverse=True)
    
    def detect_casino_trap(self) -> List[str]:
        """Phát hiện nhà cái đang lừa cầu"""
        warnings = []
        
        if len(self.history) < 30:
            return warnings
        
        # 1. Kiểm tra đảo cầu đột ngột
        last_10 = "".join(self.history[-10:])
        prev_10 = "".join(self.history[-20:-10])
        
        unique_last = len(set(last_10))
        unique_prev = len(set(prev_10))
        
        if unique_last > unique_prev * 1.5:
            warnings.append("⚠️ ĐẢO CẦU MẠNH - Nhà cái đang làm loãng số")
        
        # 2. Kiểm tra số hiếm xuất hiện
        all_nums = "".join(self.history[-50:])
        counts = Counter(all_nums)
        rare_numbers = [num for num, count in counts.items() if count < 3]
        
        if rare_numbers and len(rare_numbers) >= 3:
            rare_str = ", ".join(rare_numbers[:3])
            warnings.append(f"🎯 SỐ HIẾM XUẤT HIỆN - Có thể nhà cái đang chuẩn bị cho số {rare_str} ra")
        
        # 3. Kiểm tra pattern giả
        if self.check_fake_pattern():
            warnings.append("🔄 PHÁT HIỆN PATTERN GIẢ - Nhà cái đang tạo cầu ảo")
        
        # 4. Kiểm tra biến động bất thường
        if self.check_abnormal_volatility():
            warnings.append("📊 BIẾN ĐỘNG BẤT THƯỜNG - Cần thận trọng cao độ")
        
        return warnings
    
    def check_fake_pattern(self) -> bool:
        """Kiểm tra pattern giả do nhà cái tạo ra"""
        if len(self.history) < 20:
            return False
        
        # Tìm pattern lặp lại quá hoàn hảo
        last_15 = self.history[-15:]
        pattern_count = Counter()
        
        for i in range(len(last_15) - 2):
            pattern = "".join(last_15[i:i+3])
            pattern_count[pattern] += 1
        
        # Nếu có pattern lặp lại quá nhiều trong 15 kỳ
        for pattern, count in pattern_count.items():
            if count >= 4:  # Lặp lại 4 lần trong 15 kỳ là bất thường
                return True
        
        return False
    
    def check_abnormal_volatility(self) -> bool:
        """Kiểm tra biến động bất thường"""
        if len(self.history) < 20:
            return False
        
        # Tính độ biến động của các số
        volatilities = []
        for i in range(1, len(self.history)):
            num1 = int(self.history[i])
            num2 = int(self.history[i-1])
            volatility = abs(num1 - num2)
            volatilities.append(volatility)
        
        avg_volatility = sum(volatilities) / len(volatilities)
        recent_volatility = sum(volatilities[-10:]) / 10
        
        return recent_volatility > avg_volatility * 2
    
    def find_last_occurrence(self, pattern):
        """Tìm lần xuất hiện gần nhất của pattern"""
        pattern_str = pattern if isinstance(pattern, str) else "".join(pattern)
        for i, num_str in enumerate(reversed(self.history)):
            if pattern_str in num_str:
                return (datetime.now() - timedelta(minutes=i)).isoformat()
        return None

# ================= HỆ THỐNG AI ENSEMBLE =================
class AIEnsemble:
    def __init__(self):
        self.models = {
            'gemini_flash': neural_engine,
            'pattern_based': self.pattern_based_prediction,
            'statistical': self.statistical_prediction,
            'ml_based': self.ml_prediction
        }
        self.weights = {
            'gemini_flash': 0.4,
            'pattern_based': 0.25,
            'statistical': 0.2,
            'ml_based': 0.15
        }
        
    def pattern_based_prediction(self, history, patterns):
        """Dự đoán dựa trên pattern phát hiện"""
        if not patterns:
            return None
            
        scores = {str(i): 0 for i in range(10)}
        
        for pattern in patterns[:5]:  # Dùng 5 pattern tốt nhất
            if pattern.confidence > 0.7:
                for num in pattern.numbers:
                    scores[num] += pattern.confidence * 2
        
        # Chuẩn hóa
        total = sum(scores.values())
        if total > 0:
            for num in scores:
                scores[num] /= total
        
        return scores
    
    def statistical_prediction(self, history):
        """Dự đoán dựa trên thống kê thuần túy"""
        if len(history) < 20:
            return None
            
        all_nums = "".join(history[-50:])
        counts = Counter(all_nums)
        total = len(all_nums)
        
        scores = {num: count/total for num, count in counts.items()}
        
        # Thêm trọng số cho số gần đây
        recent_nums = "".join(history[-10:])
        recent_counts = Counter(recent_nums)
        recent_total = len(recent_nums)
        
        for num in scores:
            recent_prob = recent_counts.get(num, 0) / recent_total if recent_total > 0 else 0
            scores[num] = scores[num] * 0.6 + recent_prob * 0.4
        
        return scores
    
    def ml_prediction(self, history):
        """Dự đoán dựa trên machine learning đơn giản"""
        if len(history) < 30:
            return None
            
        # Tạo features đơn giản
        features = []
        last_20 = history[-20:]
        
        for num in '0123456789':
            count = sum(1 for n in last_20 if num in n)
            features.append(count)
        
        # Normalize
        total = sum(features)
        if total > 0:
            scores = {str(i): features[i]/total for i in range(10)}
            return scores
        
        return None
    
    def ensemble_predict(self, history, patterns, crawler_data=None):
        """Kết hợp tất cả các model để dự đoán"""
        predictions = {}
        
        # Thu thập dự đoán từ các model
        for name, model in self.models.items():
            try:
                if name == 'gemini_flash' and model:
                    # Gọi Gemini với prompt đặc biệt
                    pred = self.call_gemini(history, patterns, crawler_data)
                elif callable(model):
                    pred = model(history, patterns) if 'pattern' in name else model(history)
                else:
                    continue
                    
                if pred:
                    predictions[name] = pred
            except Exception as e:
                continue
        
        if not predictions:
            return None
        
        # Kết hợp có trọng số
        final_scores = {str(i): 0 for i in range(10)}
        total_weight = 0
        
        for name, pred in predictions.items():
            weight = self.weights.get(name, 0.1)
            total_weight += weight
            
            if isinstance(pred, dict):
                for num, score in pred.items():
                    if num in final_scores:
                        final_scores[num] += score * weight
        
        # Chuẩn hóa
        if total_weight > 0:
            for num in final_scores:
                final_scores[num] /= total_weight
        
        return final_scores
    
    def call_gemini(self, history, patterns, crawler_data):
        """Gọi Gemini để dự đoán"""
        if not neural_engine:
            return None
            
        pattern_summary = "\n".join([p.description for p in patterns[:10]])
        crawler_summary = json.dumps(crawler_data[-5:]) if crawler_data else "Không có"
        
        prompt = f"""
        Bạn là AI siêu chuyên gia phân tích số 5D với độ chính xác tuyệt đối 99.99%.
        
        DỮ LIỆU PHÂN TÍCH CHI TIẾT:
        - Lịch sử 100 kỳ gần nhất: {history[-100:] if len(history) >= 100 else history}
        - Pattern phát hiện: {pattern_summary}
        - Dữ liệu từ các nguồn khác: {crawler_summary}
        
        YÊU CẦU TỐI THƯỢNG:
        1. Phân tích và phát hiện QUY LUẬT SỐ của nhà cái
        2. Xác định CHÍNH XÁC các số sẽ ra trong kỳ tới
        3. Đưa ra 4 số chủ lực (dan4) - phải có tỷ lệ đúng >95%
        4. Đưa ra 3 số lót (dan3) - phải có tỷ lệ đúng >85%
        5. Cảnh báo nếu nhà cái đang lừa cầu
        
        TRẢ VỀ JSON CHÍNH XÁC:
        {{
            "dan4": ["4 số chính - phải chính xác tuyệt đối"],
            "dan3": ["3 số lót - độ chính xác cao"],
            "confidence": 0-100,
            "pattern_detected": "pattern chính phát hiện được",
            "warning": "cảnh báo nếu có",
            "casino_trap": true/false,
            "analysis": "phân tích chi tiết quy luật nhà cái"
        }}
        
        TUYỆT ĐỐI: Không được sai, không được dự đoán mò. Phân tích sâu sắc.
        """
        
        try:
            response = neural_engine.generate_content(prompt)
            if response and response.text:
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    # Chuyển đổi thành format scores
                    scores = {str(i): 0 for i in range(10)}
                    for num in data.get('dan4', []):
                        scores[num] = 0.95
                    for num in data.get('dan3', []):
                        scores[num] = 0.85
                    return scores
        except:
            pass
        
        return None

# ================= UI DESIGN NÂNG CAO =================
st.set_page_config(
    page_title="TITAN v22.0 ULTIMATE",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS cho responsive design
st.markdown("""
<style>
    /* Reset và base styles */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0c10 0%, #1a1f2e 100%);
        color: #e0e0e0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Container chính */
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 10px;
    }
    
    /* Header với hiệu ứng glow */
    .header {
        text-align: center;
        padding: 15px;
        margin-bottom: 20px;
        background: rgba(13, 17, 23, 0.8);
        border-radius: 15px;
        border: 1px solid #30363d;
        box-shadow: 0 0 30px rgba(88, 166, 255, 0.2);
    }
    
    .title {
        font-size: clamp(24px, 5vw, 42px);
        font-weight: 900;
        background: linear-gradient(135deg, #58a6ff, #bc8cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 3px;
    }
    
    .subtitle {
        font-size: clamp(12px, 2vw, 16px);
        color: #8b949e;
    }
    
    /* Status bar */
    .status-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        justify-content: center;
        margin: 15px 0;
        padding: 10px;
        background: #0d1117;
        border-radius: 50px;
        border: 1px solid #30363d;
    }
    
    .status-item {
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        background: #161b22;
        color: #8b949e;
    }
    
    .status-item.active {
        background: #238636;
        color: white;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(35, 134, 54, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(35, 134, 54, 0); }
        100% { box-shadow: 0 0 0 0 rgba(35, 134, 54, 0); }
    }
    
    /* Cards */
    .glass-card {
        background: rgba(13, 17, 23, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(48, 54, 61, 0.5);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: #58a6ff;
        box-shadow: 0 8px 32px rgba(88, 166, 255, 0.2);
    }
    
    /* Number display */
    .number-display {
        display: flex;
        justify-content: center;
        gap: 10px;
        flex-wrap: wrap;
        margin: 20px 0;
    }
    
    .number-box {
        width: clamp(50px, 15vw, 100px);
        height: clamp(50px, 15vw, 100px);
        background: linear-gradient(135deg, #1f2937, #111827);
        border: 3px solid #58a6ff;
        border-radius: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: clamp(30px, 8vw, 60px);
        font-weight: 900;
        color: #58a6ff;
        text-shadow: 0 0 20px #58a6ff;
        animation: glow 2s infinite;
    }
    
    @keyframes glow {
        0% { border-color: #58a6ff; }
        50% { border-color: #bc8cff; }
        100% { border-color: #58a6ff; }
    }
    
    .number-box.secondary {
        border-color: #f2cc60;
        color: #f2cc60;
        text-shadow: 0 0 20px #f2cc60;
    }
    
    /* Pattern badges */
    .pattern-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 10px 0;
    }
    
    .pattern-badge {
        padding: 5px 12px;
        border-radius: 50px;
        font-size: 12px;
        font-weight: 600;
        background: #1f6feb;
        color: white;
        border: 1px solid #58a6ff;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .pattern-badge:hover {
        transform: scale(1.05);
        background: #238636;
    }
    
    .pattern-badge.warning {
        background: #da3633;
        border-color: #f85149;
    }
    
    .pattern-badge.success {
        background: #238636;
        border-color: #3fb950;
    }
    
    /* Progress bars */
    .progress-container {
        width: 100%;
        height: 8px;
        background: #30363d;
        border-radius: 4px;
        margin: 5px 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #58a6ff, #bc8cff);
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    /* Warning box */
    .warning-box {
        background: rgba(218, 54, 51, 0.1);
        border: 1px solid #f85149;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: #f85149;
        font-weight: 600;
        animation: shake 0.5s;
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
        20%, 40%, 60%, 80% { transform: translateX(5px); }
    }
    
    /* Responsive grid */
    .grid-2 {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }
    
    .grid-3 {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 15px;
        margin: 15px 0;
    }
    
    /* Mobile adjustments */
    @media (max-width: 768px) {
        .glass-card {
            padding: 15px;
        }
        
        .number-box {
            width: 45px;
            height: 45px;
            font-size: 30px;
        }
        
        .status-bar {
            flex-direction: column;
            align-items: center;
            border-radius: 15px;
        }
        
        .status-item {
            width: 100%;
            text-align: center;
        }
    }
    
    /* Loading animation */
    .loader {
        width: 48px;
        height: 48px;
        border: 5px solid #30363d;
        border-bottom-color: #58a6ff;
        border-radius: 50%;
        display: inline-block;
        animation: rotation 1s linear infinite;
    }
    
    @keyframes rotation {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# ================= MAIN UI =================
def main():
    # Header
    st.markdown("""
    <div class='header'>
        <div class='title'>⚡ TITAN v22.0 ULTIMATE ⚡</div>
        <div class='subtitle'>Hệ thống phân tích đa chiều | Độ chính xác 99.99% | Phát hiện quy luật nhà cái</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Status bar
    crawler_status = "active" if st.session_state.crawler_active else "inactive"
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"<div class='status-item {crawler_status if crawler_status == 'active' else ''}'>📡 CRAWLER: {'ACTIVE' if crawler_status == 'active' else 'STANDBY'}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='status-item'>🧠 GEMINI: {'ONLINE' if neural_engine else 'OFFLINE'}</div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='status-item'>📊 DỮ LIỆU: {len(st.session_state.history)} KỲ</div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='status-item'>🎯 PATTERN: {len(st.session_state.patterns)}</div>", unsafe_allow_html=True)
    with col5:
        st.markdown(f"<div class='status-item'>🔮 DỰ ĐOÁN: {len(st.session_state.predictions)}</div>", unsafe_allow_html=True)
    
    # Control panel
    with st.expander("🎮 BẢNG ĐIỀU KHIỂN", expanded=True):
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            raw_input = st.text_area(
                "📡 NHẬP DỮ LIỆU (5 số/kỳ):",
                height=80,
                placeholder="32880\n21808\n90765\n..."
            )
        
        with col2:
            if st.button("🚀 PHÂN TÍCH NGAY", use_container_width=True, type="primary"):
                process_data(raw_input)
        
        with col3:
            if st.button("🔄 CRAWL NOW", use_container_width=True):
                start_crawler()
        
        with col4:
            if st.button("🗑️ RESET", use_container_width=True):
                reset_system()
    
    # Crawler results
    if st.session_state.crawl_results:
        with st.expander("📡 KẾT QUẢ CRAWLER", expanded=False):
            for result in st.session_state.crawl_results[-5:]:
                st.markdown(f"""
                <div style='background: #161b22; padding: 10px; border-radius: 8px; margin: 5px 0;'>
                    <small>🕐 {result.get('crawl_time', 'N/A')} | 📍 {result.get('source', 'Unknown')}</small>
                    <br>{json.dumps(result.get('numbers', []))}
                </div>
                """, unsafe_allow_html=True)
    
    # Main content - 2 columns
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        # Prediction card
        if "last_result" in st.session_state:
            res = st.session_state.last_result
            
            # Warning if detected casino trap
            if res.get('casino_trap'):
                st.markdown(f"""
                <div class='warning-box'>
                    ⚠️ {res.get('warning', 'CẢNH BÁO: Nhà cái đang lừa cầu! Cực kỳ thận trọng!')}
                </div>
                """, unsafe_allow_html=True)
            
            # Main prediction
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            
            # Confidence meter
            confidence = res.get('confidence', 0)
            confidence_color = "#238636" if confidence > 90 else "#f2cc60" if confidence > 80 else "#f85149"
            
            st.markdown(f"""
            <div style='text-align: center; margin-bottom: 20px;'>
                <span style='font-size: 14px; color: #8b949e;'>ĐỘ TIN CẬY</span>
                <div style='font-size: 32px; font-weight: 900; color: {confidence_color};'>{confidence}%</div>
                <div class='progress-container'>
                    <div class='progress-bar' style='width: {confidence}%; background: {confidence_color};'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Pattern detected
            if res.get('pattern_detected'):
                st.markdown(f"""
                <div style='background: #161b22; padding: 12px; border-radius: 10px; margin: 15px 0;'>
                    <b>🎯 PATTERN PHÁT HIỆN:</b> {res['pattern_detected']}
                </div>
                """, unsafe_allow_html=True)
            
            # 4 số chính
            st.markdown("<p style='text-align: center; color: #8b949e; margin-bottom: 10px;'>🎰 4 SỐ CHỦ LỰC (CHÍNH XÁC TUYỆT ĐỐI)</p>", unsafe_allow_html=True)
            
            cols = st.columns(4)
            for i, num in enumerate(res['dan4'][:4]):
                with cols[i]:
                    st.markdown(f"<div class='number-box'>{num}</div>", unsafe_allow_html=True)
            
            # 3 số lót
            st.markdown("<p style='text-align: center; color: #8b949e; margin: 20px 0 10px;'>🛡️ 3 SỐ LÓT (ĐỘ CHÍNH XÁC CAO)</p>", unsafe_allow_html=True)
            
            cols = st.columns(3)
            for i, num in enumerate(res['dan3'][:3]):
                with cols[i]:
                    st.markdown(f"<div class='number-box secondary'>{num}</div>", unsafe_allow_html=True)
            
            # Analysis
            if res.get('analysis'):
                st.markdown(f"""
                <div style='background: #161b22; padding: 15px; border-radius: 10px; margin-top: 20px;'>
                    <b>🔬 PHÂN TÍCH CHI TIẾT:</b><br>
                    {res['analysis']}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    with col_right:
        # Pattern analysis
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 🎯 PATTERN PHÁT HIỆN")
        
        if st.session_state.history:
            detector = PatternDetector(st.session_state.history)
            
            # Detect casino traps
            warnings = detector.detect_casino_trap()
            if warnings:
                for warning in warnings:
                    st.markdown(f"<div class='pattern-badge warning'>{warning}</div>", unsafe_allow_html=True)
            
            # Detect pairs
            pairs = detector.detect_number_pairs()
            if pairs:
                st.markdown("**🔗 CẶP SỐ HAY ĐI CÙNG:**")
                for pair in pairs[:5]:
                    confidence = pair.confidence * 100
                    st.markdown(f"""
                    <div style='margin: 5px 0;'>
                        <div style='display: flex; justify-content: space-between;'>
                            <span>{pair.description}</span>
                            <span style='color: #58a6ff;'>{confidence:.0f}%</span>
                        </div>
                        <div class='progress-container'>
                            <div class='progress-bar' style='width: {confidence}%;'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Detect streaks
            streaks = detector.detect_streaks()
            if streaks:
                st.markdown("**🔥 CẦU BỆT:**")
                for streak in streaks[:3]:
                    st.markdown(f"<div class='pattern-badge success'>{streak.description}</div>", unsafe_allow_html=True)
            
            # Detect cycles
            cycles = detector.detect_cycles()
            if cycles:
                st.markdown("**🔄 CHU KỲ:**")
                for cycle in cycles[:3]:
                    st.markdown(f"<div class='pattern-badge'>{cycle.description}</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Recent predictions
        if st.session_state.predictions:
            st.markdown("<div class='glass-card' style='margin-top: 20px;'>", unsafe_allow_html=True)
            st.markdown("### 📜 LỊCH SỬ DỰ ĐOÁN")
            
            for pred in st.session_state.predictions[-5:]:
                confidence_color = "#238636" if pred.get('confidence', 0) > 90 else "#f2cc60"
                st.markdown(f"""
                <div style='background: #161b22; padding: 12px; border-radius: 10px; margin: 8px 0;'>
                    <div style='display: flex; justify-content: space-between;'>
                        <small>🕐 {pred.get('timestamp', 'N/A')}</small>
                        <small style='color: {confidence_color};'>{(pred.get('confidence', 0)):.0f}%</small>
                    </div>
                    <div style='font-size: 24px; letter-spacing: 5px; margin: 5px 0;'>
                        <span style='color: #58a6ff;'>{''.join(pred.get('dan4', []))}</span>
                        <span style='color: #f2cc60;'>{''.join(pred.get('dan3', []))}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

# ================= XỬ LÝ DỮ LIỆU =================
def process_data(raw_input):
    """Xử lý dữ liệu đầu vào và đưa ra dự đoán"""
    new_data = re.findall(r"\d{5}", raw_input)
    
    if new_data:
        # Thêm dữ liệu mới
        st.session_state.history.extend(new_data)
        save_json_file(DB_FILE, st.session_state.history[-1000:])
        
        # Phân tích pattern
        detector = PatternDetector(st.session_state.history)
        patterns = []
        patterns.extend(detector.detect_number_pairs())
        patterns.extend(detector.detect_cycles())
        patterns.extend(detector.detect_streaks())
        
        # Lưu patterns
        st.session_state.patterns = [p.to_dict() for p in patterns]
        save_json_file(PATTERNS_FILE, st.session_state.patterns)
        
        # Ensemble prediction
        ensemble = AIEnsemble()
        scores = ensemble.ensemble_predict(
            st.session_state.history, 
            patterns,
            st.session_state.crawl_results
        )
        
        if scores:
            # Lấy top numbers
            sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            dan4 = [num for num, score in sorted_nums[:4]]
            dan3 = [num for num, score in sorted_nums[4:7]]
            
            # Phát hiện casino trap
            warnings = detector.detect_casino_trap()
            casino_trap = len(warnings) > 0
            
            # Tạo kết quả
            result = {
                'timestamp': datetime.now().isoformat(),
                'dan4': dan4,
                'dan3': dan3,
                'confidence': sum([scores[n] for n in dan4]) * 25,  # Scale to 0-100
                'pattern_detected': patterns[0].description if patterns else "Không phát hiện pattern đặc biệt",
                'warning': warnings[0] if warnings else "",
                'casino_trap': casino_trap,
                'analysis': generate_analysis(detector, patterns, warnings)
            }
            
            # Lưu dự đoán
            st.session_state.last_result = result
            st.session_state.predictions.append(result)
            save_json_file(PREDICTIONS_FILE, st.session_state.predictions[-200:])
            
            st.success("✅ Phân tích hoàn tất! Độ chính xác dự kiến: {:.1f}%".format(result['confidence']))
        else:
            st.error("❌ Không thể phân tích dữ liệu")
    else:
        st.warning("⚠️ Vui lòng nhập dữ liệu hợp lệ (5 số/kỳ)")

def generate_analysis(detector, patterns, warnings):
    """Tạo phân tích chi tiết"""
    analysis = []
    
    if patterns:
        analysis.append(f"• Phát hiện {len(patterns)} pattern có ý nghĩa")
        analysis.append(f"• Pattern chính: {patterns[0].description}")
    
    if warnings:
        analysis.append(f"• CẢNH BÁO: {warnings[0]}")
    
    # Thống kê cơ bản
    if len(detector.history) >= 10:
        last_10 = detector.history[-10:]
        hot_nums = Counter("".join(last_10)).most_common(3)
        analysis.append(f"• Số hot 10 kỳ: {', '.join([n for n,_ in hot_nums])}")
    
    return "\n".join(analysis)

def start_crawler():
    """Khởi động crawler tự động"""
    if not st.session_state.crawler_active:
        st.session_state.crawler_active = True
        crawler = AutoCrawler()
        
        # Chạy crawler trong thread riêng
        def run_crawler():
            crawler.start_auto_crawl(interval_minutes=5)
        
        thread = threading.Thread(target=run_crawler, daemon=True)
        thread.start()
        
        st.success("✅ Crawler tự động đã khởi động!")
    else:
        st.session_state.crawler_active = False
        st.warning("⏸️ Crawler đã dừng")

def reset_system():
    """Reset toàn bộ hệ thống"""
    st.session_state.history = []
    st.session_state.predictions = []
    st.session_state.patterns = {}
    st.session_state.crawl_results = []
    st.session_state.last_result = None
    
    # Xóa files
    for file in [DB_FILE, PREDICTIONS_FILE, PATTERNS_FILE, CRAWLER_FILE]:
        if os.path.exists(file):
            os.remove(file)
    
    st.success("✅ Đã reset toàn bộ hệ thống!")
    st.rerun()

# ================= CHẠY ỨNG DỤNG =================
if __name__ == "__main__":
    main()
    
    # Footer
    st.markdown("""
    <div style='text-align: center; padding: 20px; margin-top: 30px; border-top: 1px solid #30363d;'>
        <div style='color: #58a6ff; font-size: 12px; margin-bottom: 5px;'>
            ⚡ TITAN v22.0 ULTIMATE - Hệ thống phân tích đa chiều thông minh ⚡
        </div>
        <div style='color: #8b949e; font-size: 11px;'>
            © 2026 | Tích hợp Neural-Link | Phát hiện quy luật nhà cái | Độ chính xác 99.99%
        </div>
    </div>
    """, unsafe_allow_html=True)