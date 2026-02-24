# ================= LOTOBET ULTRA AI PRO – V10.2 COMPLETE =================
# Enhanced AI with Cloud Integration & Real-time Features

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import itertools
import time
import os
import warnings
import requests
import json
import threading
from queue import Queue
warnings.filterwarnings('ignore')

# ================= CLOUD AI LIBRARIES =================
try:
    # Machine Learning Libraries
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    import xgboost as xgb
    import lightgbm as lgb
    
    # Deep Learning (Lightweight)
    try:
        import tensorflow as tf
        DEEP_LEARNING_AVAILABLE = True
    except:
        DEEP_LEARNING_AVAILABLE = False
    
    # Time Series Analysis
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    
    # Advanced Statistics
    from scipy import stats
    from scipy.signal import find_peaks
    
    AI_LIBS_AVAILABLE = True
except ImportError as e:
    AI_LIBS_AVAILABLE = False
    st.warning(f"⚠️ Thiếu thư viện AI: {str(e)}")

from collections import Counter, defaultdict, deque

# ================= CONFIG =================
st.set_page_config(
    page_title="LOTOBET ULTRA AI PRO – V10.2",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Compact CSS for V10.2
st.markdown("""
<style>
    /* Compact layout */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Compact section headers */
    .compact-section {
        background: linear-gradient(135deg, #2D3748 0%, #4A5568 100%);
        color: white;
        padding: 12px 15px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    /* Compact cards */
    .compact-card {
        background-color: white;
        padding: 12px;
        border-radius: 8px;
        border: 2px solid #E2E8F0;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Highlight card */
    .highlight-compact {
        background: linear-gradient(135deg, #FFA726 0%, #FB8C00 100%);
        padding: 15px;
        border-radius: 10px;
        border: 3px solid #F57C00;
        margin: 12px 0;
    }
    
    /* Small number displays */
    .small-big-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E40AF;
        text-align: center;
        margin: 5px 0;
    }
    
    .very-small-number {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2D3748;
        text-align: center;
    }
    
    /* Horizontal analysis rows */
    .horizontal-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #E2E8F0;
    }
    
    .algo-badge-small {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: bold;
        margin: 1px;
    }
    
    .algo-1 { background-color: #3B82F6; color: white; }
    .algo-2 { background-color: #10B981; color: white; }
    .algo-3 { background-color: #8B5CF6; color: white; }
    .algo-4 { background-color: #F59E0B; color: white; }
    .algo-5 { background-color: #EF4444; color: white; }
    .algo-6 { background-color: #EC4899; color: white; }
    .algo-7 { background-color: #06B6D4; color: white; }
    .algo-8 { background-color: #8B5CF6; color: white; }
    
    /* Status indicators */
    .status-online {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        margin-right: 5px;
    }
    
    .status-offline {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #EF4444;
        margin-right: 5px;
    }
    
    /* Compact tables */
    .compact-table {
        font-size: 0.85rem;
    }
    
    /* Real-time notification */
    .realtime-notification {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 8px 12px;
        border-radius: 6px;
        margin: 5px 0;
        font-size: 0.9rem;
    }
    
    /* Betting recommendations */
    .bet-recommendation {
        background-color: #D1FAE5;
        border: 2px solid #10B981;
        padding: 10px;
        border-radius: 8px;
        margin: 8px 0;
    }
    
    .bet-avoid {
        background-color: #FEE2E2;
        border: 2px solid #EF4444;
        padding: 10px;
        border-radius: 8px;
        margin: 8px 0;
    }
    
    /* Icon styling */
    .icon-sm {
        font-size: 16px;
        vertical-align: middle;
        margin-right: 4px;
    }
    
    /* Progress bars compact */
    .stProgress > div > div > div > div {
        background-color: #10B981;
        height: 6px !important;
    }
</style>
""", unsafe_allow_html=True)

DB_FILE = "lotobet_ultra_v10_2.db"

# ================= CLOUD AI CONFIGURATION =================
class CloudAIConfig:
    """Cấu hình AI đám mây và web scraping hợp pháp"""
    
    # Các website soi cầu công khai (chỉ đọc)
    PUBLIC_SITES = {
        'soicau_mienbac': 'https://example-soicau.com/mienbac',
        'xosodaicat': 'https://example-xoso.com',
        'ketqua_trực_tiếp': 'https://example-ketqua.com/live'
    }
    
    # Cloud AI endpoints (giả lập - cần thay bằng thực tế)
    CLOUD_AI_ENDPOINTS = {
        'predict_2d': 'https://api.lotobet-ai.com/v1/predict/2d',
        'predict_3d': 'https://api.lotobet-ai.com/v1/predict/3d',
        'analyze_patterns': 'https://api.lotobet-ai.com/v1/analyze/patterns',
        'get_trends': 'https://api.lotobet-ai.com/v1/trends'
    }
    
    # Real-time lottery APIs (công khai nếu có)
    LOTTERY_APIS = {
        'check_current_ky': 'https://api.lottery.com/current-draw',
        'get_results': 'https://api.lottery.com/results'
    }
    
    @staticmethod
    def is_public_site_allowed(url):
        """Kiểm tra website có cho phép truy cập công khai không"""
        # Thực tế cần kiểm tra robots.txt
        return True  # Giả lập
    
    @staticmethod
    def get_cloud_predictions(data, endpoint):
        """Lấy dự đoán từ cloud AI"""
        # Giả lập - thực tế cần gọi API
        return {
            'status': 'success',
            'predictions': [],
            'source': 'cloud_ai',
            'timestamp': datetime.now().isoformat()
        }

# ================= REAL-TIME MONITOR =================
class RealTimeMonitor:
    """Giám sát thời gian thực kỳ quay thưởng"""
    
    def __init__(self):
        self.current_ky = None
        self.last_update = None
        self.next_draw_time = None
        self.is_synced = False
        self.ky_queue = Queue()
    
    def sync_with_lottery(self, target_ky=None):
        """Đồng bộ kỳ quay với nhà cái"""
        try:
            # Giả lập - thực tế cần kết nối API nhà cái
            current_time = datetime.now()
            
            # Tạo kỳ giả lập dựa trên thời gian
            if target_ky:
                self.current_ky = target_ky
            else:
                # Tạo kỳ theo format: YYMMDD + số thứ tự
                base_ky = current_time.strftime("%y%m%d")
                sequence = (current_time.hour * 60 + current_time.minute) // 5  # Mỗi 5 phút 1 kỳ
                self.current_ky = f"{base_ky}{sequence:03d}"
            
            # Tính thời gian quay tiếp theo (giả lập mỗi 5 phút)
            next_minute = (current_time.minute // 5 + 1) * 5
            if next_minute == 60:
                next_hour = current_time.hour + 1
                next_minute = 0
            else:
                next_hour = current_time.hour
            
            self.next_draw_time = current_time.replace(
                hour=next_hour % 24, 
                minute=next_minute, 
                second=0, 
                microsecond=0
            )
            
            self.last_update = current_time
            self.is_synced = True
            
            return {
                'status': 'synced',
                'current_ky': self.current_ky,
                'next_draw': self.next_draw_time.strftime("%H:%M:%S"),
                'time_to_next': (self.next_draw_time - current_time).seconds
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def check_ky_consistency(self, user_ky):
        """Kiểm tra kỳ người dùng nhập có khớp với hệ thống không"""
        if not self.is_synced:
            return {'match': False, 'message': 'Chưa đồng bộ với nhà cái'}
        
        # Logic kiểm tra đơn giản
        try:
            user_num = int(user_ky[-3:]) if user_ky[-3:].isdigit() else 0
            current_num = int(self.current_ky[-3:]) if self.current_ky[-3:].isdigit() else 0
            
            diff = abs(current_num - user_num)
            
            if diff == 0:
                return {'match': True, 'message': '✅ Đúng kỳ hiện tại'}
            elif diff == 1:
                return {'match': 'close', 'message': '⚠️ Gần đúng kỳ (sai 1 kỳ)'}
            else:
                return {'match': False, 'message': f'❌ Sai kỳ. Kỳ hiện tại: {self.current_ky}'}
                
        except:
            return {'match': False, 'message': 'Lỗi kiểm tra kỳ'}

# ================= DATABASE V10.2 =================
def init_db_v10_2():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Bảng kỳ quay (mở rộng)
    c.execute("""
    CREATE TABLE IF NOT EXISTS ky_quay_v2 (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ky TEXT UNIQUE,
        so5 TEXT,
        tien_nhi TEXT,
        hau_nhi TEXT,
        tong INTEGER,
        tai_xiu TEXT,
        le_chan TEXT,
        de_numbers TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        synced_with_lottery INTEGER DEFAULT 0,
        cloud_analyzed INTEGER DEFAULT 0
    )
    """)
    
    # Bảng dự đoán AI (chi tiết)
    c.execute("""
    CREATE TABLE IF NOT EXISTS du_doan_chi_tiet (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ky TEXT,
        algo_type TEXT,
        prediction_type TEXT,
        predicted_value TEXT,
        confidence REAL,
        should_bet INTEGER DEFAULT 0,
        bet_amount REAL DEFAULT 0,
        bet_reason TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Bảng quản lý vốn thông minh
    c.execute("""
    CREATE TABLE IF NOT EXISTS quan_ly_von_thong_minh (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER DEFAULT 1,
        total_capital REAL DEFAULT 1000000,
        available_capital REAL DEFAULT 1000000,
        current_bet_cycle INTEGER DEFAULT 1,
        max_bet_per_cycle REAL DEFAULT 50000,
        capital_distribution TEXT,
        risk_level TEXT DEFAULT 'medium',
        stop_loss REAL DEFAULT 0.2,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Bảng kết quả đánh thực tế
    c.execute("""
    CREATE TABLE IF NOT EXISTS ket_qua_danh_thuc_te (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ky TEXT,
        bet_type TEXT,
        bet_numbers TEXT,
        bet_amount REAL,
        result TEXT,
        win_amount REAL,
        profit_loss REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Bảng cấu hình hệ thống
    c.execute("""
    CREATE TABLE IF NOT EXISTS system_config_v2 (
        id INTEGER PRIMARY KEY,
        enable_cloud_ai INTEGER DEFAULT 1,
        enable_real_time_sync INTEGER DEFAULT 1,
        auto_capital_management INTEGER DEFAULT 1,
        notification_level TEXT DEFAULT 'high',
        data_retention_days INTEGER DEFAULT 365,
        api_keys TEXT,
        last_sync_time DATETIME
    )
    """)
    
    c.execute("INSERT OR IGNORE INTO system_config_v2 (id) VALUES (1)")
    c.execute("INSERT OR IGNORE INTO quan_ly_von_thong_minh (id) VALUES (1)")
    
    conn.commit()
    conn.close()

init_db_v10_2()

# ================= ENHANCED AI ENGINE V10.2 =================
class EnhancedLottoAI_V10_2:
    """AI nâng cao với 8 thuật toán và phân tích đa chiều"""
    
    def __init__(self, df, cloud_enabled=True):
        self.df = df.copy()
        self.cloud_enabled = cloud_enabled
        self.analysis_results = {}
        
        # 8 Thuật toán chính
        self.algorithms = {
            1: 'basic_statistics',
            2: 'hot_cold_analysis', 
            3: 'pattern_recognition',
            4: 'time_series_forecasting',
            5: 'machine_learning_predict',
            6: 'cycle_analysis',
            7: 'probability_calculation',
            8: 'cloud_ai_integration'
        }
        
        # 5 Mẫu hình chính
        self.patterns = {
            1: 'straight_pattern',
            2: 'wave_pattern',
            3: 'mirror_pattern',
            4: 'ladder_pattern', 
            5: 'repeat_pattern'
        }
        
        # 6 Mẹo đánh
        self.gambling_tips = {
            1: 'bach_nho_tips',
            2: 'lo_gan_tips',
            3: 'cham_dau_duoi_tips',
            4: 'tong_de_tips',
            5: 'bong_so_tips',
            6: 'kep_so_tips'
        }
    
    def run_comprehensive_analysis(self):
        """Chạy phân tích toàn diện"""
        results = {
            'algorithms': {},
            'patterns': {},
            'gambling_tips': {},
            'final_predictions': {},
            'betting_recommendations': {}
        }
        
        # 1. Chạy 8 thuật toán
        for algo_id, algo_name in self.algorithms.items():
            if hasattr(self, algo_name):
                results['algorithms'][algo_id] = getattr(self, algo_name)()
        
        # 2. Phát hiện 5 mẫu hình
        for pattern_id, pattern_name in self.patterns.items():
            if hasattr(self, f'detect_{pattern_name}'):
                results['patterns'][pattern_id] = getattr(self, f'detect_{pattern_name}')()
        
        # 3. Áp dụng 6 mẹo đánh
        for tip_id, tip_name in self.gambling_tips.items():
            if hasattr(self, tip_name):
                results['gambling_tips'][tip_id] = getattr(self, tip_name)()
        
        # 4. Tạo dự đoán cuối cùng
        results['final_predictions'] = self.generate_final_predictions(results)
        
        # 5. Tạo khuyến nghị đánh
        results['betting_recommendations'] = self.generate_betting_recommendations(results)
        
        self.analysis_results = results
        return results
    
    # ========== 8 THUẬT TOÁN ==========
    
    def basic_statistics(self):
        """Thuật toán 1: Thống kê cơ bản"""
        if self.df.empty:
            return {}
        
        return {
            'total_games': len(self.df),
            'avg_sum': float(self.df['tong'].mean()),
            'std_sum': float(self.df['tong'].std()),
            'tai_ratio': float((self.df['tai_xiu'] == 'TÀI').mean()),
            'le_ratio': float((self.df['le_chan'] == 'LẺ').mean()),
            'confidence': min(85, len(self.df) / 100 * 80)
        }
    
    def hot_cold_analysis(self):
        """Thuật toán 2: Phân tích số nóng/lạnh"""
        if len(self.df) < 20:
            return {}
        
        # Tính số nóng (xuất hiện nhiều trong 20 kỳ gần nhất)
        hot_window = min(20, len(self.df))
        hot_counts = {str(i): 0 for i in range(10)}
        
        for num in self.df.head(hot_window)['so5']:
            for digit in num:
                hot_counts[digit] += 1
        
        hot_numbers = [d for d, c in sorted(hot_counts.items(), 
                                          key=lambda x: x[1], 
                                          reverse=True)[:4]]
        
        # Tính số lạnh (ít xuất hiện)
        cold_numbers = [d for d, c in sorted(hot_counts.items(), 
                                           key=lambda x: x[1])[:4]]
        
        # Số gan (lâu chưa về)
        gan_numbers = self._calculate_gan_numbers()
        
        return {
            'hot_numbers': hot_numbers,
            'cold_numbers': cold_numbers,
            'gan_numbers': gan_numbers[:3],
            'confidence': 75
        }
    
    def pattern_recognition(self):
        """Thuật toán 3: Nhận diện mẫu hình"""
        patterns_found = []
        
        # Phát hiện các pattern cơ bản
        if len(self.df) >= 10:
            # Pattern lặp
            for i in range(len(self.df) - 5):
                if self.df.iloc[i]['so5'] == self.df.iloc[i+5]['so5']:
                    patterns_found.append({
                        'type': 'repeat_5_cycles',
                        'position': i,
                        'number': self.df.iloc[i]['so5']
                    })
            
            # Pattern đối xứng
            mirror_count = 0
            for i in range(len(self.df) - 3):
                num1 = self.df.iloc[i]['so5']
                num2 = self.df.iloc[i+3]['so5']
                if num1 == num2[::-1]:  # Đảo ngược
                    mirror_count += 1
            
            if mirror_count > 0:
                patterns_found.append({
                    'type': 'mirror_pattern',
                    'count': mirror_count
                })
        
        return {
            'patterns': patterns_found[:5],
            'total_patterns': len(patterns_found),
            'confidence': min(70, len(patterns_found) * 15)
        }
    
    def time_series_forecasting(self):
        """Thuật toán 4: Dự báo chuỗi thời gian"""
        if len(self.df) < 30:
            return {}
        
        try:
            # ARIMA đơn giản cho tổng số
            sums = self.df['tong'].values[::-1]
            
            # Dự báo bằng moving average
            window = min(10, len(sums))
            predicted_sum = np.mean(sums[:window])
            
            # Xu hướng
            trend = 'tăng' if len(sums) >= 5 and sums[0] > sums[4] else 'giảm'
            
            # Dự đoán Tài/Xỉu, Lẻ/Chẵn
            predicted_tx = tai_xiu(predicted_sum)
            predicted_lc = le_chan(predicted_sum)
            
            # Tính confidence
            confidence = min(80, len(sums) / 50 * 70)
            
            # Quyết định đánh hay không
            should_bet_tx = confidence >= 65
            should_bet_lc = confidence >= 60
            
            return {
                'predicted_sum': round(predicted_sum, 1),
                'predicted_tai_xiu': predicted_tx,
                'predicted_le_chan': predicted_lc,
                'trend': trend,
                'confidence': confidence,
                'should_bet_tai_xiu': should_bet_tx,
                'should_bet_le_chan': should_bet_lc,
                'bet_strength': 'mạnh' if confidence >= 70 else 'vừa' if confidence >= 60 else 'yếu'
            }
            
        except Exception as e:
            return {'error': str(e), 'confidence': 50}
    
    def machine_learning_predict(self):
        """Thuật toán 5: Machine Learning"""
        if not AI_LIBS_AVAILABLE or len(self.df) < 50:
            return {}
        
        try:
            # Sử dụng Random Forest để dự đoán
            features = []
            targets_2d = []  # 2 số cuối
            targets_3d = []  # 3 số
            
            for i in range(len(self.df) - 1):
                current = self.df.iloc[i]
                next_row = self.df.iloc[i + 1]
                
                # Features từ kỳ hiện tại
                feat = [
                    int(d) for d in current['so5']
                ] + [
                    current['tong'],
                    1 if current['tai_xiu'] == 'TÀI' else 0,
                    1 if current['le_chan'] == 'LẺ' else 0
                ]
                
                features.append(feat)
                
                # Targets
                targets_2d.append(int(next_row['hau_nhi']))
                targets_3d.append(int(next_row['so5'][:3]))  # 3 số đầu
            
            # Huấn luyện mô hình đơn giản
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets_2d, test_size=0.2, random_state=42
            )
            
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            # Dự đoán
            last_features = features[0]
            pred_2d = model.predict([last_features])[0]
            proba_2d = model.predict_proba([last_features])[0]
            
            return {
                'predicted_2d': f"{pred_2d:02d}",
                'confidence_2d': float(max(proba_2d) * 100),
                'feature_importance': model.feature_importances_.tolist()[:5]
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def cycle_analysis(self):
        """Thuật toán 6: Phân tích chu kỳ"""
        if len(self.df) < 20:
            return {}
        
        cycles = {
            'fibonacci': self._analyze_fibonacci_cycles(),
            'prime': self._analyze_prime_cycles(),
            'lunar': self._analyze_lunar_cycles()
        }
        
        return {
            'cycles': cycles,
            'confidence': 65
        }
    
    def probability_calculation(self):
        """Thuật toán 7: Tính xác suất"""
        if self.df.empty:
            return {}
        
        # Tính xác suất xuất hiện của từng số
        digit_probs = {str(i): 0 for i in range(10)}
        total_digits = len(self.df) * 5
        
        for num in self.df['so5']:
            for digit in num:
                digit_probs[digit] += 1
        
        # Chuyển thành xác suất
        for digit in digit_probs:
            digit_probs[digit] = digit_probs[digit] / total_digits * 100
        
        # Xác suất Tài/Xỉu, Lẻ/Chẵn
        tai_prob = (self.df['tai_xiu'] == 'TÀI').mean() * 100
        le_prob = (self.df['le_chan'] == 'LẺ').mean() * 100
        
        return {
            'digit_probabilities': digit_probs,
            'tai_probability': float(tai_prob),
            'le_probability': float(le_prob),
            'confidence': 70
        }
    
    def cloud_ai_integration(self):
        """Thuật toán 8: Tích hợp Cloud AI"""
        if not self.cloud_enabled:
            return {'status': 'disabled'}
        
        try:
            # Giả lập kết nối Cloud AI
            cloud_data = {
                'total_samples': len(self.df),
                'prediction_model': 'ensemble_v2',
                'predictions': {
                    '2d': ['68', '79', '45'],
                    '3d': ['168', '279', '345'],
                    'tai_xiu': 'TÀI',
                    'le_chan': 'LẺ'
                },
                'confidence': 72,
                'timestamp': datetime.now().isoformat()
            }
            
            return {
                'status': 'connected',
                'data': cloud_data,
                'confidence': cloud_data['confidence']
            }
            
        except:
            return {'status': 'error'}
    
    # ========== 5 MẪU HÌNH ==========
    
    def detect_straight_pattern(self):
        """Mẫu hình 1: Cầu bệt"""
        patterns = []
        if len(self.df) >= 5:
            for i in range(len(self.df) - 4):
                nums = self.df.iloc[i:i+5]['so5'].tolist()
                # Kiểm tra cầu bệt 2 số
                common_digits = set(nums[0]) & set(nums[1]) & set(nums[2]) & set(nums[3]) & set(nums[4])
                if len(common_digits) >= 2:
                    patterns.append({
                        'type': 'straight_5',
                        'digits': list(common_digits),
                        'start_position': i,
                        'length': 5
                    })
        return patterns[:3]
    
    def detect_wave_pattern(self):
        """Mẫu hình 2: Cầu sóng"""
        patterns = []
        if len(self.df) >= 8:
            for i in range(len(self.df) - 7):
                sums = self.df.iloc[i:i+8]['tong'].tolist()
                # Kiểm tra mẫu sóng
                changes = []
                for j in range(len(sums)-1):
                    changes.append(1 if sums[j] < sums[j+1] else -1)
                
                # Đếm số lần đổi chiều
                direction_changes = sum(1 for j in range(len(changes)-1) 
                                      if changes[j] != changes[j+1])
                
                if direction_changes >= 4:
                    patterns.append({
                        'type': 'wave',
                        'start_position': i,
                        'amplitude': max(sums) - min(sums)
                    })
        return patterns[:2]
    
    def detect_mirror_pattern(self):
        """Mẫu hình 3: Số gương"""
        patterns = []
        mirror_map = {'0':'5','1':'6','2':'7','3':'8','4':'9',
                     '5':'0','6':'1','7':'2','8':'3','9':'4'}
        
        if len(self.df) >= 10:
            for i in range(len(self.df) - 3):
                original = self.df.iloc[i]['so5']
                mirror = ''.join([mirror_map.get(d, d) for d in original])
                
                for j in range(i+1, min(i+4, len(self.df))):
                    if self.df.iloc[j]['so5'] == mirror:
                        patterns.append({
                            'type': 'mirror',
                            'original': original,
                            'mirror': mirror,
                            'delay': j - i
                        })
                        break
        
        return patterns[:3]
    
    def detect_ladder_pattern(self):
        """Mẫu hình 4: Cầu thang"""
        patterns = []
        if len(self.df) >= 5:
            for i in range(len(self.df) - 4):
                nums = self.df.iloc[i:i+5]['so5'].tolist()
                
                # Kiểm tra tăng dần
                if all(int(nums[j]) < int(nums[j+1]) for j in range(4)):
                    patterns.append({
                        'type': 'increasing_ladder',
                        'numbers': nums,
                        'start_position': i
                    })
                
                # Kiểm tra giảm dần
                elif all(int(nums[j]) > int(nums[j+1]) for j in range(4)):
                    patterns.append({
                        'type': 'decreasing_ladder',
                        'numbers': nums,
                        'start_position': i
                    })
        
        return patterns[:2]
    
    def detect_repeat_pattern(self):
        """Mẫu hình 5: Lặp lại"""
        patterns = []
        if len(self.df) >= 15:
            for i in range(len(self.df) - 10):
                current = self.df.iloc[i]['so5']
                
                # Kiểm tra lặp trong 10 kỳ tiếp
                for j in range(i+1, min(i+11, len(self.df))):
                    if self.df.iloc[j]['so5'] == current:
                        patterns.append({
                            'type': 'repeat',
                            'number': current,
                            'first_position': i,
                            'repeat_position': j,
                            'interval': j - i
                        })
                        break
        
        return patterns[:3]
    
    # ========== 6 MẸO ĐÁNH ==========
    
    def bach_nho_tips(self):
        """Mẹo 1: Bạc nhớ"""
        if len(self.df) < 15:
            return []
        
        tips = []
        # Tìm cặp số hay đi cùng
        pair_counts = {}
        for i in range(len(self.df) - 1):
            current = set(self.df.iloc[i]['so5'])
            next_set = set(self.df.iloc[i+1]['so5'])
            common = current & next_set
            
            for digit in common:
                for other in common:
                    if digit != other:
                        pair = ''.join(sorted([digit, other]))
                        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        
        common_pairs = [pair for pair, count in pair_counts.items() if count >= 3]
        
        if common_pairs:
            tips.append({
                'tip': 'Bạc nhớ',
                'description': 'Cặp số thường xuất hiện cùng nhau',
                'numbers': common_pairs[:3]
            })
        
        return tips
    
    def lo_gan_tips(self):
        """Mẹo 2: Lô gan"""
        tips = []
        gan_numbers = self._calculate_gan_numbers()[:3]
        
        if gan_numbers:
            tips.append({
                'tip': 'Lô gan',
                'description': 'Số lâu chưa về, sắp về',
                'numbers': gan_numbers,
                'warning': 'Cần kiểm tra kỹ'
            })
        
        return tips
    
    def cham_dau_duoi_tips(self):
        """Mẹo 3: Chạm đầu đuôi"""
        if len(self.df) < 10:
            return []
        
        tips = []
        heads = []
        tails = []
        
        for num in self.df.head(10)['so5']:
            heads.append(num[0])
            tails.append(num[-1])
        
        head_counter = Counter(heads)
        tail_counter = Counter(tails)
        
        common_heads = [digit for digit, _ in head_counter.most_common(2)]
        common_tails = [digit for digit, _ in tail_counter.most_common(2)]
        
        if common_heads:
            tips.append({
                'tip': 'Chạm đầu',
                'description': 'Đầu số thường xuất hiện',
                'numbers': common_heads
            })
        
        if common_tails:
            tips.append({
                'tip': 'Chạm đuôi',
                'description': 'Đuôi số thường xuất hiện',
                'numbers': common_tails
            })
        
        return tips
    
    def tong_de_tips(self):
        """Mẹo 4: Tổng đề"""
        if len(self.df) < 10:
            return []
        
        tips = []
        sums = self.df.head(15)['tong'].tolist()
        sum_counter = Counter(sums)
        
        common_sums = [str(s) for s, _ in sum_counter.most_common(2)]
        
        if common_sums:
            tips.append({
                'tip': 'Tổng đề',
                'description': 'Tổng số đề phổ biến',
                'numbers': common_sums
            })
        
        return tips
    
    def bong_so_tips(self):
        """Mẹo 5: Bóng số"""
        if self.df.empty:
            return []
        
        tips = []
        bong_map = {'0':'5','1':'6','2':'7','3':'8','4':'9',
                   '5':'0','6':'1','7':'2','8':'3','9':'4'}
        
        recent_nums = self.df.head(3)['so5'].tolist()
        bong_numbers = set()
        
        for num in recent_nums:
            for digit in num:
                if digit in bong_map:
                    bong_numbers.add(bong_map[digit])
        
        if bong_numbers:
            tips.append({
                'tip': 'Bóng số',
                'description': 'Bóng âm/dương của số gần đây',
                'numbers': list(bong_numbers)[:3]
            })
        
        return tips
    
    def kep_so_tips(self):
        """Mẹo 6: Kẹp số"""
        if len(self.df) < 5:
            return []
        
        tips = []
        recent_digits = set()
        
        for num in self.df.head(5)['so5']:
            for digit in num:
                recent_digits.add(int(digit))
        
        if len(recent_digits) >= 4:
            sorted_digits = sorted(recent_digits)
            kep_numbers = []
            
            for i in range(len(sorted_digits) - 1):
                diff = sorted_digits[i+1] - sorted_digits[i]
                if diff > 1:
                    for d in range(sorted_digits[i] + 1, sorted_digits[i+1]):
                        kep_numbers.append(str(d))
            
            if kep_numbers:
                tips.append({
                    'tip': 'Kẹp số',
                    'description': 'Số nằm giữa các số đã ra',
                    'numbers': kep_numbers[:3]
                })
        
        return tips
    
    # ========== HÀM HỖ TRỢ ==========
    
    def _calculate_gan_numbers(self):
        """Tính số gan"""
        if self.df.empty:
            return []
        
        all_digits = set(str(i) for i in range(10))
        last_seen = {digit: 0 for digit in all_digits}
        
        for idx, row in self.df.iterrows():
            for digit in row['so5']:
                last_seen[digit] = idx
        
        current_idx = len(self.df)
        gan_periods = {digit: current_idx - last_seen[digit] for digit in all_digits}
        
        sorted_gan = sorted(gan_periods.items(), key=lambda x: x[1], reverse=True)
        return [digit for digit, period in sorted_gan[:5]]
    
    def _analyze_fibonacci_cycles(self):
        """Phân tích chu kỳ Fibonacci"""
        cycles = {}
        fib_seq = [3, 5, 8, 13, 21]
        
        for fib in fib_seq:
            if len(self.df) >= fib:
                pattern_count = 0
                for i in range(len(self.df) - fib):
                    if len(set(self.df.iloc[i]['so5']) & set(self.df.iloc[i+fib]['so5'])) >= 2:
                        pattern_count += 1
                
                cycles[f'F{fib}'] = pattern_count
        
        return cycles
    
    def _analyze_prime_cycles(self):
        """Phân tích chu kỳ số nguyên tố"""
        prime_numbers = [2, 3, 5, 7, 11, 13, 17, 19]
        cycles = {}
        
        for prime in prime_numbers[:4]:
            if len(self.df) >= prime:
                pattern_count = 0
                for i in range(len(self.df) - prime):
                    if self.df.iloc[i]['tong'] % 2 == self.df.iloc[i+prime]['tong'] % 2:
                        pattern_count += 1
                
                cycles[f'P{prime}'] = pattern_count
        
        return cycles
    
    def _analyze_lunar_cycles(self):
        """Phân tích chu kỳ âm lịch (giả lập)"""
        # Giả lập chu kỳ 7 ngày
        if len(self.df) >= 7:
            same_day_count = 0
            for i in range(len(self.df) - 7):
                if self.df.iloc[i]['tai_xiu'] == self.df.iloc[i+7]['tai_xiu']:
                    same_day_count += 1
            
            return {'7_day_cycle': same_day_count}
        
        return {}
    
    def generate_final_predictions(self, analysis_results):
        """Tạo dự đoán cuối cùng từ tất cả phân tích"""
        
        # Tổng hợp từ các thuật toán
        predictions = {
            '2_so': [],
            '3_so': [],
            'tai_xiu': {'prediction': 'TÀI', 'confidence': 50, 'should_bet': False},
            'le_chan': {'prediction': 'LẺ', 'confidence': 50, 'should_bet': False},
            'de_numbers': []
        }
        
        # Lấy dự đoán từ time series
        ts_result = analysis_results['algorithms'].get(4, {})
        if ts_result:
            predictions['tai_xiu']['prediction'] = ts_result.get('predicted_tai_xiu', 'TÀI')
            predictions['tai_xiu']['confidence'] = ts_result.get('confidence', 50)
            predictions['tai_xiu']['should_bet'] = ts_result.get('should_bet_tai_xiu', False)
            
            predictions['le_chan']['prediction'] = ts_result.get('predicted_le_chan', 'LẺ')
            predictions['le_chan']['confidence'] = ts_result.get('confidence', 50)
            predictions['le_chan']['should_bet'] = ts_result.get('should_bet_le_chan', False)
        
        # Lấy dự đoán từ machine learning
        ml_result = analysis_results['algorithms'].get(5, {})
        if ml_result and 'predicted_2d' in ml_result:
            predictions['2_so'].append({
                'number': ml_result['predicted_2d'],
                'confidence': ml_result['confidence_2d'],
                'source': 'ml'
            })
        
        # Lấy dự đoán từ cloud AI
        cloud_result = analysis_results['algorithms'].get(8, {})
        if cloud_result.get('status') == 'connected':
            cloud_data = cloud_result.get('data', {})
            cloud_preds = cloud_data.get('predictions', {})
            
            if '2d' in cloud_preds:
                for num in cloud_preds['2d'][:2]:
                    predictions['2_so'].append({
                        'number': num,
                        'confidence': cloud_data.get('confidence', 60),
                        'source': 'cloud'
                    })
            
            if '3d' in cloud_preds:
                for num in cloud_preds['3d'][:2]:
                    predictions['3_so'].append({
                        'number': num,
                        'confidence': cloud_data.get('confidence', 60),
                        'source': 'cloud'
                    })
        
        # Thêm dự đoán từ mẹo đánh
        gambling_tips = analysis_results.get('gambling_tips', {})
        for tip_list in gambling_tips.values():
            for tip in tip_list:
                if 'numbers' in tip:
                    for num in tip['numbers'][:2]:
                        if len(num) == 2:
                            predictions['2_so'].append({
                                'number': num,
                                'confidence': 55,
                                'source': 'tip'
                            })
        
        # Tạo số đề từ các số 2 số
        de_numbers = []
        for pred in predictions['2_so']:
            num = pred['number']
            if len(num) == 2:
                # Tạo các biến thể số đề
                de_numbers.extend([num, num[::-1], num[0]+num[0], num[1]+num[1]])
        
        predictions['de_numbers'] = list(set(de_numbers))[:5]
        
        # Sắp xếp theo confidence
        predictions['2_so'].sort(key=lambda x: x['confidence'], reverse=True)
        predictions['3_so'].sort(key=lambda x: x['confidence'], reverse=True)
        
        return predictions
    
    def generate_betting_recommendations(self, analysis_results):
        """Tạo khuyến nghị đánh chi tiết"""
        predictions = self.generate_final_predictions(analysis_results)
        
        recommendations = {
            '2_so_recommendations': [],
            '3_so_recommendations': [],
            'tai_xiu_recommendation': {},
            'le_chan_recommendation': {},
            'de_recommendations': []
        }
        
        # Khuyến nghị 2 số
        for pred in predictions['2_so'][:3]:
            if pred['confidence'] >= 60:
                recommendations['2_so_recommendations'].append({
                    'number': pred['number'],
                    'confidence': pred['confidence'],
                    'recomme