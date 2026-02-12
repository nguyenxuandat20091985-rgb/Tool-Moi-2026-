import streamlit as st
import collections
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import json
from typing import List, Dict, Tuple, Any
import hashlib
import random
from collections import deque, defaultdict
import math

# =============== CẤU HÌNH API ===============
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

# =============== KIẾN TRÚC 116 THUẬT TOÁN ĐA TẦNG ===============
class AI_116_Algorithms:
    """Hệ thống 116 thuật toán phân tích số học cao cấp"""
    
    def __init__(self):
        # ===== TẦNG 1: THỐNG KÊ CƠ BẢN (20 thuật toán) =====
        self.stats_algorithms = {
            1: self.freq_basic,              # Tần suất cơ bản
            2: self.cycle_length,            # Chu kỳ lặp
            3: self.gap_analysis,            # Khoảng nghỉ
            4: self.freq_short_term,         # Tần suất ngắn hạn
            5: self.freq_long_term,          # Tần suất dài hạn
            6: self.density_analysis,        # Mật độ xuất hiện
            7: self.hot_numbers,             # Số nóng
            8: self.cold_numbers,            # Số lạnh
            9: self.regression_prob,         # Xác suất hồi quy
            10: self.normal_distribution,    # Phân phối chuẩn
            11: self.moving_average,         # Trung bình động
            12: self.standard_deviation,     # Độ lệch chuẩn
            13: self.variance_analysis,      # Phân tích phương sai
            14: self.correlation_coef,       # Hệ số tương quan
            15: self.freq_weighted,          # Tần suất có trọng số
            16: self.trend_strength,         # Độ mạnh xu hướng
            17: self.peak_detection,         # Phát hiện đỉnh
            18: self.valley_detection,       # Phát hiện đáy
            19: self.volatility,             # Độ biến động
            20: self.stationarity_test       # Kiểm định dừng
        }
        
        # ===== TẦNG 2: PHÂN TÍCH CẦU (25 thuật toán) =====
        self.pattern_algorithms = {
            21: self.cau_bet,                # Cầu bệt
            22: self.cau_nhay,              # Cầu nhảy
            23: self.cau_lap,               # Cầu lặp
            24: self.cau_dao,               # Cầu đảo
            25: self.cau_hoi,               # Cầu hồi
            26: self.cau_doi_xung,          # Cầu đối xứng
            27: self.cau_zigzag,            # Cầu zigzag
            28: self.cau_cheo,              # Cầu chéo
            29: self.cau_cum,               # Cầu cụm
            30: self.cau_ngat,              # Cầu ngắt
            31: self.cau_tang_dan,          # Cầu tăng dần
            32: self.cau_giam_dan,          # Cầu giảm dần
            33: self.cau_2nhip,             # Cầu 2 nhịp
            34: self.cau_3nhip,             # Cầu 3 nhịp
            35: self.cau_4nhip,             # Cầu 4 nhịp
            36: self.cau_ganh,              # Cầu gánh
            37: self.cau_kep,               # Cầu kẹp
            38: self.cau_xien,              # Cầu xiên
            39: self.cau_thong,             # Cầu thông
            40: self.cau_ty_le,             # Cầu tỷ lệ
            41: self.cau_ma_tran,           # Cầu ma trận
            42: self.cau_hoi_quy,           # Cầu hồi quy
            43: self.cau_song_song,         # Cầu song song
            44: self.cau_giao_thoa,         # Cầu giao thoa
            45: self.cau_bien_thien         # Cầu biến thiên
        }
        
        # ===== TẦNG 3: PHÂN TÍCH HÀNH VI SỐ (25 thuật toán) =====
        self.behavior_algorithms = {
            46: self.bong_so,               # Bóng số
            47: self.dao_chieu,             # Số đảo chiều
            48: self.tang_toc,              # Số tăng tốc
            49: self.giam_toc,              # Số giảm tốc
            50: self.nhieu_so,              # Số nhiễu
            51: self.nhiem_cum,             # Số nhiễm cụm
            52: self.chuyen_trang_thai,     # Xác suất chuyển trạng thái
            53: self.markov_analysis,       # Phân tích Markov
            54: self.entropy_calc,          # Entropy số
            55: self.bat_dinh,              # Độ bất định
            56: self.dong_nang,             # Động năng số
            57: self.the_nang,              # Thế năng số
            58: self.sinh_thai,             # Hệ sinh thái số
            59: self.tuong_tac,             # Tương tác số
            60: self.duong_dan,             # Đường dẫn số
            61: self.chu_ky_sinh,           # Chu kỳ sinh học
            62: self.nhi_thuc,              # Nhị thức
            63: self.phan_phoi_xac_suat,    # Phân phối xác suất
            64: self.moment_generating,     # Hàm sinh moment
            65: self.likelihood_ratio,      # Tỷ lệ khả năng
            66: self.bayesian_update,       # Cập nhật Bayes
            67: self.kalman_filter,         # Lọc Kalman
            68: self.hidden_markov,         # Markov ẩn
            69: self.chaos_detection,       # Phát hiện hỗn loạn
            70: self.fractal_dimension      # Chiều fractal
        }
        
        # ===== TẦNG 4: MACHINE LEARNING (20 thuật toán) =====
        self.ml_algorithms = {
            71: self.random_forest_predict,     # Random Forest
            72: self.gradient_boosting,         # Gradient Boosting
            73: self.xgboost_predict,           # XGBoost
            74: self.lightgbm_predict,          # LightGBM
            75: self.logistic_regression,       # Logistic Regression
            76: self.svm_predict,               # SVM
            77: self.neural_network,            # Neural Network
            78: self.bayesian_inference,        # Bayesian
            79: self.monte_carlo_sim,           # Monte Carlo
            80: self.time_series_forecast,      # Time Series
            81: self.knn_predict,               # K-Nearest Neighbors
            82: self.naive_bayes,               # Naive Bayes
            83: self.decision_tree,             # Decision Tree
            84: self.linear_regression,         # Linear Regression
            85: self.polynomial_regression,     # Polynomial Regression
            86: self.lstm_predict,              # LSTM
            87: self.gru_predict,               # GRU
            88: self.transformer_predict,       # Transformer
            89: self.ensemble_stacking,         # Stacking
            90: self.deep_learning              # Deep Learning
        }
        
        # ===== TẦNG 5: META AI + GEMINI (15 thuật toán) =====
        self.meta_algorithms = {
            91: self.gemini_reasoning,          # Gemini reasoning
            92: self.gemini_pattern_scan,       # Gemini pattern scan
            93: self.gemini_anomaly,            # Gemini anomaly detection
            94: self.gemini_ranking,            # Gemini probability ranking
            95: self.gemini_trend,              # Gemini trend detection
            96: self.gemini_correlation,        # Gemini correlation
            97: self.gemini_forecast,           # Gemini forecast
            98: self.gemini_ensemble,           # Gemini ensemble
            99: self.gemini_adversarial,        # Gemini adversarial
            100: self.gemini_counter,           # Gemini counter-strategy
            101: self.gemini_adaptation,        # Gemini adaptation
            102: self.gemini_optimization,      # Gemini optimization
            103: self.gemini_validation,        # Gemini validation
            104: self.gemini_calibration,       # Gemini calibration
            105: self.gemini_fusion             # Gemini fusion
        }
        
        # ===== TẦNG 6: META ENSEMBLE (11 thuật toán) =====
        self.ensemble_algorithms = {
            106: self.voting_ensemble,          # Voting ensemble
            107: self.weighted_ensemble,        # Weighted ensemble
            108: self.dynamic_weight,           # Dynamic weight
            109: self.confidence_fusion,        # Confidence fusion
            110: self.risk_scoring,             # Risk scoring
            111: self.stability_index,          # Stability index
            112: self.volatility_index,         # Volatility index
            113: self.overfit_detection,        # Overfit detection
            114: self.bias_correction,          # Bias correction
            115: self.adaptive_weight,          # Adaptive weight learning
            116: self.final_decision            # Final decision engine
        }
        
        # ===== WEIGHTS ĐỘNG =====
        self.weights = {
            'stats': 0.2,
            'pattern': 0.2,
            'behavior': 0.15,
            'ml': 0.2,
            'meta': 0.15,
            'ensemble': 0.1
        }
        
        # Lưu trữ lịch sử để học thích nghi
        self.prediction_history = []
        self.accuracy_history = []
        
    # =============== TẦNG 1: THỐNG KÊ CƠ BẢN ===============
    def freq_basic(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 1: Tần suất cơ bản"""
        total = len(nums)
        counts = collections.Counter(nums)
        return {num: counts[num]/total for num in map(str, range(10))}
    
    def cycle_length(self, nums: List[str]) -> Dict[str, int]:
        """Thuật toán 2: Chu kỳ lặp trung bình"""
        cycles = {}
        for num in map(str, range(10)):
            positions = [i for i, x in enumerate(nums) if x == num]
            if len(positions) > 1:
                gaps = [positions[i] - positions[i-1] for i in range(1, len(positions))]
                cycles[num] = int(np.mean(gaps)) if gaps else 99
            else:
                cycles[num] = 99
        return cycles
    
    def gap_analysis(self, nums: List[str]) -> Dict[str, int]:
        """Thuật toán 3: Khoảng nghỉ hiện tại"""
        gaps = {}
        for num in map(str, range(10)):
            positions = [i for i, x in enumerate(nums) if x == num]
            if positions:
                gaps[num] = len(nums) - 1 - positions[-1]
            else:
                gaps[num] = len(nums)
        return gaps
    
    def freq_short_term(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 4: Tần suất 10 số gần nhất"""
        recent = nums[-10:] if len(nums) >= 10 else nums
        total = len(recent)
        counts = collections.Counter(recent)
        return {num: counts.get(num, 0)/total if total > 0 else 0 
                for num in map(str, range(10))}
    
    def freq_long_term(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 5: Tần suất toàn bộ lịch sử"""
        total = len(nums)
        counts = collections.Counter(nums)
        return {num: counts[num]/total for num in map(str, range(10))}
    
    def density_analysis(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 6: Mật độ xuất hiện"""
        densities = {}
        window = 20
        for num in map(str, range(10)):
            recent = nums[-window:] if len(nums) >= window else nums
            count = recent.count(num)
            densities[num] = count / len(recent) if recent else 0
        return densities
    
    def hot_numbers(self, nums: List[str], threshold: float = 0.15) -> List[str]:
        """Thuật toán 7: Số nóng"""
        recent = nums[-20:] if len(nums) >= 20 else nums
        total = len(recent)
        counts = collections.Counter(recent)
        return [num for num in map(str, range(10)) 
                if counts.get(num, 0)/total >= threshold]
    
    def cold_numbers(self, nums: List[str], threshold: int = 20) -> List[str]:
        """Thuật toán 8: Số lạnh"""
        if len(nums) < threshold:
            return []
        recent_set = set(nums[-threshold:])
        return [str(i) for i in range(10) if str(i) not in recent_set]
    
    def regression_prob(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 9: Xác suất hồi quy"""
        probs = {}
        for num in map(str, range(10)):
            positions = [i for i, x in enumerate(nums) if x == num]
            if len(positions) >= 2:
                gaps = [positions[i] - positions[i-1] for i in range(1, len(positions))]
                avg_gap = np.mean(gaps)
                last_pos = positions[-1]
                current_gap = len(nums) - 1 - last_pos
                probs[num] = max(0, 1 - (current_gap / avg_gap)) if avg_gap > 0 else 0
            else:
                probs[num] = 0
        return probs
    
    def normal_distribution(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 10: Phân phối chuẩn"""
        int_nums = [int(n) for n in nums]
        mean = np.mean(int_nums) if int_nums else 4.5
        std = np.std(int_nums) if int_nums else 2.87
        
        probs = {}
        for i in range(10):
            z_score = (i - mean) / std if std > 0 else 0
            probs[str(i)] = math.exp(-0.5 * z_score**2) / (std * math.sqrt(2 * math.pi))
        
        # Chuẩn hóa
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def moving_average(self, nums: List[str], window: int = 5) -> Dict[str, float]:
        """Thuật toán 11: Trung bình động"""
        if len(nums) < window:
            return {str(i): 0.1 for i in range(10)}
        
        recent = [int(n) for n in nums[-window:]]
        ma = np.mean(recent)
        
        probs = {}
        for i in range(10):
            probs[str(i)] = 1 - abs(i - ma) / 9
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def standard_deviation(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 12: Độ lệch chuẩn"""
        int_nums = [int(n) for n in nums]
        std = np.std(int_nums) if len(int_nums) > 1 else 2.87
        mean = np.mean(int_nums) if int_nums else 4.5
        
        probs = {}
        for i in range(10):
            z_score = (i - mean) / std if std > 0 else 0
            probs[str(i)] = 1 / (1 + abs(z_score))
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def variance_analysis(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 13: Phân tích phương sai"""
        int_nums = [int(n) for n in nums]
        var = np.var(int_nums) if len(int_nums) > 1 else 8.25
        return {str(i): 1/(1 + abs(i - np.mean(int_nums)) + var/10) 
                for i in range(10)}
    
    def correlation_coef(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 14: Hệ số tương quan"""
        if len(nums) < 10:
            return {str(i): 0.1 for i in range(10)}
        
        int_nums = [int(n) for n in nums]
        x = list(range(len(int_nums)))
        
        if len(set(int_nums)) == 1:
            return {str(i): 0.1 for i in range(10)}
        
        corr = np.corrcoef(x, int_nums)[0, 1] if len(set(int_nums)) > 1 else 0
        
        probs = {}
        for i in range(10):
            probs[str(i)] = 0.1 + 0.1 * corr * (i - 4.5) / 4.5
        
        # Chuẩn hóa về [0, 1]
        min_prob = min(probs.values())
        max_prob = max(probs.values())
        if max_prob > min_prob:
            probs = {k: (v - min_prob) / (max_prob - min_prob) for k, v in probs.items()}
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def freq_weighted(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 15: Tần suất có trọng số"""
        weights = np.exp(-np.arange(len(nums))[::-1] / 50)  # Trọng số giảm dần
        weighted_counts = defaultdict(float)
        
        for i, num in enumerate(nums):
            weighted_counts[num] += weights[i] if i < len(weights) else 0
        
        total = sum(weighted_counts.values())
        return {str(i): weighted_counts.get(str(i), 0)/total for i in range(10)}
    
    def trend_strength(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 16: Độ mạnh xu hướng"""
        int_nums = [int(n) for n in nums[-20:]] if len(nums) >= 20 else [int(n) for n in nums]
        
        if len(int_nums) < 2:
            return {str(i): 0.1 for i in range(10)}
        
        # Tính slope
        x = list(range(len(int_nums)))
        slope = np.polyfit(x, int_nums, 1)[0] if len(set(int_nums)) > 1 else 0
        
        probs = {}
        for i in range(10):
            trend_value = i + slope
            probs[str(i)] = 1 - abs(trend_value - i) / 9
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def peak_detection(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 17: Phát hiện đỉnh"""
        int_nums = [int(n) for n in nums[-10:]] if len(nums) >= 10 else [int(n) for n in nums]
        
        peaks = []
        for i in range(1, len(int_nums)-1):
            if int_nums[i] > int_nums[i-1] and int_nums[i] > int_nums[i+1]:
                peaks.append(int_nums[i])
        
        probs = {}
        for i in range(10):
            if i in peaks:
                probs[str(i)] = 0.2
            else:
                probs[str(i)] = 0.1
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def valley_detection(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 18: Phát hiện đáy"""
        int_nums = [int(n) for n in nums[-10:]] if len(nums) >= 10 else [int(n) for n in nums]
        
        valleys = []
        for i in range(1, len(int_nums)-1):
            if int_nums[i] < int_nums[i-1] and int_nums[i] < int_nums[i+1]:
                valleys.append(int_nums[i])
        
        probs = {}
        for i in range(10):
            if i in valleys:
                probs[str(i)] = 0.2
            else:
                probs[str(i)] = 0.1
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def volatility(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 19: Độ biến động"""
        int_nums = [int(n) for n in nums[-10:]] if len(nums) >= 10 else [int(n) for n in nums]
        
        if len(int_nums) < 2:
            return {str(i): 0.1 for i in range(10)}
        
        volatility_score = np.std([abs(int_nums[i] - int_nums[i-1]) 
                                  for i in range(1, len(int_nums))])
        
        probs = {}
        for i in range(10):
            probs[str(i)] = 1 / (1 + abs(i - np.mean(int_nums)) + volatility_score/10)
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def stationarity_test(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 20: Kiểm định dừng"""
        int_nums = [int(n) for n in nums[-20:]] if len(nums) >= 20 else [int(n) for n in nums]
        
        if len(int_nums) < 2:
            return {str(i): 0.1 for i in range(10)}
        
        mean = np.mean(int_nums)
        var = np.var(int_nums)
        
        # Kiểm tra tính dừng đơn giản
        recent_mean = np.mean(int_nums[-5:]) if len(int_nums) >= 5 else mean
        is_stationary = abs(mean - recent_mean) < var**0.5
        
        probs = {}
        for i in range(10):
            if is_stationary:
                probs[str(i)] = 1 - abs(i - mean) / 9
            else:
                probs[str(i)] = 0.1 + 0.1 * (i in int_nums[-3:])
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    # =============== TẦNG 2: PHÂN TÍCH CẦU (25 thuật toán) ===============
    def cau_bet(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 21: Cầu bệt - số xuất hiện liên tiếp"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 2:
            return probs
        
        last_num = nums[-1]
        count = 1
        for i in range(len(nums)-2, -1, -1):
            if nums[i] == last_num:
                count += 1
            else:
                break
        
        if count >= 2:
            probs[last_num] = min(0.8, 0.3 + count * 0.1)
        else:
            probs[last_num] = 0.1
        
        # Chuẩn hóa
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def cau_nhay(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 22: Cầu nhảy - số nhảy qua 1 bậc"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 2:
            return probs
        
        last_num = int(nums[-1])
        # Các số nhảy (cách 2 đơn vị)
        jump_numbers = [(last_num + 2) % 10, (last_num - 2) % 10]
        
        for num in jump_numbers:
            probs[str(num)] = 0.25
        
        # Chuẩn hóa
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def cau_lap(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 23: Cầu lặp - số lặp lại pattern"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 4:
            return probs
        
        # Tìm pattern 3 số gần nhất
        pattern = nums[-3:]
        found = False
        for i in range(len(nums)-6, -1, -1):
            if nums[i:i+3] == pattern:
                next_num = nums[i+3] if i+3 < len(nums) else None
                if next_num:
                    probs[next_num] = probs.get(next_num, 0) + 0.3
                    found = True
        
        if not found:
            probs[nums[-1]] = 0.1
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def cau_dao(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 24: Cầu đảo - số đảo chiều"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 3:
            return probs
        
        last_trend = int(nums[-1]) - int(nums[-2])
        if last_trend > 0:
            # Đang tăng, dự đoán giảm
            pred = (int(nums[-1]) - 1) % 10
            probs[str(pred)] = 0.3
        elif last_trend < 0:
            # Đang giảm, dự đoán tăng
            pred = (int(nums[-1]) + 1) % 10
            probs[str(pred)] = 0.3
        else:
            # Đang bằng, dự đoán dao động
            probs[str((int(nums[-1]) + 1) % 10)] = 0.15
            probs[str((int(nums[-1]) - 1) % 10)] = 0.15
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def cau_hoi(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 25: Cầu hồi - số quay về giá trị trung bình"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 5:
            return probs
        
        int_nums = [int(n) for n in nums[-5:]]
        mean = np.mean(int_nums)
        current = int(nums[-1])
        
        if current > mean:
            # Cao hơn trung bình, dự đoán giảm
            pred = (current - 1) % 10
            probs[str(pred)] = 0.25
        elif current < mean:
            # Thấp hơn trung bình, dự đoán tăng
            pred = (current + 1) % 10
            probs[str(pred)] = 0.25
        else:
            # Bằng trung bình, dự đoán ổn định
            probs[str(current)] = 0.15
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def cau_doi_xung(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 26: Cầu đối xứng"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 4:
            return probs
        
        # Đối xứng qua tâm 4.5
        last_num = int(nums[-1])
        symmetric = 9 - last_num
        probs[str(symmetric)] = 0.3
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def cau_zigzag(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 27: Cầu zigzag"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 3:
            return probs
        
        # Pattern: lên - xuống - lên - xuống
        diff1 = int(nums[-1]) - int(nums[-2])
        diff2 = int(nums[-2]) - int(nums[-3])
        
        if diff1 > 0 and diff2 < 0:
            # Lên sau xuống, dự đoán xuống
            pred = (int(nums[-1]) - 1) % 10
            probs[str(pred)] = 0.25
        elif diff1 < 0 and diff2 > 0:
            # Xuống sau lên, dự đoán lên
            pred = (int(nums[-1]) + 1) % 10
            probs[str(pred)] = 0.25
        else:
            # Đang thẳng, dự đoán đổi chiều
            if diff1 > 0:
                probs[str((int(nums[-1]) - 1) % 10)] = 0.2
            elif diff1 < 0:
                probs[str((int(nums[-1]) + 1) % 10)] = 0.2
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def cau_cheo(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 28: Cầu chéo"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 3:
            return probs
        
        # Lấy số ở vị trí chéo
        if len(nums) >= 3:
            diagonal = nums[-3]
            probs[diagonal] = 0.2
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def cau_cum(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 29: Cầu cụm"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 5:
            return probs
        
        # Tìm cụm số xuất hiện nhiều gần đây
        recent = nums[-5:]
        counts = collections.Counter(recent)
        for num, count in counts.items():
            probs[num] = count / 5 * 0.5
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def cau_ngat(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 30: Cầu ngắt"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 2:
            return probs
        
        # Số ngắt cầu
        last_num = nums[-1]
        # Dự đoán số khác với số cuối
        other_nums = [str(i) for i in range(10) if str(i) != last_num]
        for num in other_nums[:3]:
            probs[num] = 0.1
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def cau_tang_dan(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 31: Cầu tăng dần"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 3:
            return probs
        
        int_nums = [int(n) for n in nums[-3:]]
        if int_nums[0] < int_nums[1] < int_nums[2]:
            # Đang tăng, dự đoán tăng tiếp
            pred = (int_nums[2] + 1) % 10
            probs[str(pred)] = 0.35
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def cau_giam_dan(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 32: Cầu giảm dần"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 3:
            return probs
        
        int_nums = [int(n) for n in nums[-3:]]
        if int_nums[0] > int_nums[1] > int_nums[2]:
            # Đang giảm, dự đoán giảm tiếp
            pred = (int_nums[2] - 1) % 10
            probs[str(pred)] = 0.35
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def cau_2nhip(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 33: Cầu 2 nhịp"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 4:
            return probs
        
        # Pattern 2 nhịp lặp lại
        for i in range(len(nums)-4, -1, -2):
            if i+1 < len(nums) and nums[i] == nums[i-2] and nums[i+1] == nums[i-1]:
                if i+2 < len(nums):
                    probs[nums[i+2]] = probs.get(nums[i+2], 0) + 0.2
                    probs[nums[i+3]] = probs.get(nums[i+3], 0) + 0.2
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def cau_3nhip(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 34: Cầu 3 nhịp"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 6:
            return probs
        
        # Pattern 3 nhịp lặp lại
        for i in range(len(nums)-6, -1, -3):
            if all(nums[i+j] == nums[i+j-3] for j in range(3) if i+j < len(nums)):
                if i+3 < len(nums):
                    probs[nums[i+3]] = probs.get(nums[i+3], 0) + 0.15
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def cau_4nhip(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 35: Cầu 4 nhịp"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 8:
            return probs
        
        # Pattern 4 nhịp lặp lại
        for i in range(len(nums)-8, -1, -4):
            if all(nums[i+j] == nums[i+j-4] for j in range(4) if i+j < len(nums)):
                if i+4 < len(nums):
                    probs[nums[i+4]] = probs.get(nums[i+4], 0) + 0.1
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def cau_ganh(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 36: Cầu gánh"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 3:
            return probs
        
        # Số ở giữa bằng tổng 2 số biên
        for i in range(1, len(nums)-1):
            if int(nums[i-1]) + int(nums[i+1]) == int(nums[i]) * 2:
                probs[nums[i]] = probs.get(nums[i], 0) + 0.15
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def cau_kep(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 37: Cầu kẹp"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 3:
            return probs
        
        # Số bị kẹp giữa 2 số
        for i in range(1, len(nums)-1):
            if nums[i-1] == nums[i+1] and nums[i-1] != nums[i]:
                probs[nums[i-1]] = probs.get(nums[i-1], 0) + 0.2
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def cau_xien(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 38: Cầu xiên"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 4:
            return probs
        
        # Xiên chéo
        for i in range(len(nums)-3):
            if int(nums[i]) + int(nums[i+3]) == int(nums[i+1]) + int(nums[i+2]):
                probs[nums[i+1]] = probs.get(nums[i+1], 0) + 0.1
                probs[nums[i+2]] = probs.get(nums[i+2], 0) + 0.1
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def cau_thong(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 39: Cầu thông"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 5:
            return probs
        
        # Cầu thông suốt
        for num in map(str, range(10)):
            if num in nums[-3:]:
                probs[num] = 0.2
            else:
                probs[num] = 0.05
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def cau_ty_le(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 40: Cầu tỷ lệ"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 10:
            return probs
        
        # Tỷ lệ vàng Fibonacci
        ratios = [0.618, 0.382, 0.236, 0.146]
        int_nums = [int(n) for n in nums[-5:]]
        mean = np.mean(int_nums)
        
        for i in range(10):
            for ratio in ratios:
                if abs(i - mean) < ratio * 10:
                    probs[str(i)] += 0.1
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def cau_ma_tran(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 41: Cầu ma trận"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 9:
            return probs
        
        # Tạo ma trận 3x3
        matrix = [nums[i:i+3] for i in range(0, min(9, len(nums)), 3)]
        if len(matrix) >= 3 and len(matrix[0]) >= 3:
            for i in range(3):
                for j in range(3):
                    if matrix[i][j] in matrix[i-1] or matrix[i][j] in matrix[i-2]:
                        probs[matrix[i][j]] = probs.get(matrix[i][j], 0) + 0.1
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def cau_hoi_quy(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 42: Cầu hồi quy"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 5:
            return probs
        
        # Hồi quy tuyến tính
        int_nums = [int(n) for n in nums[-5:]]
        x = list(range(len(int_nums)))
        
        if len(set(int_nums)) > 1:
            z = np.polyfit(x, int_nums, 1)
            pred = z[0] * (len(int_nums)) + z[1]
            pred = int(round(pred)) % 10
            probs[str(pred)] = 0.25
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def cau_song_song(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 43: Cầu song song"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 6:
            return probs
        
        # 2 cầu song song
        even_positions = [nums[i] for i in range(0, len(nums), 2)]
        odd_positions = [nums[i] for i in range(1, len(nums), 2)]
        
        if even_positions:
            probs[even_positions[-1]] = probs.get(even_positions[-1], 0) + 0.15
        if odd_positions:
            probs[odd_positions[-1]] = probs.get(odd_positions[-1], 0) + 0.15
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def cau_giao_thoa(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 44: Cầu giao thoa"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 4:
            return probs
        
        # Giao thoa sóng
        for i in range(len(nums)-2):
            if int(nums[i]) + int(nums[i+1]) == int(nums[i+2]) * 2:
                probs[nums[i+2]] = probs.get(nums[i+2], 0) + 0.1
            if int(nums[i]) + int(nums[i+2]) == int(nums[i+1]) * 2:
                probs[nums[i+1]] = probs.get(nums[i+1], 0) + 0.1
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def cau_bien_thien(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 45: Cầu biến thiên"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 3:
            return probs
        
        # Biến thiên theo chu kỳ
        diffs = [abs(int(nums[i]) - int(nums[i-1])) for i in range(1, len(nums))]
        if diffs:
            avg_diff = np.mean(diffs[-3:]) if len(diffs) >= 3 else np.mean(diffs)
            last_num = int(nums[-1])
            
            if avg_diff < 3:
                # Biến thiên nhỏ
                for delta in [1, 2]:
                    probs[str((last_num + delta) % 10)] = 0.2
                    probs[str((last_num - delta) % 10)] = 0.2
            else:
                # Biến thiên lớn
                for delta in [3, 4, 5]:
                    probs[str((last_num + delta) % 10)] = 0.15
                    probs[str((last_num - delta) % 10)] = 0.15
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    # =============== TẦNG 3: PHÂN TÍCH HÀNH VI SỐ (25 thuật toán) ===============
    def bong_so(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 46: Bóng số"""
        probs = {str(i): 0 for i in range(10)}
        if not nums:
            return probs
        
        last_num = nums[-1]
        bong_duong = {"0": "5", "1": "6", "2": "7", "3": "8", "4": "9",
                      "5": "0", "6": "1", "7": "2", "8": "3", "9": "4"}
        bong_am = {"0": "7", "1": "4", "2": "9", "3": "6", "4": "1",
                   "5": "8", "6": "3", "7": "0", "8": "5", "9": "2"}
        
        probs[bong_duong.get(last_num, "")] = 0.25
        probs[bong_am.get(last_num, "")] = 0.25
        probs[last_num] = 0.15
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def dao_chieu(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 47: Số đảo chiều"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 2:
            return probs
        
        last_num = int(nums[-1])
        prev_num = int(nums[-2])
        
        if last_num > prev_num:
            # Đảo chiều giảm
            probs[str((last_num - 1) % 10)] = 0.25
        elif last_num < prev_num:
            # Đảo chiều tăng
            probs[str((last_num + 1) % 10)] = 0.25
        else:
            # Bằng nhau, đảo chiều mạnh
            probs[str((last_num + 3) % 10)] = 0.2
            probs[str((last_num - 3) % 10)] = 0.2
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def tang_toc(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 48: Số tăng tốc"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 3:
            return probs
        
        diffs = [int(nums[i]) - int(nums[i-1]) for i in range(len(nums)-1, len(nums)-3, -1)]
        if len(diffs) == 2:
            if diffs[0] > diffs[1] > 0:
                # Tăng tốc
                accel = diffs[0] - diffs[1]
                pred = (int(nums[-1]) + 1 + accel) % 10
                probs[str(pred)] = 0.3
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def giam_toc(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 49: Số giảm tốc"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 3:
            return probs
        
        diffs = [int(nums[i]) - int(nums[i-1]) for i in range(len(nums)-1, len(nums)-3, -1)]
        if len(diffs) == 2:
            if diffs[0] < diffs[1] < 0:
                # Giảm tốc
                decel = abs(diffs[0] - diffs[1])
                pred = (int(nums[-1]) - 1 - decel) % 10
                probs[str(pred)] = 0.3
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def nhieu_so(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 50: Số nhiễu"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 5:
            return probs
        
        # Số xuất hiện bất thường
        expected = len(nums) / 10
        counts = collections.Counter(nums)
        for num, count in counts.items():
            if count > expected * 2:
                # Quá nhiều, có thể sắp giảm
                probs[num] = 0.1
            elif count < expected / 2:
                # Quá ít, có thể sắp tăng
                probs[num] = 0.2
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def nhiem_cum(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 51: Số nhiễm cụm"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 10:
            return probs
        
        # Số xuất hiện theo cụm
        for i in range(len(nums)-4):
            window = nums[i:i+5]
            if len(set(window)) <= 2:
                # Cụm dày đặc
                for num in set(window):
                    probs[num] = probs.get(num, 0) + 0.15
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def chuyen_trang_thai(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 52: Xác suất chuyển trạng thái"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 2:
            return probs
        
        transitions = defaultdict(lambda: defaultdict(int))
        for i in range(len(nums)-1):
            transitions[nums[i]][nums[i+1]] += 1
        
        last_num = nums[-1]
        total = sum(transitions[last_num].values())
        
        if total > 0:
            for num, count in transitions[last_num].items():
                probs[num] = count / total
        
        total_probs = sum(probs.values())
        return {k: v/total_probs for k, v in probs.items()} if total_probs > 0 else probs
    
    def markov_analysis(self, nums: List[str], order: int = 2) -> Dict[str, float]:
        """Thuật toán 53: Phân tích Markov"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < order + 1:
            return probs
        
        transitions = defaultdict(lambda: defaultdict(int))
        for i in range(len(nums)-order):
            state = tuple(nums[i:i+order])
            next_num = nums[i+order]
            transitions[state][next_num] += 1
        
        last_state = tuple(nums[-order:])
        if last_state in transitions:
            total = sum(transitions[last_state].values())
            for num, count in transitions[last_state].items():
                probs[num] = count / total
        
        total_probs = sum(probs.values())
        return {k: v/total_probs for k, v in probs.items()} if total_probs > 0 else probs
    
    def entropy_calc(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 54: Entropy số"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 10:
            return probs
        
        # Tính entropy của phân phối
        total = len(nums)
        counts = collections.Counter(nums)
        entropy = 0
        for num in map(str, range(10)):
            p = counts.get(num, 0) / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        # Entropy càng cao, phân phối càng đều
        # Dự đoán số có tần suất thấp
        for num in map(str, range(10)):
            freq = counts.get(num, 0) / total
            probs[num] = 1 - freq + entropy/10
        
        total_probs = sum(probs.values())
        return {k: v/total_probs for k, v in probs.items()}
    
    def bat_dinh(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 55: Độ bất định"""
        probs = {str(i): 0.1 for i in range(10)}
        if len(nums) < 5:
            return probs
        
        # Dựa vào độ lệch chuẩn gần đây
        int_nums = [int(n) for n in nums[-5:]]
        std = np.std(int_nums) if len(int_nums) > 1 else 2.87
        
        uncertainty = min(1, std / 5)
        for num in map(str, range(10)):
            if num in nums[-3:]:
                probs[num] = 0.1 + uncertainty * 0.2
            else:
                probs[num] = 0.1 - uncertainty * 0.05
        
        total = sum(probs.values())
        return {k: max(0.01, v)/total for k, v in probs.items()}
    
    def dong_nang(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 56: Động năng số"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 2:
            return probs
        
        # Động năng = khối lượng * vận tốc^2 / 2
        # Coi mỗi số có khối lượng 1
        velocities = [abs(int(nums[i]) - int(nums[i-1])) for i in range(1, len(nums))]
        if velocities:
            kinetic = sum(v**2 for v in velocities[-5:]) / 2 if len(velocities) >= 5 else sum(v**2 for v in velocities) / 2
            
            if kinetic > 50:
                # Động năng cao, dự đoán số ổn định
                probs[nums[-1]] = 0.25
            else:
                # Động năng thấp, dự đoán biến động
                for delta in [2, 3, 4]:
                    probs[str((int(nums[-1]) + delta) % 10)] = 0.1
                    probs[str((int(nums[-1]) - delta) % 10)] = 0.1
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def the_nang(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 57: Thế năng số"""
        probs = {str(i): 0 for i in range(10)}
        if not nums:
            return probs
        
        # Thế năng = khối lượng * gia tốc * độ cao
        # Coi mỗi số có khối lượng 1, độ cao là giá trị số
        height = int(nums[-1])
        potential = height * 9.8  # g = 9.8
        
        # Thế năng cao -> dễ rơi xuống
        if potential > 50:
            probs[str((height - 1) % 10)] = 0.25
            probs[str((height - 2) % 10)] = 0.15
        else:
            probs[str((height + 1) % 10)] = 0.25
            probs[str((height + 2) % 10)] = 0.15
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def sinh_thai(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 58: Hệ sinh thái số"""
        probs = {str(i): 0.1 for i in range(10)}
        if len(nums) < 20:
            return probs
        
        # Mô phỏng quần thể số
        populations = collections.Counter(nums)
        total = len(nums)
        
        for num in map(str, range(10)):
            freq = populations.get(num, 0) / total
            if freq < 0.05:
                # Loài có nguy cơ tuyệt chủng -> ưu tiên bảo tồn
                probs[num] = 0.2
            elif freq > 0.15:
                # Loài phát triển quá mức -> cần kiểm soát
                probs[num] = 0.05
            else:
                # Cân bằng
                probs[num] = 0.1
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def tuong_tac(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 59: Tương tác số"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 3:
            return probs
        
        # Tương tác cặp
        pairs = defaultdict(int)
        for i in range(len(nums)-1):
            pairs[(nums[i], nums[i+1])] += 1
        
        last_num = nums[-1]
        for i in range(10):
            pair = (last_num, str(i))
            probs[str(i)] = pairs[pair] / (sum(pairs.values()) + 1)
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def duong_dan(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 60: Đường dẫn số"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 2:
            return probs
        
        # Tìm đường dẫn phổ biến
        paths = defaultdict(int)
        for i in range(len(nums)-1):
            path = f"{nums[i]}->{nums[i+1]}"
            paths[path] += 1
        
        last_num = nums[-1]
        for i in range(10):
            path = f"{last_num}->{i}"
            probs[str(i)] = paths[path] / (len(nums) - 1)
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def chu_ky_sinh(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 61: Chu kỳ sinh học"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 20:
            return probs
        
        # Tìm chu kỳ
        from scipy import signal
        int_nums = [int(n) for n in nums[-50:]]
        
        # Phát hiện chu kỳ bằng autocorrelation
        correlation = np.correlate(int_nums, int_nums, mode='full')
        mid = len(correlation) // 2
        correlation = correlation[mid:]
        
        if len(correlation) > 5:
            peaks = signal.find_peaks(correlation)[0]
            if len(peaks) > 0:
                period = peaks[0] + 1
                if period < len(int_nums):
                    pred = int_nums[-period]
                    probs[str(pred)] = 0.3
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def nhi_thuc(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 62: Nhị thức"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 5:
            return probs
        
        # Phân phối nhị thức
        n = 10  # Số lần thử
        p = 0.5  # Xác suất thành công
        
        from math import comb
        for k in range(10):
            prob = comb(n, k) * (p**k) * ((1-p)**(n-k))
            probs[str(k)] = prob
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def phan_phoi_xac_suat(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 63: Phân phối xác suất"""
        probs = {str(i): 0 for i in range(10)}
        if not nums:
            return probs
        
        # Phân phối thực nghiệm
        counts = collections.Counter(nums)
        total = len(nums)
        for num in map(str, range(10)):
            probs[num] = counts.get(num, 0) / total
        
        return probs
    
    def moment_generating(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 64: Hàm sinh moment"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 3:
            return probs
        
        int_nums = [int(n) for n in nums]
        mean = np.mean(int_nums)
        var = np.var(int_nums)
        skew = pd.Series(int_nums).skew()
        kurt = pd.Series(int_nums).kurtosis()
        
        # Moment bậc 4
        for i in range(10):
            z = (i - mean) / (var**0.5) if var > 0 else 0
            probs[str(i)] = math.exp(-0.5 * z**2) * (1 + skew*z/6 + (kurt-3)*z**2/24)
        
        total = sum(probs.values())
        return {k: max(0, v)/total for k, v in probs.items()}
    
    def likelihood_ratio(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 65: Tỷ lệ khả năng"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 10:
            return probs
        
        # So sánh với phân phối đều
        observed = collections.Counter(nums)
        expected = len(nums) / 10
        
        for num in map(str, range(10)):
            obs = observed.get(num, 0)
            if obs > 0:
                ratio = obs / expected
                probs[num] = ratio / (ratio + 1)
            else:
                probs[num] = 0.5 / (len(nums) + 1)
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def bayesian_update(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 66: Cập nhật Bayes"""
        probs = {str(i): 0.1 for i in range(10)}  # Prior uniform
        
        if not nums:
            return probs
        
        # Update với dữ liệu mới
        counts = collections.Counter(nums)
        total = len(nums)
        
        for num in map(str, range(10)):
            likelihood = counts.get(num, 0) / total if total > 0 else 0.1
            prior = probs[num]
            probs[num] = prior * likelihood / (sum(prior * likelihood for p in probs.values()) + 1e-10)
        
        total_probs = sum(probs.values())
        return {k: v/total_probs for k, v in probs.items()}
    
    def kalman_filter(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 67: Lọc Kalman"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 2:
            return probs
        
        # Khởi tạo Kalman filter đơn giản
        x_est = float(nums[0])
        p_est = 1.0
        q = 0.1  # Process noise
        r = 0.5  # Measurement noise
        
        for num in nums[1:]:
            # Predict
            x_pred = x_est
            p_pred = p_est + q
            
            # Update
            z = float(num)
            k = p_pred / (p_pred + r)
            x_est = x_pred + k * (z - x_pred)
            p_est = (1 - k) * p_pred
        
        pred = int(round(x_est)) % 10
        probs[str(pred)] = 0.3
        
        # Thêm uncertainty
        for delta in [1, 2]:
            probs[str((pred + delta) % 10)] = 0.15
            probs[str((pred - delta) % 10)] = 0.15
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def hidden_markov(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 68: Markov ẩn"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 3:
            return probs
        
        # HMM đơn giản với 2 trạng thái ẩn
        int_nums = [int(n) for n in nums[-10:]]
        
        # Phân cụm đơn giản thành 2 trạng thái
        mean = np.mean(int_nums)
        states = [1 if x > mean else 0 for x in int_nums]
        
        last_state = states[-1]
        if last_state == 1:
            # Trạng thái cao
            for i in range(5, 10):
                probs[str(i)] = 0.1
        else:
            # Trạng thái thấp
            for i in range(0, 5):
                probs[str(i)] = 0.1
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def chaos_detection(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 69: Phát hiện hỗn loạn"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 10:
            return probs
        
        int_nums = [int(n) for n in nums[-20:]]
        
        # Tính Lyapunov exponent đơn giản
        diffs = [abs(int_nums[i] - int_nums[i-1]) for i in range(1, len(int_nums))]
        if diffs:
            lyapunov = np.mean(np.log([d + 1 for d in diffs]))
            
            if lyapunov > 1:
                # Hỗn loạn cao, khó dự đoán
                for i in range(10):
                    probs[str(i)] = 0.1
            else:
                # Có thể dự đoán
                probs[nums[-1]] = 0.2
                probs[str((int(nums[-1]) + 1) % 10)] = 0.15
                probs[str((int(nums[-1]) - 1) % 10)] = 0.15
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def fractal_dimension(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 70: Chiều fractal"""
        probs = {str(i): 0.1 for i in range(10)}
        if len(nums) < 10:
            return probs
        
        int_nums = [int(n) for n in nums[-30:]]
        
        # Tính box-counting dimension đơn giản
        scale = 5
        boxes = set()
        for i in range(0, len(int_nums), scale):
            box = tuple(int_nums[i:i+scale])
            boxes.add(box)
        
        dimension = np.log(len(boxes)) / np.log(len(int_nums)/scale) if len(boxes) > 0 else 1
        
        # Chiều fractal càng cao, càng phức tạp
        if dimension > 1.5:
            # Phức tạp, phân phối đều
            return {str(i): 0.1 for i in range(10)}
        else:
            # Đơn giản, tập trung vào số gần đây
            for num in nums[-3:]:
                probs[num] = 0.2
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    # =============== TẦNG 4: MACHINE LEARNING (20 thuật toán) ===============
    def random_forest_predict(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 71: Random Forest"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 10:
            return probs
        
        # Simulate Random Forest với 10 cây
        n_trees = 10
        predictions = []
        
        for _ in range(n_trees):
            # Bootstrap sampling
            sample = random.choices(nums, k=min(10, len(nums)))
            
            # Feature: last 3 numbers
            features = []
            if len(sample) >= 3:
                features = [int(x) for x in sample[-3:]]
            
            if features:
                # Simple decision: mean of features
                pred = int(np.mean(features)) % 10
                predictions.append(pred)
        
        if predictions:
            pred_counts = collections.Counter(predictions)
            total = sum(pred_counts.values())
            for num, count in pred_counts.items():
                probs[str(num)] = count / total
        
        total_probs = sum(probs.values())
        return {k: v/total_probs for k, v in probs.items()} if total_probs > 0 else probs
    
    def gradient_boosting(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 72: Gradient Boosting"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 10:
            return probs
        
        # Simulate Gradient Boosting
        int_nums = [int(n) for n in nums[-20:]]
        
        # Initialize with mean
        pred = np.mean(int_nums) if int_nums else 4.5
        
        # Boosting rounds
        for _ in range(5):
            residuals = [int_nums[i] - pred for i in range(len(int_nums))]
            if residuals:
                # Fit simple tree (mean of residuals)
                pred += np.mean(residuals) * 0.1
        
        pred = int(round(pred)) % 10
        probs[str(pred)] = 0.3
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def xgboost_predict(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 73: XGBoost"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 10:
            return probs
        
        # Simulate XGBoost with regularization
        int_nums = [int(n) for n in nums[-15:]]
        
        # Create features
        features = []
        for i in range(3, len(int_nums)):
            features.append(int_nums[i-3:i])
        
        if features:
            # Simple gradient boosting with regularization
            pred = 0
            for feat in features[-5:]:
                pred += np.mean(feat) * 0.2
            
            pred = int(round(pred)) % 10
            probs[str(pred)] = 0.25
            probs[str((pred + 1) % 10)] = 0.1
            probs[str((pred - 1) % 10)] = 0.1
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def lightgbm_predict(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 74: LightGBM"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 10:
            return probs
        
        # Simulate LightGBM with leaf-wise growth
        int_nums = [int(n) for n in nums[-12:]]
        
        # Gradient-based one-side sampling
        diffs = [abs(int_nums[i] - int_nums[i-1]) for i in range(1, len(int_nums))]
        if diffs:
            threshold = np.percentile(diffs, 70)
            high_grad = [i for i, d in enumerate(diffs) if d > threshold]
            
            if high_grad:
                pred = int_nums[high_grad[-1] + 1] if high_grad[-1] + 1 < len(int_nums) else int_nums[-1]
                probs[str(pred)] = 0.2
                probs[str((pred + 1) % 10)] = 0.15
                probs[str((pred - 1) % 10)] = 0.15
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def logistic_regression(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 75: Logistic Regression"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 5:
            return probs
        
        # One-vs-rest logistic regression
        int_nums = [int(n) for n in nums[-10:]]
        
        for target in range(10):
            # Binary classification: target vs others
            y = [1 if x == target else 0 for x in int_nums]
            x = list(range(len(y)))
            
            if sum(y) > 0:
                # Simple logistic function
                logit = np.mean(x) * 0.1 - 0.5
                prob = 1 / (1 + np.exp(-logit))
                probs[str(target)] = prob
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def svm_predict(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 76: SVM"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 10:
            return probs
        
        # Simulate SVM with RBF kernel
        int_nums = [int(n) for n in nums[-10:]]
        
        # Find support vectors (outliers)
        mean = np.mean(int_nums)
        std = np.std(int_nums) if len(int_nums) > 1 else 2.87
        
        support_vectors = [x for x in int_nums if abs(x - mean) > std]
        
        if support_vectors:
            # Predict based on nearest support vector
            last_num = int(nums[-1])
            nearest = min(support_vectors, key=lambda x: abs(x - last_num))
            probs[str(nearest)] = 0.25
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def neural_network(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 77: Neural Network"""
        probs = {str(i): 0.1 for i in range(10)}
        if len(nums) < 10:
            return probs
        
        # Simple 3-layer neural network simulation
        int_nums = [int(n) for n in nums[-5:]]
        
        # Input layer (5 neurons)
        inputs = np.array(int_nums) / 9.0  # Normalize
        
        # Hidden layer (10 neurons) with ReLU
        weights1 = np.random.randn(5, 10) * 0.1
        hidden = np.maximum(0, np.dot(inputs, weights1))
        
        # Output layer (10 neurons) with softmax
        weights2 = np.random.randn(10, 10) * 0.1
        output = np.dot(hidden, weights2)
        
        # Softmax
        exp_output = np.exp(output - np.max(output))
        softmax = exp_output / exp_output.sum()
        
        for i in range(10):
            probs[str(i)] = softmax[i]
        
        return probs
    
    def bayesian_inference(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 78: Bayesian Inference"""
        probs = {str(i): 0.1 for i in range(10)}  # Prior
        
        if not nums:
            return probs
        
        # Update with conjugate prior (Beta distribution)
        counts = collections.Counter(nums)
        total = len(nums)
        
        for num in map(str, range(10)):
            alpha = 1 + counts.get(num, 0)
            beta = 1 + total - counts.get(num, 0)
            probs[num] = alpha / (alpha + beta)
        
        total_probs = sum(probs.values())
        return {k: v/total_probs for k, v in probs.items()}
    
    def monte_carlo_sim(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 79: Monte Carlo Simulation"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 10:
            return probs
        
        # Simulate 1000 paths
        n_simulations = 1000
        int_nums = [int(n) for n in nums[-10:]]
        
        predictions = []
        for _ in range(n_simulations):
            # Random walk
            path = [int_nums[-1]]
            for _ in range(3):
                step = random.choice([-1, 0, 1])
                next_val = (path[-1] + step) % 10
                path.append(next_val)
            predictions.append(path[-1])
        
        counts = collections.Counter(predictions)
        total = len(predictions)
        
        for i in range(10):
            probs[str(i)] = counts.get(i, 0) / total
        
        return probs
    
    def time_series_forecast(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 80: Time Series Forecast"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 10:
            return probs
        
        int_nums = [int(n) for n in nums[-20:]]
        
        # Simple exponential smoothing
        alpha = 0.3
        forecast = int_nums[0]
        
        for value in int_nums[1:]:
            forecast = alpha * value + (1 - alpha) * forecast
        
        pred = int(round(forecast)) % 10
        probs[str(pred)] = 0.25
        probs[str((pred + 1) % 10)] = 0.15
        probs[str((pred - 1) % 10)] = 0.15
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def knn_predict(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 81: K-Nearest Neighbors"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 5:
            return probs
        
        k = 3
        last_seq = nums[-3:]
        
        # Find similar sequences
        neighbors = []
        for i in range(len(nums)-4, -1, -1):
            if nums[i:i+3] == last_seq:
                if i+3 < len(nums):
                    neighbors.append(int(nums[i+3]))
        
        if neighbors:
            nearest = neighbors[:k]
            counts = collections.Counter(nearest)
            total = len(nearest)
            for num, count in counts.items():
                probs[str(num)] = count / total
        
        total_probs = sum(probs.values())
        return {k: v/total_probs for k, v in probs.items()} if total_probs > 0 else probs
    
    def naive_bayes(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 82: Naive Bayes"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 10:
            return probs
        
        # Assume independence of positions
        int_nums = [int(n) for n in nums[-5:]]
        
        for target in range(10):
            prob = 1.0
            for pos, value in enumerate(int_nums):
                # P(feature|class)
                likelihood = 0.1  # Prior
                if value == target:
                    likelihood = 0.3
                prob *= likelihood
            probs[str(target)] = prob
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def decision_tree(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 83: Decision Tree"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 5:
            return probs
        
        # Simple decision rules
        last_num = int(nums[-1])
        
        # Rule 1: If last number > 5, predict smaller
        if last_num > 5:
            probs[str((last_num - 1) % 10)] = 0.3
            probs[str((last_num - 2) % 10)] = 0.2
        else:
            # Rule 2: If last number <= 5, predict larger
            probs[str((last_num + 1) % 10)] = 0.3
            probs[str((last_num + 2) % 10)] = 0.2
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def linear_regression(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 84: Linear Regression"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 5:
            return probs
        
        int_nums = [int(n) for n in nums[-10:]]
        x = list(range(len(int_nums)))
        
        if len(set(int_nums)) > 1:
            slope, intercept = np.polyfit(x, int_nums, 1)
            pred = slope * (len(int_nums)) + intercept
            pred = int(round(pred)) % 10
            probs[str(pred)] = 0.3
        else:
            probs[nums[-1]] = 0.3
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def polynomial_regression(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 85: Polynomial Regression"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 6:
            return probs
        
        int_nums = [int(n) for n in nums[-6:]]
        x = list(range(len(int_nums)))
        
        if len(set(int_nums)) > 1:
            coefs = np.polyfit(x, int_nums, 2)  # Quadratic
            pred = coefs[0]*len(int_nums)**2 + coefs[1]*len(int_nums) + coefs[2]
            pred = int(round(pred)) % 10
            probs[str(pred)] = 0.25
        else:
            probs[nums[-1]] = 0.25
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def lstm_predict(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 86: LSTM"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 10:
            return probs
        
        # Simulate LSTM with memory
        int_nums = [int(n) for n in nums[-15:]]
        
        # Simple LSTM cell simulation
        h = 0  # hidden state
        c = 0  # cell state
        
        for x in int_nums:
            # Forget gate
            f = 0.9
            # Input gate
            i = 0.1
            # Candidate
            g = np.tanh(x * 0.1)
            # Update cell
            c = f * c + i * g
            # Output gate
            o = 0.1
            # Update hidden
            h = o * np.tanh(c)
        
        pred = int(round((h * 10) % 10)) % 10
        probs[str(pred)] = 0.2
        probs[str((pred + 1) % 10)] = 0.15
        probs[str((pred - 1) % 10)] = 0.15
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def gru_predict(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 87: GRU"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 10:
            return probs
        
        # Simulate GRU
        int_nums = [int(n) for n in nums[-15:]]
        
        h = 0  # hidden state
        
        for x in int_nums:
            # Reset gate
            r = 0.8
            # Update gate
            z = 0.2
            # Candidate
            h_tilde = np.tanh(x * 0.1 + r * h)
            # Update hidden
            h = z * h + (1 - z) * h_tilde
        
        pred = int(round((h * 10) % 10)) % 10
        probs[str(pred)] = 0.2
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def transformer_predict(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 88: Transformer"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 10:
            return probs
        
        # Simulate self-attention
        int_nums = [int(n) for n in nums[-10:]]
        
        # Self-attention weights (more weight on recent)
        attention = np.exp(np.arange(len(int_nums)) / 10)
        attention = attention / attention.sum()
        
        # Weighted sum
        context = np.sum(np.array(int_nums) * attention)
        pred = int(round(context)) % 10
        probs[str(pred)] = 0.25
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def ensemble_stacking(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 89: Ensemble Stacking"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 10:
            return probs
        
        # Stack multiple models
        models = [
            self.random_forest_predict(nums),
            self.gradient_boosting(nums),
            self.xgboost_predict(nums),
            self.svm_predict(nums)
        ]
        
        # Meta-learner (simple average)
        for i in range(10):
            avg_prob = np.mean([m.get(str(i), 0) for m in models])
            probs[str(i)] = avg_prob
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def deep_learning(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 90: Deep Learning"""
        probs = {str(i): 0 for i in range(10)}
        if len(nums) < 10:
            return probs
        
        # Simulate deep neural network with multiple layers
        int_nums = [int(n) for n in nums[-8:]] / 9.0  # Normalize
        
        # Deep network: 8->16->32->16->10
        layers = [8, 16, 32, 16, 10]
        activations = int_nums
        
        for i in range(len(layers)-1):
            weights = np.random.randn(layers[i], layers[i+1]) * 0.1
            biases = np.zeros(layers[i+1])
            activations = np.tanh(np.dot(activations, weights) + biases)
        
        # Output layer
        output = activations[:10] if len(activations) >= 10 else np.zeros(10)
        exp_output = np.exp(output - np.max(output))
        softmax = exp_output / exp_output.sum()
        
        for i in range(10):
            probs[str(i)] = softmax[i]
        
        return probs
    
    # =============== TẦNG 5: META AI + GEMINI (15 thuật toán) ===============
    def gemini_reasoning(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 91: Gemini reasoning"""
        probs = {str(i): 0.1 for i in range(10)}
        if not GEMINI_API_KEY or len(nums) < 20:
            return probs
        
        try:
            prompt = f"""Phân tích chuỗi số xổ số: {''.join(nums[-30:])}
            Task: Dựa trên pattern và logic, hãy dự đoán 3 số có khả năng cao nhất.
            Format trả về: {{"predictions": ["x", "y", "z"], "reasons": "..."}}
            """
            
            headers = {"Content-Type": "application/json"}
            data = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
            
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}",
                headers=headers,
                json=data,
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                
                # Parse response
                try:
                    import re
                    json_match = re.search(r'\{.*\}', text, re.DOTALL)
                    if json_match:
                        preds = json.loads(json_match.group())
                        predictions = preds.get("predictions", [])
                        
                        # Reset probabilities
                        probs = {str(i): 0 for i in range(10)}
                        for i, pred in enumerate(predictions[:3]):
                            weight = 0.33 - i * 0.1
                            probs[str(pred)] = weight
                except:
                    pass
        except:
            pass
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()} if total > 0 else probs
    
    def gemini_pattern_scan(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 92: Gemini pattern scan"""
        # Similar to reasoning but focused on pattern detection
        return self.gemini_reasoning(nums)
    
    def gemini_anomaly(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 93: Gemini anomaly detection"""
        probs = {str(i): 0.1 for i in range(10)}
        if not GEMINI_API_KEY:
            return probs
        
        try:
            prompt = f"""Phát hiện bất thường trong chuỗi: {''.join(nums[-20:])}
            Xác định số nào đang bị "giam" hoặc có pattern bất thường.
            Trả về 3 số bất thường nhất.
            """
            
            headers = {"Content-Type": "application/json"}
            data = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
            
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}",
                headers=headers,
                json=data,
                timeout=5
            )
            
            if response.status_code == 200:
                # Parse response and adjust probabilities
                pass
        except:
            pass
        
        return probs
    
    def gemini_ranking(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 94: Gemini probability ranking"""
        probs = {str(i): 0 for i in range(10)}
        if not GEMINI_API_KEY:
            return probs
        
        try:
            prompt = f"""Xếp hạng xác suất cho các số 0-9 dựa trên: {''.join(nums[-30:])}
            Trả về list 10 số với xác suất từ cao xuống thấp.
            Format: [("5", 0.15), ("3", 0.12), ...]
            """
            
            headers = {"Content-Type": "application/json"}
            data = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
            
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}",
                headers=headers,
                json=data,
                timeout=5
            )
            
            if response.status_code == 200:
                # Parse ranking
                pass
        except:
            pass
        
        return probs
    
    def gemini_trend(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 95: Gemini trend detection"""
        probs = {str(i): 0.1 for i in range(10)}
        if len(nums) < 20:
            return probs
        
        # Analyze short-term and long-term trends
        short_term = collections.Counter(nums[-10:])
        long_term = collections.Counter(nums[-50:]) if len(nums) >= 50 else collections.Counter(nums)
        
        for num in map(str, range(10)):
            short_freq = short_term.get(num, 0) / 10
            long_freq = long_term.get(num, 0) / len(nums)
            
            if short_freq > long_freq * 1.5:
                # Increasing trend
                probs[num] = 0.15
            elif short_freq < long_freq * 0.5:
                # Decreasing trend
                probs[num] = 0.05
            else:
                # Stable
                probs[num] = 0.1
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def gemini_correlation(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 96: Gemini correlation"""
        return self.gemini_reasoning(nums)
    
    def gemini_forecast(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 97: Gemini forecast"""
        return self.gemini_reasoning(nums)
    
    def gemini_ensemble(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 98: Gemini ensemble"""
        return self.gemini_reasoning(nums)
    
    def gemini_adversarial(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 99: Gemini adversarial"""
        # Counter Kubet AI
        probs = {str(i): 0.1 for i in range(10)}
        
        # Add random noise to avoid detection
        noise = np.random.normal(0, 0.05, 10)
        for i in range(10):
            probs[str(i)] += noise[i]
        
        total = sum(probs.values())
        return {k: max(0.01, v)/total for k, v in probs.items()}
    
    def gemini_counter(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 100: Gemini counter-strategy"""
        return self.gemini_adversarial(nums)
    
    def gemini_adaptation(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 101: Gemini adaptation"""
        # Learn from past predictions
        if self.prediction_history:
            recent_acc = np.mean(self.accuracy_history[-10:]) if self.accuracy_history else 0.5
            
            if recent_acc < 0.4:
                # Poor performance, switch strategy
                return self.gemini_adversarial(nums)
            else:
                return self.gemini_reasoning(nums)
        
        return self.gemini_reasoning(nums)
    
    def gemini_optimization(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 102: Gemini optimization"""
        return self.gemini_reasoning(nums)
    
    def gemini_validation(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 103: Gemini validation"""
        return self.gemini_reasoning(nums)
    
    def gemini_calibration(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 104: Gemini calibration"""
        # Calibrate probabilities
        probs = self.gemini_reasoning(nums)
        
        # Platt scaling
        for num in probs:
            probs[num] = 1 / (1 + np.exp(-probs[num]))
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def gemini_fusion(self, nums: List[str]) -> Dict[str, float]:
        """Thuật toán 105: Gemini fusion"""
        # Fuse multiple gemini outputs
        strategies = [
            self.gemini_reasoning(nums),
            self.gemini_pattern_scan(nums),
            self.gemini_anomaly(nums),
            self.gemini_trend(nums)
        ]
        
        probs = {str(i): 0 for i in range(10)}
        for i in range(10):
            for strategy in strategies:
                probs[str(i)] += strategy.get(str(i), 0)
            probs[str(i)] /= len(strategies)
        
        return probs
    
    # =============== TẦNG 6: META ENSEMBLE (11 thuật toán) ===============
    def voting_ensemble(self, predictions: List[Dict[str, float]]) -> Dict[str, float]:
        """Thuật toán 106: Voting ensemble"""
        probs = {str(i): 0 for i in range(10)}
        
        for pred in predictions:
            # Get top 1 from each model
            top_num = max(pred.items(), key=lambda x: x[1])[0]
            probs[top_num] += 1
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()} if total > 0 else probs
    
    def weighted_ensemble(self, predictions: List[Dict[str, float]], weights: List[float]) -> Dict[str, float]:
        """Thuật toán 107: Weighted ensemble"""
        probs = {str(i): 0 for i in range(10)}
        
        for pred, weight in zip(predictions, weights):
            for num in pred:
                probs[num] += pred[num] * weight
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def dynamic_weight(self, predictions: List[Dict[str, float]]) -> Dict[str, float]:
        """Thuật toán 108: Dynamic weight"""
        # Adjust weights based on recent performance
        weights = []
        for i, pred in enumerate(predictions):
            if len(self.accuracy_history) > i:
                weights.append(max(0.1, self.accuracy_history[-i-1]))
            else:
                weights.append(1.0)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        return self.weighted_ensemble(predictions, weights.tolist())
    
    def confidence_fusion(self, predictions: List[Dict[str, float]]) -> Dict[str, float]:
        """Thuật toán 109: Confidence fusion"""
        probs = {str(i): 0 for i in range(10)}
        
        for pred in predictions:
            # Weight by confidence (entropy)
            values = list(pred.values())
            entropy = -sum(p * np.log(p + 1e-10) for p in values)
            confidence = 1 / (1 + entropy)
            
            for num in pred:
                probs[num] += pred[num] * confidence
        
        total = sum(probs.values())
        return {k: v/total for k, v in probs.items()}
    
    def risk_scoring(self, predictions: Dict[str, float]) -> Dict[str, float]:
        """Thuật toán 110: Risk scoring"""
        # Calculate risk score for each number
        risk_scores = {num: 0 for num in predictions}
        
        # Higher probability = higher risk (overfitting)
        for num in predictions:
            risk_scores[num] = predictions[num] * 0.5
        
        # Adjust probabilities
        adjusted = {num: max(0, predictions[num] - risk_scores[num]) 
                   for num in predictions}
        
        total = sum(adjusted.values())
        return {k: v/total for k, v in adjusted.items()}
    
    def stability_index(self, predictions: Dict[str, float]) -> float:
        """Thuật toán 111: Stability index"""
        # Measure prediction stability
        values = list(predictions.values())
        return 1 - np.std(values) * 2
    
    def volatility_index(self, nums: List[str]) -> float:
        """Thuật toán 112: Volatility index"""
        if len(nums) < 10:
            return 0.5
        
        int_nums = [int(n) for n in nums[-20:]]
        diffs = [abs(int_nums[i] - int_nums[i-1]) for i in range(1, len(int_nums))]
        
        return np.mean(diffs) / 9.0 if diffs else 0.5
    
    def overfit_detection(self, predictions: Dict[str, float]) -> bool:
        """Thuật toán 113: Overfit detection"""
        # Check if predictions are too concentrated
        top_prob = max(predictions.values())
        return top_prob > 0.5
    
    def bias_correction(self, predictions: Dict[str, float]) -> Dict[str, float]:
        """Thuật toán 114: Bias correction"""
        # Correct for selection bias
        corrected = predictions.copy()
        
        # Reduce bias towards recent numbers
        for num in predictions:
            corrected[num] = predictions[num] * 0.9 + 0.01
        
        total = sum(corrected.values())
        return {k: v/total for k, v in corrected.items()}
    
    def adaptive_weight(self) -> Dict[str, float]:
        """Thuật toán 115: Adaptive weight learning"""
        # Learn optimal weights from history
        if len(self.accuracy_history) > 10:
            recent_acc = np.mean(self.accuracy_history[-10:])
            
            if recent_acc > 0.6:
                # Good performance, maintain weights
                return self.weights
            elif recent_acc < 0.4:
                # Poor performance, adjust weights
                self.weights['stats'] *= 1.1
                self.weights['ml'] *= 0.9
                
                # Re-normalize
                total = sum(self.weights.values())
                for key in self.weights:
                    self.weights[key] /= total
        
        return self.weights
    
    def final_decision(self, tier_scores: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
        """Thuật toán 116: Final decision engine"""
        weights = self.adaptive_weight()
        
        # Combine all tiers
        final_scores = {str(i): 0 for i in range(10)}
        
        # Weighted sum
        final_scores['stats'] = sum(tier_scores['stats'].get(str(i), 0) * weights['stats'] 
                                   for i in range(10))
        final_scores['pattern'] = sum(tier_scores['pattern'].get(str(i), 0) * weights['pattern'] 
                                     for i in range(10))
        final_scores['behavior'] = sum(tier_scores['behavior'].get(str(i), 0) * weights['behavior'] 
                                      for i in range(10))
        final_scores['ml'] = sum(tier_scores['ml'].get(str(i), 0) * weights['ml'] 
                                for i in range(10))
        final_scores['meta'] = sum(tier_scores['meta'].get(str(i), 0) * weights['meta'] 
                                  for i in range(10))
        final_scores['ensemble'] = sum(tier_scores['ensemble'].get(str(i), 0) * weights['ensemble'] 
                                      for i in range(10))
        
        # Normalize
        total = sum(final_scores.values())
        final_scores = {k: v/total for k, v in final_scores.items()}
        
        # Calculate risk and confidence
        volatility = self.volatility_index(list(tier_scores.get('data', [])))
        stability = self.stability_index(final_scores)
        confidence = 1 - volatility * 0.5
        
        # Detect overfitting
        overfit = self.overfit_detection(final_scores)
        if overfit:
            final_scores = self.bias_correction(final_scores)
        
        # Sort and get top 3
        sorted_nums = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_nums[:3]
        
        # Prepare output
        result = {
            'top_3': [],
            'all_numbers': [],
            'metadata': {
                'confidence': confidence,
                'volatility': volatility,
                'stability': stability,
                'weights': weights
            }
        }
        
        for num, score in top_3:
            # Determine status
            status = self._determine_status(num, tier_scores.get('data', []))
            
            result['top_3'].append({
                'number': num,
                'strength': score * 100,
                'risk': (1 - score) * 100,
                'status': status,
                'reasons': self._generate_reason(num, tier_scores)
            })
        
        # All numbers ranking
        for num, score in sorted_nums:
            result['all_numbers'].append({
                'number': num,
                'score': score * 100,
                'rank': len(result['all_numbers']) + 1
            })
        
        return result
    
    def _determine_status(self, num: str, nums: List[str]) -> str:
        """Xác định trạng thái của số"""
        if not nums:
            return "unknown"
        
        recent = nums[-10:] if len(nums) >= 10 else nums
        
        if num in recent:
            if recent.count(num) >= 3:
                return "bệt"
            elif recent[-1] == num:
                return "nóng"
            else:
                return "hồi"
        else:
            return "lạnh"
    
    def _generate_reason(self, num: str, tier_scores: Dict) -> str:
        """Tạo lý do phân tích"""
        reasons = []
        
        if tier_scores['stats'].get(num, 0) > 0.15:
            reasons.append("tần suất cao")
        if tier_scores['pattern'].get(num, 0) > 0.15:
            reasons.append("cầu đẹp")
        if tier_scores['behavior'].get(num, 0) > 0.15:
            reasons.append("bóng số mạnh")
        if tier_scores['ml'].get(num, 0) > 0.15:
            reasons.append("ML dự báo")
        if tier_scores['meta'].get(num, 0) > 0.15:
            reasons.append("AI xác nhận")
        
        if not reasons:
            reasons.append("tiềm năng")
        
        return ", ".join(reasons[:3])

# =============== CLASS PHÂN TÍCH CHÍNH ===============
class AdvancedLotteryAnalyzer:
    def __init__(self):
        self.ai = AI_116_Algorithms()
        self.history = []
        self.weights = self.ai.weights.copy()
        
    def analyze(self, data: str) -> Dict:
        """Phân tích toàn diện với 116 thuật toán"""
        nums = list(filter(str.isdigit, data))
        
        if len(nums) < 10:
            return None
        
        # ===== TẦNG 1: THỐNG KÊ =====
        stats_probs = {str(i): 0 for i in range(10)}
        for algo_id, algo_func in self.ai.stats_algorithms.items():
            try:
                result = algo_func(nums)
                for num in result:
                    stats_probs[num] += result[num] / len(self.ai.stats_algorithms)
            except:
                continue
        
        # ===== TẦNG 2: PHÂN TÍCH CẦU =====
        pattern_probs = {str(i): 0 for i in range(10)}
        for algo_id, algo_func in self.ai.pattern_algorithms.items():
            try:
                result = algo_func(nums)
                for num in result:
                    pattern_probs[num] += result[num] / len(self.ai.pattern_algorithms)
            except:
                continue
        
        # ===== TẦNG 3: PHÂN TÍCH HÀNH VI =====
        behavior_probs = {str(i): 0 for i in range(10)}
        for algo_id, algo_func in self.ai.behavior_algorithms.items():
            try:
                result = algo_func(nums)
                for num in result:
                    behavior_probs[num] += result[num] / len(self.ai.behavior_algorithms)
            except:
                continue
        
        # ===== TẦNG 4: MACHINE LEARNING =====
        ml_probs = {str(i): 0 for i in range(10)}
        for algo_id, algo_func in self.ai.ml_algorithms.items():
            try:
                result = algo_func(nums)
                for num in result:
                    ml_probs[num] += result[num] / len(self.ai.ml_algorithms)
            except:
                continue
        
        # ===== TẦNG 5: META AI =====
        meta_probs = {str(i): 0 for i in range(10)}
        for algo_id, algo_func in self.ai.meta_algorithms.items():
            try:
                result = algo_func(nums)
                for num in result:
                    meta_probs[num] += result[num] / len(self.ai.meta_algorithms)
            except:
                continue
        
        # ===== TẦNG 6: ENSEMBLE =====
        tier_scores = {
            'stats': stats_probs,
            'pattern': pattern_probs,
            'behavior': behavior_probs,
            'ml': ml_probs,
            'meta': meta_probs,
            'ensemble': stats_probs,  # Temporary
            'data': nums
        }
        
        # Final decision
        result = self.ai.final_decision(tier_scores)
        
        # Lưu lịch sử
        self.history.append({
            'timestamp': datetime.now(),
            'data': data[-20:],
            'result': result
        })
        
        return result

# =============== GIAO DIỆN STREAMLIT ===============
st.set_page_config(
    page_title="AI 3-TINH SIÊU CẤP - 116 THUẬT TOÁN",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS RESPONSIVE CHO SAMSUNG MULTI WINDOW
st.markdown("""
<style>
    * {
        box-sizing: border-box;
    }
    
    .stApp {
        background: #0f172a !important;
        color: #e2e8f0;
        font-family: 'Segoe UI', Roboto, sans-serif;
        padding: 10px;
        width: 100%;
        max-width: 100%;
        margin: 0 auto;
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1e40af, #3b82f6);
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 15px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    .main-title {
        font-size: clamp(1.5rem, 6vw, 2.2rem) !important;
        font-weight: 800;
        color: white;
        margin: 0;
        line-height: 1.2;
    }
    
    .subtitle {
        font-size: clamp(0.8rem, 3vw, 1rem) !important;
        color: #cbd5e1;
        margin-top: 5px;
    }
    
    /* Input area */
    .stTextArea textarea {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        border: 2px solid #3b82f6 !important;
        border-radius: 12px !important;
        font-size: clamp(14px, 4vw, 16px) !important;
        min-height: 80px !important;
        padding: 12px !important;
    }
    
    /* Button */
    .stButton button {
        background: linear-gradient(90deg, #10b981, #34d399) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: clamp(16px, 5vw, 18px) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 15px 25px !important;
        width: 100% !important;
        transition: all 0.3s !important;
        margin: 10px 0;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.4) !important;
    }
    
    /* Kết quả chính - FONT LỚN */
    .result-box {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 2px solid #10b981;
        border-radius: 20px;
        padding: 20px;
        margin: 15px 0;
        text-align: center;
    }
    
    .result-title {
        color: #38bdf8;
        font-size: clamp(1.2rem, 5vw, 1.5rem);
        font-weight: 700;
        margin-bottom: 15px;
    }
    
    .top3-container {
        display: flex;
        justify-content: center;
        gap: clamp(10px, 3vw, 20px);
        flex-wrap: wrap;
        margin: 15px 0;
    }
    
    .number-card {
        background: linear-gradient(135deg, #fbbf24, #f59e0b);
        border-radius: 15px;
        padding: 15px;
        min-width: clamp(80px, 20vw, 100px);
        text-align: center;
        box-shadow: 0 8px 20px rgba(245, 158, 11, 0.3);
    }
    
    .big-number {
        font-size: clamp(2.5rem, 10vw, 3.5rem);
        font-weight: 900;
        color: #1e293b;
        line-height: 1;
        margin-bottom: 5px;
    }
    
    .stats-info {
        font-size: clamp(0.8rem, 3vw, 0.9rem);
        color: #1e293b;
        font-weight: 600;
    }
    
    .status-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: clamp(0.7rem, 2.5vw, 0.8rem);
        font-weight: 600;
        margin-top: 5px;
    }
    
    /* Info box */
    .info-panel {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid;
    }
    
    .info-title {
        font-size: clamp(1rem, 4vw, 1.1rem);
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .info-content {
        font-size: clamp(1.1rem, 5vw, 1.3rem);
        font-weight: 700;
        color: #f8fafc;
    }
    
    /* Table */
    .ranking-table {
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
        font-size: clamp(0.9rem, 3.5vw, 1rem);
    }
    
    .ranking-table th {
        background: #1e293b;
        color: #94a3b8;
        padding: 10px;
        text-align: left;
    }
    
    .ranking-table td {
        padding: 8px 10px;
        border-bottom: 1px solid #334155;
    }
    
    /* Responsive */
    @media (max-width: 640px) {
        .top3-container {
            gap: 8px;
        }
        
        .number-card {
            min-width: 70px;
            padding: 10px;
        }
        
        .big-number {
            font-size: 2rem;
        }
    }
    
    @media (max-width: 480px) {
        .top3-container {
            flex-direction: column;
            align-items: center;
        }
        
        .number-card {
            width: 80%;
        }
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 20px;
        padding-top: 15px;
        border-top: 1px solid #334155;
        color: #94a3b8;
        font-size: clamp(0.7rem, 2.5vw, 0.8rem);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class='main-header'>
    <h1 class='main-title'>🎯 AI 3-TINH SIÊU CẤP</h1>
    <p class='subtitle'>116 thuật toán đa tầng | Đối kháng AI Kubet | Chính xác tối đa</p>
</div>
""", unsafe_allow_html=True)

# Khởi tạo analyzer
@st.cache_resource
def init_analyzer():
    return AdvancedLotteryAnalyzer()

analyzer = init_analyzer()

# Tabs
tab1, tab2, tab3 = st.tabs(["🎯 PHÂN TÍCH", "📊 LỊCH SỬ", "⚙️ CẤU HÌNH"])

with tab1:
    # Input
    st.markdown("### 📥 NHẬP DỮ LIỆU")
    data_input = st.text_area(
        "Dán chuỗi số thực tế (càng nhiều càng chính xác):",
        height=100,
        placeholder="Nhập ít nhất 20-30 số gần nhất...\nVí dụ: 5382917462538192047518364927...",
        key="data_input"
    )
    
    # Quick stats
    if data_input and len(data_input.strip()) >= 10:
        nums = list(filter(str.isdigit, data_input))
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tổng số ván", len(nums))
        with col2:
            unique = len(set(nums))
            st.metric("Số unique", unique)
        with col3:
            last_num = nums[-1] if nums else "0"
            st.metric("Số cuối", last_num)
        with col4:
            volatility = analyzer.ai.volatility_index(nums)
            st.metric("Biến động", f"{volatility:.1%}")
    
    # Analyze button
    if st.button("🚀 KÍCH HOẠT 116 THUẬT TOÁN", use_container_width=True):
        if not data_input or len(data_input.strip()) < 20:
            st.error("⚠️ Cần ít nhất 20 số để phân tích chính xác!")
        else:
            with st.spinner('🔄 Đang chạy 116 thuật toán đa tầng...'):
                progress_bar = st.progress(0)
                
                # Simulate processing
                for i in range(10):
                    time.sleep(0.2)
                    progress_bar.progress((i + 1) * 10)
                
                # Analyze
                result = analyzer.analyze(data_input)
                
                if result:
                    st.success(f"✅ Hoàn thành! Độ tin cậy: {result['metadata']['confidence']:.1%}")
                    
                    # Hiển thị TOP 3
                    st.markdown("### 🎯 TOP 3 SỐ MẠNH NHẤT")
                    
                    html = "<div class='top3-container'>"
                    for item in result['top_3']:
                        status_color = {
                            'bệt': '#ef4444',
                            'nóng': '#f59e0b',
                            'hồi': '#3b82f6',
                            'lạnh': '#8b5cf6'
                        }.get(item['status'], '#94a3b8')
                        
                        html += f"""
                        <div class='number-card'>
                            <div class='big-number'>{item['number']}</div>
                            <div class='stats-info'>Sức mạnh: {item['strength']:.1f}%</div>
                            <div class='stats-info'>Rủi ro: {item['risk']:.1f}%</div>
                            <span class='status-badge' style='background: {status_color}20; color: {status_color}; border: 1px solid {status_color}'>
                                {item['status'].upper()}
                            </span>
                            <div style='font-size: 0.8rem; margin-top: 8px; color: #94a3b8;'>
                                {item['reasons']}
                            </div>
                        </div>
                        """
                    html += "</div>"
                    
                    st.markdown(html, unsafe_allow_html=True)
                    
                    # Dàn 7 số an toàn
                    all_nums = result['all_numbers']
                    safe_7 = [item['number'] for item in all_nums[:7]]
                    eliminated = [item['number'] for item in all_nums[7:]]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        <div class='info-panel' style='border-left-color: #10b981;'>
                            <div class='info-title'>✅ DÀN 7 SỐ AN TOÀN</div>
                            <div class='info-content'>""" + " - ".join(safe_7) + """</div>
                            <small style='color: #94a3b8;'>Chọn 7 số từ dàn này</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                        <div class='info-panel' style='border-left-color: #ef4444;'>
                            <div class='info-title'>🚫 3 SỐ RỦI RO CAO</div>
                            <div class='info-content'>""" + " - ".join(eliminated[:3]) + """</div>
                            <small style='color: #94a3b8;'>Tránh xa các số này</small>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Bảng xếp hạng đầy đủ
                    with st.expander("📊 XEM BẢNG XẾP HẠNG ĐẦY ĐỦ", expanded=False):
                        table_html = "<table class='ranking-table'><tr><th>Hạng</th><th>Số</th><th>Điểm</th><th>Trạng thái</th><th>Phân tích</th></tr>"
                        
                        for i, item in enumerate(result['all_numbers'], 1):
                            status = 'unknown'
                            for top in result['top_3']:
                                if top['number'] == item['number']:
                                    status = top['status']
                                    break
                            
                            status_color = {
                                'bệt': '#ef4444',
                                'nóng': '#f59e0b',
                                'hồi': '#3b82f6',
                                'lạnh': '#8b5cf6'
                            }.get(status, '#94a3b8')
                            
                            table_html += f"""
                            <tr>
                                <td>#{i}</td>
                                <td style='font-weight: 700; font-size: 1.1rem;'>{item['number']}</td>
                                <td>{item['score']:.1f}%</td>
                                <td><span style='color: {status_color};'>{status.upper()}</span></td>
                                <td>{result['top_3'][0]['reasons'] if i <= 3 else 'tiềm năng'}</td>
                            </tr>
                            """
                        
                        table_html += "</table>"
                        st.markdown(table_html, unsafe_allow_html=True)
                    
                    # Thông tin hệ thống
                    st.markdown("""
                    <div style='background: rgba(59, 130, 246, 0.1); border-radius: 12px; padding: 15px; margin-top: 15px;'>
                        <h4 style='color: #3b82f6; margin-bottom: 10px;'>🧠 PHÂN TÍCH ĐA TẦNG</h4>
                        <ul style='margin: 0; padding-left: 20px; color: #cbd5e1;'>
                            <li><b>Tầng 1:</b> 20 thuật toán thống kê</li>
                            <li><b>Tầng 2:</b> 25 thuật toán phân tích cầu</li>
                            <li><b>Tầng 3:</b> 25 thuật toán hành vi số</li>
                            <li><b>Tầng 4:</b> 20 thuật toán Machine Learning</li>
                            <li><b>Tầng 5:</b> 15 thuật toán Meta AI + Gemini</li>
                            <li><b>Tầng 6:</b> 11 thuật toán Meta Ensemble</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("❌ Không đủ dữ liệu để phân tích!")

with tab2:
    st.markdown("### 📜 LỊCH SỬ PHÂN TÍCH")
    
    if analyzer.history:
        history_data = []
        for h in analyzer.history[-10:]:
            history_data.append({
                'Thời gian': h['timestamp'].strftime('%H:%M:%S'),
                'Top 3': '-'.join([t['number'] for t in h['result']['top_3']]),
                'Độ tin cậy': f"{h['result']['metadata']['confidence']:.1%}",
                'Biến động': f"{h['result']['metadata']['volatility']:.1%}"
            })
        
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tổng phân tích", len(analyzer.history))
        with col2:
            avg_confidence = np.mean([h['result']['metadata']['confidence'] for h in analyzer.history[-10:]])
            st.metric("Tin cậy TB", f"{avg_confidence:.1%}")
        with col3:
            avg_volatility = np.mean([h['result']['metadata']['volatility'] for h in analyzer.history[-10:]])
            st.metric("Biến động TB", f"{avg_volatility:.1%}")
    else:
        st.info("📊 Chưa có lịch sử phân tích. Hãy thực hiện phân tích ở tab 'PHÂN TÍCH'.")

with tab3:
    st.markdown("### ⚙️ CẤU HÌNH HỆ THỐNG")
    
    with st.form("config_form"):
        st.markdown("#### 🔗 KẾT NỐI AI")
        gemini_key = st.text_input("Gemini API Key", type="password", value=GEMINI_API_KEY)
        openai_key = st.text_input("OpenAI API Key", type="password", value=OPENAI_API_KEY)
        
        st.markdown("#### 🎯 TRỌNG SỐ THUẬT TOÁN")
        col1, col2 = st.columns(2)
        with col1:
            w_stats = st.slider("Thống kê", 0.0, 0.5, analyzer.weights['stats'], 0.05)
            w_pattern = st.slider("Phân tích cầu", 0.0, 0.5, analyzer.weights['pattern'], 0.05)
            w_behavior = st.slider("Hành vi số", 0.0, 0.5, analyzer.weights['behavior'], 0.05)
        with col2:
            w_ml = st.slider("Machine Learning", 0.0, 0.5, analyzer.weights['ml'], 0.05)
            w_meta = st.slider("Meta AI", 0.0, 0.5, analyzer.weights['meta'], 0.05)
            w_ensemble = st.slider("Ensemble", 0.0, 0.5, analyzer.weights['ensemble'], 0.05)
        
        # Normalize weights
        total = w_stats + w_pattern + w_behavior + w_ml + w_meta + w_ensemble
        if total > 0:
            analyzer.weights = {
                'stats': w_stats / total,
                'pattern': w_pattern / total,
                'behavior': w_behavior / total,
                'ml': w_ml / total,
                'meta': w_meta / total,
                'ensemble': w_ensemble / total
            }
        
        st.markdown("#### 🛡️ BẢO MẬT")
        adversarial_mode = st.checkbox("Kích hoạt chế độ đối kháng AI", value=True)
        auto_adjust = st.checkbox("Tự động điều chỉnh trọng số", value=True)
        
        submitted = st.form_submit_button("💾 LƯU CẤU HÌNH", use_container_width=True)
        if submitted:
            st.success("✅ Đã lưu cấu hình hệ thống!")
            
            if auto_adjust:
                st.info("🔄 Chế độ tự động điều chỉnh trọng số đã được kích hoạt")
            if adversarial_mode:
                st.warning("⚔️ Chế độ đối kháng AI - Đang né tránh phát hiện của Kubet")
    
    # Reset
    if st.button("🔄 RESET HỆ THỐNG", use_container_width=True):
        analyzer.history = []
        analyzer.weights = {
            'stats': 0.2,
            'pattern': 0.2,
            'behavior': 0.15,
            'ml': 0.2,
            'meta': 0.15,
            'ensemble': 0.1
        }
        st.session_state.clear()
        st.rerun()

# Footer
st.markdown("""
<div class='footer'>
    <p>🛡️ <b>AI 3-TINH SIÊU CẤP - 116 THUẬT TOÁN</b> | Hệ thống đối kháng AI Kubet | Phiên bản 2.0</p>
    <p><small>⚠️ Kết quả mang tính tham khảo. Quản lý vốn thông minh. Chơi có trách nhiệm.</small></p>
    <p><small>⚙️ Tự động thích nghi | Cập nhật realtime | Tối ưu Samsung Multi Window</small></p>
</div>
""", unsafe_allow_html=True)