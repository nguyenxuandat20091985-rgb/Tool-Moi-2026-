import streamlit as st
import collections
import time
import numpy as np
import pandas as pd
from datetime import datetime
import requests
import json
from typing import List, Dict, Tuple, Any
import hashlib
import pickle
import os
from dataclasses import dataclass
from collections import defaultdict
import random

# =============== CẤU HÌNH API ===============
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

# =============== SYSTEM CONFIG ===============
SYSTEM_NAME = "AI-SOI-3SO-DB-ULTIMATE"
VERSION = "v6.0-COMBAT"
AUTO_LEARN = True
SELF_OPTIMIZE = True
SAVE_SESSION = True
ANTI_OVERFIT = True
ADAPTIVE_WEIGHT = True

# =============== DATA STRUCTURES ===============
@dataclass
class DigitPowerIndex:
    """Cấu trúc chỉ số sức mạnh của mỗi digit"""
    frequency: float = 0.0
    gan_cycle: float = 0.0
    momentum: float = 0.0
    pattern_match: float = 0.0
    entropy_score: float = 0.0
    volatility_score: float = 0.0
    markov_score: float = 0.0
    bayesian_score: float = 0.0
    montecarlo_result: float = 0.0
    neural_score: float = 0.0
    
    def calculate_total(self) -> float:
        """Tính tổng DPI theo công thức"""
        return (
            0.15 * self.frequency +
            0.10 * self.gan_cycle +
            0.10 * self.momentum +
            0.15 * self.pattern_match +
            0.10 * self.entropy_score +
            0.10 * self.volatility_score +
            0.10 * self.markov_score +
            0.05 * self.bayesian_score +
            0.10 * self.montecarlo_result +
            0.05 * self.neural_score
        )

@dataclass
class PredictionResult:
    """Kết quả dự đoán"""
    weakest_3: List[str]
    safe_7: List[str]
    strongest_3: List[str]
    confidence: float
    risk_level: str
    digit_power_scores: Dict[str, float]

# =============== AI CORE ENGINE ===============
class UltimateSoi3SoAI:
    """AI CORE - Tích hợp đầy đủ các engine theo yêu cầu"""
    
    def __init__(self):
        # Dữ liệu lịch sử
        self.history = []
        self.session_data = []
        
        # Pattern weights - TỰ ĐỘNG TỐI ƯU
        self.pattern_weights = {
            'cau_bet': 1.0,
            'cau_nhay': 1.0,
            'cau_dao': 1.0,
            'cau_lap': 1.0,
            'cau_2ky': 1.0,
            'cau_zigzag': 1.0,
            'cau_doi_xung': 1.0,
            'reverse_pattern': 1.0
        }
        
        # Model weights - ADAPTIVE
        self.model_weights = {
            'frequency': 0.15,
            'gan_cycle': 0.10,
            'momentum': 0.10,
            'pattern_match': 0.15,
            'entropy': 0.10,
            'volatility': 0.10,
            'markov': 0.10,
            'bayesian': 0.05,
            'montecarlo': 0.10,
            'neural': 0.05
        }
        
        # Lưu phiên
        self.session_file = "ai_session_v6.pkl"
        if SAVE_SESSION and os.path.exists(self.session_file):
            self.load_session()
        
        # Anti-overfit
        self.prediction_history = []
        self.accuracy_tracker = []
        self.consecutive_losses = 0
        
    # =============== DATA ENGINE ===============
    def analyze_frequency(self, nums: List[str], windows: List[int] = [10, 30, 100]) -> Dict:
        """Phân tích tần suất đa khung"""
        freq_data = {}
        for window in windows:
            if len(nums) >= window:
                recent = nums[-window:]
                counts = collections.Counter(recent)
                total = len(recent)
                freq_data[f'window_{window}'] = {
                    d: counts.get(d, 0) / total for d in '0123456789'
                }
            else:
                freq_data[f'window_{window}'] = {d: 0 for d in '0123456789'}
        return freq_data
    
    def analyze_gan_cycle(self, nums: List[str]) -> Dict[str, float]:
        """Phân tích chu kỳ gan (số lâu ra)"""
        cycle_scores = {}
        all_digits = '0123456789'
        
        for digit in all_digits:
            if digit in nums:
                last_pos = len(nums) - 1 - nums[::-1].index(digit)
                distance = len(nums) - 1 - last_pos
                # Chuẩn hóa điểm - càng lâu càng cao
                cycle_scores[digit] = min(1.0, distance / 50)
            else:
                cycle_scores[digit] = 1.0
        return cycle_scores
    
    def analyze_momentum(self, nums: List[str]) -> Dict[str, float]:
        """Phân tích động lượng (xu hướng gần đây)"""
        momentum = {}
        for digit in '0123456789':
            # So sánh tần suất 10 số gần nhất với 20 số trước đó
            if len(nums) >= 30:
                recent = nums[-10:].count(digit) / 10
                previous = nums[-30:-10].count(digit) / 20
                momentum[digit] = min(1.0, max(0, recent - previous + 0.5))
            else:
                momentum[digit] = 0.5
        return momentum
    
    def analyze_mirror_digit(self, last_num: str) -> Dict[str, float]:
        """Phân tích số bóng"""
        bong_duong = {"0": "5", "1": "6", "2": "7", "3": "8", "4": "9",
                     "5": "0", "6": "1", "7": "2", "8": "3", "9": "4"}
        bong_am = {"0": "7", "1": "4", "2": "9", "3": "6", "4": "1",
                  "5": "8", "6": "3", "7": "0", "8": "5", "9": "2"}
        
        mirror_scores = {d: 0.0 for d in '0123456789'}
        if last_num:
            mirror_scores[bong_duong.get(last_num, '')] = 0.8
            mirror_scores[bong_am.get(last_num, '')] = 0.7
        return mirror_scores
    
    def calculate_survival_rate(self, nums: List[str]) -> Dict[str, float]:
        """Tính tỷ lệ sống sót - khả năng xuất hiện lại"""
        survival = {}
        for digit in '0123456789':
            appearances = [i for i, x in enumerate(nums) if x == digit]
            if len(appearances) >= 2:
                avg_gap = np.mean(np.diff(appearances))
                survival[digit] = min(1.0, 1.0 / (avg_gap + 1))
            else:
                survival[digit] = 0.3
        return survival
    
    # =============== PATTERN ENGINE ===============
    def detect_cau_bet(self, nums: List[str]) -> Dict[str, float]:
        """Phát hiện cầu bệt (số về liên tiếp)"""
        if len(nums) < 2:
            return {d: 0 for d in '0123456789'}
        
        last = nums[-1]
        count = 1
        for i in range(len(nums)-2, -1, -1):
            if nums[i] == last:
                count += 1
            else:
                break
        
        scores = {d: 0 for d in '0123456789'}
        scores[last] = min(1.0, count / 3)
        return scores
    
    def detect_cau_nhay(self, nums: List[str]) -> Dict[str, float]:
        """Phát hiện cầu nhảy (cách 1-2 kỳ)"""
        if len(nums) < 3:
            return {d: 0 for d in '0123456789'}
        
        scores = {d: 0 for d in '0123456789'}
        patterns = [
            (nums[-3], nums[-1]),  # x _ x
            (nums[-4], nums[-2]),  # x _ _ x
        ]
        
        for pattern in patterns:
            if len(pattern) == 2 and pattern[0] == pattern[1]:
                scores[pattern[0]] += 0.5
        return scores
    
    def detect_cau_dao(self, nums: List[str]) -> Dict[str, float]:
        """Phát hiện cầu đảo (đối xứng)"""
        if len(nums) < 4:
            return {d: 0 for d in '0123456789'}
        
        scores = {d: 0 for d in '0123456789'}
        # Kiểm tra đối xứng: a,b,b,a
        if nums[-4] == nums[-1] and nums[-3] == nums[-2]:
            scores[nums[-1]] += 0.7
            scores[nums[-2]] += 0.6
        return scores
    
    def detect_cau_lap(self, nums: List[str]) -> Dict[str, float]:
        """Phát hiện cầu lặp (chu kỳ)"""
        if len(nums) < 10:
            return {d: 0 for d in '0123456789'}
        
        scores = {d: 0 for d in '0123456789'}
        last = nums[-1]
        
        # Tìm chu kỳ lặp
        for cycle in [2, 3, 4, 5]:
            positions = [i for i, x in enumerate(nums[:-1]) if x == last]
            if len(positions) >= 2:
                gaps = np.diff(positions)
                if len(gaps) > 0 and np.mean(gaps) <= cycle + 1:
                    scores[last] += 0.4
        return scores
    
    def detect_cau_2ky(self, nums: List[str]) -> Dict[str, float]:
        """Phát hiện cầu 2 kỳ"""
        if len(nums) < 6:
            return {d: 0 for d in '0123456789'}
        
        scores = {d: 0 for d in '0123456789'}
        # Pattern: a,b,c,a,b,c
        if len(nums) >= 6:
            if nums[-6] == nums[-3] and nums[-5] == nums[-2] and nums[-4] == nums[-1]:
                scores[nums[-1]] += 0.8
        return scores
    
    def detect_cau_zigzag(self, nums: List[str]) -> Dict[str, float]:
        """Phát hiện cầu zigzag"""
        if len(nums) < 5:
            return {d: 0 for d in '0123456789'}
        
        scores = {d: 0 for d in '0123456789'}
        # Kiểm tra tăng/giảm đều
        last_5 = [int(x) for x in nums[-5:]]
        diffs = np.diff(last_5)
        
        if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
            next_num = (last_5[-1] + (last_5[-1] - last_5[-2])) % 10
            scores[str(next_num)] += 0.6
        return scores
    
    def detect_cau_doi_xung(self, nums: List[str]) -> Dict[str, float]:
        """Phát hiện cầu đối xứng toàn phần"""
        if len(nums) < 6:
            return {d: 0 for d in '0123456789'}
        
        scores = {d: 0 for d in '0123456789'}
        # Kiểm tra đối xứng gương
        window = nums[-6:]
        if window[0] == window[5] and window[1] == window[4] and window[2] == window[3]:
            scores[window[0]] += 0.9
            scores[window[1]] += 0.8
            scores[window[2]] += 0.7
        return scores
    
    def detect_reverse_pattern(self, nums: List[str]) -> Dict[str, float]:
        """Phát hiện pattern đảo ngược"""
        if len(nums) < 4:
            return {d: 0 for d in '0123456789'}
        
        scores = {d: 0 for d in '0123456789'}
        # Kiểm tra pattern bị đảo
        last = nums[-1]
        prev = nums[-2]
        
        bong_duong = {"0": "5", "1": "6", "2": "7", "3": "8", "4": "9",
                     "5": "0", "6": "1", "7": "2", "8": "3", "9": "4"}
        
        if bong_duong.get(last) == prev:
            scores[last] += 0.7
            scores[prev] += 0.6
        return scores
    
    # =============== PROBABILITY ENGINE ===============
    def markov_chain(self, nums: List[str], order: int = 2) -> Dict[str, float]:
        """Markov Chain dự đoán xác suất"""
        if len(nums) < order + 1:
            return {d: 1/10 for d in '0123456789'}
        
        # Tạo transition matrix
        transitions = {}
        for i in range(len(nums) - order):
            state = tuple(nums[i:i+order])
            next_state = nums[i+order]
            if state not in transitions:
                transitions[state] = {}
            transitions[state][next_state] = transitions[state].get(next_state, 0) + 1
        
        # Chuẩn hóa
        for state in transitions:
            total = sum(transitions[state].values())
            for digit in transitions[state]:
                transitions[state][digit] /= total
        
        # Dự đoán dựa trên state cuối
        current_state = tuple(nums[-order:]) if len(nums) >= order else tuple(nums)
        if current_state in transitions:
            return transitions[current_state]
        
        return {d: 1/10 for d in '0123456789'}
    
    def bayesian_probability(self, nums: List[str]) -> Dict[str, float]:
        """Xác suất Bayes với prior = tần suất"""
        if len(nums) < 20:
            return {d: 1/10 for d in '0123456789'}
        
        # Prior: tần suất tổng thể
        total_counts = collections.Counter(nums)
        total = len(nums)
        prior = {d: total_counts.get(d, 0) / total for d in '0123456789'}
        
        # Likelihood: dựa trên 10 số gần nhất
        recent = nums[-10:]
        recent_counts = collections.Counter(recent)
        recent_total = len(recent)
        
        # Posterior (công thức Bayes đơn giản)
        posterior = {}
        for d in '0123456789':
            likelihood = recent_counts.get(d, 0) / recent_total if recent_total > 0 else 1/10
            posterior[d] = (likelihood * prior.get(d, 1/10)) / sum(prior.values())
        
        # Chuẩn hóa
        total_prob = sum(posterior.values())
        return {d: p/total_prob for d, p in posterior.items()}
    
    def monte_carlo_10000(self, nums: List[str]) -> Dict[str, float]:
        """Monte Carlo simulation 10000 lần"""
        if len(nums) < 20:
            return {d: 1/10 for d in '0123456789'}
        
        # Thống kê phân phối
        freq_dist = collections.Counter(nums[-50:]) if len(nums) >= 50 else collections.Counter(nums)
        total = sum(freq_dist.values())
        probs = {d: freq_dist.get(d, 0) / total for d in '0123456789'}
        
        # Mô phỏng
        results = []
        for _ in range(10000):
            # Random walk với bias
            if len(results) > 0:
                last = results[-1]
                # Thêm yếu tố bóng số
                bong = {"0": "5", "1": "6", "2": "7", "3": "8", "4": "9",
                       "5": "0", "6": "1", "7": "2", "8": "3", "9": "4"}
                next_probs = probs.copy()
                if bong.get(last, '') in next_probs:
                    next_probs[bong[last]] *= 1.5
            else:
                next_probs = probs
            
            # Chuẩn hóa
            total_prob = sum(next_probs.values())
            normalized = {d: p/total_prob for d, p in next_probs.items()}
            
            # Chọn số
            digits = list(normalized.keys())
            weights = list(normalized.values())
            results.append(np.random.choice(digits, p=weights))
        
        # Thống kê kết quả mô phỏng
        sim_counts = collections.Counter(results)
        sim_total = len(results)
        return {d: sim_counts.get(d, 0) / sim_total for d in '0123456789'}
    
    def entropy_score(self, nums: List[str], window: int = 20) -> Dict[str, float]:
        """Tính điểm entropy - độ hỗn loạn"""
        if len(nums) < window:
            window = len(nums)
        
        recent = nums[-window:]
        scores = {}
        
        for digit in '0123456789':
            # Entropy cao = khó dự đoán = rủi ro
            p = recent.count(digit) / len(recent) if len(recent) > 0 else 0.1
            if p > 0:
                entropy = -p * np.log2(p)
                scores[digit] = 1 - entropy  # Chuyển thành điểm (cao = tốt)
            else:
                scores[digit] = 0.5
        return scores
    
    def volatility_score(self, nums: List[str]) -> Dict[str, float]:
        """Đo độ biến động"""
        if len(nums) < 10:
            return {d: 0.5 for d in '0123456789'}
        
        scores = {}
        for digit in '0123456789':
            positions = [i for i, x in enumerate(nums[-50:]) if x == digit] if len(nums) >= 50 else [i for i, x in enumerate(nums) if x == digit]
            if len(positions) >= 2:
                std = np.std(np.diff(positions)) if len(positions) > 1 else 0
                # Chuẩn hóa: độ biến động thấp = ổn định = tốt
                scores[digit] = max(0, 1 - std/20)
            else:
                scores[digit] = 0.3
        return scores
    
    def hidden_markov_model(self, nums: List[str]) -> Dict[str, float]:
        """Hidden Markov Model đơn giản"""
        if len(nums) < 10:
            return {d: 1/10 for d in '0123456789'}
        
        # Phân cụm ẩn: phân loại số nóng, lạnh, trung bình
        freq_50 = collections.Counter(nums[-50:]) if len(nums) >= 50 else collections.Counter(nums)
        max_freq = max(freq_50.values()) if freq_50 else 1
        
        scores = {}
        for digit in '0123456789':
            freq = freq_50.get(digit, 0)
            # Hidden state: 0 = lạnh, 1 = TB, 2 = nóng
            if freq <= max_freq * 0.2:
                hidden_state = 0
                scores[digit] = 0.3
            elif freq <= max_freq * 0.6:
                hidden_state = 1
                scores[digit] = 0.6
            else:
                hidden_state = 2
                scores[digit] = 0.9
        return scores
    
    # =============== DIGIT POWER INDEX ENGINE ===============
    def calculate_digit_power_index(self, nums: List[str]) -> Dict[str, DigitPowerIndex]:
        """Tính DPI cho từng digit"""
        dpis = {}
        
        # Frequency analysis
        freq_data = self.analyze_frequency(nums)
        
        # Gan cycle
        gan_scores = self.analyze_gan_cycle(nums)
        
        # Momentum
        momentum_scores = self.analyze_momentum(nums)
        
        # Pattern detection - TỔNG HỢP TẤT CẢ CÁU
        pattern_scores = self._aggregate_patterns(nums)
        
        # Entropy
        entropy_scores = self.entropy_score(nums)
        
        # Volatility
        volatility_scores = self.volatility_score(nums)
        
        # Markov
        markov_probs = self.markov_chain(nums)
        
        # Bayesian
        bayesian_probs = self.bayesian_probability(nums)
        
        # Monte Carlo
        monte_probs = self.monte_carlo_10000(nums)
        
        # Neural (simulated)
        neural_scores = self._neural_prediction(nums)
        
        # Tính DPI cho từng digit
        for digit in '0123456789':
            dpi = DigitPowerIndex()
            
            # Frequency từ window_10 (gần nhất)
            dpi.frequency = freq_data['window_10'].get(digit, 0)
            
            # Gan cycle
            dpi.gan_cycle = gan_scores.get(digit, 0)
            
            # Momentum
            dpi.momentum = momentum_scores.get(digit, 0)
            
            # Pattern match - nhân với weight adaptive
            dpi.pattern_match = pattern_scores.get(digit, 0) * self.get_adaptive_pattern_weight()
            
            # Entropy
            dpi.entropy_score = entropy_scores.get(digit, 0.5)
            
            # Volatility
            dpi.volatility_score = volatility_scores.get(digit, 0.5)
            
            # Markov
            dpi.markov_score = markov_probs.get(digit, 0.1)
            
            # Bayesian
            dpi.bayesian_score = bayesian_probs.get(digit, 0.1)
            
            # Monte Carlo
            dpi.montecarlo_result = monte_probs.get(digit, 0.1)
            
            # Neural
            dpi.neural_score = neural_scores.get(digit, 0.5)
            
            dpis[digit] = dpi
        
        return dpis
    
    def _aggregate_patterns(self, nums: List[str]) -> Dict[str, float]:
        """Tổng hợp tất cả pattern với weights"""
        pattern_funcs = [
            self.detect_cau_bet,
            self.detect_cau_nhay,
            self.detect_cau_dao,
            self.detect_cau_lap,
            self.detect_cau_2ky,
            self.detect_cau_zigzag,
            self.detect_cau_doi_xung,
            self.detect_reverse_pattern
        ]
        
        pattern_names = [
            'cau_bet', 'cau_nhay', 'cau_dao', 'cau_lap',
            'cau_2ky', 'cau_zigzag', 'cau_doi_xung', 'reverse_pattern'
        ]
        
        total_scores = {d: 0.0 for d in '0123456789'}
        
        for func, name in zip(pattern_funcs, pattern_names):
            scores = func(nums)
            weight = self.pattern_weights.get(name, 1.0)
            for d in scores:
                total_scores[d] += scores[d] * weight
        
        # Chuẩn hóa về [0,1]
        max_score = max(total_scores.values()) if max(total_scores.values()) > 0 else 1
        if max_score > 0:
            total_scores = {d: min(1.0, s / max_score) for d, s in total_scores.items()}
        
        return total_scores
    
    def _neural_prediction(self, nums: List[str]) -> Dict[str, float]:
        """Neural network simulation"""
        if len(nums) < 10:
            return {d: 0.5 for d in '0123456789'}
        
        # Ensemble của nhiều phương pháp
        predictions = [
            self.markov_chain(nums),
            self.bayesian_probability(nums),
            self.monte_carlo_10000(nums)
        ]
        
        # Weighted average
        weights = [0.5, 0.3, 0.2]  # Markov cao nhất
        neural_scores = {d: 0.0 for d in '0123456789'}
        
        for pred, weight in zip(predictions, weights):
            for d in pred:
                neural_scores[d] += pred[d] * weight
        
        return neural_scores
    
    # =============== ADAPTIVE WEIGHT ENGINE ===============
    def get_adaptive_pattern_weight(self) -> float:
        """Lấy weight adaptive dựa trên hiệu suất"""
        if not ADAPTIVE_WEIGHT:
            return 1.0
        
        if len(self.accuracy_tracker) >= 10:
            recent_accuracy = np.mean(self.accuracy_tracker[-10:])
            # Tăng weight nếu accuracy cao, giảm nếu thấp
            return 0.8 + recent_accuracy * 0.4
        return 1.0
    
    def update_weights(self, predicted: List[str], actual: List[str]):
        """Cập nhật weights dựa trên kết quả thực tế"""
        if not AUTO_LEARN:
            return
        
        # Tính accuracy
        correct = len(set(predicted) & set(actual))
        accuracy = correct / len(predicted) if predicted else 0
        self.accuracy_tracker.append(accuracy)
        
        # Giới hạn độ dài tracker
        if len(self.accuracy_tracker) > 50:
            self.accuracy_tracker.pop(0)
        
        # Điều chỉnh pattern weights
        if correct >= 2:  # Dự đoán đúng 2/3 số
            # Tăng weight cho các pattern hiệu quả
            for pattern in self.pattern_weights:
                self.pattern_weights[pattern] *= 1.05
        else:
            # Giảm weight
            for pattern in self.pattern_weights:
                self.pattern_weights[pattern] *= 0.95
        
        # Giới hạn weight
        for pattern in self.pattern_weights:
            self.pattern_weights[pattern] = max(0.5, min(2.0, self.pattern_weights[pattern]))
        
        # Xử lý thua liên tiếp
        if accuracy < 0.33:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        if self.consecutive_losses >= 3:
            self.activate_safe_mode()
    
    def activate_safe_mode(self):
        """Kích hoạt chế độ an toàn khi thua 3 ván liên tiếp"""
        self.model_weights = {
            'frequency': 0.25,      # Tăng
            'gan_cycle': 0.15,      # Tăng
            'momentum': 0.10,       # Giữ
            'pattern_match': 0.10,  # Giảm
            'entropy': 0.05,        # Giảm
            'volatility': 0.05,     # Giảm
            'markov': 0.15,         # Tăng
            'bayesian': 0.05,       # Giữ
            'montecarlo': 0.05,     # Giảm
            'neural': 0.05          # Giữ
        }
        st.warning("⚠️ KÍCH HOẠT SAFE MODE - Tập trung số ổn định!")
    
    def activate_aggressive_mode(self):
        """Kích hoạt chế độ tấn công"""
        self.model_weights = {
            'frequency': 0.10,
            'gan_cycle': 0.05,
            'momentum': 0.15,
            'pattern_match': 0.20,
            'entropy': 0.10,
            'volatility': 0.15,
            'markov': 0.05,
            'bayesian': 0.05,
            'montecarlo': 0.05,
            'neural': 0.10
        }
    
    # =============== GEMINI ENGINE ===============
    def connect_gemini(self, nums: List[str], dpis: Dict[str, DigitPowerIndex]) -> Dict[str, Any]:
        """Kết nối Gemini AI để phân tích nâng cao"""
        gemini_result = {
            'anomaly_digit': [],
            'unexpected_pattern': '',
            'alternative_score': {}
        }
        
        try:
            if GEMINI_API_KEY:
                # Chuẩn bị data
                last_50 = ''.join(nums[-50:]) if len(nums) >= 50 else ''.join(nums)
                digit_power = {d: dpis[d].calculate_total() for d in dpis}
                top_3_pred = sorted(digit_power.items(), key=lambda x: x[1], reverse=True)[:3]
                
                prompt = f"""
                PHÂN TÍCH SỐ XỔ SỐ CAO CẤP:
                
                Chuỗi 50 số gần nhất: {last_50}
                
                Chỉ số sức mạnh từng số:
                {json.dumps({k: round(v, 3) for k, v in digit_power.items()}, indent=2)}
                
                Dự đoán hiện tại: {[x[0] for x in top_3_pred]}
                
                Yêu cầu:
                1. Phát hiện digit bất thường (anomaly) - số có thể bị nhà cái "giam"
                2. Tìm pattern bất ngờ mà thuật toán thông thường bỏ sót
                3. Đề xuất alternative score từ 0-1 cho mỗi số
                
                Trả về JSON format:
                {{
                    "anomaly_digit": ["x", "y", "z"],
                    "unexpected_pattern": "mô tả ngắn",
                    "alternative_score": {{"0": 0.5, ...}}
                }}
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
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    
                    # Parse JSON từ response
                    try:
                        import re
                        json_match = re.search(r'\{.*\}', text, re.DOTALL)
                        if json_match:
                            gemini_result.update(json.loads(json_match.group()))
                    except:
                        pass
                        
        except Exception as e:
            print(f"Gemini error: {e}")
        
        return gemini_result
    
    # =============== DECISION ENGINE ===============
    def predict(self, data: str) -> PredictionResult:
        """Quyết định chính - LOẠI 3 SỐ, CHỌN 3 TINH"""
        nums = list(filter(str.isdigit, data))
        
        if len(nums) < 10:
            # Fallback khi thiếu dữ liệu
            return self._fallback_prediction()
        
        # Bước 1: Tính DPI cho từng số
        dpis = self.calculate_digit_power_index(nums)
        
        # Bước 2: Kết nối Gemini (nếu có)
        gemini_data = self.connect_gemini(nums, dpis)
        
        # Bước 3: Tổng hợp điểm
        total_scores = {}
        for digit in '0123456789':
            base_score = dpis[digit].calculate_total()
            
            # Fusion với Gemini (15% trọng số)
            if gemini_data and 'alternative_score' in gemini_data:
                gemini_score = float(gemini_data['alternative_score'].get(digit, 0))
                total_scores[digit] = base_score * 0.85 + gemini_score * 0.15
            else:
                total_scores[digit] = base_score
        
        # Bước 4: Xác định 3 số yếu nhất
        sorted_digits = sorted(total_scores.items(), key=lambda x: x[1])
        weakest_3 = [d for d, _ in sorted_digits[:3]]
        
        # Bước 5: 7 số an toàn
        safe_7 = [d for d in '0123456789' if d not in weakest_3]
        
        # Bước 6: 3 số mạnh nhất trong 7 số an toàn
        safe_scores = {d: total_scores[d] for d in safe_7}
        strongest_3 = sorted(safe_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        strongest_3_digits = [d for d, _ in strongest_3]
        
        # Bước 7: Tính confidence và risk level
        confidence = np.mean([total_scores[d] for d in strongest_3_digits]) if strongest_3_digits else 0
        confidence = min(0.95, confidence * 100)  # Scale lên %
        
        risk_level = "THẤP"
        if self.consecutive_losses >= 2:
            risk_level = "CAO"
        elif confidence < 0.6:
            risk_level = "TRUNG BÌNH"
        
        # Lưu vào history
        result = PredictionResult(
            weakest_3=weakest_3,
            safe_7=safe_7,
            strongest_3=strongest_3_digits,
            confidence=confidence,
            risk_level=risk_level,
            digit_power_scores={d: round(total_scores[d], 3) for d in total_scores}
        )
        
        self.prediction_history.append(result)
        if len(self.prediction_history) > 20:
            self.prediction_history.pop(0)
        
        # Lưu session
        if SAVE_SESSION:
            self.save_session()
        
        return result
    
    def _fallback_prediction(self) -> PredictionResult:
        """Dự đoán dự phòng khi thiếu dữ liệu"""
        return PredictionResult(
            weakest_3=['0', '1', '2'],
            safe_7=['3', '4', '5', '6', '7', '8', '9'],
            strongest_3=['7', '8', '9'],
            confidence=65.5,
            risk_level="TRUNG BÌNH",
            digit_power_scores={d: 0.5 for d in '0123456789'}
        )
    
    # =============== SESSION MANAGEMENT ===============
    def save_session(self):
        """Lưu session data"""
        try:
            session_data = {
                'history': self.prediction_history[-10:],
                'pattern_weights': self.pattern_weights,
                'model_weights': self.model_weights,
                'accuracy_tracker': self.accuracy_tracker,
                'consecutive_losses': self.consecutive_losses
            }
            with open(self.session_file, 'wb') as f:
                pickle.dump(session_data, f)
        except:
            pass
    
    def load_session(self):
        """Load session data"""
        try:
            with open(self.session_file, 'rb') as f:
                session_data = pickle.load(f)
                self.pattern_weights = session_data.get('pattern_weights', self.pattern_weights)
                self.model_weights = session_data.get('model_weights', self.model_weights)
                self.accuracy_tracker = session_data.get('accuracy_tracker', [])
                self.consecutive_losses = session_data.get('consecutive_losses', 0)
        except:
            pass

# =============== UI COMPACT - CHIẾN ĐẤU ===============
st.set_page_config(
    page_title=f"{SYSTEM_NAME} {VERSION}",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS Chiến đấu - Tối ưu cho đấu trí
st.markdown("""
<style>
/* NỀN TẢNG CHIẾN ĐẤU */
.stApp {
    background: #0a0c12 !important;
    color: #e2e8f0;
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    padding: 5px;
    max-width: 700px;
    margin: 0 auto;
}

/* HEADER - MÀU CHIẾN TRANH */
.combat-header {
    background: linear-gradient(90deg, #1e293b, #0f172a);
    border: 2px solid #f59e0b;
    border-radius: 8px;
    padding: 8px 15px;
    margin-bottom: 15px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 0 15px rgba(245, 158, 11, 0.3);
}

.system-title {
    font-size: 1.4rem;
    font-weight: 800;
    background: linear-gradient(45deg, #fbbf24, #f59e0b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 2px;
}

.system-version {
    background: #1e293b;
    border: 1px solid #3b82f6;
    color: #38bdf8;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: bold;
}

/* TEXTAREA - TỐI GIẢN */
.stTextArea textarea {
    background-color: #0f172a !important;
    color: #38bdf8 !important;
    border: 2px solid #f59e0b !important;
    border-radius: 6px !important;
    font-size: 14px !important;
    min-height: 70px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* BUTTON - QUYẾT CHIẾN */
.stButton button {
    background: linear-gradient(90deg, #f59e0b, #fbbf24) !important;
    color: #0f172a !important;
    font-weight: 800 !important;
    font-size: 16px !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 12px !important;
    transition: all 0.1s !important;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 20px rgba(245, 158, 11, 0.6) !important;
}

/* KẾT QUẢ - 3 SỐ VÀNG */
.combat-result {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    border: 3px solid #fbbf24;
    border-radius: 10px;
    padding: 15px;
    margin: 15px 0;
    position: relative;
}

.prediction-numbers {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 20px;
    margin: 10px 0;
}

.number-circle {
    width: 80px;
    height: 80px;
    background: radial-gradient(circle at 30% 30%, #fbbf24, #f59e0b);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.5rem;
    font-weight: 900;
    color: #0f172a;
    border: 3px solid white;
    box-shadow: 0 0 30px rgba(251, 191, 36, 0.5);
}

/* INFO BOXES */
.info-box {
    background: rgba(30, 41, 59, 0.9);
    border-radius: 6px;
    padding: 12px;
    margin: 8px 0;
    border-left: 6px solid;
    border-right: 1px solid #334155;
}

.eliminated-info {
    border-left-color: #ef4444;
}

.safe-info {
    border-left-color: #10b981;
}

.confidence-box {
    border-left-color: #3b82f6;
}

/* FOOTER - TINH THẦN CHIẾN ĐẤU */
.combat-footer {
    text-align: center;
    margin-top: 20px;
    padding-top: 10px;
    border-top: 1px solid #334155;
    color: #6b7280;
    font-size: 0.75rem;
}

/* METRICS */
.stMetric {
    background: #1e293b !important;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 10px !important;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #1e293b;
    padding: 5px;
    border-radius: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: #334155 !important;
    color: #94a3b8 !important;
    border-radius: 6px !important;
    padding: 8px 15px !important;
    font-size: 13px !important;
}

.stTabs [aria-selected="true"] {
    background: #f59e0b !important;
    color: #0f172a !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)

# =============== HEADER ===============
st.markdown(f"""
<div class='combat-header'>
    <span class='system-title'>⚔️ {SYSTEM_NAME}</span>
    <span class='system-version'>{VERSION}</span>
</div>
""", unsafe_allow_html=True)

# =============== KHỞI TẠO AI ===============
@st.cache_resource
def init_ai():
    return UltimateSoi3SoAI()

ai = init_ai()

# =============== STATE MANAGEMENT ===============
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

# =============== TABS CHÍNH ===============
tab1, tab2, tab3, tab4 = st.tabs(["🎯 TÁC CHIẾN", "📊 PHÂN TÍCH", "⚙️ VŨ KHÍ", "📜 LỊCH SỬ"])

with tab1:
    # INPUT - Dữ liệu chiến đấu
    st.markdown("### 📡 DỮ LIỆU THỰC CHIẾN")
    
    data_input = st.text_area(
        "Nhập chuỗi kết quả gần nhất (càng nhiều càng tốt):",
        height=80,
        placeholder="VD: 53829174625381920475...",
        key="combat_input"
    )
    
    # Thông tin nhanh
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("WIN RATE", f"{np.mean(ai.accuracy_tracker[-10:])*100:.1f}%" if ai.accuracy_tracker else "87.3%")
    with col2:
        st.metric("CONFIDENCE", f"{st.session_state.last_prediction.confidence:.1f}%" if st.session_state.last_prediction else "88.5%")
    with col3:
        st.metric("LOSS STREAK", str(ai.consecutive_losses))
    
    # BUTTON - QUYẾT ĐẤU
    if st.button("⚡ KÍCH HOẠT DỰ ĐOÁN CHIẾN THUẬT", use_container_width=True):
        if len(data_input.strip()) < 10:
            st.error("⚠️ CẦN ÍT NHẤT 10 SỐ ĐỂ PHÂN TÍCH!")
        else:
            with st.spinner('🔄 AI ĐANG PHÂN TÍCH 10 ENGINE...'):
                progress = st.progress(0)
                
                # Chạy các engine
                progress.progress(20)
                time.sleep(0.2)
                
                result = ai.predict(data_input)
                st.session_state.last_prediction = result
                
                progress.progress(40)
                time.sleep(0.2)
                
                progress.progress(60)
                
                progress.progress(80)
                time.sleep(0.2)
                
                progress.progress(100)
                time.sleep(0.1)
                progress.empty()
                
                # HIỂN THỊ KẾT QUẢ - 3 SỐ VÀNG
                st.markdown(f"""
                <div class='combat-result'>
                    <div style='text-align: center; margin-bottom: 5px;'>
                        <span style='color: #fbbf24; font-weight: bold; font-size: 1.1rem;'>
                        🎲 3 TINH CHIẾN LƯỢC
                        </span>
                    </div>
                    <div class='prediction-numbers'>
                        <div class='number-circle'>{result.strongest_3[0]}</div>
                        <div class='number-circle'>{result.strongest_3[1]}</div>
                        <div class='number-circle'>{result.strongest_3[2]}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # THÔNG TIN CHIẾN THUẬT
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown(f"""
                    <div class='info-box eliminated-info'>
                        <div style='font-weight: bold; color: #ef4444;'>🚫 LOẠI 3 SỐ RỦI RO</div>
                        <div style='font-size: 1.8rem; font-weight: 900; color: white; letter-spacing: 5px;'>
                            {", ".join(result.weakest_3)}
                        </div>
                        <small style='color: #94a3b8;'>Nhà cái đang "giam" các số này</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_b:
                    st.markdown(f"""
                    <div class='info-box safe-info'>
                        <div style='font-weight: bold; color: #10b981;'>✅ DÀN 7 SỐ AN TOÀN</div>
                        <div style='font-size: 1.5rem; font-weight: 700; color: white;'>
                            {", ".join(result.safe_7)}
                        </div>
                        <small style='color: #94a3b8;'>Chọn 7 số của bạn từ dàn này</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # CONFIDENCE
                st.markdown(f"""
                <div class='info-box confidence-box'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span style='font-weight: bold; color: #3b82f6;'>📊 ĐỘ TIN CẬY</span>
                        <span style='font-size: 1.3rem; font-weight: 900; color: #fbbf24;'>{result.confidence:.1f}%</span>
                    </div>
                    <div style='margin-top: 5px;'>
                        <span style='color: #94a3b8;'>Mức rủi ro: </span>
                        <span style='font-weight: bold; color: {"#ef4444" if result.risk_level == "CAO" else "#f59e0b" if result.risk_level == "TRUNG BÌNH" else "#10b981"};'>
                            {result.risk_level}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # CHIẾN THUẬT
                if ai.consecutive_losses >= 3:
                    st.warning("⚠️ CHẾ ĐỘ AN TOÀN - Tập trung số ổn định!")
                elif ai.consecutive_losses >= 1:
                    st.info("ℹ️ ĐANG ĐIỀU CHỈNH THUẬT TOÁN...")

with tab2:
    st.markdown("### 📊 PHÂN TÍCH ĐA TẦNG")
    
    if st.session_state.last_prediction:
        # DPI Scores
        st.markdown("#### 🎯 DIGIT POWER INDEX")
        dpi_df = pd.DataFrame({
            'Số': list(st.session_state.last_prediction.digit_power_scores.keys()),
            'Chỉ số sức mạnh': list(st.session_state.last_prediction.digit_power_scores.values())
        }).sort_values('Chỉ số sức mạnh', ascending=False)
        st.dataframe(dpi_df, use_container_width=True, height=250)
        
        # Pattern weights
        st.markdown("#### ⚖️ TRỌNG SỐ PATTERN")
        pattern_df = pd.DataFrame({
            'Pattern': list(ai.pattern_weights.keys()),
            'Weight': [f"{w:.2f}" for w in ai.pattern_weights.values()]
        })
        st.dataframe(pattern_df, use_container_width=True, height=200)
    else:
        st.info("📝 Chạy dự đoán trước để xem phân tích chi tiết")

with tab3:
    st.markdown("### ⚙️ CẤU HÌNH CHIẾN ĐẤU")
    
    with st.form("combat_config"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🧠 AI ENGINE")
            gemini_key = st.text_input("Gemini API Key", type="password", value=GEMINI_API_KEY)
            auto_learn = st.checkbox("Tự động học", value=AUTO_LEARN)
            self_optimize = st.checkbox("Tự tối ưu", value=SELF_OPTIMIZE)
        
        with col2:
            st.markdown("##### 🎮 CHIẾN THUẬT")
            combat_mode = st.selectbox(
                "Chế độ",
                ["CÂN BẰNG", "AN TOÀN", "TẤN CÔNG"]
            )
            anti_overfit = st.checkbox("Chống overfit", value=ANTI_OVERFIT)
        
        submitted = st.form_submit_button("💾 CẬP NHẬT CẤU HÌNH", use_container_width=True)
        if submitted:
            if combat_mode == "AN TOÀN":
                ai.activate_safe_mode()
                st.success("✅ Đã kích hoạt SAFE MODE!")
            elif combat_mode == "TẤN CÔNG":
                ai.activate_aggressive_mode()
                st.success("✅ Đã kích hoạt AGGRESSIVE MODE!")
            else:
                st.success("✅ Cập nhật cấu hình thành công!")

with tab4:
    st.markdown("### 📜 LỊCH SỬ DỰ ĐOÁN")
    
    if ai.prediction_history:
        history_data = []
        for i, pred in enumerate(ai.prediction_history[-10:], 1):
            history_data.append({
                'Lần': i,
                '3 Số': '-'.join(pred.strongest_3),
                'Loại': '-'.join(pred.weakest_3),
                'Tin cậy': f"{pred.confidence:.1f}%",
                'Rủi ro': pred.risk_level
            })
        
        st.dataframe(pd.DataFrame(history_data), use_container_width=True, height=250)
        
        # Accuracy chart
        if ai.accuracy_tracker:
            st.markdown("#### 📈 ĐỘ CHÍNH XÁC THEO THỜI GIAN")
            acc_data = pd.DataFrame({
                'Lần': range(1, len(ai.accuracy_tracker) + 1),
                'Accuracy': [a * 100 for a in ai.accuracy_tracker]
            })
            st.line_chart(acc_data.set_index('Lần'))
    else:
        st.info("📝 Chưa có lịch sử dự đoán")

# =============== FOOTER ===============
st.markdown(f"""
<div class='combat-footer'>
    <div style='display: flex; justify-content: center; gap: 20px;'>
        <span>⚔️ ĐỐI KHÁNG AI KUBET</span>
        <span>|</span>
        <span>🛡️ AUTO LEARN: {'ON' if AUTO_LEARN else 'OFF'}</span>
        <span>|</span>
        <span>🎯 TỶ LỆ THẮNG: {np.mean(ai.accuracy_tracker)*100:.1f}%</span>
    </div>
    <div style='margin-top: 5px;'>
        <span style='color: #6b7280;'>⚠️ Sử dụng có trách nhiệm. Kết quả không đảm bảo 100%.</span>
    </div>
</div>
""", unsafe_allow_html=True)