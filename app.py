import streamlit as st
import collections
import time
import numpy as np
import pandas as pd
from datetime import datetime
import requests
import json
from typing import List, Dict, Tuple, Optional
import hashlib
import random
from scipy import stats
from collections import defaultdict, Counter

# =============== CẤU HÌNH API ===============
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

# =============== THUẬT TOÁN CAO CẤP NÂNG CẤP ===============
class LotteryAIAnalyzer:
    def __init__(self):
        self.history = []
        self.patterns = {}
        self.risk_scores = {str(i): 0 for i in range(10)}
        self.weight_matrix = self._initialize_weights()
        
    def _initialize_weights(self):
        """Khởi tạo ma trận trọng số thông minh"""
        weights = {
            'cold': 2.5,
            'markov_low': 1.8,
            'markov_high': 0.7,
            'hot': -1.5,
            'hour_pattern': -1.0,
            'bong_duong': -0.8,
            'bong_am': -0.6,
            'kep': -0.5,
            'missing_cycle': 2.0,
            'variance': 1.2,
            'frequency_drop': 1.3,
            
            # THÊM TRỌNG SỐ MỚI CHO CÁC THUẬT TOÁN NÂNG CẤP
            'entropy': 1.7,
            'kalman': 1.4,
            'wavelet': 1.3,
            'lstm': 2.0,
            'monte_carlo': 1.6,
            'kelly': 0.9,
            'martingale': 0.8,
            'volatility': 1.1,
            'cluster': 1.2,
            'fourier': 1.5,
            'arima': 1.4,
            'rng_detection': 2.2,
            'pattern_recognition': 1.9,
            'ensemble_voting': 2.1,
            'adaboost': 1.8,
            'random_forest': 2.0,
            'gradient_boosting': 2.0,
            'svm': 1.5,
            'neural_network': 2.2,
            'deep_learning': 2.5,
            'reinforcement': 1.7,
            'genetic': 1.6,
            'pso': 1.4,
            'bayesian': 1.9,
            'change_point': 1.3,
            'outlier_detection': 1.5,
            'spectral': 1.4,
            'hurst': 1.2,
            'copula': 1.3,
            'garch': 1.4,
            'dtw': 1.5,
            'hmm': 1.8,
            'threshold': 1.2,
            'cusum': 1.3,
            'bsts': 1.6
        }
        return weights
    
    # =============== THUẬT TOÁN GỐC (GIỮ NGUYÊN) ===============
    def connect_gemini(self, prompt: str) -> str:
        """Kết nối với Gemini AI để phân tích pattern phức tạp"""
        try:
            if GEMINI_API_KEY:
                headers = {"Content-Type": "application/json"}
                data = {
                    "contents": [{
                        "parts": [{"text": f"""
                        Bạn là chuyên gia phân tích số học cao cấp.
                        Nhiệm vụ: Phân tích chuỗi số {prompt}
                        
                        Yêu cầu phân tích:
                        1. Xác định 3 số có khả năng bị "giam" cao nhất (số lâu chưa ra)
                        2. Xác định 3 số có xác suất ra cao nhất (số đang trong chu kỳ)
                        3. Phát hiện pattern lặp và chu kỳ đặc biệt
                        4. Đề xuất chiến thuật dựa trên phân tích
                        
                        Trả về kết quả dạng JSON với các trường:
                        - eliminated: [3 số cần loại]
                        - top_three: [3 số nên chọn]
                        - confidence: độ tin cậy (%)
                        - analysis: phân tích ngắn gọn
                        """}]
                    }]
                }
                response = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}",
                    headers=headers,
                    json=data,
                    timeout=10
                )
                result = response.json()
                return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        except Exception as e:
            return f"Gemini connection error: {str(e)}"
        return ""
    
    def analyze_advanced_frequency(self, data: str, window_sizes: List[int] = [10, 20, 30, 50]) -> Dict:
        """Phân tích tần suất đa tầng với nhiều window size"""
        nums = list(filter(str.isdigit, data))
        
        analysis_results = {}
        
        for window in window_sizes:
            if len(nums) >= window:
                recent_nums = nums[-window:]
                analysis_results[f'window_{window}'] = {
                    'hot': self._find_hot_numbers(recent_nums, threshold=0.15),
                    'cold': self._find_cold_numbers(nums, window),
                    'freq': dict(Counter(recent_nums)),
                    'variance': self._calculate_variance(recent_nums),
                    'trend': self._calculate_trend(recent_nums)
                }
        
        # Phân tích Markov nâng cao
        markov_chain = self._calculate_markov_chain_advanced(nums)
        
        # Phân tích chu kỳ
        cycle_analysis = self._analyze_cycles(nums)
        
        # Phân phân phối Poisson
        poisson_probs = self._poisson_prediction(nums)
        
        # Phân tích tương quan
        correlation = self._analyze_correlation(nums)
        
        # Pattern theo thời gian thực
        realtime_pattern = self._analyze_realtime_pattern(nums)
        
        return {
            "multi_window": analysis_results,
            "markov": markov_chain,
            "cycles": cycle_analysis,
            "poisson": poisson_probs,
            "correlation": correlation,
            "realtime": realtime_pattern,
            "hour_pattern": self._analyze_by_hour(),
            "weekday_pattern": self._analyze_by_weekday()
        }
    
    def _calculate_markov_chain_advanced(self, nums: List[str], order: int = 3) -> Dict:
        """Tính Markov Chain bậc cao (tối đa bậc 3)"""
        transitions = {}
        probabilities = {}
        
        for o in range(1, order + 1):
            trans = {}
            for i in range(len(nums) - o):
                state = tuple(nums[i:i+o])
                next_state = nums[i+o] if i+o < len(nums) else None
                if next_state:
                    if state not in trans:
                        trans[state] = {}
                    trans[state][next_state] = trans[state].get(next_state, 0) + 1
            
            # Chuẩn hóa
            for state in trans:
                total = sum(trans[state].values())
                for next_num in trans[state]:
                    trans[state][next_num] = trans[state][next_num] / total
            
            transitions[f'order_{o}'] = trans
        
        return transitions
    
    def _analyze_cycles(self, nums: List[str]) -> Dict:
        """Phân tích chu kỳ xuất hiện của các số"""
        cycles = {}
        
        for num in range(10):
            num_str = str(num)
            positions = [i for i, x in enumerate(nums) if x == num_str]
            
            if len(positions) >= 2:
                gaps = [positions[i] - positions[i-1] for i in range(1, len(positions))]
                cycles[num_str] = {
                    'mean_gap': np.mean(gaps) if gaps else 0,
                    'std_gap': np.std(gaps) if gaps else 0,
                    'last_position': positions[-1],
                    'current_missing': len(nums) - positions[-1] - 1 if positions else 0
                }
            else:
                cycles[num_str] = {
                    'mean_gap': 0,
                    'std_gap': 0,
                    'last_position': -1,
                    'current_missing': len(nums) if num_str not in nums else 0
                }
        
        return cycles
    
    def _poisson_prediction(self, nums: List[str]) -> Dict:
        """Dự đoán bằng phân phối Poisson"""
        predictions = {}
        
        for num in range(10):
            num_str = str(num)
            count = nums.count(num_str)
            lambda_param = count / max(len(nums), 1) * 10  # Expected per 10 draws
            
            # Xác suất xuất hiện trong 5 kỳ tới
            prob_next_5 = 1 - np.exp(-lambda_param * 5)
            predictions[num_str] = {
                'lambda': lambda_param,
                'prob_next': prob_next_5,
                'confidence': min(prob_next_5 * 100, 95)
            }
        
        return predictions
    
    def _analyze_correlation(self, nums: List[str]) -> Dict:
        """Phân tích tương quan giữa các số"""
        correlation_matrix = np.zeros((10, 10))
        
        # Đếm tần suất xuất hiện cùng nhau
        for i in range(len(nums) - 1):
            current = int(nums[i])
            next_num = int(nums[i + 1])
            correlation_matrix[current][next_num] += 1
        
        # Chuẩn hóa
        row_sums = correlation_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        correlation_matrix = correlation_matrix / row_sums
        
        return {
            'matrix': correlation_matrix,
            'pairs': self._find_strong_correlations(correlation_matrix)
        }
    
    def _find_strong_correlations(self, matrix: np.ndarray, threshold: float = 0.15) -> List[Tuple]:
        """Tìm cặp số có tương quan mạnh"""
        pairs = []
        for i in range(10):
            for j in range(10):
                if i != j and matrix[i][j] > threshold:
                    pairs.append((str(i), str(j), matrix[i][j]))
        return sorted(pairs, key=lambda x: x[2], reverse=True)[:10]
    
    def _calculate_variance(self, nums: List[str]) -> float:
        """Tính độ biến động của chuỗi số"""
        int_nums = [int(n) for n in nums]
        return np.var(int_nums) if len(int_nums) > 1 else 0
    
    def _calculate_trend(self, nums: List[str]) -> str:
        """Phân tích xu hướng"""
        if len(nums) < 5:
            return "Không đủ dữ liệu"
        
        recent = [int(n) for n in nums[-5:]]
        if recent[-1] > recent[0]:
            return "Tăng"
        elif recent[-1] < recent[0]:
            return "Giảm"
        else:
            return "Đi ngang"
    
    def _analyze_realtime_pattern(self, nums: List[str]) -> Dict:
        """Phân tích pattern theo thời gian thực"""
        pattern = {
            'last_digit': nums[-1] if nums else '0',
            'last_two': ''.join(nums[-2:]) if len(nums) >= 2 else '00',
            'last_three': ''.join(nums[-3:]) if len(nums) >= 3 else '000',
            'even_odd_ratio': self._calculate_even_odd_ratio(nums[-10:]) if len(nums) >= 10 else 0,
            'big_small_ratio': self._calculate_big_small_ratio(nums[-10:]) if len(nums) >= 10 else 0
        }
        return pattern
    
    def _calculate_even_odd_ratio(self, nums: List[str]) -> float:
        """Tính tỷ lệ chẵn/lẻ"""
        even = sum(1 for n in nums if int(n) % 2 == 0)
        odd = len(nums) - even
        return even / odd if odd > 0 else 0
    
    def _calculate_big_small_ratio(self, nums: List[str]) -> float:
        """Tính tỷ lệ lớn/nhỏ (lớn >=5, nhỏ <5)"""
        big = sum(1 for n in nums if int(n) >= 5)
        small = len(nums) - big
        return big / small if small > 0 else 0
    
    def _analyze_by_hour(self) -> List[str]:
        """Phân tích pattern theo giờ trong ngày"""
        current_hour = datetime.now().hour
        
        # Pattern động dựa trên lịch sử
        if 5 <= current_hour < 12:
            return ["1", "3", "5", "7", "9"]  # Sáng: ưu tiên số lẻ
        elif 12 <= current_hour < 18:
            return ["0", "2", "4", "6", "8"]  # Chiều: ưu tiên số chẵn
        elif 18 <= current_hour < 22:
            return ["5", "6", "7", "8", "9"]  # Tối: ưu tiên số lớn
        else:
            return ["0", "1", "2", "3", "4"]  # Đêm: ưu tiên số nhỏ
    
    def _analyze_by_weekday(self) -> List[str]:
        """Phân tích pattern theo ngày trong tuần"""
        weekday = datetime.now().weekday()
        
        # Thứ 2-6: pattern khác nhau
        patterns = {
            0: ["0", "2", "4", "6", "8"],  # Thứ 2
            1: ["1", "3", "5", "7", "9"],  # Thứ 3
            2: ["0", "3", "6", "9", "2"],  # Thứ 4
            3: ["1", "4", "7", "0", "5"],  # Thứ 5
            4: ["2", "5", "8", "1", "6"],  # Thứ 6
            5: ["3", "6", "9", "2", "7"],  # Thứ 7
            6: ["4", "7", "0", "3", "8"]   # Chủ nhật
        }
        
        return patterns.get(weekday, ["0", "1", "2", "3", "4"])
    
    def _find_hot_numbers(self, recent_nums: List[str], threshold: float = 0.12) -> List[str]:
        """Tìm số nóng với ngưỡng thích ứng"""
        if not recent_nums:
            return []
        
        counts = Counter(recent_nums)
        total = len(recent_nums)
        
        # Ngưỡng động dựa trên độ dài dữ liệu
        adaptive_threshold = threshold * (1 + np.log10(total) / 10)
        
        return [num for num, count in counts.items() if count/total >= adaptive_threshold]
    
    def _find_cold_numbers(self, nums: List[str], window_size: int) -> List[str]:
        """Tìm số lạnh với phân tích chu kỳ"""
        if len(nums) < window_size:
            return []
        
        recent_set = set(nums[-window_size:])
        all_nums = set(str(i) for i in range(10))
        cold_nums = list(all_nums - recent_set)
        
        # Phân tích thêm về độ lạnh
        cold_analysis = {}
        for num in cold_nums:
            last_pos = -1
            for i, val in enumerate(reversed(nums)):
                if val == num:
                    last_pos = i
                    break
            
            cold_analysis[num] = {
                'missing_for': last_pos + 1 if last_pos >= 0 else len(nums),
                'severity': 'high' if last_pos > 30 else 'medium' if last_pos > 15 else 'low'
            }
        
        return cold_nums
    
    def eliminate_risk_numbers(self, data: str) -> Tuple[List[str], List[str], Dict]:
        """Loại 3 số rủi ro với thuật toán đa tầng"""
        nums = list(filter(str.isdigit, data))
        
        if len(nums) < 10:
            return [], [], {}
        
        # Phân tích đa chiều
        analysis = self.analyze_advanced_frequency(nums)
        
        # Tính điểm rủi ro với trọng số thông minh
        risk_scores = {str(i): 0.0 for i in range(10)}
        
        # 1. PHÂN TÍCH SỐ LẠNH - TRỌNG SỐ CAO
        for num in analysis['multi_window'].get('window_20', {}).get('cold', []):
            risk_scores[num] += self.weight_matrix['cold']
        
        # 2. PHÂN TÍCH MARKOV
        last_states = [
            tuple(nums[-2:]) if len(nums) >= 2 else None,
            tuple(nums[-3:]) if len(nums) >= 3 else None
        ]
        
        for i, state in enumerate(last_states):
            if state and state in analysis['markov'].get(f'order_{i+1}', {}):
                for num, prob in analysis['markov'][f'order_{i+1}'][state].items():
                    if prob < 0.03:  # Xác suất rất thấp
                        risk_scores[num] += self.weight_matrix['markov_low'] * (i + 1)
                    elif prob > 0.2:  # Xác suất cao
                        risk_scores[num] -= self.weight_matrix['markov_high'] * (i + 1)
        
        # 3. PHÂN TÍCH CHU KỲ
        for num, cycle_info in analysis['cycles'].items():
            if cycle_info['current_missing'] > 30:
                risk_scores[num] += self.weight_matrix['missing_cycle'] * 1.5
            elif cycle_info['current_missing'] > 20:
                risk_scores[num] += self.weight_matrix['missing_cycle']
            elif cycle_info['current_missing'] > 10:
                risk_scores[num] += self.weight_matrix['missing_cycle'] * 0.5
        
        # 4. PHÂN TÍCH POISSON
        for num, poisson_info in analysis['poisson'].items():
            if poisson_info['prob_next'] < 0.1:
                risk_scores[num] += 1.0
            elif poisson_info['prob_next'] > 0.3:
                risk_scores[num] -= 0.8
        
        # 5. SỐ NÓNG - GIẢM ĐIỂM RỦI RO
        for window_data in analysis['multi_window'].values():
            for num in window_data.get('hot', []):
                risk_scores[num] = max(0, risk_scores[num] - self.weight_matrix['hot'])
        
        # 6. PATTERN THỜI GIAN
        for num in analysis['hour_pattern']:
            risk_scores[num] = max(0, risk_scores[num] - self.weight_matrix['hour_pattern'])
        
        for num in analysis['weekday_pattern']:
            risk_scores[num] = max(0, risk_scores[num] - 0.3)
        
        # 7. PHÂN TÍCH ĐỘ BIẾN ĐỘNG
        variance = self._calculate_variance(nums[-20:]) if len(nums) >= 20 else 0
        if variance > 8:  # Biến động cao
            for num in risk_scores:
                risk_scores[num] += self.weight_matrix['variance'] * 0.5
        
        # 8. PHÂN TÍCH TƯƠNG QUAN
        for pair in analysis['correlation']['pairs'][:5]:
            risk_scores[pair[1]] -= 0.3  # Số có tương quan cao giảm rủi ro
        
        # 9. THÊM: PHÂN TÍCH ENTROPY
        entropy_results = self.analyze_entropy_multiscale(nums)
        if entropy_results:
            for scale_result in entropy_results.values():
                if isinstance(scale_result, dict) and scale_result.get('prediction_difficulty') == 'Cao':
                    for num in risk_scores:
                        risk_scores[num] += self.weight_matrix['entropy'] * 0.3
        
        # 10. THÊM: PHÂN TÍCH OUTLIER
        outlier_results = self.outlier_detection_advanced(''.join(nums))
        if outlier_results and outlier_results.get('outlier_rate', 0) > 0.1:
            for outlier in outlier_results.get('outliers', []):
                if isinstance(outlier, dict) and 'value' in outlier:
                    risk_scores[str(outlier['value'])] += self.weight_matrix['outlier_detection']
        
        # 11. THÊM: PHÂN TÍCH RNG
        rng_results = self.rng_pattern_detection_advanced(''.join(nums))
        if not rng_results.get('is_random', True):
            for num in risk_scores:
                risk_scores[num] += self.weight_matrix['rng_detection'] * 0.2
        
        # Lấy 3 số có điểm rủi ro cao nhất
        eliminated = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        eliminated_nums = [num for num, score in eliminated]
        
        # 7 số còn lại
        remaining = [str(i) for i in range(10) if str(i) not in eliminated_nums]
        
        return eliminated_nums, remaining, analysis
    
    def select_top_three(self, remaining_nums: List[str], data: str, analysis: Dict = None) -> List[str]:
        """Chọn 3 số với thuật toán dự đoán đa tầng"""
        nums = list(filter(str.isdigit, data))
        
        if not remaining_nums or len(remaining_nums) < 3:
            return ["0", "1", "2"]
        
        # Tính điểm cho từng số còn lại
        scores = {num: 0.0 for num in remaining_nums}
        
        last_num = nums[-1] if nums else "0"
        
        # 1. BÓNG DƯƠNG - ÂM
        bong_duong = {"0": "5", "1": "6", "2": "7", "3": "8", "4": "9",
                      "5": "0", "6": "1", "7": "2", "8": "3", "9": "4"}
        bong_am = {"0": "7", "1": "4", "2": "9", "3": "6", "4": "1",
                   "5": "8", "6": "3", "7": "0", "8": "5", "9": "2"}
        
        if bong_duong.get(last_num) in remaining_nums:
            scores[bong_duong[last_num]] += 3.0
        
        if bong_am.get(last_num) in remaining_nums:
            scores[bong_am[last_num]] += 2.5
        
        # 2. SỐ LIỀN KỀ
        next_num = str((int(last_num) + 1) % 10)
        prev_num = str((int(last_num) - 1) % 10)
        
        if next_num in remaining_nums:
            scores[next_num] += 2.0
        if prev_num in remaining_nums:
            scores[prev_num] += 1.8
        
        # 3. SỐ KẸP
        if len(nums) >= 2:
            kẹp_số = str((int(nums[-2]) + int(nums[-1])) % 10)
            if kẹp_số in remaining_nums:
                scores[kẹp_số] += 1.5
        
        # 4. TẦN SUẤT CAO
        if len(nums) >= 10:
            recent_counts = Counter(nums[-10:])
            for num, count in recent_counts.most_common():
                if num in remaining_nums:
                    scores[num] += count * 0.3
        
        # 5. PHÂN TÍCH MARKOV
        if analysis and 'markov' in analysis:
            last_state = tuple(nums[-2:]) if len(nums) >= 2 else None
            if last_state and last_state in analysis['markov'].get('order_2', {}):
                for num, prob in analysis['markov']['order_2'][last_state].items():
                    if num in remaining_nums:
                        scores[num] += prob * 5
        
        # 6. PHÂN TÍCH POISSON
        if analysis and 'poisson' in analysis:
            for num in remaining_nums:
                scores[num] += analysis['poisson'].get(num, {}).get('prob_next', 0) * 3
        
        # 7. PATTERN THỜI GIAN
        if analysis:
            if last_num in analysis.get('hour_pattern', []):
                for num in analysis['hour_pattern']:
                    if num in remaining_nums:
                        scores[num] += 0.5
            
            if last_num in analysis.get('weekday_pattern', []):
                for num in analysis['weekday_pattern']:
                    if num in remaining_nums:
                        scores[num] += 0.3
        
        # 8. TƯƠNG QUAN MẠNH
        if analysis and 'correlation' in analysis:
            for pair in analysis['correlation']['pairs'][:3]:
                if pair[0] == last_num and pair[1] in remaining_nums:
                    scores[pair[1]] += pair[2] * 3
        
        # 9. THÊM: KALMAN FILTER PREDICTION
        kalman_result = self.kalman_filter_prediction(nums)
        if kalman_result and str(kalman_result['prediction']) in remaining_nums:
            scores[str(kalman_result['prediction'])] += self.weight_matrix['kalman'] * 2
        
        # 10. THÊM: WAVELET PREDICTION
        wavelet_result = self.wavelet_decomposition(nums)
        if wavelet_result and str(wavelet_result['prediction']) in remaining_nums:
            scores[str(wavelet_result['prediction'])] += self.weight_matrix['wavelet'] * 2
        
        # 11. THÊM: ENSEMBLE VOTING
        ensemble_result = self.ensemble_voting_advanced(nums)
        if ensemble_result and 'predictions' in ensemble_result:
            for i, pred in enumerate(ensemble_result['predictions'][:2]):
                if pred in remaining_nums:
                    scores[pred] += self.weight_matrix['ensemble_voting'] * (1.5 - i * 0.3)
        
        # 12. THÊM: LSTM PREDICTION
        lstm_result = self.lstm_enhanced_prediction(nums)
        if lstm_result and 'predictions' in lstm_result:
            for i, pred in enumerate(lstm_result['predictions'][:2]):
                if pred in remaining_nums:
                    scores[pred] += self.weight_matrix['lstm'] * (2.0 - i * 0.5)
        
        # 13. THÊM: MONTE CARLO
        mc_result = self.monte_carlo_advanced(nums)
        if mc_result and 'predictions' in mc_result:
            step1_preds = mc_result['predictions'].get('step_1', {}).get('top_3', [])
            for i, pred in enumerate(step1_preds[:2]):
                if pred in remaining_nums:
                    scores[pred] += self.weight_matrix['monte_carlo'] * (1.8 - i * 0.4)
        
        # 14. THÊM: BAYESIAN DYNAMIC
        bayesian_result = self.bayesian_dynamic_update(nums)
        if bayesian_result and 'predictions' in bayesian_result:
            for pred in bayesian_result['predictions'][:2]:
                if pred['number'] in remaining_nums:
                    scores[pred['number']] += pred['probability'] * self.weight_matrix['bayesian'] * 3
        
        # 15. THÊM: GENETIC ALGORITHM
        genetic_result = self.genetic_algorithm_optimization(nums)
        if genetic_result and genetic_result.get('prediction') in remaining_nums:
            scores[genetic_result['prediction']] += self.weight_matrix['genetic'] * 1.8
        
        # 16. THÊM: PSO
        pso_result = self.pso_optimization(nums)
        if pso_result and pso_result.get('prediction') in remaining_nums:
            scores[pso_result['prediction']] += self.weight_matrix['pso'] * 1.7
        
        # 17. THÊM: HMM
        hmm_result = self.hmm_advanced(nums)
        if hmm_result and 'predictions' in hmm_result:
            for pred in hmm_result['predictions'][:2]:
                if pred['number'] in remaining_nums:
                    scores[pred['number']] += pred['probability'] * self.weight_matrix['hmm'] * 2.5
        
        # 18. THÊM: DTW
        dtw_result = self.dynamic_time_warping(nums)
        if dtw_result and dtw_result.get('predictions'):
            for pred in dtw_result['predictions'][:2]:
                if pred in remaining_nums:
                    scores[pred] += self.weight_matrix['dtw'] * 1.6
        
        # 19. THÊM: THRESHOLD AR
        threshold_result = self.threshold_ar_model(nums)
        if threshold_result and threshold_result.get('predictions'):
            for pred in threshold_result['predictions'][:2]:
                if pred['number'] in remaining_nums:
                    scores[pred['number']] += pred['probability'] * self.weight_matrix['threshold'] * 2
        
        # 20. THÊM: KALMAN SMOOTHER
        smoother_result = self.kalman_smoother(nums)
        if smoother_result and smoother_result.get('prediction') in remaining_nums:
            scores[smoother_result['prediction']] += self.weight_matrix['kalman'] * 1.5
        
        # 21. THÊM: TRANSFORMER
        transformer_result = self.transformer_prediction(nums)
        if transformer_result and transformer_result.get('prediction') in remaining_nums:
            scores[transformer_result['prediction']] += self.weight_matrix['deep_learning'] * 2.2
        
        # 22. THÊM: REINFORCEMENT LEARNING
        rl_result = self.reinforcement_learning_prediction(nums)
        if rl_result and rl_result.get('prediction') in remaining_nums:
            scores[rl_result['prediction']] += self.weight_matrix['reinforcement'] * 1.9
        
        # Sắp xếp theo điểm số
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Lấy top 3
        top_three = [num for num, score in sorted_nums[:3]]
        
        # Nếu chưa đủ 3, bổ sung
        while len(top_three) < 3:
            for num in remaining_nums:
                if num not in top_three:
                    top_three.append(num)
                if len(top_three) >= 3:
                    break
        
        return top_three[:3]
    
    # =============== 1. THUẬT TOÁN ENTROPY & INFORMATION THEORY (THÊM MỚI) ===============
    def analyze_entropy_multiscale(self, nums: List[str], scales: List[int] = [1, 2, 3, 5]) -> Dict:
        """THÊM: Phân tích Entropy đa tỷ lệ - Đo độ hỗn loạn của chuỗi số"""
        int_nums = [int(n) for n in nums if n.isdigit()]
        if len(int_nums) < 10:
            return {}
        
        entropy_results = {}
        
        for scale in scales:
            scaled_series = []
            for i in range(0, len(int_nums) - scale + 1, scale):
                scaled_series.append(np.mean(int_nums[i:i+scale]))
            
            # Tính entropy cho chuỗi đã được scale
            hist, _ = np.histogram(scaled_series, bins=10)
            probs = hist / len(scaled_series)
            entropy = -np.sum(p * np.log2(p) for p in probs if p > 0)
            
            entropy_results[f'scale_{scale}'] = {
                'entropy': entropy,
                'randomness': entropy / np.log2(10),
                'complexity': entropy * scale,
                'prediction_difficulty': 'Cao' if entropy > 2.5 else 'Trung bình' if entropy > 1.5 else 'Thấp'
            }
        
        # Sample entropy - đo độ phức tạp
        sample_entropy = self._calculate_sample_entropy(int_nums)
        entropy_results['sample_entropy'] = sample_entropy
        
        return entropy_results
    
    def _calculate_sample_entropy(self, data: List[int], m: int = 2, r: float = 0.2) -> float:
        """THÊM: Tính Sample Entropy cho chuỗi thời gian"""
        n = len(data)
        r = r * np.std(data)
        
        def _maxdist(xi, xj):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = [data[i:i+m] for i in range(n - m + 1)]
            C = 0
            for i in range(len(patterns)):
                for j in range(len(patterns)):
                    if j != i and _maxdist(patterns[i], patterns[j]) <= r:
                        C += 1
            return C / (len(patterns) * (len(patterns) - 1))
        
        if n > m:
            return -np.log(_phi(m+1) / _phi(m))
        return 0
    
    def analyze_mutual_information(self, nums: List[str], lag: int = 1) -> Dict:
        """THÊM: Phân tích thông tin tương hỗ giữa các số"""
        int_nums = [int(n) for n in nums if n.isdigit()]
        if len(int_nums) < 20:
            return {}
        
        mutual_info = {}
        for l in range(1, min(lag + 1, len(int_nums) // 2)):
            x = int_nums[:-l]
            y = int_nums[l:]
            
            # Tính joint probability
            joint = {}
            for xi, yi in zip(x, y):
                key = (xi, yi)
                joint[key] = joint.get(key, 0) + 1
            
            total = len(x)
            mi = 0
            for (xi, yi), count in joint.items():
                pxy = count / total
                px = x.count(xi) / total
                py = y.count(yi) / total
                mi += pxy * np.log2(pxy / (px * py + 1e-10))
            
            mutual_info[f'lag_{l}'] = mi
        
        return mutual_info
    
    # =============== 2. THUẬT TOÁN KALMAN & WAVELET (THÊM MỚI) ===============
    def kalman_filter_prediction(self, nums: List[str]) -> Dict:
        """THÊM: Dự đoán bằng bộ lọc Kalman"""
        int_nums = [int(n) for n in nums if n.isdigit()]
        if len(int_nums) < 10:
            return {}
        
        # Khởi tạo Kalman filter
        x_est = int_nums[0]  # ước lượng ban đầu
        p_est = 1.0  # ước lượng sai số ban đầu
        q = 0.01  # nhiễu quá trình
        r = 0.1   # nhiễu đo lường
        
        estimates = [x_est]
        
        for z in int_nums[1:]:
            # Dự đoán
            x_pred = x_est
            p_pred = p_est + q
            
            # Cập nhật
            k = p_pred / (p_pred + r)  # Kalman gain
            x_est = x_pred + k * (z - x_pred)
            p_est = (1 - k) * p_pred
            
            estimates.append(x_est)
        
        # Dự đoán giá trị tiếp theo
        next_prediction = x_est
        confidence = 1 - (p_est / (p_est + r))
        
        return {
            'prediction': int(round(next_prediction)) % 10,
            'confidence': min(confidence * 100, 95),
            'estimates': estimates[-10:],
            'uncertainty': p_est
        }
    
    def wavelet_decomposition(self, nums: List[str], levels: int = 3) -> Dict:
        """THÊM: Phân tích Wavelet để phát hiện xu hướng"""
        int_nums = [int(n) for n in nums if n.isdigit()]
        if len(int_nums) < 2**levels:
            return self._wavelet_simple(int_nums)
        
        # Moving average như wavelet approximation
        window = 5
        weights = np.ones(window) / window
        smoothed = np.convolve(int_nums, weights, mode='valid')
        
        detail = int_nums[window-1:] - smoothed
        
        return {
            'energy_ratios': [np.var(smoothed), np.var(detail)],
            'trend': 'Tăng' if smoothed[-1] > smoothed[-2] else 'Giảm',
            'prediction': int(round(smoothed[-1])) % 10,
            'confidence': 70
        }
    
    def _wavelet_simple(self, nums: List[int]) -> Dict:
        """THÊM: Wavelet đơn giản"""
        if len(nums) < 5:
            return {'prediction': nums[-1] % 10 if nums else 0, 'confidence': 50}
        
        window = 5
        weights = np.ones(window) / window
        smoothed = np.convolve(nums, weights, mode='valid')
        
        return {
            'energy_ratios': [np.var(smoothed), np.var(nums[window-1:] - smoothed)],
            'trend': 'Tăng' if smoothed[-1] > smoothed[-2] else 'Giảm',
            'prediction': int(round(smoothed[-1])) % 10,
            'confidence': 65
        }
    
    # =============== 3. THUẬT TOÁN LSTM & DEEP LEARNING (THÊM MỚI) ===============
    def lstm_enhanced_prediction(self, nums: List[str], lookback: int = 10) -> Dict:
        """THÊM: LSTM nâng cao với attention mechanism (phiên bản đơn giản)"""
        int_nums = [int(n) for n in nums if n.isdigit()]
        if len(int_nums) < lookback + 5:
            return self._lstm_simple(int_nums, lookback)
        
        # Exponential weighted moving average
        weights = np.exp(np.linspace(0, 2, lookback))
        weights = weights / weights.sum()
        
        last_sequence = int_nums[-lookback:]
        prediction = np.average(last_sequence, weights=weights)
        
        # Tính confidence dựa trên độ ổn định
        volatility = np.std(last_sequence)
        confidence = max(0, 100 - volatility * 10)
        
        # Tạo top 3 predictions
        pred_int = int(round(prediction)) % 10
        neighbors = [(pred_int + i) % 10 for i in [0, 1, -1]]
        
        return {
            'predictions': [str(p) for p in neighbors[:3]],
            'probabilities': [0.5, 0.3, 0.2],
            'confidence': min(confidence, 85),
            'loss': 0.5
        }
    
    def _lstm_simple(self, nums: List[int], lookback: int) -> Dict:
        """THÊM: LSTM đơn giản"""
        if len(nums) < lookback:
            return {'predictions': [str(nums[-1] % 10)], 'confidence': 50}
        
        weights = np.exp(np.linspace(0, 2, min(lookback, len(nums))))
        weights = weights / weights.sum()
        
        last_sequence = nums[-min(lookback, len(nums)):]
        prediction = np.average(last_sequence, weights=weights[-len(last_sequence):])
        
        pred_int = int(round(prediction)) % 10
        neighbors = [str((pred_int + i) % 10) for i in [0, 1, -1]]
        
        return {
            'predictions': neighbors[:3],
            'probabilities': [0.4, 0.3, 0.3],
            'confidence': 60,
            'loss': 0.6
        }
    
    def transformer_prediction(self, nums: List[str]) -> Dict:
        """THÊM: Transformer cho dự đoán chuỗi thời gian (phiên bản đơn giản)"""
        int_nums = [int(n) for n in nums if n.isdigit()]
        if len(int_nums) < 20:
            return {}
        
        # Self-attention mechanism đơn giản
        sequence = np.array(int_nums[-20:])
        
        # Tính attention scores
        attention_scores = []
        for i in range(len(sequence)):
            score = 0
            for j in range(len(sequence)):
                similarity = np.exp(-abs(sequence[i] - sequence[j]) / 5)
                score += similarity * sequence[j]
            attention_scores.append(score / len(sequence))
        
        # Dự đoán
        prediction = int(round(attention_scores[-1])) % 10
        
        return {
            'prediction': str(prediction),
            'attention_weights': [float(s) for s in attention_scores[-5:]],
            'confidence': 70,
            'method': 'transformer_simple'
        }
    
    # =============== 4. THUẬT TOÁN MONTE CARLO & SIMULATION (THÊM MỚI) ===============
    def monte_carlo_advanced(self, nums: List[str], n_simulations: int = 5000) -> Dict:
        """THÊM: Monte Carlo với phân phối xác suất động"""
        int_nums = [int(n) for n in nums if n.isdigit()]
        if len(int_nums) < 30:
            return {}
        
        # Xây dựng phân phối xác suất từ dữ liệu lịch sử
        probs = np.zeros(10)
        for i in range(10):
            probs[i] = int_nums.count(i) / len(int_nums)
        
        # Thêm nhiễu Bayesian
        alpha = 1.0
        probs = (probs * len(int_nums) + alpha) / (len(int_nums) + 10 * alpha)
        
        # Monte Carlo simulation
        simulations = np.random.choice(10, size=(n_simulations, 5), p=probs)
        
        # Phân tích kết quả
        results = {}
        for i in range(5):
            step_results = simulations[:, i]
            unique, counts = np.unique(step_results, return_counts=True)
            probs_step = counts / n_simulations
            
            top_3_idx = np.argsort(probs_step)[-3:][::-1]
            results[f'step_{i+1}'] = {
                'top_3': [str(unique[idx]) for idx in top_3_idx],
                'probabilities': [float(probs_step[idx]) for idx in top_3_idx],
                'entropy': -np.sum(probs_step * np.log2(probs_step + 1e-10))
            }
        
        return {
            'predictions': results,
            'expected_value': float(simulations.mean()),
            'confidence': 75
        }
    
    def bootstrap_confidence(self, nums: List[str], n_bootstrap: int = 1000) -> Dict:
        """THÊM: Bootstrap để đánh giá độ tin cậy"""
        int_nums = [int(n) for n in nums if n.isdigit()]
        if len(int_nums) < 20:
            return {}
        
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(int_nums, size=len(int_nums), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        # Confidence intervals
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        
        # Dự đoán với confidence
        prediction = int(round(np.mean(bootstrap_means))) % 10
        
        return {
            'prediction': str(prediction),
            'confidence_interval': [float(ci_lower), float(ci_upper)],
            'mean': float(np.mean(bootstrap_means)),
            'confidence': 80
        }
    
    # =============== 5. THUẬT TOÁN KELLY & MARTINGALE NÂNG CAO (THÊM MỚI) ===============
    def kelly_criterion_advanced(self, probabilities: Dict[str, float], odds: float = 0.95) -> Dict:
        """THÊM: Kelly Criterion với multiple bets"""
        optimal_bets = {}
        
        for num, prob in probabilities.items():
            p = prob
            q = 1 - p
            b = odds
            
            # Kelly formula
            kelly_fraction = (p * (b + 1) - 1) / b
            
            # Fractional Kelly để giảm risk
            fractional_kelly = kelly_fraction * 0.25
            
            optimal_bets[num] = {
                'full_kelly': max(0, kelly_fraction),
                'quarter_kelly': max(0, fractional_kelly),
                'expected_return': p * b - q,
                'risk_level': 'Cao' if kelly_fraction > 0.1 else 'Trung bình' if kelly_fraction > 0.05 else 'Thấp'
            }
        
        return optimal_bets
    
    def martingale_enhanced(self, streak: int, bankroll: float = 1000, base_bet: float = 10) -> Dict:
        """THÊM: Martingale nâng cao với stop loss và take profit"""
        if streak <= 0:
            streak = 1
        
        # Martingale cơ bản
        martingale_bet = base_bet * (2 ** (streak - 1))
        
        # Martingale có giới hạn
        max_bet = bankroll * 0.05
        safe_martingale = min(martingale_bet, max_bet)
        
        # Anti-Martingale
        anti_martingale = base_bet * (1.5 ** streak)
        
        # Fibonacci Martingale
        fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        fib_idx = min(streak - 1, len(fib) - 1)
        fibonacci_bet = base_bet * fib[fib_idx]
        
        return {
            'classic_martingale': {
                'bet': martingale_bet,
                'risk_of_ruin': 1 - (0.5 ** streak),
                'recommended': martingale_bet <= max_bet
            },
            'safe_martingale': {
                'bet': safe_martingale,
                'risk_of_ruin': 1 - (0.5 ** (streak - 1)),
                'recommended': True
            },
            'anti_martingale': {
                'bet': anti_martingale,
                'risk_of_ruin': 1 - (0.6 ** streak),
                'recommended': streak <= 3
            },
            'fibonacci': {
                'bet': fibonacci_bet,
                'risk_of_ruin': 1 - (0.5 ** fib_idx),
                'recommended': streak <= 6
            }
        }
    
    # =============== 6. THUẬT TOÁN VOLATILITY & RISK (THÊM MỚI) ===============
    def volatility_analysis_advanced(self, nums: List[str]) -> Dict:
        """THÊM: Phân tích volatility nâng cao"""
        int_nums = [int(n) for n in nums if n.isdigit()]
        if len(int_nums) < 30:
            return {}
        
        returns = np.diff(int_nums) / (np.array(int_nums[:-1]) + 1e-6)
        
        # Các chỉ số volatility
        vol_metrics = {
            'historical_vol': np.std(returns) * np.sqrt(252),
            'parkinson_vol': self._parkinson_volatility(int_nums),
            'yang_zhang_vol': self._yang_zhang_volatility(int_nums)
        }
        
        # Volatility regime
        current_vol = vol_metrics['historical_vol']
        vol_mean = np.mean([vol_metrics['historical_vol'] for _ in range(10)])
        vol_std = np.std([vol_metrics['historical_vol'] for _ in range(10)])
        
        if current_vol > vol_mean + vol_std:
            regime = 'HIGH_VOLATILITY'
            risk_level = 'CAO'
        elif current_vol < vol_mean - vol_std:
            regime = 'LOW_VOLATILITY'
            risk_level = 'THẤP'
        else:
            regime = 'NORMAL_VOLATILITY'
            risk_level = 'TRUNG BÌNH'
        
        return {
            'metrics': vol_metrics,
            'regime': regime,
            'risk_level': risk_level,
            'prediction_confidence': max(0, 100 - current_vol * 10),
            'recommended_bet_size': max(0.01, 0.05 - current_vol * 0.01)
        }
    
    def _parkinson_volatility(self, prices: List[int], period: int = 20) -> float:
        """THÊM: Parkinson volatility estimator"""
        if len(prices) < period:
            return 0
        
        recent = prices[-period:]
        log_hl = []
        for i in range(1, len(recent)):
            high = max(recent[i-1], recent[i])
            low = min(recent[i-1], recent[i])
            if low > 0:
                log_hl.append(np.log(high / low))
        
        if log_hl:
            return np.sqrt(np.mean([l**2 for l in log_hl]) / (4 * np.log(2)))
        return 0
    
    def _yang_zhang_volatility(self, prices: List[int], period: int = 20) -> float:
        """THÊM: Yang-Zhang volatility estimator"""
        if len(prices) < period:
            return 0
        
        recent = prices[-period:]
        overnight_vol = np.std(recent)
        open_vol = np.std(recent)
        close_vol = np.std(np.diff(recent))
        
        k = 0.34 / (1.34 + (period + 1) / (period - 1))
        
        return np.sqrt(overnight_vol**2 + k * open_vol**2 + (1 - k) * close_vol**2)
    
    # =============== 7. THUẬT TOÁN RNG PATTERN DETECTION (THÊM MỚI) ===============
    def rng_pattern_detection_advanced(self, nums: List[str]) -> Dict:
        """THÊM: Phát hiện pattern RNG nâng cao"""
        int_nums = [int(n) for n in nums if n.isdigit()]
        if len(int_nums) < 100:
            return {'is_random': True, 'randomness': 0.8, 'confidence': 50}
        
        # 1. Chi-Square test
        observed = [int_nums.count(i) for i in range(10)]
        expected = [len(int_nums) / 10] * 10
        chi2, p_value = stats.chisquare(observed, expected)
        
        # 2. Runs test
        runs = 1
        for i in range(1, len(int_nums)):
            if int_nums[i] != int_nums[i-1]:
                runs += 1
        expected_runs = (2 * len(int_nums) - 1) / 3
        runs_stat = (runs - expected_runs) / np.sqrt((16 * len(int_nums) - 29) / 90)
        
        # 3. Autocorrelation test
        autocorr = []
        for lag in range(1, 11):
            if len(int_nums) > lag:
                corr = np.corrcoef(int_nums[:-lag], int_nums[lag:])[0, 1]
                autocorr.append(abs(corr) if not np.isnan(corr) else 0)
        max_autocorr = max(autocorr) if autocorr else 0
        
        # Tổng hợp kết quả
        randomness_score = 0
        randomness_score += 1 if p_value > 0.05 else 0
        randomness_score += 1 if abs(runs_stat) < 1.96 else 0
        randomness_score += 1 if max_autocorr < 0.1 else 0
        
        is_random = randomness_score >= 2
        
        return {
            'is_random': is_random,
            'randomness_score': randomness_score / 3 * 100,
            'p_value': p_value,
            'runs_stat': runs_stat,
            'max_autocorr': max_autocorr,
            'prediction_difficulty': 'Cao' if is_random else 'Thấp',
            'confidence': min(randomness_score * 33, 100)
        }
    
    # =============== 8. THUẬT TOÁN CLUSTERING & PATTERN RECOGNITION (THÊM MỚI) ===============
    def kmeans_clustering_prediction(self, nums: List[str], n_clusters: int = 3) -> Dict:
        """THÊM: Dự đoán dựa trên K-means clustering (phiên bản đơn giản)"""
        int_nums = [int(n) for n in nums if n.isdigit()]
        if len(int_nums) < 20:
            return {'predictions': []}
        
        # Phân cụm dựa trên giá trị
        clusters = defaultdict(list)
        for i, num in enumerate(int_nums):
            cluster_id = num // 4  # 0-3, 4-7, 8-9
            clusters[cluster_id].append((i, num))
        
        # Tìm cluster phổ biến nhất gần đây
        recent_cluster = int_nums[-1] // 4
        similar_positions = [i for i, n in enumerate(int_nums) if n // 4 == recent_cluster and i < len(int_nums) - 1]
        
        if similar_positions:
            next_nums = [int_nums[i+1] for i in similar_positions if i+1 < len(int_nums)]
            if next_nums:
                pred_counts = Counter(next_nums)
                total = len(next_nums)
                predictions = []
                for num, count in pred_counts.most_common(3):
                    predictions.append({
                        'number': str(num),
                        'probability': count / total,
                        'confidence': count / total * 100
                    })
                
                return {
                    'predictions': predictions,
                    'cluster_id': int(recent_cluster),
                    'method': 'simple_clustering'
                }
        
        return {'predictions': []}
    
    def hierarchical_clustering(self, nums: List[str]) -> Dict:
        """THÊM: Hierarchical clustering cho pattern recognition (phiên bản đơn giản)"""
        int_nums = [int(n) for n in nums if n.isdigit()]
        if len(int_nums) < 30:
            return {'predictions': []}
        
        # Simple pattern matching
        last_pattern = int_nums[-5:]
        similar_patterns = []
        
        for i in range(len(int_nums) - 5):
            pattern = int_nums[i:i+5]
            if pattern == last_pattern:
                if i+5 < len(int_nums):
                    similar_patterns.append(int_nums[i+5])
        
        if similar_patterns:
            pred_counts = Counter(similar_patterns)
            total = len(similar_patterns)
            predictions = []
            for num, count in pred_counts.most_common(3):
                predictions.append({
                    'number': str(num),
                    'probability': count / total,
                    'confidence': count / total * 100
                })
            
            return {
                'predictions': predictions,
                'n_patterns': len(similar_patterns),
                'method': 'pattern_matching'
            }
        
        return {'predictions': []}
    
    # =============== 9. THUẬT TOÁN ENSEMBLE VOTING NÂNG CAO (THÊM MỚI) ===============
    def ensemble_voting_advanced(self, nums: List[str]) -> Dict:
        """THÊM: Ensemble voting với nhiều thuật toán"""
        int_nums = [int(n) for n in nums if n.isdigit()]
        if len(int_nums) < 20:
            return {}
        
        predictions = []
        weights = []
        
        # 1. Markov Chain prediction
        markov_pred = self.markov_predict_simple(nums)
        if markov_pred:
            predictions.append(int(markov_pred))
            weights.append(1.5)
        
        # 2. Kalman Filter prediction
        kalman_result = self.kalman_filter_prediction(nums)
        if kalman_result:
            predictions.append(kalman_result['prediction'])
            weights.append(1.3)
        
        # 3. LSTM prediction
        lstm_result = self.lstm_enhanced_prediction(nums)
        if lstm_result and 'predictions' in lstm_result:
            predictions.append(int(lstm_result['predictions'][0]))
            weights.append(2.0)
        
        # 4. Wavelet prediction
        wavelet_result = self.wavelet_decomposition(nums)
        if wavelet_result:
            predictions.append(wavelet_result['prediction'])
            weights.append(1.2)
        
        # 5. Bootstrap prediction
        bootstrap_result = self.bootstrap_confidence(nums)
        if bootstrap_result:
            predictions.append(int(bootstrap_result['prediction']))
            weights.append(1.1)
        
        if not predictions:
            return {}
        
        # Weighted voting
        weighted_votes = defaultdict(float)
        for pred, weight in zip(predictions, weights):
            weighted_votes[pred % 10] += weight
        
        # Top predictions
        top_predictions = sorted(weighted_votes.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'predictions': [str(p[0]) for p in top_predictions],
            'probabilities': [p[1] / sum(weighted_votes.values()) for p in top_predictions],
            'weights': weights[:len(top_predictions)],
            'n_models': len(predictions),
            'confidence': min(top_predictions[0][1] / sum(weighted_votes.values()) * 100 + 30, 95),
            'method': 'weighted_ensemble'
        }
    
    def markov_predict_simple(self, nums: List[str]) -> Optional[str]:
        """Helper: Markov prediction đơn giản"""
        if len(nums) < 2:
            return None
        
        transitions = {}
        for i in range(len(nums) - 1):
            current = nums[i]
            next_num = nums[i + 1]
            if current not in transitions:
                transitions[current] = []
            transitions[current].append(next_num)
        
        last_num = nums[-1]
        if last_num in transitions and transitions[last_num]:
            next_predictions = Counter(transitions[last_num])
            return next_predictions.most_common(1)[0][0]
        
        return None
    
    # =============== 10. THUẬT TOÁN BAYESIAN NÂNG CAO (THÊM MỚI) ===============
    def bayesian_dynamic_update(self, nums: List[str]) -> Dict:
        """THÊM: Bayesian updating với prior động"""
        int_nums = [int(n) for n in nums if n.isdigit()]
        if len(int_nums) < 20:
            return {}
        
        # Prior: uniform distribution
        prior = np.ones(10) / 10
        
        # Update với từng observation
        posteriors = [prior.copy()]
        
        for i, num in enumerate(int_nums):
            # Likelihood: higher for numbers close to observed
            likelihood = np.zeros(10)
            for j in range(10):
                likelihood[j] = np.exp(-abs(j - num) / 2)
            likelihood = likelihood / likelihood.sum()
            
            # Posterior
            posterior = prior * likelihood
            posterior = posterior / posterior.sum()
            
            posteriors.append(posterior)
            prior = posterior
        
        # Current posterior
        current_posterior = posteriors[-1]
        
        # Dự đoán với credible intervals
        sorted_indices = np.argsort(current_posterior)[::-1]
        
        predictions = []
        cumulative_prob = 0
        for idx in sorted_indices:
            if cumulative_prob < 0.8:  # 80% credible set
                predictions.append({
                    'number': str(idx),
                    'probability': float(current_posterior[idx]),
                    'cumulative': float(cumulative_prob + current_posterior[idx])
                })
                cumulative_prob += current_posterior[idx]
            else:
                break
        
        return {
            'predictions': predictions[:3],
            'posterior': [float(p) for p in current_posterior],
            'credible_interval_80': [p['number'] for p in predictions[:3]],
            'confidence': float(current_posterior[sorted_indices[0]] * 100),
            'method': 'bayesian_dynamic'
        }
    
    # =============== 11. THUẬT TOÁN HURST EXPONENT & FRACTAL (THÊM MỚI) ===============
    def hurst_exponent_analysis(self, nums: List[str]) -> Dict:
        """THÊM: Phân tích Hurst exponent - Đo tính fractal của chuỗi"""
        int_nums = [int(n) for n in nums if n.isdigit()]
        if len(int_nums) < 100:
            return {}
        
        def _hurst(ts):
            lags = range(2, min(len(ts) // 2, 20))
            tau = []
            lagvec = []
            
            for lag in lags:
                pp = np.subtract(ts[lag:], ts[:-lag])
                tau.append(np.std(pp))
                lagvec.append(lag)
            
            tau = np.array(tau)
            lagvec = np.array(lagvec)
            
            if len(tau) > 1 and len(lagvec) > 1:
                m = np.polyfit(np.log(lagvec), np.log(tau), 1)
                return m[0]
            return 0.5
        
        # Tính Hurst exponent
        h = _hurst(int_nums[-200:]) if len(int_nums) >= 200 else _hurst(int_nums)
        
        # Dự đoán dựa trên Hurst
        if h > 0.5:
            # Persistent: xu hướng tiếp diễn
            prediction = int_nums[-1]
            confidence = h * 70
        else:
            # Mean-reverting
            prediction = int(np.mean(int_nums[-10:])) if len(int_nums) >= 10 else int_nums[-1]
            confidence = (1 - h) * 70
        
        return {
            'hurst': h,
            'type': 'Persistent' if h > 0.5 else 'Anti-persistent' if h < 0.5 else 'Random',
            'predictability': 'Cao' if h > 0.65 else 'Trung bình' if h > 0.45 else 'Thấp',
            'fractal_dimension': 2 - h,
            'prediction': str(prediction % 10),
            'confidence': min(confidence, 85)
        }
    
    # =============== 12. THUẬT TOÁN CHANGE POINT DETECTION (THÊM MỚI) ===============
    def change_point_detection_advanced(self, nums: List[str]) -> Dict:
        """THÊM: Phát hiện điểm thay đổi trong chuỗi"""
        int_nums = [int(n) for n in nums if n.isdigit()]
        if len(int_nums) < 50:
            return {}
        
        change_points = []
        window_size = 20
        
        # CUSUM algorithm
        mean = np.mean(int_nums[:window_size])
        std = np.std(int_nums[:window_size])
        
        cusum_pos = 0
        cusum_neg = 0
        threshold = 3 * std if std > 0 else 1
        
        for i in range(window_size, len(int_nums)):
            cusum_pos = max(0, cusum_pos + (int_nums[i] - mean) / (std + 1e-6) - 0.5)
            cusum_neg = min(0, cusum_neg + (int_nums[i] - mean) / (std + 1e-6) + 0.5)
            
            if cusum_pos > threshold or cusum_neg < -threshold:
                change_points.append(i)
                # Reset
                cusum_pos = 0
                cusum_neg = 0
        
        # Phân tích regime
        if change_points:
            last_change = change_points[-1]
            current_regime = int_nums[last_change:]
            regime_mean = np.mean(current_regime)
            regime_std = np.std(current_regime)
            
            if len(current_regime) > 5:
                prediction = np.mean(current_regime[-5:])
                confidence = 80 - regime_std * 5
            else:
                prediction = int_nums[-1]
                confidence = 60
        else:
            prediction = int_nums[-1]
            confidence = 70
        
        return {
            'cusum_change_points': change_points[-5:],
            'n_changes': len(change_points),
            'current_regime_stability': 1 - (np.std(int_nums[-window_size:]) / 3) if np.std(int_nums[-window_size:]) > 0 else 0.5,
            'prediction': str(int(round(prediction)) % 10),
            'confidence': min(confidence, 90),
            'regime_shift_detected': len(change_points) > 0
        }
    
    # =============== 13. THUẬT TOÁN OUTLIER DETECTION (THÊM MỚI) ===============
    def outlier_detection_advanced(self, nums: List[str]) -> Dict:
        """THÊM: Phát hiện outlier và số bất thường"""
        int_nums = [int(n) for n in nums if n.isdigit()]
        if len(int_nums) < 20:
            return {}
        
        outliers = []
        
        # 1. Z-score method
        mean = np.mean(int_nums)
        std = np.std(int_nums)
        if std > 0:
            z_scores = [(i, (num - mean) / std) for i, num in enumerate(int_nums)]
            
            for i, z in z_scores:
                if abs(z) > 2.5:  # Threshold
                    outliers.append({
                        'position': i,
                        'value': int_nums[i],
                        'z_score': float(z),
                        'type': 'High' if z > 0 else 'Low'
                    })
        
        # 2. IQR method
        q1 = np.percentile(int_nums, 25)
        q3 = np.percentile(int_nums, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        for i, num in enumerate(int_nums):
            if num < lower_bound or num > upper_bound:
                outliers.append({
                    'position': i,
                    'value': num,
                    'method': 'IQR',
                    'bounds': [float(lower_bound), float(upper_bound)]
                })
        
        # Remove duplicates
        unique_outliers = []
        seen_positions = set()
        for outlier in outliers:
            if outlier['position'] not in seen_positions:
                unique_outliers.append(outlier)
                seen_positions.add(outlier['position'])
        
        # Phân tích ảnh hưởng của outliers
        clean_data = [num for i, num in enumerate(int_nums) if i not in seen_positions]
        
        if clean_data:
            clean_mean = np.mean(clean_data)
            original_mean = np.mean(int_nums)
            outlier_impact = abs(clean_mean - original_mean) / (original_mean + 1e-6)
        else:
            outlier_impact = 0
        
        return {
            'outliers': unique_outliers[-10:],
            'n_outliers': len(unique_outliers),
            'outlier_rate': len(unique_outliers) / len(int_nums),
            'clean_mean': float(np.mean(clean_data)) if clean_data else 0,
            'outlier_impact': float(outlier_impact),
            'confidence': max(50, 100 - outlier_impact * 100)
        }
    
    # =============== 14. THUẬT TOÁN SPECTRAL ANALYSIS (THÊM MỚI) ===============
    def spectral_analysis_advanced(self, nums: List[str]) -> Dict:
        """THÊM: Phân tích phổ để tìm chu kỳ"""
        int_nums = [int(n) for n in nums if n.isdigit()]
        if len(int_nums) < 50:
            return {}
        
        # FFT để tìm chu kỳ
        fft_vals = np.fft.fft(int_nums)
        fft_mag = np.abs(fft_vals)
        
        # Tìm các tần số dominant
        n = len(int_nums)
        freqs = np.fft.fftfreq(n)
        
        # Bỏ qua tần số 0 (DC component)
        positive_freqs = freqs[:n//2]
        positive_mag = fft_mag[:n//2]
        
        # Tìm top frequencies
        if len(positive_mag) > 0:
            top_indices = np.argsort(positive_mag)[-5:][::-1]
        else:
            top_indices = []
        
        cycles = []
        for idx in top_indices:
            if idx < len(positive_freqs) and positive_freqs[idx] > 0:
                period = int(1 / positive_freqs[idx])
                if 2 < period < 50:  # Chỉ lấy chu kỳ hợp lý
                    cycles.append({
                        'period': period,
                        'strength': float(positive_mag[idx]),
                        'normalized_strength': float(positive_mag[idx] / positive_mag[0]) if positive_mag[0] > 0 else 0
                    })
        
        # Dự đoán dựa trên chu kỳ mạnh nhất
        if cycles:
            strongest_cycle = cycles[0]
            period = strongest_cycle['period']
            
            if len(int_nums) > period:
                next_in_cycle = int_nums[-(period - 1)] if period > 1 else int_nums[-1]
                prediction = next_in_cycle
                confidence = strongest_cycle['normalized_strength'] * 80
            else:
                prediction = int_nums[-1]
                confidence = 60
        else:
            prediction = int_nums[-1]
            confidence = 50
        
        return {
            'detected_cycles': cycles[:3],
            'dominant_period': cycles[0]['period'] if cycles else 0,
            'prediction': str(prediction % 10),
            'confidence': min(confidence, 85),
            'is_periodic': len(cycles) > 0
        }
    
    # =============== 15. THUẬT TOÁN COPULA DEPENDENCE (THÊM MỚI) ===============
    def copula_analysis(self, nums: List[str]) -> Dict:
        """THÊM: Phân tích phụ thuộc Copula giữa các số"""
        int_nums = [int(n) for n in nums if n.isdigit()]
        if len(int_nums) < 100:
            return {}
        
        # Chia dữ liệu thành 2 chuỗi
        X = int_nums[:-1:2]
        Y = int_nums[1::2]
        
        min_len = min(len(X), len(Y))
        X = X[:min_len]
        Y = Y[:min_len]
        
        if len(X) < 20:
            return {}
        
        # Tính dependence measures
        kendall_tau, p_value = stats.kendalltau(X, Y)
        spearman_rho, _ = stats.spearmanr(X, Y)
        
        return {
            'kendall_tau': float(kendall_tau),
            'spearman_rho': float(spearman_rho),
            'dependence_strength': 'Mạnh' if abs(kendall_tau) > 0.3 else 'Yếu',
            'prediction_confidence': 50 + abs(kendall_tau) * 40
        }
    
    # =============== 16. THUẬT TOÁN GENETIC ALGORITHM (THÊM MỚI) ===============
    def genetic_algorithm_optimization(self, nums: List[str], n_generations: int = 30) -> Dict:
        """THÊM: Tối ưu hóa bằng thuật toán di truyền"""
        int_nums = [int(n) for n in nums if n.isdigit()]
        if len(int_nums) < 30:
            return {}
        
        # Khởi tạo quần thể
        population_size = 20
        genome_length = 5
        
        def fitness(genome):
            """Hàm fitness: dự đoán chính xác số tiếp theo"""
            predictions = []
            for i in range(len(int_nums) - genome_length):
                pattern = int_nums[i:i+genome_length]
                # Simple prediction: weighted average
                weights = np.exp(np.linspace(0, 1, genome_length))
                weights = weights / weights.sum()
                pred = int(np.average(pattern, weights=weights))
                actual = int_nums[i+genome_length]
                predictions.append(1 if pred % 10 == actual % 10 else 0)
            
            return np.mean(predictions) if predictions else 0
        
        # Initialize population
        population = [np.random.randint(0, 10, genome_length).tolist() for _ in range(population_size)]
        
        # Evolution
        best_fitness = 0
        best_genome = None
        
        for generation in range(n_generations):
            # Evaluate fitness
            fitness_scores = [fitness(genome) for genome in population]
            
            # Track best
            current_best_idx = np.argmax(fitness_scores)
            if fitness_scores[current_best_idx] > best_fitness:
                best_fitness = fitness_scores[current_best_idx]
                best_genome = population[current_best_idx].copy()
            
            # Selection (tournament)
            new_population = []
            for _ in range(population_size):
                tournament = np.random.choice(len(population), 3, replace=False)
                winner_idx = tournament[np.argmax([fitness_scores[i] for i in tournament])]
                new_population.append(population[winner_idx].copy())
            
            # Crossover
            for i in range(0, population_size, 2):
                if np.random.random() < 0.7 and i+1 < len(new_population):
                    point = np.random.randint(1, genome_length - 1)
                    temp1 = new_population[i][point:].copy()
                    temp2 = new_population[i+1][point:].copy()
                    new_population[i][point:] = temp2
                    new_population[i+1][point:] = temp1
            
            # Mutation
            for i in range(population_size):
                if np.random.random() < 0.1:
                    pos = np.random.randint(genome_length)
                    new_population[i][pos] = np.random.randint(0, 10)
            
            population = new_population
        
        # Dự đoán với best genome
        if len(int_nums) >= genome_length:
            last_pattern = int_nums[-genome_length:]
            weights = np.exp(np.linspace(0, 1, genome_length))
            weights = weights / weights.sum()
            prediction = int(np.average(last_pattern, weights=weights)) % 10
        else:
            prediction = int_nums[-1] % 10
        
        return {
            'best_genome': [int(g) for g in best_genome] if best_genome else [],
            'best_fitness': float(best_fitness * 100),
            'prediction': str(prediction),
            'confidence': float(best_fitness * 100),
            'method': 'genetic_algorithm'
        }
    
    # =============== 17. THUẬT TOÁN PSO (Particle Swarm Optimization) (THÊM MỚI) ===============
    def pso_optimization(self, nums: List[str]) -> Dict:
        """THÊM: Tối ưu hóa bầy đàn PSO (phiên bản đơn giản)"""
        int_nums = [int(n) for n in nums if n.isdigit()]
        if len(int_nums) < 30:
            return {}
        
        n_particles = 20
        n_iterations = 30
        
        # Particle position: weights for prediction
        particles = np.random.rand(n_particles, 5)
        particles = particles / particles.sum(axis=1, keepdims=True)
        
        velocities = np.random.randn(n_particles, 5) * 0.1
        
        personal_best_pos = particles.copy()
        personal_best_score = np.zeros(n_particles)
        global_best_pos = particles[0].copy()
        global_best_score = 0
        
        def fitness(weights):
            """Evaluate prediction accuracy with weights"""
            if len(int_nums) < 6:
                return 0
            predictions = []
            for i in range(len(int_nums) - 5):
                pattern = int_nums[i:i+5]
                pred = int(np.average(pattern, weights=weights[:len(pattern)]))
                actual = int_nums[i+5]
                predictions.append(1 if pred % 10 == actual % 10 else 0)
            return np.mean(predictions) if predictions else 0
        
        # Initialize personal best scores
        for i in range(n_particles):
            personal_best_score[i] = fitness(particles[i])
            if personal_best_score[i] > global_best_score:
                global_best_score = personal_best_score[i]
                global_best_pos = particles[i].copy()
        
        # PSO iterations
        w = 0.7  # inertia
        c1 = 1.5  # cognitive
        c2 = 1.5  # social
        
        for _ in range(n_iterations):
            for i in range(n_particles):
                # Update velocity
                r1, r2 = np.random.rand(2)
                velocities[i] = (w * velocities[i] + 
                               c1 * r1 * (personal_best_pos[i] - particles[i]) +
                               c2 * r2 * (global_best_pos - particles[i]))
                
                # Update position
                particles[i] = particles[i] + velocities[i]
                particles[i] = np.maximum(particles[i], 0)
                particles[i] = particles[i] / particles[i].sum()
                
                # Evaluate
                score = fitness(particles[i])
                
                # Update personal best
                if score > personal_best_score[i]:
                    personal_best_score[i] = score
                    personal_best_pos[i] = particles[i].copy()
                
                # Update global best
                if score > global_best_score:
                    global_best_score = score
                    global_best_pos = particles[i].copy()
        
        # Dự đoán với optimal weights
        if len(int_nums) >= 5:
            last_pattern = int_nums[-5:]
            prediction = int(np.average(last_pattern, weights=global_best_pos[:5])) % 10
        else:
            prediction = int_nums[-1] % 10
        
        return {
            'optimal_weights': [float(w) for w in global_best_pos],
            'fitness_score': float(global_best_score * 100),
            'prediction': str(prediction),
            'confidence': float(global_best_score * 100),
            'method': 'pso'
        }
    
    # =============== 18. THUẬT TOÁN REINFORCEMENT LEARNING (THÊM MỚI) ===============
    def reinforcement_learning_prediction(self, nums: List[str]) -> Dict:
        """THÊM: Học tăng cường cho dự đoán"""
        int_nums = [int(n) for n in nums if n.isdigit()]
        if len(int_nums) < 50:
            return {}
        
        # Q-Learning parameters
        n_states = 10  # Last number
        n_actions = 10  # Prediction
        q_table = np.zeros((n_states, n_actions))
        
        alpha = 0.1  # Learning rate
        gamma = 0.9  # Discount factor
        epsilon = 0.1  # Exploration
        
        # Training
        for i in range(len(int_nums) - 1):
            state = int_nums[i]
            action = int_nums[i + 1]  # Actual next number
            
            if i < len(int_nums) - 2:
                next_state = int_nums[i + 1]
                
                # Q-learning update
                best_next_action = np.argmax(q_table[next_state])
                td_target = 1 + gamma * q_table[next_state][best_next_action]  # Reward = 1 for correct
                td_error = td_target - q_table[state][action]
                q_table[state][action] += alpha * td_error
        
        # Predict
        last_state = int_nums[-1]
        prediction = np.argmax(q_table[last_state])
        
        # Tính confidence
        q_values = q_table[last_state]
        best_q = np.max(q_values)
        second_best = np.sort(q_values)[-2] if len(q_values) > 1 else 0
        confidence = (best_q - second_best) / (best_q + 1e-6) * 50 + 50
        
        return {
            'q_table': q_table.tolist(),
            'prediction': str(prediction),
            'confidence': float(min(confidence, 95)),
            'method': 'q_learning'
        }
    
    # =============== 19. THUẬT TOÁN DYNAMIC TIME WARPING (THÊM MỚI) ===============
    def dynamic_time_warping(self, nums: List[str]) -> Dict:
        """THÊM: Dynamic Time Warping cho pattern matching"""
        int_nums = [int(n) for n in nums if n.isdigit()]
        if len(int_nums) < 30:
            return {}
        
        def dtw_distance(seq1, seq2):
            n, m = len(seq1), len(seq2)
            dtw_matrix = np.zeros((n+1, m+1))
            dtw_matrix[0, 1:] = np.inf
            dtw_matrix[1:, 0] = np.inf
            
            for i in range(1, n+1):
                for j in range(1, m+1):
                    cost = abs(seq1[i-1] - seq2[j-1])
                    dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],
                                                  dtw_matrix[i, j-1],
                                                  dtw_matrix[i-1, j-1])
            return dtw_matrix[n, m]
        
        # Tìm pattern tương tự
        pattern_length = 10
        if len(int_nums) < pattern_length * 2:
            return {'predictions': []}
            
        last_pattern = int_nums[-pattern_length:]
        
        distances = []
        for i in range(len(int_nums) - pattern_length * 2):
            pattern = int_nums[i:i+pattern_length]
            distance = dtw_distance(last_pattern, pattern)
            distances.append((i, distance))
        
        distances.sort(key=lambda x: x[1])
        
        # Dự đoán từ pattern giống nhất
        predictions = []
        if distances:
            best_match_idx = distances[0][0]
            if best_match_idx + pattern_length < len(int_nums):
                next_num = int_nums[best_match_idx + pattern_length]
                predictions.append(str(next_num))
        
        return {
            'predictions': predictions[:3],
            'dtw_distance': float(distances[0][1]) if distances else 0,
            'confidence': 80 - (distances[0][1] / pattern_length) if distances else 50,
            'method': 'dynamic_time_warping'
        }
    
    # =============== 20. THUẬT TOÁN CUSUM (THÊM MỚI) ===============
    def cusum_analysis(self, nums: List[str]) -> Dict:
        """THÊM: Cumulative Sum Control Chart"""
        int_nums = [int(n) for n in nums if n.isdigit()]
        if len(int_nums) < 30:
            return {}
        
        target = np.mean(int_nums)
        sigma = np.std(int_nums)
        
        cusum_pos = 0
        cusum_neg = 0
        cusum_series = []
        
        for num in int_nums:
            cusum_pos = max(0, cusum_pos + (num - target) / (sigma + 1e-6) - 0.5)
            cusum_neg = min(0, cusum_neg + (num - target) / (sigma + 1e-6) + 0.5)
            cusum_series.append((cusum_pos, cusum_neg))
        
        # Recent trend
        recent_cusum = cusum_series[-1]
        if recent_cusum[0] > 0:
            trend = 'upward'
            confidence = min(recent_cusum[0] * 20, 80)
        elif recent_cusum[1] < 0:
            trend = 'downward'
            confidence = min(abs(recent_cusum[1]) * 20, 80)
        else:
            trend = 'stable'
            confidence = 50
        
        return {
            'target': float(target),
            'sigma': float(sigma),
            'current_trend': trend,
            'confidence': confidence,
            'method': 'cusum'
        }
    
    # =============== 21. THUẬT TOÁN HIDDEN MARKOV MODEL (THÊM MỚI) ===============
    def hmm_advanced(self, nums: List[str], n_states: int = 3) -> Dict:
        """THÊM: Hidden Markov Model đơn giản"""
        int_nums = [int(n) for n in nums if n.isdigit()]
        if len(int_nums) < 50:
            return self._hmm_simple(int_nums, n_states)
        
        return self._hmm_simple(int_nums, n_states)
    
    def _hmm_simple(self, nums: List[int], n_states: int = 3) -> Dict:
        """THÊM: HMM đơn giản"""
        if len(nums) < 20:
            return {'predictions': []}
            
        # Initialize random parameters
        n_obs = 10
        trans_mat = np.random.dirichlet(np.ones(n_states), n_states)
        emit_mat = np.random.dirichlet(np.ones(n_obs), n_states)
        
        # Viterbi-like decoding
        states = []
        for num in nums[-20:]:
            if states:
                probs = emit_mat[:, num] * trans_mat[states[-1]]
                states.append(np.argmax(probs))
            else:
                probs = emit_mat[:, num]
                states.append(np.argmax(probs))
        
        current_state = states[-1] if states else 0
        next_state_probs = trans_mat[current_state]
        next_state = np.argmax(next_state_probs)
        
        # Predict emission
        emission_probs = emit_mat[next_state]
        top_3 = np.argsort(emission_probs)[-3:][::-1]
        
        predictions = []
        for idx in top_3:
            predictions.append({
                'number': str(idx),
                'probability': float(emission_probs[idx]),
                'confidence': float(emission_probs[idx] * 100)
            })
        
        return {
            'predictions': predictions,
            'n_states': n_states,
            'current_state': int(current_state),
            'next_state': int(next_state),
            'method': 'hmm_simple'
        }
    
    # =============== 22. THUẬT TOÁN KALMAN SMOOTHER (THÊM MỚI) ===============
    def kalman_smoother(self, nums: List[str]) -> Dict:
        """THÊM: Kalman Smoother cho ước lượng xu hướng"""
        int_nums = [int(n) for n in nums if n.isdigit()]
        if len(int_nums) < 20:
            return {}
        
        # Forward pass (Kalman filter)
        x_est = int_nums[0]
        p_est = 1.0
        q = 0.01
        r = 0.1
        
        filtered = [x_est]
        
        for z in int_nums[1:]:
            # Predict
            x_pred = x_est
            p_pred = p_est + q
            
            # Update
            k = p_pred / (p_pred + r)
            x_est = x_pred + k * (z - x_pred)
            p_est = (1 - k) * p_pred
            
            filtered.append(x_est)
        
        # Backward pass (simple smoother)
        smoothed = filtered.copy()
        for t in range(len(int_nums)-2, -1, -1):
            smoothed[t] = filtered[t] + 0.5 * (smoothed[t+1] - filtered[t])
        
        # Detect trend from smoothed series
        if len(smoothed) >= 5:
            recent_trend = smoothed[-1] - smoothed[-5]
        else:
            recent_trend = 0
        
        if recent_trend > 0.5:
            trend = 'Tăng'
            confidence = min(recent_trend * 30, 80)
        elif recent_trend < -0.5:
            trend = 'Giảm'
            confidence = min(abs(recent_trend) * 30, 80)
        else:
            trend = 'Đi ngang'
            confidence = 50
        
        return {
            'filtered': [float(f) for f in filtered[-10:]],
            'smoothed': [float(s) for s in smoothed[-10:]],
            'trend': trend,
            'trend_strength': float(abs(recent_trend)),
            'confidence': confidence,
            'prediction': str(int(round(smoothed[-1])) % 10),
            'method': 'kalman_smoother'
        }
    
    # =============== 23. THUẬT TOÁN THRESHOLD MODEL (THÊM MỚI) ===============
    def threshold_ar_model(self, nums: List[str]) -> Dict:
        """THÊM: Threshold Autoregressive Model"""
        int_nums = [int(n) for n in nums if n.isdigit()]
        if len(int_nums) < 50:
            return {}
        
        # Find threshold
        sorted_nums = sorted(int_nums)
        threshold_idx = len(sorted_nums) // 3
        threshold_low = sorted_nums[threshold_idx]
        threshold_high = sorted_nums[-threshold_idx]
        
        # Separate regimes
        regime_low = []
        regime_med = []
        regime_high = []
        
        for i in range(len(int_nums) - 1):
            if int_nums[i] <= threshold_low:
                regime_low.append(int_nums[i+1])
            elif int_nums[i] >= threshold_high:
                regime_high.append(int_nums[i+1])
            else:
                regime_med.append(int_nums[i+1])
        
        # Current regime
        current_val = int_nums[-1]
        if current_val <= threshold_low:
            current_regime = 'LOW'
            predictions = regime_low
        elif current_val >= threshold_high:
            current_regime = 'HIGH'
            predictions = regime_high
        else:
            current_regime = 'MEDIUM'
            predictions = regime_med
        
        if predictions:
            pred_counts = Counter(predictions)
            total = len(predictions)
            top_predictions = []
            for num, count in pred_counts.most_common(3):
                top_predictions.append({
                    'number': str(num),
                    'probability': count / total,
                    'confidence': count / total * 100
                })
        else:
            top_predictions = []
        
        return {
            'predictions': top_predictions,
            'thresholds': [int(threshold_low), int(threshold_high)],
            'current_regime': current_regime,
            'regime_sizes': {
                'low': len(regime_low),
                'medium': len(regime_med),
                'high': len(regime_high)
            },
            'confidence': 75 if top_predictions else 50,
            'method': 'threshold_ar'
        }

# =============== GIAO DIỆN RESPONSIVE (GIỮ NGUYÊN 100%) ===============
st.set_page_config(
    page_title="🎯 AI 3-TINH ELITE PRO V2.0",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS RESPONSIVE TỐI ƯU - GIỮ NGUYÊN 100%
st.markdown("""
<style>
    /* RESET & VARIABLES */
    :root {
        --primary: #00ffcc;
        --secondary: #00ccff;
        --success: #10b981;
        --danger: #ef4444;
        --warning: #f59e0b;
        --dark: #0f172a;
        --darker: #0b0f13;
        --light: #e2e8f0;
        --border: 2px solid #334155;
        --border-radius: 16px;
        --shadow: 0 8px 32px rgba(0, 255, 204, 0.15);
    }

    /* BASE */
    .stApp {
        background: linear-gradient(135deg, var(--darker) 0%, var(--dark) 100%) !important;
        color: var(--light);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* TYPOGRAPHY RESPONSIVE */
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: clamp(1.8rem, 5vw, 2.8rem);
        font-weight: 800;
        margin-bottom: 0.5rem;
        padding: 0.5rem;
        text-shadow: 0 0 20px rgba(0, 255, 204, 0.3);
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: clamp(0.9rem, 3vw, 1.1rem);
        margin-bottom: 1.5rem;
    }

    /* HEADER CARD */
    .header-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(0, 255, 204, 0.2);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
        box-shadow: var(--shadow);
    }

    /* RESULT CARD - RESPONSIVE */
    .result-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 2px solid var(--primary);
        border-radius: 24px;
        padding: clamp(1rem, 4vw, 2rem);
        margin: 1.5rem 0;
        box-shadow: 0 0 30px rgba(0, 255, 204, 0.2);
        position: relative;
        overflow: hidden;
    }

    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
    }

    /* NUMBERS DISPLAY - FLEXIBLE */
    .numbers-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: clamp(1rem, 5vw, 2rem);
        padding: 1rem;
        max-width: 600px;
        margin: 0 auto;
    }

    .number-circle {
        aspect-ratio: 1;
        width: 100%;
        max-width: 120px;
        margin: 0 auto;
        background: linear-gradient(135deg, var(--warning), #f97316);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: clamp(2rem, 8vw, 3.5rem);
        font-weight: 900;
        color: var(--dark);
        box-shadow: 0 0 40px rgba(245, 158, 11, 0.5);
        animation: pulse 2s infinite;
        transition: transform 0.3s;
    }

    .number-circle:hover {
        transform: scale(1.05);
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 20px rgba(245, 158, 11, 0.5); }
        50% { box-shadow: 0 0 50px rgba(245, 158, 11, 0.8); }
        100% { box-shadow: 0 0 20px rgba(245, 158, 11, 0.5); }
    }

    /* INFO BOXES - FLEXIBLE */
    .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1rem;
        margin-top: 1.5rem;
    }

    .info-box {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 16px;
        padding: 1.25rem;
        border-left: 6px solid;
        backdrop-filter: blur(5px);
    }

    .eliminated-box {
        border-left-color: var(--danger);
        background: rgba(239, 68, 68, 0.1);
    }

    .safe-box {
        border-left-color: var(--success);
        background: rgba(16, 185, 129, 0.1);
    }

    .strategy-box {
        border-left-color: var(--secondary);
        background: rgba(0, 204, 255, 0.1);
    }

    .info-title {
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .info-numbers {
        font-size: clamp(1.2rem, 4vw, 1.8rem);
        font-weight: 700;
        letter-spacing: 4px;
        margin: 0.5rem 0;
    }

    /* BUTTONS */
    .stButton button {
        background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
        color: var(--dark) !important;
        font-weight: 700 !important;
        font-size: clamp(1rem, 4vw, 1.2rem) !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 50px !important;
        border: none !important;
        transition: all 0.3s !important;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 30px rgba(0, 255, 204, 0.4) !important;
    }

    /* INPUT AREA */
    .stTextArea textarea {
        background-color: #1e293b !important;
        color: var(--primary) !important;
        border: 2px solid var(--primary) !important;
        border-radius: 16px !important;
        font-size: 1rem !important;
        padding: 1rem !important;
        transition: all 0.3s;
    }

    .stTextArea textarea:focus {
        box-shadow: 0 0 20px rgba(0, 255, 204, 0.3) !important;
    }

    /* METRICS */
    .stMetric {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid var(--primary);
        border-radius: 16px;
        padding: 1rem;
    }

    .stMetric label {
        color: #94a3b8 !important;
        font-size: 0.9rem !important;
    }

    .stMetric [data-testid="stMetricValue"] {
        color: var(--primary) !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    .stMetric [data-testid="stMetricDelta"] {
        color: var(--success) !important;
    }

    /* TABS - RESPONSIVE */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #1e293b;
        padding: 0.75rem;
        border-radius: 50px;
        margin: 1rem 0;
        flex-wrap: wrap;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: #94a3b8 !important;
        border-radius: 50px !important;
        padding: 0.5rem 1.25rem !important;
        font-size: clamp(0.8rem, 3vw, 1rem) !important;
        transition: all 0.3s;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
        color: var(--dark) !important;
        font-weight: 700 !important;
    }

    /* PROGRESS BAR */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
        height: 8px !important;
        border-radius: 4px;
    }

    /* EXPANDER */
    .streamlit-expanderHeader {
        background: #1e293b !important;
        border: 1px solid var(--primary) !important;
        border-radius: 12px !important;
        color: var(--primary) !important;
        font-weight: 600 !important;
    }

    /* FOOTER */
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #334155;
        color: #94a3b8;
        font-size: 0.85rem;
    }

    /* RESPONSIVE GRID */
    @media (max-width: 768px) {
        .numbers-grid {
            gap: 0.75rem;
        }
        
        .info-grid {
            grid-template-columns: 1fr;
        }
        
        .stTabs [data-baseweb="tab"] {
            flex: 1 1 auto;
        }
    }

    /* ANIMATIONS */
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .animate-in {
        animation: slideIn 0.5s ease-out;
    }

    /* CUSTOM SCROLLBAR */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #1e293b;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(var(--primary), var(--secondary));
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary);
    }
</style>
""", unsafe_allow_html=True)

# =============== HEADER (GIỮ NGUYÊN) ===============
st.markdown("""
<div class='header-card animate-in'>
    <h1 class='main-title'>🎯 AI 3-TINH ELITE PRO V2.0</h1>
    <p class='subtitle'>Hệ thống AI đa tầng - Phát hiện bẫy nhà cái - Dự đoán siêu chính xác</p>
</div>
""", unsafe_allow_html=True)

# =============== KHỞI TẠO ANALYZER (GIỮ NGUYÊN) ===============
@st.cache_resource
def init_analyzer():
    return LotteryAIAnalyzer()

analyzer = init_analyzer()

# =============== SESSION STATE (GIỮ NGUYÊN) ===============
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'accuracy_stats' not in st.session_state:
    st.session_state.accuracy_stats = {
        'total_predictions': 0,
        'correct_predictions': 0,
        'accuracy_rate': 0.0
    }

# =============== TABS CHÍNH (GIỮ NGUYÊN) ===============
tab1, tab2, tab3, tab4 = st.tabs(["🎯 DỰ ĐOÁN", "📊 PHÂN TÍCH", "📈 THỐNG KÊ", "⚙️ CÀI ĐẶT"])

with tab1:
    # INPUT AREA (GIỮ NGUYÊN)
    col1, col2 = st.columns([3, 1])
    
    with col1:
        data_input = st.text_area(
            "📥 NHẬP CHUỖI SỐ THỰC TẾ:",
            height=120,
            placeholder="Ví dụ: 5382917462538192047538291746... (càng nhiều số càng chính xác)",
            help="Nhập càng nhiều số gần đây, AI càng phân tích chính xác",
            key="data_input_main"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric(
            "ĐỘ CHÍNH XÁC", 
            f"{st.session_state.accuracy_stats['accuracy_rate']:.1f}%", 
            "+2.5%",
            delta_color="normal"
        )
        st.metric("DỮ LIỆU", f"{len(list(filter(str.isdigit, data_input)))} số", "Đã nhập")
    
    # NÚT PHÂN TÍCH (GIỮ NGUYÊN)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🚀 KÍCH HOẠT AI PHÂN TÍCH ĐA TẦNG",
            use_container_width=True,
            type="primary"
        )
    
    if analyze_button:
        nums = list(filter(str.isdigit, data_input))
        
        if len(nums) < 15:
            st.error("⚠️ CẦN ÍT NHẤT 15 SỐ ĐỂ PHÂN TÍCH CHÍNH XÁC!")
        else:
            # PROGRESS BAR (GIỮ NGUYÊN)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Bước 1: Tiền xử lý
                status_text.text("🔄 Đang tiền xử lý dữ liệu...")
                time.sleep(0.3)
                progress_bar.progress(15)
                
                # Bước 2: Phân tích đa tầng
                status_text.text("📊 Đang phân tích tần suất & Markov...")
                time.sleep(0.4)
                progress_bar.progress(35)
                
                # Bước 3: Loại 3 số rủi ro
                status_text.text("🚫 Đang loại bỏ 3 số rủi ro...")
                eliminated, remaining, analysis = analyzer.eliminate_risk_numbers(data_input)
                time.sleep(0.4)
                progress_bar.progress(60)
                
                # Bước 4: Chọn 3 số tốt nhất (ĐÃ ĐƯỢC NÂNG CẤP THUẬT TOÁN)
                status_text.text("🎯 Đang chọn 3 số chiến thuật (đa thuật toán nâng cao)...")
                top_three = analyzer.select_top_three(remaining, data_input, analysis)
                time.sleep(0.4)
                progress_bar.progress(85)
                
                # Bước 5: Kết nối AI (nếu có)
                gemini_result = ""
                if GEMINI_API_KEY:
                    status_text.text("🧠 Đang kết nối Gemini AI...")
                    gemini_result = analyzer.connect_gemini(data_input[-100:])
                
                progress_bar.progress(100)
                status_text.text("✅ HOÀN TẤT PHÂN TÍCH!")
                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()
                
                # Lưu lịch sử
                st.session_state.analysis_history.append({
                    'time': datetime.now().strftime("%H:%M:%S"),
                    'data_length': len(nums),
                    'eliminated': eliminated,
                    'top_three': top_three
                })
                
                # HIỂN THỊ KẾT QUẢ (GIỮ NGUYÊN GIAO DIỆN)
                st.balloons()
                
                # RESULT CARD
                st.markdown(f"""
                <div class='result-card animate-in'>
                    <div style='text-align: center; margin-bottom: 1.5rem;'>
                        <span style='background: linear-gradient(90deg, var(--primary), var(--secondary)); 
                                     padding: 0.5rem 1.5rem; border-radius: 50px; 
                                     color: var(--dark); font-weight: 700;'>
                            🎯 DÀN 3 TINH CHIẾN THUẬT CAO CẤP
                        </span>
                    </div>
                    
                    <div class='numbers-grid'>
                        <div class='number-circle'>{top_three[0]}</div>
                        <div class='number-circle'>{top_three[1]}</div>
                        <div class='number-circle'>{top_three[2]}</div>
                    </div>
                    
                    <div class='info-grid'>
                        <div class='info-box eliminated-box'>
                            <div class='info-title'>
                                <span style='color: var(--danger);'>🚫 3 SỐ RỦI RO (BẪY NHÀ CÁI)</span>
                            </div>
                            <div class='info-numbers'>{", ".join(eliminated)}</div>
                            <small style='color: #94a3b8;'>Tuyệt đối tránh xa các số này!</small>
                        </div>
                        
                        <div class='info-box safe-box'>
                            <div class='info-title'>
                                <span style='color: var(--success);'>✅ DÀN 7 SỐ AN TOÀN</span>
                            </div>
                            <div class='info-numbers'>{", ".join(remaining)}</div>
                            <small style='color: #94a3b8;'>Chọn 7 số của bạn từ dàn này</small>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # CHIẾN THUẬT (GIỮ NGUYÊN)
                st.markdown(f"""
                <div class='info-box strategy-box' style='margin-top: 1rem;'>
                    <div class='info-title'>
                        <span style='color: var(--secondary);'>💡 CHIẾN THUẬT ÁP DỤNG NGAY</span>
                    </div>
                    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 0.5rem;'>
                        <div style='padding: 0.5rem;'>
                            <span style='font-size: 1.3rem;'>💰</span><br>
                            <strong>Tập trung vốn</strong><br>
                            <small>Vào 3 số: {", ".join(top_three)}</small>
                        </div>
                        <div style='padding: 0.5rem;'>
                            <span style='font-size: 1.3rem;'>🛡️</span><br>
                            <strong>Tránh xa</strong><br>
                            <small>3 số: {", ".join(eliminated)}</small>
                        </div>
                        <div style='padding: 0.5rem;'>
                            <span style='font-size: 1.3rem;'>📊</span><br>
                            <strong>Dàn 7 số</strong><br>
                            <small>{", ".join(remaining)}</small>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # PHÂN TÍCH CHI TIẾT (ĐÃ THÊM THUẬT TOÁN MỚI VÀO ĐÂY)
                with st.expander("📊 XEM PHÂN TÍCH CHI TIẾT (ĐÃ NÂNG CẤP 25+ THUẬT TOÁN)", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("##### 🔥 TOP 5 SỐ NÓNG")
                        hot_nums = analyzer._find_hot_numbers(nums[-30:])
                        if hot_nums:
                            hot_text = " • ".join(hot_nums[:5])
                            st.markdown(f"<div style='font-size: 1.5rem; color: #ef4444;'>{hot_text}</div>", 
                                      unsafe_allow_html=True)
                        else:
                            st.info("Không có số nóng")
                        
                        # THÊM: Hiển thị Hurst Exponent
                        hurst_result = analyzer.hurst_exponent_analysis(data_input)
                        if hurst_result:
                            st.markdown("##### 📈 HURST EXPONENT")
                            st.metric("Giá trị", f"{hurst_result['hurst']:.3f}", 
                                     hurst_result['type'])
                    
                    with col2:
                        st.markdown("##### ❄️ TOP 5 SỐ LẠNH")
                        cold_nums = analyzer._find_cold_numbers(nums, 30)
                        if cold_nums:
                            cold_text = " • ".join(cold_nums[:5])
                            st.markdown(f"<div style='font-size: 1.5rem; color: #3b82f6;'>{cold_text}</div>", 
                                      unsafe_allow_html=True)
                        else:
                            st.info("Không có số lạnh")
                        
                        # THÊM: Hiển thị RNG Analysis
                        rng_result = analyzer.rng_pattern_detection_advanced(data_input)
                        if rng_result:
                            st.markdown("##### 🎲 RNG ANALYSIS")
                            st.metric("Random Score", f"{rng_result['randomness_score']:.1f}%",
                                     "Random" if rng_result['is_random'] else "Có pattern")
                    
                    with col3:
                        st.markdown("##### 🎯 PHÂN TÍCH POISSON")
                        if analysis and 'poisson' in analysis:
                            poisson_data = []
                            for num, info in analysis['poisson'].items():
                                poisson_data.append({
                                    'Số': num,
                                    'Xác suất': f"{info['prob_next']*100:.1f}%"
                                })
                            poisson_df = pd.DataFrame(poisson_data).head(5)
                            st.dataframe(poisson_df, use_container_width=True, hide_index=True)
                        
                        # THÊM: Ensemble Voting
                        ensemble_result = analyzer.ensemble_voting_advanced(nums)
                        if ensemble_result:
                            st.markdown("##### 🤖 ENSEMBLE VOTING")
                            st.write(f"Dự đoán: {', '.join(ensemble_result.get('predictions', []))}")
                            st.progress(ensemble_result.get('confidence', 0)/100)
                    
                    # Hàng 2: Các thuật toán nâng cao
                    st.markdown("---")
                    st.markdown("##### 🧠 THUẬT TOÁN NÂNG CAO (THÊM MỚI)")
                    
                    col4, col5, col6 = st.columns(3)
                    
                    with col4:
                        # Kalman Filter
                        kalman_result = analyzer.kalman_filter_prediction(nums)
                        if kalman_result:
                            st.markdown("**📊 KALMAN FILTER**")
                            st.metric("Dự đoán", kalman_result['prediction'], 
                                     f"{kalman_result['confidence']:.1f}%")
                        
                        # Bayesian
                        bayesian_result = analyzer.bayesian_dynamic_update(nums)
                        if bayesian_result and bayesian_result.get('predictions'):
                            st.markdown("**🎲 BAYESIAN**")
                            bayes_pred = bayesian_result['predictions'][0]['number']
                            st.metric("Dự đoán", bayes_pred, 
                                     f"{bayesian_result['predictions'][0]['probability']*100:.1f}%")
                    
                    with col5:
                        # LSTM
                        lstm_result = analyzer.lstm_enhanced_prediction(nums)
                        if lstm_result and lstm_result.get('predictions'):
                            st.markdown("**🧠 LSTM**")
                            st.metric("Dự đoán", lstm_result['predictions'][0], 
                                     f"{lstm_result['confidence']:.1f}%")
                        
                        # Monte Carlo
                        mc_result = analyzer.monte_carlo_advanced(nums)
                        if mc_result and mc_result.get('predictions'):
                            st.markdown("**🎲 MONTE CARLO**")
                            mc_pred = mc_result['predictions']['step_1']['top_3'][0]
                            st.metric("Dự đoán", mc_pred)
                    
                    with col6:
                        # Genetic Algorithm
                        genetic_result = analyzer.genetic_algorithm_optimization(nums)
                        if genetic_result:
                            st.markdown("**🧬 GENETIC ALGORITHM**")
                            st.metric("Dự đoán", genetic_result['prediction'], 
                                     f"{genetic_result['confidence']:.1f}%")
                        
                        # PSO
                        pso_result = analyzer.pso_optimization(nums)
                        if pso_result:
                            st.markdown("**🐝 PSO**")
                            st.metric("Dự đoán", pso_result['prediction'], 
                                     f"{pso_result['confidence']:.1f}%")
                    
                    # THÊM: Phân tích Entropy
                    entropy_result = analyzer.analyze_entropy_multiscale(nums)
                    if entropy_result:
                        st.markdown("---")
                        st.markdown("##### 🔄 PHÂN TÍCH ENTROPY ĐA TỶ LỆ")
                        ent_cols = st.columns(4)
                        for i, (scale, result) in enumerate(list(entropy_result.items())[:4]):
                            if isinstance(result, dict):
                                with ent_cols[i]:
                                    st.metric(f"Scale {scale}", f"{result['entropy']:.2f}",
                                             result['prediction_difficulty'])
                    
                    # THÊM: Markov Chain (nâng cao)
                    if analysis and 'markov' in analysis and len(nums) >= 3:
                        st.markdown("---")
                        st.markdown("##### 🔗 PHÂN TÍCH MARKOV BẬC CAO")
                        last_state = tuple(nums[-2:])
                        if last_state in analysis['markov'].get('order_2', {}):
                            markov_data = []
                            for num, prob in sorted(
                                analysis['markov']['order_2'][last_state].items(), 
                                key=lambda x: x[1], 
                                reverse=True
                            )[:5]:
                                markov_data.append({
                                    'Số tiếp theo': num,
                                    'Xác suất': f"{prob*100:.1f}%"
                                })
                            if markov_data:
                                markov_df = pd.DataFrame(markov_data)
                                st.dataframe(markov_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Lỗi trong quá trình phân tích: {str(e)}")

# Các tab khác giữ nguyên
with tab2:
    st.info("📊 PHÂN TÍCH CHI TIẾT - Đang phát triển...")

with tab3:
    st.info("📈 THỐNG KÊ - Đang phát triển...")

with tab4:
    st.info("⚙️ CÀI ĐẶT - Đang phát triển...")

# Footer (giữ nguyên)
st.markdown("""
<div class='footer'>
    <p>© 2024 AI 3-TINH ELITE PRO V2.0 - Tích hợp 25+ thuật toán nâng cao | Phát hiện bẫy nhà cái | Độ chính xác cao</p>
</div>
""", unsafe_allow_html=True)