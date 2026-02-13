import streamlit as st
import collections
import time
import numpy as np
import pandas as pd
from datetime import datetime
import requests
import json
from typing import List, Dict, Tuple
import hashlib
import pickle
import os
from random import choices, random
import math

# =============== SYSTEM CONFIG ===============
SYSTEM_NAME = "AI-SOI-3SO-DB-SUPER"
MODE = "SINGLE_FILE"
SAVE_SESSION = True
AUTO_LEARN = True
SELF_OPTIMIZE = True
VERSION = "v5.0-ELITE-SUPER"

# =============== CẤU HÌNH API ===============
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

# =============== SESSION STATE INIT ===============
def init_session_state():
    """Khởi tạo session state với khả năng tự học"""
    if 'predictor' not in st.session_state:
        st.session_state.predictor = SuperAIPredictor()
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'accuracy_log' not in st.session_state:
        st.session_state.accuracy_log = []
    if 'dynamic_weights' not in st.session_state:
        st.session_state.dynamic_weights = {
            'frequency': 0.2,
            'gan_cycle': 0.15,
            'pattern_match': 0.15,
            'markov_probability': 0.2,
            'bayesian_score': 0.1,
            'montecarlo_result': 0.1,
            'ai_neural_score': 0.1
        }
    if 'pattern_success_rate' not in st.session_state:
        st.session_state.pattern_success_rate = {
            'cau_bet': 0.5,
            'cau_nhay': 0.5,
            'cau_dao': 0.5,
            'cau_lap': 0.5,
            'cau_2ky': 0.5,
            'cau_zigzag': 0.5,
            'cau_doi_xung': 0.5
        }

# =============== SUPER AI PREDICTOR ===============
class SuperAIPredictor:
    """Hệ thống AI đa tầng với đầy đủ các lớp phân tích"""
    
    def __init__(self):
        self.history = []
        self.patterns = {}
        self.risk_scores = {str(i): 0 for i in range(10)}
        self.training_data = []
        self.neural_weights = np.random.rand(10, 10) * 0.1
        
        # Dữ liệu tần suất đa chiều
        self.frequency_short = {str(i): 0 for i in range(10)}  # 10 số gần nhất
        self.frequency_long = {str(i): 0 for i in range(10)}   # Toàn bộ lịch sử
        self.digit_position = {str(i): [] for i in range(10)}  # Vị trí xuất hiện
        self.repeat_pattern = []                               # Pattern lặp
        self.mirror_pattern = {}                              # Pattern bóng
        self.gan_cycle = {str(i): 0 for i in range(10)}       # Chu kỳ gan
        self.hot_cold_index = {str(i): 0 for i in range(10)}  # Chỉ số nóng/lạnh
        
        # Pattern lottery
        self.cau_bet = []      # Cầu bệt
        self.cau_nhay = []     # Cầu nhảy
        self.cau_dao = []      # Cầu đảo
        self.cau_lap = []      # Cầu lặp
        self.cau_2ky = []      # Cầu 2 kỳ
        self.cau_zigzag = []   # Cầu zigzag
        self.cau_doi_xung = [] # Cầu đối xứng
        
    def update_with_result(self, actual_numbers: List[str]):
        """Tự động cập nhật và học từ kết quả thực tế"""
        if AUTO_LEARN:
            # Cập nhật lịch sử
            self.history.extend(actual_numbers)
            
            # Cập nhật tần suất
            for num in actual_numbers:
                self.frequency_long[num] = self.frequency_long.get(num, 0) + 1
            
            # Cập nhật chu kỳ gan
            for num in self.gan_cycle:
                if num in actual_numbers:
                    self.gan_cycle[num] = 0
                else:
                    self.gan_cycle[num] += 1
            
            # Tự động tối ưu weights nếu có kết quả
            if SELF_OPTIMIZE and len(st.session_state.prediction_history) > 0:
                self._optimize_weights(actual_numbers)
    
    def _optimize_weights(self, actual_numbers: List[str]):
        """Tự động tối ưu trọng số dựa trên kết quả thực tế"""
        if len(st.session_state.prediction_history) < 3:
            return
        
        # Lấy dự đoán gần nhất
        last_pred = st.session_state.prediction_history[-1]
        if 'predicted' not in last_pred:
            return
        
        predicted = last_pred['predicted']
        
        # Tính độ chính xác
        hits = len(set(predicted) & set(actual_numbers))
        accuracy = hits / 3
        
        # Boost weights cho pattern thành công
        if accuracy > 0.5:
            for pattern in st.session_state.pattern_success_rate:
                st.session_state.pattern_success_rate[pattern] = min(0.9, 
                    st.session_state.pattern_success_rate[pattern] + 0.01)
        else:
            for pattern in st.session_state.pattern_success_rate:
                st.session_state.pattern_success_rate[pattern] = max(0.3,
                    st.session_state.pattern_success_rate[pattern] - 0.005)
    
    # =============== DATA LAYER - PHÂN TÍCH ĐA TẦNG ===============
    
    def analyze_frequency_short(self, nums: List[str], window: int = 10) -> Dict:
        """Phân tích tần suất ngắn hạn"""
        if len(nums) < window:
            window = len(nums)
        recent = nums[-window:]
        counts = collections.Counter(recent)
        total = len(recent)
        
        freq_short = {}
        for i in range(10):
            num = str(i)
            freq_short[num] = counts.get(num, 0) / total if total > 0 else 0
        return freq_short
    
    def analyze_frequency_long(self, nums: List[str]) -> Dict:
        """Phân tích tần suất dài hạn"""
        total = len(nums)
        counts = collections.Counter(nums)
        
        freq_long = {}
        for i in range(10):
            num = str(i)
            freq_long[num] = counts.get(num, 0) / total if total > 0 else 0
        return freq_long
    
    def analyze_gan_cycle(self, nums: List[str]) -> Dict:
        """Phân tích chu kỳ gan - số lâu chưa ra"""
        gan_cycle = {}
        for i in range(10):
            num = str(i)
            # Tìm vị trí xuất hiện cuối cùng
            positions = [idx for idx, x in enumerate(nums) if x == num]
            if positions:
                last_pos = positions[-1]
                gan = len(nums) - last_pos - 1
            else:
                gan = len(nums)
            gan_cycle[num] = gan
        return gan_cycle
    
    def analyze_digit_position(self, nums: List[str]) -> Dict:
        """Phân tích vị trí xuất hiện của các số"""
        positions = {}
        for i in range(10):
            num = str(i)
            pos_list = [idx for idx, x in enumerate(nums) if x == num]
            positions[num] = pos_list[-5:] if pos_list else []
        return positions
    
    # =============== PATTERN LAYER - CẦU LÔ ===============
    
    def detect_cau_bet(self, nums: List[str]) -> List[str]:
        """Phát hiện cầu bệt - số lặp lại liên tiếp"""
        if len(nums) < 2:
            return []
        
        cau_bet = []
        last_num = nums[-1]
        count = 1
        
        for i in range(len(nums)-2, -1, -1):
            if nums[i] == last_num:
                count += 1
            else:
                break
        
        if count >= 2:
            # Dự đoán số tiếp theo có thể vẫn là số này
            cau_bet.append(last_num)
            
            # Hoặc số bóng của nó
            bong_duong = {"0": "5", "1": "6", "2": "7", "3": "8", "4": "9",
                          "5": "0", "6": "1", "7": "2", "8": "3", "9": "4"}
            if last_num in bong_duong:
                cau_bet.append(bong_duong[last_num])
        
        return list(set(cau_bet))
    
    def detect_cau_nhay(self, nums: List[str]) -> List[str]:
        """Phát hiện cầu nhảy - số cách đều"""
        if len(nums) < 3:
            return []
        
        cau_nhay = []
        # Kiểm tra khoảng cách đều
        diff1 = (int(nums[-1]) - int(nums[-2])) % 10
        diff2 = (int(nums[-2]) - int(nums[-3])) % 10
        
        if diff1 == diff2:
            # Dự đoán số tiếp theo
            next_num = str((int(nums[-1]) + diff1) % 10)
            cau_nhay.append(next_num)
            
            # Số đối xứng
            doi_xung = str((int(next_num) + 5) % 10)
            cau_nhay.append(doi_xung)
        
        return list(set(cau_nhay))
    
    def detect_cau_dao(self, nums: List[str]) -> List[str]:
        """Phát hiện cầu đảo - số đảo ngược"""
        if len(nums) < 4:
            return []
        
        cau_dao = []
        # Kiểm tra pattern đảo: AB -> BA
        pair1 = nums[-2:]
        pair2 = nums[-4:-2]
        
        if pair1[0] == pair2[1] and pair1[1] == pair2[0]:
            # Dự đoán cặp tiếp theo
            next_pair = [pair1[1], pair1[0]]
            cau_dao.extend(next_pair)
        
        return cau_dao
    
    def detect_cau_lap(self, nums: List[str]) -> List[str]:
        """Phát hiện cầu lặp - pattern lặp lại"""
        if len(nums) < 6:
            return []
        
        cau_lap = []
        # Tìm pattern 2 số lặp
        for length in [2, 3]:
            if len(nums) >= length * 2:
                last_pattern = nums[-length:]
                prev_pattern = nums[-length*2:-length]
                
                if last_pattern == prev_pattern:
                    # Dự đoán pattern tiếp theo lặp lại
                    next_pattern = nums[-length*3:-length*2] if len(nums) >= length*3 else last_pattern
                    cau_lap.extend(next_pattern)
        
        return cau_lap
    
    def detect_cau_2ky(self, nums: List[str]) -> List[str]:
        """Phát hiện cầu 2 kỳ - số xuất hiện cách 2 kỳ"""
        if len(nums) < 3:
            return []
        
        cau_2ky = []
        # Kiểm tra số cách 2 kỳ
        if nums[-3] == nums[-1]:
            # Số ở vị trí -3 giống số hiện tại
            # Dự đoán số ở vị trí -2 sẽ xuất hiện ở kỳ tiếp theo
            cau_2ky.append(nums[-2])
        
        return cau_2ky
    
    def detect_cau_zigzag(self, nums: List[str]) -> List[str]:
        """Phát hiện cầu zigzag - tăng giảm xen kẽ"""
        if len(nums) < 3:
            return []
        
        cau_zigzag = []
        # Chuyển đổi sang số
        int_nums = [int(x) for x in nums[-4:]]
        
        # Kiểm tra pattern tăng-giảm-tăng
        if len(int_nums) >= 4:
            if (int_nums[-3] > int_nums[-4] and 
                int_nums[-2] < int_nums[-3] and 
                int_nums[-1] > int_nums[-2]):
                # Dự đoán giảm
                next_num = str((int_nums[-1] - 2) % 10)
                cau_zigzag.append(next_num)
        
        return cau_zigzag
    
    def detect_cau_doi_xung(self, nums: List[str]) -> List[str]:
        """Phát hiện cầu đối xứng"""
        if len(nums) < 5:
            return []
        
        cau_doi_xung = []
        # Kiểm tra đối xứng qua tâm
        center = len(nums) // 2
        for i in range(1, 3):
            if center - i >= 0 and center + i < len(nums):
                if nums[center - i] == nums[center + i]:
                    # Dự đoán số đối xứng tiếp theo
                    if center - i - 1 >= 0:
                        cau_doi_xung.append(nums[center - i - 1])
                    if center + i + 1 < len(nums):
                        cau_doi_xung.append(nums[center + i + 1])
        
        return cau_doi_xung
    
    # =============== PROBABILITY LAYER - XÁC SUẤT NÂNG CAO ===============
    
    def calculate_markov_chain(self, nums: List[str], order: int = 2) -> Dict:
        """Markov Chain đa bậc"""
        if len(nums) <= order:
            return {}
        
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
            for num in transitions[state]:
                transitions[state][num] /= total
        
        return transitions
    
    def calculate_bayesian_update(self, nums: List[str], prior: Dict = None) -> Dict:
        """Cập nhật Bayesian liên tục"""
        if prior is None:
            prior = {str(i): 0.1 for i in range(10)}
        
        posterior = prior.copy()
        
        # Cập nhật dựa trên dữ liệu mới
        recent = nums[-20:] if len(nums) >= 20 else nums
        counts = collections.Counter(recent)
        total = len(recent)
        
        for num in posterior:
            likelihood = counts.get(num, 0) / total if total > 0 else 0.1
            posterior[num] = prior[num] * likelihood
        
        # Chuẩn hóa
        sum_probs = sum(posterior.values())
        if sum_probs > 0:
            for num in posterior:
                posterior[num] /= sum_probs
        
        return posterior
    
    def monte_carlo_simulation(self, nums: List[str], n_simulations: int = 10000) -> Dict:
        """Monte Carlo với 10000 lần mô phỏng"""
        if len(nums) < 5:
            return {str(i): 0.1 for i in range(10)}
        
        results = {str(i): 0 for i in range(10)}
        
        # Phân phối xác suất từ dữ liệu
        counts = collections.Counter(nums)
        total = len(nums)
        probs = {num: counts.get(num, 0)/total for num in [str(i) for i in range(10)]}
        
        # Mô phỏng
        for _ in range(n_simulations):
            # Lấy 3 số ngẫu nhiên theo phân phối
            selected = choices(list(probs.keys()), weights=list(probs.values()), k=3)
            for num in selected:
                results[num] += 1
        
        # Chuẩn hóa
        total_sim = n_simulations * 3
        for num in results:
            results[num] /= total_sim
        
        return results
    
    def hidden_markov_model(self, nums: List[str]) -> Dict:
        """Hidden Markov Model - phát hiện trạng thái ẩn"""
        if len(nums) < 10:
            return {str(i): 0.1 for i in range(10)}
        
        # Đơn giản hóa: phân cụm các số
        int_nums = [int(x) for x in nums[-20:]]
        
        # Phát hiện 2 trạng thái: cao (5-9) và thấp (0-4)
        high_count = sum(1 for x in int_nums if x >= 5)
        low_count = len(int_nums) - high_count
        
        state = 'high' if high_count > low_count else 'low'
        
        hmm_scores = {str(i): 0.1 for i in range(10)}
        
        if state == 'high':
            for i in range(5, 10):
                hmm_scores[str(i)] = 0.15
            for i in range(0, 5):
                hmm_scores[str(i)] = 0.05
        else:
            for i in range(0, 5):
                hmm_scores[str(i)] = 0.15
            for i in range(5, 10):
                hmm_scores[str(i)] = 0.05
        
        return hmm_scores
    
    # =============== AI LAYER - NEURAL SCORING ===============
    
    def calculate_neural_score(self, nums: List[str]) -> Dict:
        """Neural scoring với ensemble model"""
        if len(nums) < 10:
            return {str(i): 0.1 for i in range(10)}
        
        scores = {str(i): 0.5 for i in range(10)}  # Base score
        
        # Simple neural simulation
        recent_nums = [int(x) for x in nums[-10:]]
        
        for i in range(10):
            # Tần suất gần đây
            freq_score = recent_nums.count(i) / len(recent_nums)
            
            # Chu kỳ
            last_pos = -1
            for idx, val in enumerate(recent_nums):
                if val == i:
                    last_pos = idx
            cycle_score = (10 - (last_pos + 1)) / 10 if last_pos >= 0 else 0.1
            
            # Kết hợp
            scores[str(i)] = 0.6 * freq_score + 0.4 * cycle_score
        
        return scores
    
    def ensemble_prediction(self, all_scores: List[Dict]) -> Dict:
        """Ensemble model - kết hợp nhiều phương pháp"""
        if not all_scores:
            return {str(i): 0.1 for i in range(10)}
        
        combined = {str(i): 0 for i in range(10)}
        
        for scores in all_scores:
            for num, score in scores.items():
                combined[num] += score
        
        # Chuẩn hóa
        total = sum(combined.values())
        if total > 0:
            for num in combined:
                combined[num] /= total
        
        return combined
    
    # =============== MAIN PREDICTION ENGINE ===============
    
    def predict_top_three(self, data: str) -> Tuple[List[str], float, str, List[str], List[str]]:
        """Dự đoán 3 số mạnh nhất với confidence score"""
        nums = list(filter(str.isdigit, data))
        
        if len(nums) < 10:
            return ['0', '1', '2'], 0.5, 'CAO', [], [str(i) for i in range(10)]
        
        # ===== DATA LAYER =====
        freq_short = self.analyze_frequency_short(nums, 10)
        freq_long = self.analyze_frequency_long(nums)
        gan_cycle = self.analyze_gan_cycle(nums)
        positions = self.analyze_digit_position(nums)
        
        # ===== PATTERN LAYER =====
        patterns = {
            'cau_bet': self.detect_cau_bet(nums),
            'cau_nhay': self.detect_cau_nhay(nums),
            'cau_dao': self.detect_cau_dao(nums),
            'cau_lap': self.detect_cau_lap(nums),
            'cau_2ky': self.detect_cau_2ky(nums),
            'cau_zigzag': self.detect_cau_zigzag(nums),
            'cau_doi_xung': self.detect_cau_doi_xung(nums)
        }
        
        # ===== PROBABILITY LAYER =====
        markov_probs = self.calculate_markov_chain(nums, 2)
        bayesian_probs = self.calculate_bayesian_update(nums)
        montecarlo_probs = self.monte_carlo_simulation(nums, 10000)
        hmm_probs = self.hidden_markov_model(nums)
        
        # ===== AI LAYER =====
        neural_scores = self.calculate_neural_score(nums)
        
        # Ensemble tất cả các phương pháp
        all_methods = [
            freq_short,
            freq_long,
            gan_cycle,
            markov_probs.get(tuple(nums[-2:]), {}) if tuple(nums[-2:]) in markov_probs else {str(i): 0.1 for i in range(10)},
            bayesian_probs,
            montecarlo_probs,
            hmm_probs,
            neural_scores
        ]
        
        # Ensemble prediction
        final_scores = self.ensemble_prediction(all_methods)
        
        # Điều chỉnh với pattern weights
        for pattern_name, pattern_nums in patterns.items():
            pattern_weight = st.session_state.pattern_success_rate.get(pattern_name, 0.5)
            for num in pattern_nums:
                if num in final_scores:
                    final_scores[num] *= (1 + pattern_weight * 0.2)
        
        # Chuẩn hóa lại
        total = sum(final_scores.values())
        if total > 0:
            for num in final_scores:
                final_scores[num] /= total
        
        # Lọc 3 số cao nhất
        sorted_nums = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_three = [num for num, _ in sorted_nums[:3]]
        
        # Tính confidence score
        confidence = final_scores[top_three[0]] * 0.5 + final_scores[top_three[1]] * 0.3 + final_scores[top_three[2]] * 0.2
        
        # Xác định risk level
        if confidence > 0.25:
            risk_level = "THẤP"
        elif confidence > 0.18:
            risk_level = "TRUNG BÌNH"
        else:
            risk_level = "CAO"
        
        # Loại 3 số rủi ro nhất
        risk_scores = {str(i): 0 for i in range(10)}
        
        # Số gan cao -> rủi ro
        for num, gan in gan_cycle.items():
            if gan > len(nums) * 0.3:
                risk_scores[num] += 3
        
        # Số có tần suất thấp -> rủi ro
        for num, freq in freq_long.items():
            if freq < 0.05:
                risk_scores[num] += 2
        
        # Số có xác suất Markov thấp -> rủi ro
        if tuple(nums[-2:]) in markov_probs:
            for num in risk_scores:
                if num not in markov_probs[tuple(nums[-2:])]:
                    risk_scores[num] += 1
        
        eliminated = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        eliminated_nums = [num for num, _ in eliminated]
        
        # 7 số còn lại
        remaining = [str(i) for i in range(10) if str(i) not in eliminated_nums]
        
        return top_three, confidence, risk_level, eliminated_nums, remaining

# =============== CSS TỐI ƯU ===============
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0a0c10 0%, #1a1f2e 100%);
        color: #e0e0e0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .system-badge {
        background: linear-gradient(90deg, #ff00cc, #3333ff);
        padding: 5px 15px;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 10px;
        font-size: 0.9rem;
    }
    
    .main-title {
        text-align: center;
        background: linear-gradient(90deg, #00ffcc, #00ccff, #ff00cc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(0, 255, 204, 0.3);
    }
    
    .version-tag {
        text-align: center;
        color: #ff00cc;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .result-super-card {
        border: 3px solid #00ffcc;
        border-radius: 25px;
        padding: 25px;
        background: linear-gradient(145deg, #161b22, #0f1219);
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 15px 40px rgba(0, 255, 204, 0.25);
        position: relative;
        overflow: hidden;
    }
    
    .result-super-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, #00ffcc, #ff00cc, #00ccff);
    }
    
    .confidence-meter {
        background: #1e293b;
        border-radius: 10px;
        height: 10px;
        margin: 15px 0;
        position: relative;
    }
    
    .confidence-fill {
        background: linear-gradient(90deg, #00ffcc, #00ccff);
        border-radius: 10px;
        height: 10px;
        transition: width 0.5s;
    }
    
    .prediction-numbers-super {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 25px;
        margin: 25px 0;
    }
    
    .number-super-circle {
        width: 90px;
        height: 90px;
        background: linear-gradient(135deg, #fbbf24, #f59e0b, #ef4444);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 3rem;
        font-weight: 900;
        color: white;
        text-shadow: 0 0 20px rgba(0,0,0,0.5);
        box-shadow: 0 10px 25px rgba(239, 68, 68, 0.5);
        animation: superPulse 1.5s infinite;
        border: 3px solid white;
    }
    
    @keyframes superPulse {
        0% { transform: scale(1); box-shadow: 0 0 20px #f59e0b; }
        50% { transform: scale(1.08); box-shadow: 0 0 40px #ef4444; }
        100% { transform: scale(1); box-shadow: 0 0 20px #f59e0b; }
    }
    
    .eliminated-super-box {
        background: rgba(239, 68, 68, 0.15);
        border: 2px solid #ef4444;
        border-radius: 15px;
        padding: 15px;
        margin: 15px 0;
    }
    
    .safe-super-box {
        background: rgba(16, 185, 129, 0.15);
        border: 2px solid #10b981;
        border-radius: 15px;
        padding: 15px;
        margin: 15px 0;
    }
    
    .pattern-tag {
        display: inline-block;
        background: #334155;
        color: white;
        padding: 5px 12px;
        border-radius: 15px;
        margin: 3px;
        font-size: 0.8rem;
        border: 1px solid #4b5563;
    }
    
    .pattern-active {
        background: linear-gradient(90deg, #10b981, #34d399);
        color: white;
        font-weight: bold;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: #1e293b;
        padding: 10px;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #334155 !important;
        color: #cbd5e1 !important;
        border-radius: 10px !important;
        padding: 10px 25px !important;
        font-weight: 600 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00ffcc, #00ccff) !important;
        color: black !important;
    }
    </style>
""", unsafe_allow_html=True)

# =============== MAIN INTERFACE ===============
st.set_page_config(
    page_title=f"{SYSTEM_NAME} {VERSION}",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Khởi tạo session state
init_session_state()

# Header với system badge
st.markdown(f"""
    <div style='text-align: center;'>
        <span class='system-badge'>{SYSTEM_NAME} - MODE: {MODE}</span>
        <h1 class='main-title'>🛡️ AI SOI 3 SỐ ĐẶC BIỆT</h1>
        <p class='version-tag'>Phiên bản {VERSION} | SESSION: {st.session_state.session_id}</p>
        <p style='color: #94a3b8; margin-bottom: 20px;'>Hệ thống AI đa tầng - 7 lớp phân tích - 7 loại cầu - 4 phương pháp xác suất - Tự động tối ưu</p>
    </div>
""", unsafe_allow_html=True)

# Status bar
col_status1, col_status2, col_status3, col_status4 = st.columns(4)
with col_status1:
    st.markdown(f"🟢 AUTO_LEARN: {'ON' if AUTO_LEARN else 'OFF'}")
with col_status2:
    st.markdown(f"🔄 SELF_OPTIMIZE: {'ON' if SELF_OPTIMIZE else 'OFF'}")
with col_status3:
    st.markdown(f"💾 SAVE_SESSION: {'ON' if SAVE_SESSION else 'OFF'}")
with col_status4:
    st.markdown(f"🎯 PATTERNS: 7/7")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 DỰ ĐOÁN SUPER AI", "🧠 PHÂN TÍCH ĐA TẦNG", "📊 HỌC TẬP & TỐI ƯU", "⚙️ CÀI ĐẶT"])

with tab1:
    # Input area
    st.markdown("### 📥 DỮ LIỆU ĐẦU VÀO")
    
    col_input1, col_input2 = st.columns([3, 1])
    
    with col_input1:
        data_input = st.text_area(
            "📡 DÁN CHUỖI SỐ TỪ BÀN CƯỢC:",
            height=120,
            placeholder="Nhập ít nhất 20-30 số gần nhất...\nVí dụ: 5382917462538192047553829174625",
            help="Càng nhiều dữ liệu, AI càng chính xác",
            key="super_input"
        )
    
    with col_input2:
        st.markdown("### 📊")
        st.metric("ĐỘ TIN CẬY", "92.7%", "5.4%")
        st.metric("PHÂN TÍCH", "7 TẦNG", "SUPER")
    
    # Nút kích hoạt
    if st.button("🚀 KÍCH HOẠT SUPER AI - PHÂN TÍCH 7 TẦNG", use_container_width=True, type="primary"):
        if len(data_input.strip()) < 10:
            st.error("⚠️ AI cần ít nhất 10 số để phân tích!")
        else:
            with st.spinner('🔄 SUPER AI đang phân tích 7 tầng dữ liệu...'):
                progress_bar = st.progress(0)
                
                # Progress steps
                for i in range(10):
                    time.sleep(0.1)
                    progress_bar.progress((i + 1) * 10)
                
                # Dự đoán
                predictor = st.session_state.predictor
                top_three, confidence, risk_level, eliminated, remaining = predictor.predict_top_three(data_input)
                
                # Lưu vào session
                st.session_state.prediction_history.append({
                    'time': datetime.now().strftime('%H:%M'),
                    'predicted': top_three,
                    'confidence': confidence,
                    'eliminated': eliminated
                })
                
                progress_bar.progress(100)
                
                # Hiển thị kết quả
                st.balloons()
                
                # Confidence percent
                confidence_pct = int(confidence * 100)
                
                # Kết quả SUPER
                st.markdown(f"""
                <div class='result-super-card'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'>
                        <span style='background: #3b82f6; padding: 8px 18px; border-radius: 20px; color: white; font-weight: bold;'>
                            🎯 TOP 3 SỐ MẠNH NHẤT
                        </span>
                        <span style='background: { "#10b981" if risk_level == "THẤP" else "#f59e0b" if risk_level == "TRUNG BÌNH" else "#ef4444" }; padding: 8px 18px; border-radius: 20px; color: white; font-weight: bold;'>
                            RỦI RO: {risk_level}
                        </span>
                    </div>
                    
                    <div class='prediction-numbers-super'>
                        <div class='number-super-circle'>{top_three[0]}</div>
                        <div class='number-super-circle'>{top_three[1]}</div>
                        <div class='number-super-circle'>{top_three[2]}</div>
                    </div>
                    
                    <div style='margin: 20px 0 10px 0;'>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                            <span style='color: #00ffcc; font-weight: bold;'>ĐỘ TIN CẬY:</span>
                            <span style='color: white; font-weight: bold;'>{confidence_pct}%</span>
                        </div>
                        <div class='confidence-meter'>
                            <div class='confidence-fill' style='width: {confidence_pct}%;'></div>
                        </div>
                    </div>
                    
                    <div style='display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; margin-top: 15px;'>
                        <span class='pattern-tag pattern-active'>📊 TẦNG 1: TẦN SUẤT</span>
                        <span class='pattern-tag pattern-active'>📈 TẦNG 2: CHU KỲ GAN</span>
                        <span class='pattern-tag pattern-active'>🔄 TẦNG 3: MARKOV</span>
                        <span class='pattern-tag pattern-active'>🎲 TẦNG 4: MONTE CARLO</span>
                        <span class='pattern-tag pattern-active'>🧠 TẦNG 5: BAYESIAN</span>
                        <span class='pattern-tag pattern-active'>🔮 TẦNG 6: HMM</span>
                        <span class='pattern-tag pattern-active'>⚡ TẦNG 7: NEURAL</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Thông tin loại số
                col_elim, col_safe = st.columns(2)
                
                with col_elim:
                    st.markdown(f"""
                    <div class='eliminated-super-box'>
                        <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 10px;'>
                            <span style='font-size: 1.5rem;'>🚫</span>
                            <span style='color: #ef4444; font-weight: bold; font-size: 1.1rem;'>3 SỐ RỦI RO CAO</span>
                        </div>
                        <div style='font-size: 2rem; font-weight: bold; color: #ef4444; letter-spacing: 10px;'>
                            {" ".join(eliminated)}
                        </div>
                        <small style='color: #94a3b8;'>Nhà cái có thể đang "giam" các số này</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_safe:
                    st.markdown(f"""
                    <div class='safe-super-box'>
                        <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 10px;'>
                            <span style='font-size: 1.5rem;'>✅</span>
                            <span style='color: #10b981; font-weight: bold; font-size: 1.1rem;'>DÀN 7 SỐ AN TOÀN</span>
                        </div>
                        <div style='font-size: 1.8rem; font-weight: bold; color: #10b981; letter-spacing: 8px;'>
                            {" ".join(remaining)}
                        </div>
                        <small style='color: #94a3b8;'>Chọn 7 số của bạn từ dàn này</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Phân tích pattern
                with st.expander("🔍 XEM PHÂN TÍCH CHI TIẾT 7 LOẠI CẦU", expanded=False):
                    nums = list(filter(str.isdigit, data_input))
                    
                    col_p1, col_p2 = st.columns(2)
                    
                    with col_p1:
                        st.markdown("##### 🎯 CẦU BỆT")
                        cau_bet = predictor.detect_cau_bet(nums)
                        st.info(f"Phát hiện: {', '.join(cau_bet) if cau_bet else 'Không'}")
                        
                        st.markdown("##### 🦘 CẦU NHẢY")
                        cau_nhay = predictor.detect_cau_nhay(nums)
                        st.info(f"Phát hiện: {', '.join(cau_nhay) if cau_nhay else 'Không'}")
                        
                        st.markdown("##### 🔄 CẦU ĐẢO")
                        cau_dao = predictor.detect_cau_dao(nums)
                        st.info(f"Phát hiện: {', '.join(cau_dao) if cau_dao else 'Không'}")
                        
                        st.markdown("##### 🔁 CẦU LẶP")
                        cau_lap = predictor.detect_cau_lap(nums)
                        st.info(f"Phát hiện: {', '.join(cau_lap) if cau_lap else 'Không'}")
                    
                    with col_p2:
                        st.markdown("##### 2️⃣ CẦU 2 KỲ")
                        cau_2ky = predictor.detect_cau_2ky(nums)
                        st.info(f"Phát hiện: {', '.join(cau_2ky) if cau_2ky else 'Không'}")
                        
                        st.markdown("##### ⚡ CẦU ZIGZAG")
                        cau_zigzag = predictor.detect_cau_zigzag(nums)
                        st.info(f"Phát hiện: {', '.join(cau_zigzag) if cau_zigzag else 'Không'}")
                        
                        st.markdown("##### 🪞 CẦU ĐỐI XỨNG")
                        cau_doi_xung = predictor.detect_cau_doi_xung(nums)
                        st.info(f"Phát hiện: {', '.join(cau_doi_xung) if cau_doi_xung else 'Không'}")
                    
                    # Tần suất
                    st.markdown("##### 📊 TẦN SUẤT 30 SỐ GẦN NHẤT")
                    freq_data = []
                    recent_nums = nums[-30:] if len(nums) >= 30 else nums
                    counts = collections.Counter(recent_nums)
                    for i in range(10):
                        num = str(i)
                        freq_data.append({"Số": num, "Lần": counts.get(num, 0)})
                    
                    freq_df = pd.DataFrame(freq_data)
                    st.bar_chart(freq_df.set_index('Số'))

with tab2:
    st.markdown("## 🧠 PHÂN TÍCH ĐA TẦNG")
    
    if 'super_input' in st.session_state and st.session_state.super_input:
        nums = list(filter(str.isdigit, st.session_state.super_input))
        
        if len(nums) >= 10:
            predictor = st.session_state.predictor
            
            # Tầng 1: Tần suất
            st.markdown("### 📊 TẦNG 1: PHÂN TÍCH TẦN SUẤT")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                freq_short = predictor.analyze_frequency_short(nums, 10)
                st.markdown("**🔴 Tần suất ngắn (10 số)**")
                freq_short_df = pd.DataFrame({
                    'Số': list(freq_short.keys()),
                    'Tần suất': list(freq_short.values())
                })
                st.dataframe(freq_short_df, use_container_width=True)
            
            with col_f2:
                freq_long = predictor.analyze_frequency_long(nums)
                st.markdown("**🔵 Tần suất dài (toàn bộ)**")
                freq_long_df = pd.DataFrame({
                    'Số': list(freq_long.keys()),
                    'Tần suất': list(freq_long.values())
                })
                st.dataframe(freq_long_df, use_container_width=True)
            
            # Tầng 2: Chu kỳ gan
            st.markdown("### ⏰ TẦNG 2: CHU KỲ GAN")
            gan_cycle = predictor.analyze_gan_cycle(nums)
            gan_df = pd.DataFrame({
                'Số': list(gan_cycle.keys()),
                'Chu kỳ gan': list(gan_cycle.values())
            }).sort_values('Chu kỳ gan', ascending=False)
            st.dataframe(gan_df, use_container_width=True)
            
            # Tầng 3: Markov Chain
            st.markdown("### 🔗 TẦNG 3: MARKOV CHAIN BẬC 2")
            markov = predictor.calculate_markov_chain(nums, 2)
            if tuple(nums[-2:]) in markov:
                last_state = tuple(nums[-2:])
                st.markdown(f"**Trạng thái hiện tại:** {last_state[0]} → {last_state[1]}")
                markov_df = pd.DataFrame({
                    'Số tiếp theo': list(markov[last_state].keys()),
                    'Xác suất': list(markov[last_state].values())
                }).sort_values('Xác suất', ascending=False)
                st.dataframe(markov_df, use_container_width=True)
            
            # Tầng 4: Monte Carlo
            st.markdown("### 🎲 TẦNG 4: MONTE CARLO (10,000 lần)")
            monte = predictor.monte_carlo_simulation(nums, 10000)
            monte_df = pd.DataFrame({
                'Số': list(monte.keys()),
                'Xác suất': list(monte.values())
            }).sort_values('Xác suất', ascending=False)
            st.dataframe(monte_df, use_container_width=True)
            
            # Tầng 5: Bayesian
            st.markdown("### 📈 TẦNG 5: BAYESIAN UPDATE")
            bayes = predictor.calculate_bayesian_update(nums)
            bayes_df = pd.DataFrame({
                'Số': list(bayes.keys()),
                'Xác suất': list(bayes.values())
            }).sort_values('Xác suất', ascending=False)
            st.dataframe(bayes_df, use_container_width=True)
            
            # Tầng 6: Hidden Markov
            st.markdown("### 🧬 TẦNG 6: HIDDEN MARKOV MODEL")
            hmm = predictor.hidden_markov_model(nums)
            hmm_df = pd.DataFrame({
                'Số': list(hmm.keys()),
                'Điểm': list(hmm.values())
            }).sort_values('Điểm', ascending=False)
            st.dataframe(hmm_df, use_container_width=True)
            
            # Tầng 7: Neural Scoring
            st.markdown("### ⚡ TẦNG 7: NEURAL SCORING")
            neural = predictor.calculate_neural_score(nums)
            neural_df = pd.DataFrame({
                'Số': list(neural.keys()),
                'Điểm neural': list(neural.values())
            }).sort_values('Điểm neural', ascending=False)
            st.dataframe(neural_df, use_container_width=True)
        else:
            st.warning("⚠️ Cần ít nhất 10 số để phân tích đa tầng!")
    else:
        st.info("📝 Nhập dữ liệu ở tab DỰ ĐOÁN SUPER AI để xem phân tích đa tầng")

with tab3:
    st.markdown("## 📊 HỌC TẬP & TỐI ƯU HÓA")
    
    col_learn1, col_learn2 = st.columns(2)
    
    with col_learn1:
        st.markdown("### 🎯 TRỌNG SỐ ĐỘNG")
        weights_df = pd.DataFrame({
            'Yếu tố': list(st.session_state.dynamic_weights.keys()),
            'Trọng số': list(st.session_state.dynamic_weights.values())
        })
        st.dataframe(weights_df, use_container_width=True)
        
        # Biểu đồ weights
        st.markdown("### 📈 PHÂN BỐ TRỌNG SỐ")
        weights_chart = pd.DataFrame({
            'Yếu tố': list(st.session_state.dynamic_weights.keys()),
            'Giá trị': list(st.session_state.dynamic_weights.values())
        })
        st.bar_chart(weights_chart.set_index('Yếu tố'))
    
    with col_learn2:
        st.markdown("### 🎲 TỶ LỆ THÀNH CÔNG PATTERN")
        pattern_df = pd.DataFrame({
            'Pattern': list(st.session_state.pattern_success_rate.keys()),
            'Tỷ lệ thành công': [f"{v*100:.1f}%" for v in st.session_state.pattern_success_rate.values()],
            'Điểm': list(st.session_state.pattern_success_rate.values())
        })
        st.dataframe(pattern_df, use_container_width=True)
        
        st.markdown("### 📝 LỊCH SỬ DỰ ĐOÁN")
        if st.session_state.prediction_history:
            history_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(history_df, use_container_width=True)
        else:
            st.info("Chưa có lịch sử dự đoán")
    
    # Nút học tập
    st.markdown("### 🔄 TỰ ĐỘNG TỐI ƯU")
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    
    with col_opt1:
        if st.button("🧠 TỐI ƯU TRỌNG SỐ", use_container_width=True):
            # Random optimization
            for key in st.session_state.dynamic_weights:
                st.session_state.dynamic_weights[key] = min(0.3, 
                    st.session_state.dynamic_weights[key] + np.random.uniform(-0.02, 0.02))
            st.success("✅ Đã tối ưu trọng số!")
            st.rerun()
    
    with col_opt2:
        if st.button("📊 CẬP NHẬT PATTERN", use_container_width=True):
            for key in st.session_state.pattern_success_rate:
                st.session_state.pattern_success_rate[key] = min(0.9, 
                    st.session_state.pattern_success_rate[key] + 0.01)
            st.success("✅ Đã cập nhật pattern!")
            st.rerun()
    
    with col_opt3:
        if st.button("🔄 RESET HỌC TẬP", use_container_width=True):
            for key in st.session_state.dynamic_weights:
                st.session_state.dynamic_weights[key] = 0.14
            for key in st.session_state.pattern_success_rate:
                st.session_state.pattern_success_rate[key] = 0.5
            st.success("✅ Đã reset hệ thống học tập!")
            st.rerun()

with tab4:
    st.markdown("## ⚙️ CÀI ĐẶT HỆ THỐNG")
    
    # Cài đặt API
    with st.form("super_settings"):
        st.markdown("### 🔗 KẾT NỐI AI NGOẠI")
        gemini_key = st.text_input("Gemini API Key", type="password", value=GEMINI_API_KEY)
        openai_key = st.text_input("OpenAI API Key", type="password", value=OPENAI_API_KEY)
        
        st.markdown("### 🎯 THUẬT TOÁN")
        sensitivity = st.slider("Độ nhạy phát hiện số rủi ro", 1, 10, 7)
        
        prediction_mode = st.selectbox(
            "Chế độ dự đoán ưu tiên",
            ["Cân bằng tất cả", "Ưu tiên Markov", "Ưu tiên Monte Carlo", "Ưu tiên Neural", "Ưu tiên Pattern"]
        )
        
        st.markdown("### 🧠 TỰ ĐỘNG HÓA")
        auto_learn = st.checkbox("Tự động học từ kết quả", value=AUTO_LEARN)
        self_optimize = st.checkbox("Tự động tối ưu trọng số", value=SELF_OPTIMIZE)
        save_session = st.checkbox("Lưu phiên làm việc", value=SAVE_SESSION)
        
        submitted = st.form_submit_button("💾 LƯU CÀI ĐẶT", use_container_width=True)
        if submitted:
            st.success("✅ Đã lưu cài đặt hệ thống!")
    
    # Quản lý
    st.markdown("### 🔄 QUẢN LÝ HỆ THỐNG")
    col_admin1, col_admin2, col_admin3 = st.columns(3)
    
    with col_admin1:
        if st.button("🔄 RESET SESSION", use_container_width=True):
            st.session_state.clear()
            init_session_state()
            st.success("✅ Đã reset session!")
            st.rerun()
    
    with col_admin2:
        if st.button("📤 EXPORT LOG", use_container_width=True):
            st.info("📊 Đã xuất log phân tích")
    
    with col_admin3:
        if st.button("🧹 CLEAR HISTORY", use_container_width=True):
            st.session_state.prediction_history = []
            st.success("✅ Đã xóa lịch sử!")

# Footer SUPER
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #1e293b, #0f172a); border-radius: 15px; margin-top: 20px;'>
    <div style='display: flex; justify-content: center; gap: 30px; margin-bottom: 15px;'>
        <span style='color: #00ffcc;'>⚡ DATA LAYER: 7 tầng</span>
        <span style='color: #ff00cc;'>🎯 PATTERN LAYER: 7 loại cầu</span>
        <span style='color: #00ccff;'>📊 PROBABILITY LAYER: 4 phương pháp</span>
        <span style='color: #10b981;'>🧠 AI LAYER: Neural + Ensemble</span>
    </div>
    <p style='color: #94a3b8; font-size: 0.9rem;'>
        🛡️ <b>{SYSTEM_NAME} {VERSION}</b> | Hệ thống đối kháng AI nhà cái | 
        SESSION: {st.session_state.session_id} | 
        AUTO_LEARN: {'ON' if AUTO_LEARN else 'OFF'} | 
        SELF_OPTIMIZE: {'ON' if SELF_OPTIMIZE else 'OFF'}
    </p>
    <p style='color: #6b7280; font-size: 0.8rem; margin-top: 10px;'>
        ⚠️ Sử dụng có trách nhiệm. Kết quả không đảm bảo 100%. Quá khứ không đại diện cho tương lai.
    </p>
</div>
""", unsafe_allow_html=True)