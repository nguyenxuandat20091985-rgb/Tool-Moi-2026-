import numpy as np
import pandas as pd
from collections import defaultdict

class AdvancedAIPredictor:
    def __init__(self):
        self.history = []
        self.number_stats = defaultdict(lambda: {'frequency': 0, 'last_seen': 0})
        
    def analyze_house_pattern(self, last_100_games):
        """Phân tích pattern của nhà cái trong 100 ván"""
        # 1. Tính toán số lần xuất hiện của từng số
        frequency = {}
        for game in last_100_games:
            for num in game['winning_numbers']:
                frequency[num] = frequency.get(num, 0) + 1
        
        # 2. Xác định 3 số "lạnh nhất" (ít xuất hiện nhất)
        cold_numbers = sorted(frequency.items(), key=lambda x: x[1])[:3]
        
        # 3. Xác định pattern xuất hiện
        patterns = self.detect_number_patterns(last_100_games)
        
        return {
            'cold_numbers': [num for num, _ in cold_numbers],
            'hot_numbers': sorted(frequency.items(), key=lambda x: -x[1])[:5],
            'patterns': patterns
        }
    
    def generate_optimal_7(self, analysis_result, current_trend):
        """Tạo ra 7 số tối ưu nhất"""
        # Chiến lược: Loại 3 số lạnh nhất của nhà cái
        # Chọn 7 số từ 7 số còn lại có xác suất cao nhất
        
        cold_nums = analysis_result['cold_numbers']
        all_numbers = [str(i) for i in range(10)]
        
        # Loại 3 số lạnh nhất
        candidate_numbers = [num for num in all_numbers if num not in cold_nums]
        
        # Thêm 3 số có xác suất cao từ phân tích pattern
        hot_pattern_nums = self.extract_pattern_numbers(analysis_result['patterns'])
        
        # Kết hợp để có 7 số tối ưu
        optimal_7 = list(set(candidate_numbers + hot_pattern_nums))[:7]
        
        # Đảm bảo đủ 7 số
        while len(optimal_7) < 7:
            for num in all_numbers:
                if num not in optimal_7:
                    optimal_7.append(num)
                if len(optimal_7) == 7:
                    break
        
        return optimal_7
    
    def calculate_success_probability(self, selected_7, analysis_result):
        """Tính xác suất có ít nhất 3 số trùng"""
        # Dựa trên phân phối xác suất
        total_possible = self.combinations(10, 5)  # Tổ hợp 5 số từ 10 số
        favorable = self.count_favorable_combinations(selected_7, analysis_result)
        
        probability = favorable / total_possible
        return probability * 100  # Trả về phần trăm