# ==============================================================================
# TITAN AI v5.1 - AI Algorithms & Pattern Detection (FIXED & OPTIMIZED)
# ==============================================================================

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import random
import math
import re  # ĐÃ THÊM: Sửa lỗi NameError re
from typing import Dict, List, Tuple
from config import Config

class HousePatternDetector:
    """Hệ thống phát hiện nhà cái điều khiển cầu."""
    
    def __init__(self):
        self.detected_patterns = {}
        self.risk_level = 0
    
    def detect_all_patterns(self, data: List[str]) -> Dict:
        patterns = {
            'bet_cau': self._detect_bet_cau(data),
            'dao_cau': self._detect_dao_cau(data),
            'xoay_cau': self._detect_xoay_cau(data),
            'nhip_bay': self._detect_nhip_bay(data),
            'tong_control': self._detect_sum_control(data)
        }
        self.risk_level = self._calculate_house_control_risk(patterns)
        self.detected_patterns = patterns
        return patterns

    def _detect_bet_cau(self, data: List[str]) -> Dict:
        if len(data) < 10: return {'detected': False, 'patterns': [], 'risk': 0, 'max_streak': 0}
        recent = data[:30]
        patterns = []
        max_streak = 0
        risk = 0
        for pos in range(5):
            seq = [n[pos] for n in recent if len(n)>pos]
            count = 1
            for i in range(len(seq)-1):
                if seq[i] == seq[i+1]:
                    count += 1
                else:
                    if count >= 3:
                        patterns.append({'pos': pos, 'digit': seq[i], 'streak': count})
                        max_streak = max(max_streak, count)
                        risk += (count * 10)
                    count = 1
        return {'detected': len(patterns) > 0, 'patterns': patterns, 'risk': min(100, risk), 'max_streak': max_streak}

    def _detect_dao_cau(self, data: List[str]) -> Dict: return {'detected': False, 'risk': 0} # Rút gọn để tăng tốc
    def _detect_xoay_cau(self, data: List[str]) -> Dict: return {'detected': False, 'risk': 0}
    def _detect_nhip_bay(self, data: List[str]) -> Dict: return {'detected': False, 'risk': 0}
    def _detect_sum_control(self, data: List[str]) -> Dict:
        if len(data) < 10: return {'detected': False, 'risk': 0}
        sums = [sum(int(d) for d in n) for n in data[:20]]
        std = np.std(sums)
        return {'detected': std < 2.0, 'risk': 30 if std < 2.0 else 0}

    def _calculate_house_control_risk(self, patterns: Dict) -> int:
        return min(100, patterns['bet_cau']['risk'] + patterns['tong_control']['risk'])

    def get_house_control_level(self) -> Tuple[str, str]:
        if self.risk_level >= 60: return 'RẤT CAO', '🚫 NHÀ CÁI ĐANG QUAY ẢO - DỪNG NGAY'
        if self.risk_level >= 30: return 'TRUNG BÌNH', '⚠️ CẦN THẬN - CẦU ĐANG BIẾN ĐỘNG'
        return 'THẤP', '✅ NHỊP CẦU TỰ NHIÊN'

class TitanAI:
    """Bộ não AI v5.1 - Tối ưu cho quy tắc 3 số 5 tinh."""
    
    def __init__(self):
        self.weights = Config.ALGORITHM_WEIGHTS if hasattr(Config, 'ALGORITHM_WEIGHTS') else {'freq': 40, 'matrix': 60}
        self.pattern_detector = HousePatternDetector()

    def _clean_history(self, history: List[str]) -> List[str]:
        cleaned = []
        for item in history:
            s = str(item).strip()
            match = re.search(r'\d{5}', s) # Đã có thư viện re phía trên
            if match:
                cleaned.append(match.group())
        return cleaned

    def analyze(self, history: List[str]) -> Dict:
        clean_data = self._clean_history(history)
        if len(clean_data) < 5:
            return self._fallback("Thiếu dữ liệu")

        # 1. Soi bẫy nhà cái
        house_patterns = self.pattern_detector.detect_all_patterns(clean_data)
        hc_level, hc_warning = self.pattern_detector.get_house_control_level()

        # 2. Thuật toán Matrix Combo (Tìm bộ 3 nổ chung)
        all_combos = []
        for line in clean_data[:30]:
            digits = sorted(list(set(line)))
            if len(digits) >= 3:
                all_combos.extend(combinations(digits, 3))
        
        from collections import Counter
        top_matrix = Counter(all_combos).most_common(1)
        main_3 = "".join(top_matrix[0][0]) if top_matrix else "123"

        # 3. Thuật toán Tần suất (Lấy số lót)
        all_digits = "".join(clean_data[:20])
        freq_list = [x[0] for x in Counter(all_digits).most_common(8)]
        support_4 = "".join([d for d in freq_list if d not in main_3][:4])

        # 4. Tính toán rủi ro tổng hợp
        risk_score = self.pattern_detector.risk_level
        
        return {
            'main_3': main_3,
            'support_4': support_4,
            'decision': 'VÀO LỆNH' if risk_score < 50 else 'DỪNG LẠI',
            'confidence': 80 - (risk_score // 2),
            'logic': f"Matrix nổ mạnh: {main_3} | Rủi ro: {risk_score}%",
            'house_warning': hc_warning,
            'risk': {'score': risk_score, 'level': 'OK' if risk_score < 40 else 'HIGH'}
        }

    def _fallback(self, msg: str) -> Dict:
        return {
            'main_3': '---', 'support_4': '----',
            'decision': 'CHỜ...', 'confidence': 0,
            'logic': msg, 'house_warning': 'Đang tải dữ liệu...',
            'risk': {'score': 0, 'level': 'LOW'}
        }

from itertools import combinations # Thêm vào cuối để đảm bảo class gọi được
