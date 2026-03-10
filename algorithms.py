import numpy as np
import re
from collections import Counter
import math

class TitanAI:
    def __init__(self):
        # Ma trận bóng số (Shadow Map)
        self.shadow_map = {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}

    def analyze(self, data):
        """Thuật toán Neural Matrix v10.0 cho kèo 3 số 5 tinh."""
        if len(data) < 5: 
            return {"m3": "---", "l4": "----", "win_rate": 0, "logic": "Thiếu dữ liệu"}
        
        # 1. Phân tích tần suất có trọng số (Weighted Frequency)
        # Các kỳ gần nhất (5 kỳ đầu) quan trọng gấp đôi các kỳ sau
        weighted_nums = []
        for idx, row in enumerate(data[:20]):
            weight = 3 if idx < 5 else 2 if idx < 10 else 1
            for _ in range(weight):
                weighted_nums.extend(row)
        
        freq = Counter(weighted_nums)
        
        # 2. Tính toán Nhịp Trống (Gaps) và Gia tốc rơi
        gaps = {i: 50 for i in range(10)}
        for i in range(10):
            for idx, row in enumerate(data):
                if i in row:
                    gaps[i] = idx
                    break

        # 3. Chấm điểm ma trận (Matrix Scoring)
        scores = {}
        for i in range(10):
            # A. Điểm tần suất (Max 40đ)
            f_score = min(40, freq.get(i, 0) * 1.5)
            
            # B. Điểm nhịp rơi (Max 35đ) - Ưu tiên nhịp 1 (vừa về xong nghỉ 1 kỳ)
            g_score = 35 if gaps[i] == 1 else 20 if gaps[i] == 0 else 10 if gaps[i] < 4 else 0
            
            # C. Điểm bóng số (Max 25đ) - Nếu số bóng vừa về, số chính có tỷ lệ nổ cao
            shadow_num = self.shadow_map[i]
            s_score = 25 if gaps[shadow_num] == 0 else 10 if gaps[shadow_num] == 1 else 0
            
            scores[str(i)] = f_score + g_score + s_score
            
        # 4. Phân loại kết quả
        sorted_nums = [n for n, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        
        # Chốt 3 số chính (m3) và 4 số lót (l4)
        chinh = sorted_nums[:3]
        lot = sorted_nums[3:7]
        
        # 5. Tính độ tin cậy (Dựa trên độ lệch chuẩn của bảng điểm)
        all_scores = list(scores.values())
        std_dev = np.std(all_scores)
        accuracy = int(max(30, min(98, (std_dev * 5) + 20)))
        
        return {
            "m3": "".join(chinh),
            "l4": "".join(lot),
            "win_rate": accuracy,
            "logic": f"Cầu đang chạy nhịp {gaps[int(chinh[0])]} - {gaps[int(chinh[1])]}."
        }
