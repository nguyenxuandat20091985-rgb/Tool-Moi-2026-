import numpy as np
import re
from collections import Counter
import math

class TitanAI:
    def __init__(self):
        self.shadow_map = {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}

    def _parse_data(self, raw):
        cleaned = []
        lines = str(raw).split('\n') if isinstance(raw, str) else raw
        for item in lines:
            match = re.search(r'\d{5}', str(item))
            if match: cleaned.append([int(d) for d in match.group()])
        return cleaned

    def analyze_trinity(self, data):
        """Thuật toán Trinity: Tìm 5 số phủ sóng 5 tinh tốt nhất."""
        if len(data) < 5: return ["-"] * 5, 0
        
        recent = data[:15]
        all_nums = [d for row in recent for d in row]
        freq = Counter(all_nums)
        
        # Tìm khoảng cách (Gaps)
        gaps = {i: 30 for i in range(10)}
        for i in range(10):
            for idx, row in enumerate(data):
                if i in row:
                    gaps[i] = idx
                    break

        scores = {}
        for i in range(10):
            # Điểm nổ (Số đang nóng)
            f_score = freq.get(i, 0) * 3
            # Điểm rơi (Gap = 1 hoặc 0 là cực đẹp cho 3 tinh)
            g_score = 25 if gaps[i] == 1 else 10 if gaps[i] == 0 else 0
            # Điểm bóng (Shadow)
            s_score = 15 if gaps[self.shadow_map[i]] == 0 else 0
            
            scores[str(i)] = f_score + g_score + s_score
            
        top_5 = [n for n, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)][:5]
        
        # Tính độ tin cậy dựa trên Entropy
        entropy = -sum((all_nums.count(i)/len(all_nums))*math.log2(all_nums.count(i)/len(all_nums)) for i in set(all_nums))
        accuracy = int(max(0, min(100, (3.32 - entropy) * 280)))
        
        return top_5, accuracy
