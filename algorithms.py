import numpy as np
import pandas as pd
import re
from collections import Counter
import math

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class TitanAI:
    def __init__(self):
        self.api_key = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
        self.model = None
        if GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
            except: self.model = None
    
    def _parse_data(self, raw):
        cleaned = []
        lines = str(raw).split('\n') if isinstance(raw, str) else raw
        for item in lines:
            match = re.search(r'\d{5}', str(item))
            if match: cleaned.append([int(d) for d in match.group()])
        return cleaned

    def _internal_scoring(self, data):
        """Hệ thống chấm điểm v6.2 - Thuật toán 'Nhịp Rơi Gương'"""
        recent_10 = data[:10] 
        # Bóng số âm dương chuẩn xác
        shadow_map = {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}
        
        # 1. Đo độ 'Nóng' cục bộ (5 kỳ gần nhất)
        local_freq = Counter([d for row in data[:5] for d in row])
        
        # 2. Phân tích khoảng cách (Gaps)
        gaps = {i: 40 for i in range(10)}
        for i in range(10):
            for idx, row in enumerate(data):
                if i in row:
                    gaps[i] = idx
                    break

        scores = {}
        for i in range(10):
            d_str = str(i)
            # Điểm nổ rơi: Ưu tiên cực cao cho số đã gãy 2-4 kỳ (nhịp hay rơi lại nhất)
            gap_score = 0
            if gaps[i] == 0: gap_score = 5  # Đánh số bệt
            elif 1 <= gaps[i] <= 3: gap_score = 15 # Nhịp rơi lý tưởng
            elif gaps[i] > 10: gap_score = -10 # Số quá 'lạnh' - bỏ qua
            
            # Điểm đối xứng (Mirror): Nếu số bóng của nó vừa về, cộng điểm mạnh
            mirror_score = 12 if gaps[shadow_map[i]] == 0 else 0
            
            # Tần suất có trọng số
            freq_score = local_freq.get(i, 0) * 4
            
            total = gap_score + mirror_score + freq_score
            
            # Risk Filter: Nếu nổ quá 4 lần trong 5 kỳ (bệt quá dày) -> Giảm điểm tránh bẫy
            if local_freq.get(i, 0) > 4: total *= 0.3
            
            scores[d_str] = total

        return [n for n, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    def _check_rhythm_flow(self, data):
        wins = 0
        if len(data) < 12: return 0
        for i in range(1, 6):
            # Tự kiểm tra 5 kỳ gần nhất xem thuật toán có khớp không
            pred = self._internal_scoring(data[i:i+12])[:3]
            if any(int(d) in data[i-1] for d in pred):
                wins += 1
        return (wins / 5) * 100

    def analyze(self, raw_history):
        data = self._parse_data(raw_history)
        if len(data) < 10:
            return {"m3": "---", "l4": "----", "decision": "⏳ ĐỢI DATA", "risk": {"score": 0}}

        top_digits = self._internal_scoring(data)
        flow_rate = self._check_rhythm_flow(data)
        
        # Đo Entropy (Độ ảo của nhà cái)
        recent_all = [d for r in data[:10] for d in r]
        counts = Counter(recent_all)
        entropy = -sum((c/50)*math.log2(c/50) for c in counts.values() if c > 0)
        risk_val = int(max(0, min(100, (3.32 - entropy) * 220)))

        m3 = "".join(top_digits[:3])
        l4 = "".join(top_digits[3:7])
        
        # Quyết định hành động
        decision = "🔥 VÀO LỆNH" if (flow_rate >= 60 and risk_val < 50) else "⏳ QUAN SÁT"
        
        logic_msg = f"Flow: {flow_rate}% | Risk: {risk_val}%. "
        if self.model:
            try:
                prompt = f"Data 5D: {data[:10]}. Đề xuất: {m3}. Nhịp đang trượt, hãy tìm bẫy quay xe."
                res = self.model.generate_content(prompt)
                logic_msg = res.text.strip()[:150]
            except: pass

        return {
            "m3": m3, "l4": l4, "decision": decision, "logic": logic_msg,
            "risk": {"score": risk_val}, "flow_rate": flow_rate, "success": True
        }
