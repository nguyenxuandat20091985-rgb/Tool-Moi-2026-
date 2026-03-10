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
        """Hệ thống chấm điểm đa tầng v6.1 - Tập trung vào nhịp biến động."""
        recent = data[:15] # Chỉ tập trung 15 kỳ gần nhất để tránh nhiễu
        
        # 1. Tần suất trọng số (Exponential Decay)
        # Số càng mới về điểm càng cao gấp nhiều lần số cũ
        pos_scores = [Counter() for _ in range(5)]
        all_freq = Counter()
        
        for idx, row in enumerate(recent):
            weight = math.exp(-idx / 5) # Trọng số giảm dần theo thời gian
            for p, v in enumerate(row):
                pos_scores[p][v] += weight
                all_freq[v] += weight

        # 2. Phân tích nhịp 'Cầu gãy' (Gap Analysis)
        gaps = {i: 30 for i in range(10)}
        for i in range(10):
            for idx, row in enumerate(data):
                if i in row:
                    gaps[i] = idx
                    break

        # 3. Thuật toán Shadow & Hot-Zone (Vùng nóng)
        final_scores = {}
        shadow_map = {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}
        
        for i in range(10):
            d_str = str(i)
            # Điểm cơ bản từ tần suất tổng và vị trí
            base_s = (all_freq[i] * 3) + (sum(pos_scores[p][i] for p in range(5)) * 5)
            
            # Điểm chu kỳ: Ưu tiên số vừa chớm về lại sau chu kỳ gãy dài
            gap_s = (30 - gaps[i]) * 4
            if 3 <= gaps[i] <= 7: gap_s *= 1.5 # Vùng nổ tiềm năng
            
            total = base_s + gap_s
            
            # CỘNG ĐIỂM BÓNG (Shadow Logic)
            if gaps[shadow_map[i]] == 0: total += 10
            
            # BỘ LỌC CHỐNG BÃO: Nếu số nổ quá 3 lần trong 5 kỳ gần -> Giảm điểm (tránh bẫy ảo)
            recent_5 = [d for r in data[:5] for d in r]
            if recent_5.count(i) > 3: total *= 0.4
            
            final_scores[d_str] = total

        return [n for n, s in sorted(final_scores.items(), key=lambda x: x[1], reverse=True)]

    def _check_rhythm_flow(self, data):
        wins = 0
        if len(data) < 15: return 0
        for i in range(1, 6):
            if any(int(d) in data[i-1] for d in self._internal_scoring(data[i:i+15])[:3]):
                wins += 1
        return (wins / 5) * 100

    def analyze(self, raw_history):
        data = self._parse_data(raw_history)
        if len(data) < 10:
            return {"m3": "---", "l4": "----", "decision": "⏳ ĐỢI DATA", "risk": {"score": 0}}

        top_digits = self._internal_scoring(data)
        flow_rate = self._check_rhythm_flow(data)
        
        # Đo Entropy (Độ nhiễu của nhà cái)
        recent_all = [d for r in data[:12] for d in r]
        counts = Counter(recent_all)
        entropy = -sum((c/60)*math.log2(c/60) for c in counts.values() if c>0)
        risk_val = int(max(0, min(100, (3.32 - entropy) * 200)))

        m3 = "".join(top_digits[:3])
        l4 = "".join(top_digits[3:7])
        
        # Logic chốt quyết định
        decision = "🔥 VÀO LỆNH" if (flow_rate >= 60 and risk_val < 55) else "⏳ QUAN SÁT"
        
        logic_msg = f"Nhịp Flow: {flow_rate}% | Entropy: {entropy:.2f}. "
        if self.model:
            try:
                res = self.model.generate_content(f"Data 5D: {data[:10]}. Đề xuất: {m3}. Phân tích nhịp cầu gãy và bẫy ảo.")
                logic_msg = res.text.strip()[:150]
            except: pass

        return {
            "m3": m3, "l4": l4, "decision": decision, "logic": logic_msg,
            "risk": {"score": risk_val}, "flow_rate": flow_rate, "success": True
        }
