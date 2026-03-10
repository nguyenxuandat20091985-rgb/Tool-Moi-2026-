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

# Giả định Config nếu anh chưa có file config.py
class Config:
    GEMINI_MODEL = "gemini-1.5-flash"
    MIN_HISTORY_LENGTH = 15

class TitanAI:
    """TITAN AI v6.0 - Bản nâng cấp tối thượng bắt nhịp 80%."""
    
    def __init__(self):
        # API Key của anh
        self.api_key = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
        self.model = None
        
        if GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
            except:
                self.model = None
    
    # --- 1. PIPELINE: CHUẨN HÓA DỮ LIỆU ---
    def _parse_data(self, raw):
        cleaned = []
        # Tách dòng nếu user dán cả đoạn văn bản
        lines = str(raw).split('\n') if isinstance(raw, str) else raw
        for item in lines:
            # Tìm đúng cụm 5 chữ số
            match = re.search(r'\d{5}', str(item))
            if match:
                cleaned.append([int(d) for d in match.group()])
        return cleaned
    
    # --- 2. BACKTEST: KIỂM TRA NHỊP THÔNG (Flow) ---
    def _check_rhythm_flow(self, data):
        wins = 0
        if len(data) < 20: return 0
        
        # Thử nghiệm dự đoán trên 5 kỳ gần nhất
        for i in range(1, 6):
            past_segment = data[i:i+15]
            actual_result = data[i-1]
            prediction = self._internal_scoring(past_segment)
            
            # Nếu 1 trong 3 số Top 1 nổ ở bất kỳ vị trí nào -> Thắng
            if any(int(d) in actual_result for d in prediction[:3]):
                wins += 1
        
        return (wins / 5) * 100
    
    # --- 3. LÕI CHỐT SỐ (INTERNAL SCORING ENGINE) ---
    def _internal_scoring(self, data):
        # Lấy 20 kỳ gần để soi
        recent = data[:20]
        
        # Phân tích bóng số (Shadow Numbers)
        shadow_map = {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}
        
        # Tần suất vị trí
        pos_freq = [Counter() for _ in range(5)]
        flat_data = []
        for idx, row in enumerate(recent):
            # Trọng số thời gian: Kỳ càng gần (idx thấp) điểm càng cao
            time_weight = 1.0 / (idx + 1)
            for p, v in enumerate(row):
                pos_freq[p][v] += time_weight
                flat_data.append(v)
        
        freq = Counter(flat_data)
        
        # Khoảng cách gãy (Gaps)
        gaps = {i: 40 for i in range(10)}
        for i in range(10):
            for idx, row in enumerate(data):
                if i in row:
                    gaps[i] = idx
                    break
        
        scores = {}
        for i in range(10):
            d_str = str(i)
            # Công thức đột phá: Vị trí(3) + Tần suất(2) + Chu kỳ gãy(5)
            s = (freq.get(i, 0) * 2.5) + (sum(pos_freq[p][i] for p in range(5)) * 4) + ((40-gaps[i]) * 6)
            
            # Cộng điểm bóng số: Nếu bóng của nó vừa về thì nó sắp về
            shadow_val = shadow_map[i]
            if gaps[shadow_val] == 0:
                s += 15 

            # Bộ lọc chống bệt ảo (Nếu số nổ > 12 lần trong 20 kỳ -> Risk cao)
            if gaps[i] == 0 and freq.get(i, 0) > 12:
                s *= 0.5
            
            scores[d_str] = s
        
        return [n for n, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    
    # --- 4. RỦI RO & ENTROPY ---
    def _calculate_risk(self, data):
        recent_all = [d for row in data[:15] for d in row]
        if not recent_all: return 100, 0
        
        counts = Counter(recent_all)
        total = len(recent_all)
        entropy = -sum((c/total) * math.log2(c/total) for c in counts.values() if c > 0)
        
        # Risk tỉ lệ nghịch với Entropy
        risk_score = int(max(0, min(100, (3.32 - entropy) * 200)))
        return risk_score, round(entropy, 2)
    
    # --- 5. HÀM TỔNG LỆNH (MAIN ANALYZE) ---
    def analyze(self, raw_history):
        data = self._parse_data(raw_history)
        
        if len(data) < Config.MIN_HISTORY_LENGTH:
            return {
                "m3": "---", "l4": "----",
                "decision": "⏳ CHỜ DATA",
                "logic": f"Cần >{Config.MIN_HISTORY_LENGTH} kỳ để bắt nhịp",
                "risk": {"score": 0}, "flow_rate": 0, "success": False
            }
        
        # Thực thi pipeline
        top_digits = self._internal_scoring(data)
        flow_rate = self._check_rhythm_flow(data)
        risk_val, entropy = self._calculate_risk(data)
        
        m3 = "".join(top_digits[:3])
        l4 = "".join(top_digits[3:7])
        
        # Quyết định dựa trên nhịp thông và rủi ro
        if flow_rate >= 80 and risk_val < 50:
            decision = "🔥 VÀO LỆNH (CỰC MẠNH)"
        elif flow_rate >= 60 and risk_val < 60:
            decision = "✅ VÀO LỆNH"
        else:
            decision = "⏳ QUAN SÁT"
        
        # AI Logic từ Gemini
        logic_msg = f"Flow: {flow_rate}% | Risk: {risk_val}% | Entropy: {entropy}."
        if self.model:
            try:
                context = str(data[:12])
                prompt = f"Data 5D: {context}. Đề xuất: {m3}. Flow: {flow_rate}%. Phân tích bẫy nhà cái (1 câu)."
                res = self.model.generate_content(prompt)
                logic_msg = res.text.strip()[:150]
            except: pass
        
        return {
            "m3": m3,
            "l4": l4,
            "decision": decision,
            "logic": logic_msg,
            "risk": {"score": risk_val, "level": "HIGH" if risk_val >= 60 else "OK"},
            "flow_rate": flow_rate,
            "house_warning": "🚨 Cầu ảo - Nhà cái đang giấu số!" if risk_val > 65 else "",
            "success": True
        }
