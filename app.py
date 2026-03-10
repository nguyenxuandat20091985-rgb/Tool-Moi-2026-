import numpy as np
import pandas as pd
import re
import google.generativeai as genai
from collections import Counter
import math

class TitanAI:
    def __init__(self):
        # Bộ não Gemini 1.5 Pro - Giải mã nhịp bão và bẫy số
        self.api_key = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except:
            self.model = None

    # --- 1. PIPELINE: CHUẨN HÓA DỮ LIỆU ---
    def _parse_data(self, raw):
        cleaned = []
        lines = str(raw).split('\n') if isinstance(raw, str) else raw
        for item in lines:
            match = re.search(r'\d{5}', str(item))
            if match:
                cleaned.append([int(d) for d in match.group()])
        return cleaned

    # --- 2. KIỂM TRA NHỊP THÔNG (BACKTEST REAL-TIME) ---
    def _check_rhythm_flow(self, data):
        """Tự soi lại chính mình trong 5 kỳ gần nhất để báo tỷ lệ 80%"""
        wins = 0
        if len(data) < 20: return 0
        for i in range(1, 6):
            # Lấy phân đoạn quá khứ để dự đoán kỳ đã về
            past_segment = data[i:i+15]
            actual_result = data[i-1]
            # Dự đoán thử nghiệm
            prediction = self._internal_scoring(past_segment)
            # Nếu trúng ít nhất 1 số trong Top 3 -> Tính là 1 trận thắng
            if any(int(d) in actual_result for d in prediction[:3]):
                wins += 1
        return (wins / 5) * 100

    # --- 3. LÕI CHẤM ĐIỂM (INTERNAL SCORING ENGINE) ---
    def _internal_scoring(self, data):
        recent = data[:20]
        flat_data = [d for row in recent for d in row]
        freq = Counter(flat_data)
        
        pos_freq = [Counter() for _ in range(5)]
        for row in recent:
            for p, v in enumerate(row):
                pos_freq[p][v] += 1
                
        gaps = {i: 35 for i in range(10)}
        for i in range(10):
            for idx, row in enumerate(data):
                if i in row:
                    gaps[i] = idx
                    break
        
        scores = {}
        for i in range(10):
            d_str = str(i)
            # Công thức tối ưu: Vị trí(3) + Tần suất(2) + Chu kỳ gãy(5)
            s = (freq.get(i, 0) * 2) + (sum(1 for p in range(5) if pos_freq[p][i]>0) * 3) + ((35-gaps[i]) * 5)
            
            # Bộ lọc chống bệt ảo
            if gaps[i] == 0 and freq.get(i) > 12:
                s *= 0.6
            scores[d_str] = s
            
        return [n for n, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    # --- 4. ĐO ĐỘ HỖN LOẠN & RỦI RO (ENTROPY) ---
    def _calculate_risk(self, data):
        recent_all = [d for row in data[:15] for d in row]
        if not recent_all: return 100, 0
        counts = Counter(recent_all)
        total = len(recent_all)
        entropy = -sum((c/total) * math.log2(c/total) for c in counts.values() if c > 0)
        # Risk tăng khi Entropy thấp (cầu bị ép, bệt nặng)
        risk_score = int(max(0, min(100, (3.32 - entropy) * 190)))
        return risk_score, round(entropy, 2)

    # --- 5. HÀM TỔNG LỆNH (MAIN ANALYZE) ---
    def analyze(self, raw_history):
        data = self._parse_data(raw_history)
        if len(data) < 20:
            return {"m3": "---", "l4": "----", "logic": "Cần >20 kỳ để bắt nhịp", "risk": {"score": 0}}

        # Pipeline thực thi
        top_digits = self._internal_scoring(data)
        flow_rate = self._check_rhythm_flow(data)
        risk_val, entropy = self._calculate_risk(data)
        
        m3 = "".join(top_digits[:3])
        l4 = "".join(top_digits[3:7])
        
        # Quyết định dựa trên Nhịp thông (Flow Rate)
        decision = "🔥 VÀO LỆNH" if (flow_rate >= 60 and risk_val < 55) else "⏳ QUAN SÁT"
        
        # Giải mã AI Logic
        logic_msg = f"Flow: {flow_rate}% | Entropy: {entropy}. Nhịp đang {decision}."
        if self.model:
            try:
                context = str(data[:12])
                prompt = f"Data 5D: {context}. Tool đề xuất: {m3}. Flow: {flow_rate}%. Phân tích bẫy nhà cái ngắn gọn."
                res = self.model.generate_content(prompt)
                logic_msg = res.text.strip()[:150]
            except: pass

        return {
            "m3": m3,
            "l4": l4,
            "decision": decision,
            "logic": logic_msg,
            "risk": {"score": risk_val},
            "flow_rate": flow_rate,
            "house_warning": "🚨 Cảnh báo bẫy nhịp đảo!" if risk_val > 65 else ""
        }
