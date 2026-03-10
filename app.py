import numpy as np
import pandas as pd
import re
import google.generativeai as genai
from collections import Counter
import math

class TitanAI:
    def __init__(self):
        # Cấu hình bộ não Gemini 1.5 Pro để giải mã bẫy nhịp
        self.api_key = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except:
            self.model = None

    # --- 1. DATA CLEANING (Chuẩn hóa dữ liệu) ---
    def _parse_data(self, raw_input):
        cleaned = []
        # Xử lý cả dạng list hoặc chuỗi văn bản dán vào
        lines = str(raw_input).split('\n') if isinstance(raw_input, str) else raw_input
        for item in lines:
            match = re.search(r'\d{5}', str(item))
            if match:
                cleaned.append([int(d) for d in match.group()])
        return cleaned

    # --- 2. FREQUENCY & POSITION ANALYSIS (Tần suất & Vị trí) ---
    def _analyze_core(self, data):
        # Tần suất tổng hợp 30 kỳ gần nhất
        recent_30 = data[:30]
        flat_data = [d for row in recent_30 for d in row]
        freq = Counter(flat_data)
        
        # Tần suất theo từng vị trí (Hàng vạn, ngàn, trăm, chục, đơn vị)
        pos_freq = [Counter() for _ in range(5)]
        for row in recent_30:
            for pos, val in enumerate(row):
                pos_freq[pos][val] += 1
                
        # Phân tích Chu kỳ/Khoảng cách (Gaps/Delay)
        gaps = {i: 50 for i in range(10)}
        for i in range(10):
            for idx, row in enumerate(data):
                if i in row:
                    gaps[i] = idx
                    break
        return freq, pos_freq, gaps

    # --- 3. ENTROPY CHECK (Đo độ ngẫu nhiên & Bẫy nhà cái) ---
    def _calculate_entropy(self, data):
        # Lấy 15 kỳ gần nhất để đo độ hỗn loạn
        recent_all = [d for row in data[:15] for d in row]
        if not recent_all: return 3.32
        counts = Counter(recent_all)
        total = len(recent_all)
        # Entropy lý tưởng là ~3.32. Nếu thấp hơn 2.5 là nhà cái đang ép số bệt.
        entropy = -sum((c/total) * math.log2(c/total) for c in counts.values() if c > 0)
        return round(entropy, 2)

    # --- 4. SCORING ENGINE (Bộ máy chấm điểm đa tầng) ---
    def _scoring_engine(self, freq, pos_freq, gaps, entropy):
        scores = {}
        for i in range(10):
            d_str = str(i)
            # Điểm tần suất nóng (30%)
            s_freq = freq.get(i, 0) * 1.5
            # Điểm bao phủ vị trí (30%) - Số xuất hiện ở nhiều hàng khác nhau
            s_pos = sum(1 for p in range(5) if pos_freq[p][i] > 0) * 2.0
            # Điểm chu kỳ chín (40%) - Số sắp nổ theo nhịp gãy
            s_gap = (50 - gaps[i]) * 1.8
            
            total_score = s_freq + s_pos + s_gap
            
            # Risk Filter: Giảm điểm nếu số bệt quá sâu ở 1 vị trí (Dễ bị nhà cái treo)
            for p in range(5):
                if pos_freq[p][i] >= 4: 
                    total_score *= 0.6 
            
            scores[d_str] = total_score
        return scores

    # --- 5. MAIN PIPELINE (Luồng thực thi chính) ---
    def analyze(self, raw_history):
        data = self._parse_data(raw_history)
        if len(data) < 15:
            return {
                "m3": "---", "l4": "----", 
                "logic": "⚠️ Cần tối thiểu 15 kỳ để phân tích nhịp.", 
                "risk": {"score": 0, "level": "LOW"}
            }

        # Thực thi Pipeline
        freq, pos_freq, gaps = self._analyze_core(data)
        entropy = self._calculate_entropy(data)
        scores = self._scoring_engine(freq, pos_freq, gaps, entropy)
        
        # Xếp hạng số mạnh nhất (Ranking)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_digits = [n for n, s in ranked]
        
        # Trả về kết quả khớp với giao diện app.py
        m3 = "".join(top_digits[:3])  # 3 Số tinh chủ lực
        l4 = "".join(top_digits[3:7]) # 4 Số lót giữ vốn
        
        # Tính toán mức độ rủi ro (Risk)
        # Nếu Entropy thấp (cầu ảo) hoặc nhịp quá đều -> Risk tăng
        risk_val = int(max(0, min(100, (3.32 - entropy) * 160)))
        
        # Quyết định hành động
        decision = "VÀO LỆNH" if risk_val < 55 else "QUAN SÁT"
        
        # GỌI GEMINI GIẢI MÃ NHỊP CẦU
        logic_msg = f"Phân tích Entropy: {entropy}. Nhịp nổ mạnh nhất: {top_digits[0]}"
        if self.model:
            try:
                # Gửi 15 kỳ số thực tế cho AI soi bẫy
                context = str(data[:15])
                prompt = f"""
                Dữ liệu 5D Bet: {context}. 
                Dàn gợi ý: {m3} (lót {l4}). 
                Entropy: {entropy}.
                Hãy phân tích bẫy nhà cái (đảo cầu, giấu số) và chốt 1 câu logic ngắn gọn.
                """
                response = self.model.generate_content(prompt)
                logic_msg = response.text.strip()[:150]
            except:
                pass

        return {
            "m3": m3,
            "l4": l4,
            "decision": decision,
            "logic": logic_msg,
            "risk": {"score": risk_val, "level": "HIGH" if risk_val > 60 else "OK"},
            "house_warning": "🚨 Cầu đang bị ép số (Entropy thấp)!" if entropy < 2.6 else ""
        }
