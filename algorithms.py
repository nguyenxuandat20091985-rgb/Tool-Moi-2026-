import numpy as np
import pandas as pd
import re
from collections import Counter, defaultdict
from itertools import combinations
import google.generativeai as genai
from config import Config

class TitanAI:
    def __init__(self):
        self.api_key = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except: self.model = None

    # --- 1. DATA CLEANING ---
    def _parse_results(self, raw_data):
        cleaned = []
        for item in raw_data:
            match = re.search(r'\d{5}', str(item))
            if match: cleaned.append([int(d) for d in match.group()])
        return cleaned

    # --- 2. FREQUENCY & POSITION ANALYSIS ---
    def _calculate_frequency(self, data):
        all_digits = [d for sub in data[:30] for d in sub] # Ưu tiên 30 kỳ gần
        freq = Counter(all_digits)
        scores = {str(i): freq.get(i, 0) for i in range(10)}
        return scores

    # --- 3. PATTERN DETECTION (Repeat, Mirror, Pair) ---
    def _detect_patterns(self, data):
        patterns = []
        recent = data[:10]
        # Phát hiện số lặp (Repeat)
        for pos in range(5):
            if recent[0][pos] == recent[1][pos]:
                patterns.append(f"Lặp vị trí {pos}")
        return patterns

    # --- 4. CYCLE & DELAY ANALYSIS ---
    def _calculate_cycle(self, data):
        gaps = {i: 30 for i in range(10)}
        for i in range(10):
            for idx, row in enumerate(data[:30]):
                if i in row:
                    gaps[i] = idx
                    break
        return gaps

    # --- 5. RANDOMNESS / ENTROPY CHECK ---
    def _calculate_entropy(self, data):
        all_nums = [d for sub in data[:20] for d in sub]
        counts = np.array(list(Counter(all_nums).values()))
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs))
        return entropy # Càng thấp nghĩa là cầu càng bệt/ảo

    # --- 6. MOMENTUM ANALYSIS (Xu hướng gần) ---
    def _recent_trend(self, data):
        trend_score = Counter()
        for row in data[:5]: # Chỉ lấy 5 kỳ gần nhất
            for d in row:
                trend_score[str(d)] += 2
        return trend_score

    # --- 7. SCORING ENGINE & RISK FILTER ---
    def _scoring_engine(self, freq, gaps, trend, entropy):
        final_scores = {}
        for i in range(10):
            d = str(i)
            # Công thức tổng hợp điểm
            f_score = freq.get(d, 0) * 1.5
            g_score = (30 - gaps.get(i, 30)) * 2.0
            t_score = trend.get(d, 0) * 3.0
            
            total = f_score + g_score + t_score
            
            # Risk Filter: Loại số quá nóng (Overfit)
            if gaps.get(i) == 0 and freq.get(d) > 15:
                total *= 0.5 # Giảm điểm số đang bệt quá dày
                
            final_scores[d] = total
        return final_scores

    # --- 8. FINAL SELECTION & PIPELINE ---
    def analyze(self, raw_history):
        # MAIN PIPELINE
        clean_data = self._parse_results(raw_history)
        if not clean_data: return self._fallback()

        freq = self._calculate_frequency(clean_data)
        gaps = self._calculate_cycle(clean_data)
        trend = self._recent_trend(clean_data)
        entropy = self._calculate_entropy(clean_data)
        
        # Scoring
        scores = self._scoring_engine(freq, gaps, trend, entropy)
        
        # Ranking
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_digits = [n for n, s in ranked]
        
        m3 = "".join(top_digits[:3])
        l4 = "".join(top_digits[3:7])
        
        # Risk Assessment
        risk_score = int((1 - (entropy / 3.32)) * 100) # Chuẩn hóa entropy sang % risk

        # Machine Learning (Gemini hỗ trợ)
        logic = f"Nhịp gãy {gaps.get(int(top_digits[0]))} kỳ. Entropy: {entropy:.2f}"
        if self.model:
            try:
                prompt = f"Data: {clean_data[:10]}. Suggest: {m3}. Phân tích cầu bệt và bẫy."
                res = self.model.generate_content(prompt)
                logic = res.text[:100]
            except: pass

        return {
            "m3": m3,
            "l4": l4,
            "decision": "ĐÁNH" if risk_score < 60 else "QUAN SÁT",
            "logic": logic,
            "risk": {"score": risk_score},
            "house_warning": "Cầu quá đều - Dễ bị soi" if entropy < 2.5 else ""
        }

    def _fallback(self):
        return {"m3": "---", "l4": "----", "logic": "Lỗi dữ liệu", "risk": {"score": 0}}
