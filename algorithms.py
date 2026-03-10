# ==============================================================================
# TITAN AI v6.0 - Gemini Powered AI
# ==============================================================================

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

from config import Config

class TitanAI:
    """Gemini-powered AI prediction engine."""
    
    def __init__(self):
        """Initialize AI with Gemini."""
        # Gemini API Key
        self.api_key = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
        self.model = None
        
        if GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
            except Exception as e:
                self.model = None
    
    def _parse_data(self, raw):
        """Parse and normalize data."""
        cleaned = []
        lines = str(raw).split('\n') if isinstance(raw, str) else raw
        
        for item in lines:
            match = re.search(r'\d{5}', str(item))
            if match:
                cleaned.append([int(d) for d in match.group()])
        
        return cleaned
    
    def _check_rhythm_flow(self, data):
        """Backtest rhythm flow in recent 5 periods."""
        wins = 0
        if len(data) < 20:
            return 0
        
        for i in range(1, 6):
            past_segment = data[i:i+15]
            actual_result = data[i-1]
            prediction = self._internal_scoring(past_segment)
            
            if any(int(d) in actual_result for d in prediction[:3]):
                wins += 1
        
        return (wins / 5) * 100
    
    def _internal_scoring(self, data):
        """Internal scoring engine."""
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
            s = (freq.get(i, 0) * 2) + (sum(1 for p in range(5) if pos_freq[p][i]>0) * 3) + ((35-gaps[i]) * 5)
            
            # Anti-fake streak filter
            if gaps[i] == 0 and freq.get(i, 0) > 12:
                s *= 0.6
            
            scores[d_str] = s
        
        return [n for n, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    
    def _calculate_risk(self, data):
        """Calculate entropy and risk."""
        recent_all = [d for row in data[:15] for d in row]
        
        if not recent_all:
            return 100, 0
        
        counts = Counter(recent_all)
        total = len(recent_all)
        
        entropy = -sum((c/total) * math.log2(c/total) for c in counts.values() if c > 0)
        risk_score = int(max(0, min(100, (3.32 - entropy) * 190)))
        
        return risk_score, round(entropy, 2)
    
    def _gemini_analysis(self, data, m3, flow_rate):
        """Get Gemini AI analysis."""
        if not self.model:
            return None
        
        try:
            context = str(data[:12])
            prompt = f"Data 5D: {context}. Tool đề xuất: {m3}. Flow: {flow_rate}%. Phân tích bẫy nhà cái ngắn gọn (1 câu)."
            res = self.model.generate_content(prompt)
            return res.text.strip()[:150]
        except:
            return None
    
    def analyze(self, raw_history):
        """Main analysis function."""
        data = self._parse_data(raw_history)
        
        if len(data) < Config.MIN_HISTORY_LENGTH:
            return {
                "m3": "---",
                "l4": "----",
                "decision": "⏳ CHỜ DATA",
                "logic": f"Cần >{Config.MIN_HISTORY_LENGTH} kỳ để bắt nhịp",
                "risk": {"score": 0},
                "flow_rate": 0,
                "house_warning": "",
                "success": False
            }
        
        # Execute pipeline
        top_digits = self._internal_scoring(data)
        flow_rate = self._check_rhythm_flow(data)
        risk_val, entropy = self._calculate_risk(data)
        
        m3 = "".join(top_digits[:3])
        l4 = "".join(top_digits[3:7])
        
        # Decision based on Flow Rate
        decision = "🔥 VÀO LỆNH" if (flow_rate >= 60 and risk_val < 55) else "⏳ QUAN SÁT"
        
        # AI Logic
        logic_msg = f"Flow: {flow_rate}% | Entropy: {entropy}. Nhịp đang {decision}."
        
        if self.model:
            gemini_logic = self._gemini_analysis(data, m3, flow_rate)
            if gemini_logic:
                logic_msg = gemini_logic
        
        # House warning
        house_warning = "🚨 Cảnh báo bẫy nhịp đảo!" if risk_val > 65 else ""
        
        return {
            "m3": m3,
            "l4": l4,
            "decision": decision,
            "logic": logic_msg,
            "risk": {"score": risk_val, "level": "HIGH" if risk_val >= 50 else "MEDIUM" if risk_val >= 25 else "OK"},
            "flow_rate": flow_rate,
            "house_warning": house_warning,
            "success": True
        }