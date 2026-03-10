import numpy as np
import re
from collections import Counter
import math
from database import HistoryDB

class TitanAI:
    def __init__(self):
        self.db = HistoryDB()
        self.history = self.db.load_history()
        self.api_key = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"

    def add_and_save(self, raw_input):
        new_data = self._parse_data(raw_input)
        # Ưu tiên dữ liệu mới nhất lên đầu
        self.history = (new_data + self.history)[:100]
        self.db.save_history(self.history)
        return self.history

    def _parse_data(self, raw):
        cleaned = []
        lines = str(raw).split('\n') if isinstance(raw, str) else raw
        for item in lines:
            match = re.search(r'\d{5}', str(item))
            if match: cleaned.append([int(d) for d in match.group()])
        return cleaned

    def _detect_streak(self, data):
        """Thuật toán phát hiện Bệt (Streak Detection)."""
        if not data: return {}
        last_row = data[0]
        streaks = {}
        for pos in range(5):
            val = last_row[pos]
            count = 0
            for row in data:
                if row[pos] == val: count += 1
                else: break
            streaks[pos] = {"val": val, "count": count}
        return streaks

    def _internal_scoring(self, data):
        """Scoring v7.0: Ưu tiên bệt và nhịp đối xứng."""
        recent = data[:10]
        streaks = self._detect_streak(data)
        
        # Phân tích bóng số
        shadow_map = {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}
        
        scores = {}
        for i in range(10):
            d_str = str(i)
            score = 0
            
            # 1. Điểm Bệt (Nếu đang bệt thì cộng điểm để đánh tiếp bệt)
            for pos in range(5):
                if streaks[pos]["val"] == i and streaks[pos]["count"] >= 2:
                    score += 25 # Cộng điểm cực mạnh cho số đang bệt
            
            # 2. Điểm Rơi (Gap Analysis)
            gaps = 40
            for idx, row in enumerate(data):
                if i in row:
                    gaps = idx
                    break
            if 1 <= gaps <= 2: score += 15 # Nhịp rơi ngắn
            
            # 3. Điểm Bóng (Shadow)
            if any(row.count(shadow_map[i]) > 0 for row in data[:2]):
                score += 10
                
            scores[d_str] = score
            
        return [n for n, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    def analyze(self, raw_history=None):
        data = self.add_and_save(raw_history) if raw_history else self.history
        if len(data) < 5:
            return {"success": False, "logic": "Đang nạp dữ liệu..."}

        top_digits = self._internal_scoring(data)
        streaks = self._detect_streak(data)
        
        # Tổng hợp logic bệt
        streak_logic = [f"Hàng {p}:{s['val']}({s['count']}kỳ)" for p, s in streaks.items() if s['count'] >= 2]
        
        m3 = "".join(top_digits[:3])
        l4 = "".join(top_digits[3:7])
        
        return {
            "m3": m3,
            "l4": l4,
            "decision": "🔥 THEO BỆT" if streak_logic else "✅ ĐÁNH NHẸ",
            "logic": " | ".join(streak_logic) if streak_logic else "Cầu đang chuyển nhịp.",
            "flow_rate": 85 if streak_logic else 60,
            "success": True
        }
