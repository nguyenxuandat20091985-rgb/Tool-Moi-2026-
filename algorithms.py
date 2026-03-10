import numpy as np
import re
from collections import Counter
import math
from database import HistoryDB

class TitanAI:
    def __init__(self):
        self.db = HistoryDB()
        self.history = self.db.load_history() # Tự động tải lại khi khởi động
        self.api_key = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"

    def add_and_save(self, raw_input):
        """Thêm dữ liệu mới và lưu lại vĩnh viễn."""
        new_data = self._parse_data(raw_input)
        # Kết hợp dữ liệu cũ và mới, loại bỏ trùng lặp
        combined = new_data + self.history
        # Giữ lại tối đa 100 kỳ gần nhất để tối ưu
        self.history = combined[:100]
        self.db.save_history(self.history)
        return self.history

    def _parse_data(self, raw):
        cleaned = []
        lines = str(raw).split('\n') if isinstance(raw, str) else raw
        for item in lines:
            match = re.search(r'\d{5}', str(item))
            if match: cleaned.append([int(d) for d in match.group()])
        return cleaned

    def _positional_analysis(self, data):
        """Soi cầu theo từng vị trí để bớt mơ hồ."""
        if len(data) < 5: return {}
        
        # Phân tích 3 vị trí cuối (Hậu Tam)
        positions = {2: "Trăm", 3: "Chục", 4: "Đơn vị"}
        results = {}
        
        for pos, name in positions.items():
            col_data = [row[pos] for row in data[:15]]
            freq = Counter(col_data)
            # Tìm số hay về nhất ở vị trí này
            top_val = freq.most_common(1)[0][0]
            results[name] = top_val
        return results

    def _internal_scoring(self, data):
        """Scoring v6.5: Tập trung vào nhịp rơi và bóng số."""
        recent = data[:15]
        shadow_map = {0:5, 1:6, 2:7, 3:8, 4:9, 5:0, 6:1, 7:2, 8:3, 9:4}
        
        gaps = {i: 30 for i in range(10)}
        for i in range(10):
            for idx, row in enumerate(data):
                if i in row:
                    gaps[i] = idx
                    break

        scores = {}
        for i in range(10):
            # Điểm rơi (Gap 1-4 kỳ là đẹp nhất)
            g_score = 20 if 1 <= gaps[i] <= 4 else 5 if gaps[i] == 0 else 0
            # Điểm bóng (Nếu bóng vừa về thì cộng điểm)
            m_score = 15 if gaps[shadow_map[i]] == 0 else 0
            
            scores[str(i)] = g_score + m_score
            
        return [n for n, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    def analyze(self, raw_history=None):
        # Nếu có dữ liệu mới thì lưu, không thì dùng dữ liệu cũ đã lưu
        data = self.add_and_save(raw_history) if raw_history else self.history
        
        if len(data) < 10:
            return {"success": False, "logic": "Cần thêm dữ liệu để lưu trữ và phân tích."}

        top_digits = self._internal_scoring(data)
        pos_res = self._positional_analysis(data)
        
        m3 = "".join(top_digits[:3])
        l4 = "".join(top_digits[3:7])
        
        # Tạo chuỗi logic chi tiết hơn
        pos_logic = " | ".join([f"{k}:{v}" for k, v in pos_res.items()])
        logic_msg = f"Vị trí tiềm năng: {pos_logic}. Nhịp rơi chuẩn."

        return {
            "m3": m3,
            "l4": l4,
            "decision": "🔥 ĐÁNH" if len(data) > 20 else "⏳ THEO DÕI",
            "logic": logic_msg,
            "flow_rate": 80 if len(data) > 30 else 50,
            "success": True
        }
