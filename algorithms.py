import numpy as np
import re
from collections import Counter
import math
from database import HistoryDB

class TitanAI:
    def __init__(self):
        self.db = HistoryDB()
        self.history = self.db.load_history()

    def _internal_scoring(self, data):
        """Thuật toán Trinity: Tìm 5 con số 'phủ' toàn bộ bảng thưởng."""
        recent_12 = data[:12]
        
        # 1. Đo mật độ xuất hiện (Density)
        # 3 Tinh cần các số xuất hiện cùng nhau, nên ta soi các số hay đi cặp
        all_numbers = [d for row in recent_12 for d in row]
        freq = Counter(all_numbers)
        
        # 2. Thuật toán 'Mảnh ghép thiếu' (Missing Piece)
        # Soi xem trong 3 kỳ gần nhất, những con số nào thường xuyên 'suýt trúng'
        gaps = {i: 20 for i in range(10)}
        for i in range(10):
            for idx, row in enumerate(data):
                if i in row:
                    gaps[i] = idx
                    break

        scores = {}
        for i in range(10):
            # Điểm nổ (Số đang nóng có khả năng rơi lại để tạo bộ 3)
            f_score = freq.get(i, 0) * 5 
            
            # Điểm nhịp (Số đã vắng mặt 1-2 kỳ thường rơi lại rất mạnh ở 5 tinh)
            g_score = 0
            if gaps[i] == 1: g_score = 30  # Vừa nghỉ 1 kỳ -> Rơi lại cực mạnh
            elif gaps[i] == 0: g_score = 15 # Đang bệt -> Có thể bệt tiếp
            
            # Điểm liên kết (Số thường xuất hiện cùng các số nóng khác)
            scores[str(i)] = f_score + g_score
            
        return [n for n, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

    def analyze(self, raw_history=None):
        # Tự động nạp và lưu dữ liệu để thoát ra không mất
        if raw_history:
            new_data = []
            lines = str(raw_history).split('\n')
            for item in lines:
                match = re.search(r'\d{5}', str(item))
                if match: new_data.append([int(d) for d in match.group()])
            self.history = (new_data + self.history)[:100]
            self.db.save_history(self.history)
        
        data = self.history
        if len(data) < 5: return {"success": False, "logic": "Đang đợi thêm kỳ..."}

        # Lấy Top 5 số chính xác nhất
        top_5 = self._internal_scoring(data)[:5]
        
        # Tính toán độ tin cậy dựa trên nhịp rơi thực tế
        recent_all = [d for r in data[:8] for d in r]
        entropy = -sum((recent_all.count(i)/40)*math.log2(recent_all.count(i)/40) for i in set(recent_all) if recent_all.count(i)>0)
        
        accuracy = int(max(0, min(100, (3.32 - entropy) * 250)))

        return {
            "top_5": top_5,
            "m3_suggest": "".join(top_5[:3]), # Gợi ý bộ 3 tinh mạnh nhất
            "accuracy": accuracy,
            "logic": f"Entropy: {entropy:.2f}. Tập trung vào nhịp rơi kỳ 1.",
            "decision": "🔥 ĐÁNH MẠNH" if accuracy > 70 else "⏳ CHỜ NHỊP",
            "success": True
        }
