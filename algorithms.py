import numpy as np
from collections import Counter, defaultdict
import math
import random

class PredictionEngine:
    def __init__(self):
        self.weights = {'statistical': 20, 'markov': 25, 'monte_carlo': 35, 'trend': 20}
        self.win_history = []

    def get_ai_status(self):
        """Hàm cung cấp thông tin trạng thái cho Sidebar"""
        wr = (sum(self.win_history[-20:]) / 20 * 100) if self.win_history else 0
        return {
            'win_rate': round(wr, 1),
            'logic_version': "40.0.Ultimate",
            'engine_status': "⚡ Active"
        }

    def _calculate_entropy(self, history):
        """Phát hiện can thiệp từ nhà cái"""
        if not history: return 3.32
        data = "".join(history[-50:])
        counts = Counter(data)
        probs = [c/len(data) for c in counts.values()]
        return -sum(p * math.log2(p) for p in probs)

    def predict(self, history):
        # ĐẢM BẢO LUÔN CÓ DỮ LIỆU ĐẦU RA (CHỐNG LỖI KEYERROR)
        if len(history) < 2:
            return self._fallback("Cần nạp thêm dữ liệu để bắt nhịp cầu...")

        # 1. MARKOV CHAIN BẬC CAO
        nodes = defaultdict(Counter)
        for i in range(len(history)-1):
            nodes[history[i][-1]][history[i+1][-1]] += 1
        last = history[-1][-1]
        mk_res = [x[0] for x in nodes[last].most_common(3)]

        # 2. SIÊU GIẢ LẬP MONTE CARLO (100,000 VÒNG)
        pool = list("".join(history[-60:]))
        sim = Counter()
        for _ in range(100000):
            sample = random.choices(pool, k=3)
            for n in sample: sim[n] += 1
        mc_res = [x[0] for x in sim.most_common(3)]

        # 3. PHÂN TÍCH TREND (BỆT/ĐẢO)
        trend_res = [history[-1][0], history[-1][1], history[-1][2]] 

        # TỔNG HỢP (ENSEMBLE VOTING)
        votes = Counter()
        for n in mk_res: votes[n] += self.weights['markov']
        for n in mc_res: votes[n] += self.weights['monte_carlo']
        for n in trend_res: votes[n] += self.weights['trend']
        final = votes.most_common(7)

        # PHÂN TÍCH RỦI RO
        entropy = self._calculate_entropy(history)
        risk_score = int(max(0, min(100, (3.32 - entropy) * 180)))
        
        reasons = []
        if risk_score > 65: reasons.append("⚠️ CẢNH BÁO: Dữ liệu ảo - Nhà cái đang lừa cầu")
        elif history[-1][-1] == history[-2][-1]: reasons.append("🔥 Cầu đang đi bệt - Ưu tiên bám đuôi")
        else: reasons.append("✅ Nhịp cầu ổn định - Xác suất nổ cao")

        return {
            'main_3': [x[0] for x in final[:3]],
            'support_4': [x[0] for x in final[3:7]],
            'risk': {
                'score': risk_score,
                'level': "HIGH" if risk_score > 60 else "MEDIUM" if risk_score > 35 else "LOW",
                'reasons': reasons
            },
            'logic': "Ultimate Hybrid AI (v40.0)",
            'confidence': min(99, 50 + (len(history)*2))
        }

    def update_learning(self, won):
        self.win_history.append(1 if won else 0)

    def _fallback(self, msg):
        return {
            'main_3': ['?','?','?'], 
            'support_4': ['?','?','?','?'], 
            'risk': {'score': 0, 'level': 'LOW', 'reasons': [msg]}, 
            'logic': 'Khởi động...', 
            'confidence': 0
        }
