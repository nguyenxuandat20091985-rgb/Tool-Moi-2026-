import numpy as np
from collections import Counter, defaultdict
import math
import random

class PredictionEngine:
    def __init__(self):
        self.weights = {'freq': 20, 'markov': 25, 'monte': 30, 'pattern': 25}
        self.win_log = []

    def calculate_entropy(self, history):
        if not history: return 3.32 # Giá trị entropy lý tưởng cho 10 chữ số
        all_chars = "".join(history[-40:])
        counts = Counter(all_chars)
        probs = [c/len(all_chars) for c in counts.values()]
        return -sum(p * math.log2(p) for p in probs)

    def predict(self, history):
        # SỬA LỖI TRIỆT ĐỂ: Luôn đảm bảo trả về đủ 100% các Key cho dù dữ liệu ít
        if len(history) < 3:
            return self._fallback("⚠️ Hệ thống cần tối thiểu 3 kỳ dữ liệu để khởi động")

        # 1. Tầng Markov (Dự báo chuỗi liên kết)
        markov_nodes = defaultdict(Counter)
        for i in range(len(history)-1):
            markov_nodes[history[i][-1]][history[i+1][-1]] += 1
        last_digit = history[-1][-1]
        m_res = [x[0] for x in markov_nodes[last_digit].most_common(3)]

        # 2. Tầng Monte Carlo (Giả lập 20k kịch bản)
        pool = list("".join(history[-60:]))
        sim = Counter()
        for _ in range(20000):
            sample = random.choices(pool, k=3)
            for n in sample: sim[n] += 1
        mc_res = [x[0] for x in sim.most_common(3)]

        # 3. Tầng Pattern (Nhận diện Bệt/Đảo)
        p_res = [history[-1][0], history[-1][2], history[-2][4] if len(history)>1 else '5']

        # Bỏ phiếu Ensemble
        votes = Counter()
        for n in m_res: votes[n] += self.weights['markov']
        for n in mc_res: votes[n] += self.weights['monte']
        for n in p_res: votes[n] += self.weights['pattern']
        final_sorted = votes.most_common(7)

        # PHÂN TÍCH CẦU LỪA (ENTROPY)
        entropy = self.calculate_entropy(history)
        risk_score = int(max(0, min(100, (3.32 - entropy) * 150)))
        
        # Nhận diện trạng thái cầu
        status_msg = []
        if history[-1][-1] == history[-2][-1] if len(history)>1 else False:
            status_msg.append("🔥 CẦU BỆT ĐANG HÌNH THÀNH")
        if risk_score > 60:
            status_msg.append("⚠️ CẢNH BÁO: NHÀ CÁI ĐANG ĐIỀU CẦU (ẢO)")
        
        return {
            'main_3': [x[0] for x in final_sorted[:3]],
            'support_4': [x[0] for x in final_sorted[3:7]],
            'risk': {
                'score': risk_score, 
                'level': "HIGH" if risk_score > 55 else "MEDIUM" if risk_score > 30 else "LOW",
                'reasons': status_msg if status_msg else ["Cầu đang đi đúng nhịp toán học"]
            },
            'logic': "Premier v38.5 (Hybrid AI System)",
            'confidence': min(98, 50 + (len(history)*2))
        }

    def update_learning(self, won):
        self.win_log.append(1 if won else 0)
        if len(self.win_log) > 5 and sum(self.win_log[-5:]) <= 1:
            # Nếu thua quá nhiều, ưu tiên thuật toán giả lập Monte Carlo
            self.weights['monte'] += 10
            self.weights['markov'] -= 10

    def get_ai_status(self):
        wr = (sum(self.win_log[-20:]) / 20 * 100) if self.win_log else 0
        return {'wr': round(wr, 1), 'weights': self.weights}

    def _fallback(self, msg):
        return {
            'main_3': ['?','?','?'], 
            'support_4': ['?','?','?','?'], 
            'risk': {'score': 0, 'level': 'LOW', 'reasons': [msg]}, 
            'logic': 'Đang chờ dữ liệu...', 
            'confidence': 0
        }
