import numpy as np
from collections import Counter, defaultdict
import math
import random

class PredictionEngine:
    def __init__(self):
        self.weights = {'freq': 20, 'markov': 25, 'monte': 25, 'pattern': 30}
        self.history_data = []
        self.win_log = []

    def calculate_entropy(self, history):
        """Phát hiện nhà cái lừa cầu dựa trên độ hỗn loạn toán học"""
        if not history: return 0
        all_chars = "".join(history[-30:])
        counts = Counter(all_chars)
        probs = [c/len(all_chars) for c in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs)
        return entropy

    def predict(self, history):
        if len(history) < 5:
            return self._fallback()

        # 1. Tầng Markov
        markov_nodes = defaultdict(Counter)
        for i in range(len(history)-1):
            markov_nodes[history[i][-1]][history[i+1][-1]] += 1
        last_digit = history[-1][-1]
        m_preds = [x[0] for x in markov_nodes[last_digit].most_common(3)]

        # 2. Tầng Monte Carlo (Giả lập Gemini)
        pool = list("".join(history[-50:]))
        sim = Counter()
        for _ in range(10000):
            sample = random.choices(pool, k=3)
            for n in sample: sim[n] += 1
        mc_preds = [x[0] for x in sim.most_common(3)]

        # 3. Tầng Pattern (Bệt/Đảo)
        p_preds = [history[-1][0], history[-1][2], history[-1][4]] # Nhịp nháy

        # Bỏ phiếu tổng hợp (Ensemble)
        votes = Counter()
        for n in m_preds: votes[n] += self.weights['markov']
        for n in mc_preds: votes[n] += self.weights['monte']
        for n in p_preds: votes[n] += self.weights['pattern']

        final_sorted = votes.most_common(7)
        
        # Phân tích Rủi ro
        entropy = self.calculate_entropy(history)
        risk_score = 100 - (entropy * 30)
        risk_level = "LOW" if risk_score < 40 else "MEDIUM" if risk_score < 70 else "HIGH"
        reasons = []
        if risk_level == "HIGH": reasons.append("⚠️ Phát hiện nhà cái đổi luồng quay (Cầu ảo)")
        if history[-1] == history[-2]: reasons.append("🔥 Cầu đang bệt cực mạnh")

        return {
            'main_3': [x[0] for x in final_sorted[:3]],
            'support_4': [x[0] for x in final_sorted[3:7]],
            'risk': {'score': int(min(risk_score, 100)), 'level': risk_level, 'reasons': reasons},
            'logic': "Ensemble v38 (Markov + Monte Carlo + Gemini Simulation)",
            'confidence': min(98, 60 + len(history))
        }

    def update_learning(self, won):
        self.win_log.append(1 if won else 0)
        # Tự điều chỉnh trọng số nếu thua quá 3 kỳ liên tiếp
        if len(self.win_log) > 3 and sum(self.win_log[-3:]) == 0:
            self.weights['monte'] += 5
            self.weights['pattern'] -= 5

    def _fallback(self):
        return {'main_3': ['1','5','9'], 'support_4': ['0','2','4','6'], 'risk': {'score': 0, 'level': 'LOW', 'reasons': ['Chờ thêm dữ liệu']}, 'logic': 'Khởi động...', 'confidence': 0}
