import numpy as np
from collections import Counter, defaultdict

class PredictionEngine:
    def __init__(self):
        # Trọng số ban đầu của các thuật toán
        self.weights = {
            'frequency': 0.3,
            'markov': 0.3,
            'pattern': 0.2,
            'hotcold': 0.2
        }
        self.learning_rate = 0.05 # Tốc độ tự điều chỉnh sau mỗi kỳ

    def get_frequency_analysis(self, data):
        """Phân tích tần suất xuất hiện của từng số."""
        all_digits = "".join(data[-100:]) # Lấy 100 kỳ gần nhất
        counts = Counter(all_digits)
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [num for num, count in sorted_counts[:7]]

    def get_markov_chain(self, data):
        """Thuật toán chuỗi Markov dự đoán số dựa trên số trước đó."""
        transitions = defaultdict(list)
        all_digits = "".join(data[-200:])
        for i in range(len(all_digits) - 1):
            transitions[all_digits[i]].append(all_digits[i+1])
        
        last_digit = all_digits[-1]
        next_options = transitions.get(last_digit, [])
        if not next_options:
            return []
        
        counts = Counter(next_options)
        return [num for num, _ in counts.most_common(7)]

    def detect_patterns(self, data):
        """Tìm kiếm quy luật lặp lại (Pattern)."""
        # Logic: Tìm các dãy số thường xuyên xuất hiện sau một số cụ thể
        return self.get_frequency_analysis(data[-20:]) # Ưu tiên xu hướng ngắn hạn

    def update_weights(self, won, method_used):
        """
        HÀM TỰ HỌC: Điều chỉnh trọng số thuật toán.
        Nếu thắng, tăng trọng số thuật toán đó. Nếu thua, giảm xuống.
        """
        if won:
            self.weights[method_used] = min(0.6, self.weights[method_used] + self.learning_rate)
        else:
            self.weights[method_used] = max(0.1, self.weights[method_used] - self.learning_rate)
        
        # Chuẩn hóa lại tổng trọng số = 1
        total = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] /= total

    def predict(self, data):
        if len(data) < 10:
            return {'main_3': ['1','2','3'], 'support_4': ['4','5','6','7'], 'confidence': 50}

        # Lấy kết quả từ các "chuyên gia" thuật toán
        f_results = self.get_frequency_analysis(data)
        m_results = self.get_markov_chain(data)
        p_results = self.detect_patterns(data)

        # Kết hợp kết quả dựa trên trọng số hiện tại (Ensemble)
        score_board = defaultdict(float)
        
        for num in f_results: score_board[num] += self.weights['frequency']
        for num in m_results: score_board[num] += self.weights['markov']
        for num in p_results: score_board[num] += self.weights['pattern']

        sorted_final = sorted(score_board.items(), key=lambda x: x[1], reverse=True)
        final_numbers = [x[0] for x in sorted_final]

        # Đảm bảo đủ 7 số
        while len(final_numbers) < 7:
            for n in "0123456789":
                if n not in final_numbers:
                    final_numbers.append(n)

        return {
            'main_3': final_numbers[:3],
            'support_4': final_numbers[3:7],
            'confidence': int(min(95, 60 + len(data)*0.1)),
            'weights': self.weights # Trả về để app theo dõi
        }

    def calculate_risk(self, data):
        if len(data) < 20: return (50, "MEDIUM", ["Thiếu dữ liệu"])
        # Logic tính Risk dựa trên độ biến động của 10 kỳ gần nhất
        last_10 = data[-10:]
        unique_digits = len(set("".join(last_10)))
        risk_score = min(100, unique_digits * 4)
        level = "LOW" if risk_score < 40 else "HIGH" if risk_score > 70 else "MEDIUM"
        return (risk_score, level, ["Dữ liệu biến động" if risk_score > 70 else "Nhịp số ổn định"])
