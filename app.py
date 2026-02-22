class TitanEliteAnalyzer:
    def __init__(self, history):
        self.history = [str(x) for x in history]
        self.digits_history = "".join([s[-1] for s in self.history]) # Tập trung vào đuôi giải đặc biệt

    def get_smart_weights(self) -> Dict[str, float]:
        """
        Tính toán trọng số dựa trên 3 lớp: Tần suất, Độ gan (Gap) và Ma trận chuyển tiếp.
        """
        if len(self.history) < 10: return {str(i): 0.1 for i in range(10)}
        
        scores = {str(i): 0.0 for i in range(10)}
        last_20 = self.digits_history[-20:]
        
        # 1. Lớp Tần suất động (Recency Bias)
        for i, char in enumerate(last_20):
            scores[char] += (i + 1) * 0.5  # Số càng mới ra càng có trọng số cao

        # 2. Lớp Ma trận Markov (Dự đoán logic cầu tiếp nối)
        if len(self.digits_history) >= 2:
            last_pair = self.digits_history[-2:]
            # Tìm trong lịch sử xem sau cặp này thường ra số gì
            for i in range(len(self.digits_history) - 2):
                if self.digits_history[i:i+2] == last_pair:
                    next_digit = self.digits_history[i+2]
                    scores[next_digit] += 5.0 # Cộng điểm mạnh cho logic cầu lặp

        # 3. Lớp "Độ Gan" (Hạn chế số đã quá lâu không về hoặc về quá dày)
        for num in "0123456789":
            gap = self.digits_history[::-1].find(num)
            if gap > 15: # Số quá gan - thường nhà cái sẽ giấu rất kỹ
                scores[num] -= 3.0
            elif gap == 0: # Vừa về xong (bệt)
                scores[num] += 2.0 

        return scores

    def export_elite_numbers(self):
        """
        Xuất ra 3 số 99% và 4 số dự phòng
        """
        scores = self.get_smart_weights()
        # Sắp xếp theo điểm số từ cao xuống thấp
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        final_7 = [n[0] for n in sorted_nums[:7]]
        dan_3_sieu_cap = final_7[:3]
        dan_4_du_phong = final_7[3:7]
        
        # Tính toán độ tin cậy dựa trên biên độ điểm số
        top_score = sorted_nums[0][1]
        avg_score = sum(s[1] for s in sorted_nums) / 10
        confidence = min(99.0, (top_score / (avg_score + 1)) * 40 + 50)

        return dan_3_sieu_cap, dan_4_du_phong, round(confidence, 2)

# ================= PHẦN KẾT HỢP GEMINI AI =================
# Cập nhật Prompt để ép AI phân tích theo hướng "bào tiền"
def generate_titan_prompt(dan3, dan4, history, confidence):
    return f"""
    Bạn là một hacker toán học chuyên bẻ khóa thuật toán PRNG của các sàn Lotobet.
    DỮ LIỆU THỰC TẾ:
    - 20 kỳ gần nhất: {history[-20:]}
    - Thuật toán Titan lọc được Dàn 3 Siêu Cấp: {dan3}
    - Dàn 4 Dự Phòng: {dan4}
    - Độ tự tin thuật toán: {confidence}%

    NHIỆM VỤ:
    1. Kiểm tra xem 3 số {dan3} có nằm trong 'vùng chết' (số bị nhà cái khóa) không?
    2. Dựa trên lý thuyết xác suất Bayes, hãy khẳng định lại 3 số có khả năng nổ cao nhất.
    3. Thiết kế chiến thuật vào tiền (Money Management) kiểu Martingale cải tiến cho dàn này.

    TRẢ VỀ JSON:
    {{
        "top_3_fixed": ["số1", "số2", "số3"],
        "backup_4": ["số4", "số5", "số6", "số7"],
        "strategy": "cụ thể cách đi tiền vòng 1, 2, 3",
        "risk_level": "Thấp/Trung bình/Cao"
    }}
    """
