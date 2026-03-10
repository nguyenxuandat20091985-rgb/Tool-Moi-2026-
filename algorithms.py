import numpy as np
import pandas as pd
import re
import json
import google.generativeai as genai
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, List, Tuple

class TitanAI:
    def __init__(self):
        # Kết hợp API Key của anh
        self.api_key = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
        self.risk_level = 0
        self.house_warning = ""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except:
            self.model = None

    def _clean_history(self, history: List[str]) -> List[str]:
        cleaned = []
        for item in history:
            s = str(item).strip()
            match = re.search(r'\d{5}', s)
            if match:
                cleaned.append(match.group())
        return cleaned

    # --- NHÓM THUẬT TOÁN PHÁT HIỆN BẪY (HOUSE TRAPS) ---
    def _detect_house_traps(self, data: List[str]) -> Dict:
        risk = 0
        warnings = []
        recent = data[:30]
        
        # 1. Bẫy Bệt Ảo (Vị trí về quá 5 kỳ)
        for pos in range(5):
            seq = [n[pos] for n in recent[:10] if len(n) > pos]
            if len(seq) >= 5 and len(set(seq)) == 1:
                risk += 35
                warnings.append(f"Cảnh báo Bệt Ảo vị trí {pos}")

        # 2. Bẫy Đảo Nhịp (Số vừa ra lại mất hút)
        all_digits = "".join(data[:15])
        counts = Counter(all_digits)
        for d, c in counts.items():
            if c > 6: # Số nổ quá dày trong 15 kỳ thường sẽ bị "treo" ở các kỳ sau
                risk += 15
                warnings.append(f"Số {d} nổ quá dày - Dễ bị treo")

        # 3. Kiểm soát Tổng (Std Dev thấp)
        sums = [sum(int(d) for d in n) for n in data[:20]]
        if np.std(sums) < 2.2:
            risk += 25
            warnings.append("Nhà cái đang kiểm soát tổng (Số quay ảo)")

        return {"score": min(100, risk), "details": "; ".join(warnings)}

    # --- THUẬT TOÁN CHỐT SỐ (3 SỐ 5 TINH) ---
    def _matrix_evolution_analysis(self, data: List[str]) -> str:
        """Tìm bộ 3 dựa trên ma trận xác suất nổ chung"""
        combos = []
        # Quét sâu 40 kỳ để tìm nhịp nổ chung
        for line in data[:40]:
            digits = sorted(list(set(line)))
            if len(digits) >= 3:
                combos.extend(combinations(digits, 3))
        
        if not combos: return "123"
        top = Counter(combos).most_common(1)
        return "".join(top[0][0])

    def _get_support_numbers(self, data: List[str], main_3: str) -> str:
        """Lấy 4 số lót dựa trên nhịp rơi lùi"""
        all_digits = "".join(data[:25])
        freq = Counter(all_digits).most_common(10)
        support = []
        for d, c in freq:
            if d not in main_3:
                support.append(d)
            if len(support) == 4: break
        return "".join(support)

    # --- PHÂN TÍCH GEMINI AI ---
    def _ai_insight(self, history: List[str], matrix_suggestion: str) -> Dict:
        if not self.model:
            return {"l": "Cầu tự nhiên", "d": "VÀO LỆNH"}
        
        prompt = f"""
        Hệ thống TITAN AI v5.5. 
        Dữ liệu 30 kỳ gần nhất: {history[:30]}
        Gợi ý từ thuật toán Matrix: {matrix_suggestion}
        
        Nhiệm vụ:
        1. Phân tích bẫy nhà cái.
        2. Chốt lại 3 số chủ lực mạnh nhất.
        3. Đưa ra lời khuyên (ĐÁNH/DỪNG).
        
        Trả về định dạng JSON:
        {{"m": "3 số chủ lực", "s": "4 số lót", "d": "ĐÁNH/DỪNG", "l": "lý do logic", "r": "mức rủi ro 0-100"}}
        """
        try:
            response = self.model.generate_content(prompt)
            match = re.search(r'\{.*\}', response.text, re.DOTALL)
            return json.loads(match.group())
        except:
            return None

    # --- HÀM TỔNG HỢP CHÍNH ---
    def analyze(self, raw_history: List[str]) -> Dict:
        data = self._clean_history(raw_history)
        if len(data) < 10:
            return {"logic": "Cần thêm dữ liệu", "decision": "CHỜ"}

        # Bước 1: Quét bẫy nhà cái
        trap_results = self._detect_house_traps(data)
        
        # Bước 2: Chạy thuật toán nội bộ
        matrix_3 = self._matrix_evolution_analysis(data)
        support_4 = self._get_support_numbers(data, matrix_3)

        # Bước 3: Hỏi ý kiến Gemini AI
        ai_res = self._ai_insight(data, matrix_3)

        # Hợp nhất kết quả (Ưu tiên AI nếu có, nếu không dùng Matrix)
        if ai_res:
            final_main = ai_res.get("m", matrix_3)
            final_supp = ai_res.get("s", support_4)
            decision = ai_res.get("d", "VÀO LỆNH")
            logic = ai_res.get("l", "Phân tích AI")
            risk_score = ai_res.get("r", trap_results["score"])
        else:
            final_main = matrix_3
            final_supp = support_4
            decision = "VÀO LỆNH" if trap_results["score"] < 50 else "DỪNG"
            logic = "Dùng Matrix Combo (AI nghẽn)"
            risk_score = trap_results["score"]

        return {
            "main_3": final_main,
            "support_4": final_supp,
            "decision": decision,
            "logic": logic,
            "risk": {"score": risk_score, "level": "OK" if risk_score < 45 else "HIGH"},
            "house_warning": trap_results["details"]
        }
