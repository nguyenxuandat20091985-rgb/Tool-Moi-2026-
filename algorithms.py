import numpy as np
import pandas as pd
import re
import json
import google.generativeai as genai
from collections import Counter
from itertools import combinations
from typing import Dict, List

class TitanAI:
    def __init__(self):
        # API Key anh cung cấp
        self.api_key = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception:
            self.model = None

    def _clean_history(self, history: List[str]) -> List[str]:
        """Lọc dữ liệu rác, chỉ lấy chuỗi 5 số"""
        cleaned = []
        for item in history:
            s = str(item).strip()
            match = re.search(r'\d{5}', s)
            if match:
                cleaned.append(match.group())
        return cleaned

    # --- HỆ THỐNG QUÉT BẪY NHÀ CÁI (ANTI-TRAP) ---
    def _detect_house_traps(self, data: List[str]) -> Dict:
        risk = 0
        warnings = []
        recent = data[:20]
        
        # 1. Bẫy Bệt Ảo: Kiểm tra 1 vị trí ra liên tiếp > 4 lần
        for pos in range(5):
            seq = [n[pos] for n in recent[:6] if len(n) > pos]
            if len(seq) >= 5 and len(set(seq)) == 1:
                risk += 40
                warnings.append(f"Cảnh báo Bệt Ảo vị trí {pos+1}")

        # 2. Bẫy Số Treo: Một số nổ quá nhiều trong 10 kỳ
        all_digits = "".join(data[:10])
        counts = Counter(all_digits)
        for d, c in counts.items():
            if c > 7:
                risk += 20
                warnings.append(f"Số {d} đang bị 'soi' (nổ dày)")

        # 3. Bẫy Nhịp Đều (Cầu lừa): Tổng số không đổi
        sums = [sum(int(d) for d in n) for n in recent[:10]]
        if np.std(sums) < 1.5:
            risk += 30
            warnings.append("Nhà cái đang ép nhịp (Số quay không tự nhiên)")

        return {"score": min(100, risk), "details": " | ".join(warnings)}

    # --- THUẬT TOÁN MATRIX COMBO (3 SỐ 5 TINH) ---
    def _matrix_analysis(self, data: List[str]) -> str:
        combos = []
        # Quét sâu 50 kỳ để tìm cặp số hay đi cùng nhau
        for line in data[:50]:
            digits = sorted(list(set(line)))
            if len(digits) >= 3:
                combos.extend(combinations(digits, 3))
        
        if not combos: return "125" # Mặc định nếu thiếu dữ liệu
        top = Counter(combos).most_common(1)
        return "".join(top[0][0])

    # --- AI INSIGHT (GEMINI PRO) ---
    def _ai_consultant(self, history: List[str], suggest_3: str) -> Dict:
        if not self.model:
            return None
        
        prompt = f"""
        Hệ thống TITAN v24.6. Dữ liệu: {history[:20]}
        Gợi ý hiện tại: {suggest_3}
        Hãy phân tích bẫy nhà cái và chốt:
        1. 3 số 5 tinh chủ lực.
        2. 4 số lót giữ vốn.
        3. Tỷ lệ thắng (%).
        Trả về JSON: {{"m3": "...", "l4": "...", "win": ..., "reason": "..."}}
        """
        try:
            response = self.model.generate_content(prompt)
            match = re.search(r'\{.*\}', response.text, re.DOTALL)
            return json.loads(match.group())
        except:
            return None

    def analyze(self, raw_data: List[str]) -> Dict:
        data = self._clean_history(raw_data)
        if len(data) < 5:
            return {"status": "ERR", "msg": "Cần tối thiểu 5 kỳ"}

        # Chạy đồng thời 3 lớp
        traps = self._detect_house_traps(data)
        m3_logic = self._matrix_analysis(data)
        ai_res = self._ai_consultant(data, m3_logic)

        # Hợp nhất kết quả
        final_m3 = ai_res["m3"] if ai_res else m3_logic
        final_l4 = ai_res["l4"] if ai_res else "0468"
        win_rate = ai_res["win"] if ai_res else (100 - traps["score"])

        return {
            "m3": final_m3,
            "l4": final_l4,
            "win_rate": win_rate,
            "trap_msg": traps["details"] if traps["details"] else "Cầu đang ổn định",
            "decision": "ĐÁNH" if win_rate > 50 else "DỪNG/QUAN SÁT"
        }
