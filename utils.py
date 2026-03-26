"""
TITAN V27 - Utility Functions
Xử lý data, kiểm tra trúng thưởng, helper functions
"""
import re
import json
import pandas as pd
from collections import Counter
from pathlib import Path
from config import DB_FILE, PAIR_RULES, LOTTERY_CONFIG


def extract_lottery_numbers(text: str) -> list[str]:
    """Rút trích số 5 chữ số từ input người dùng"""
    return re.findall(r"\b\d{5}\b", text)


def check_win_3so5tinh(bet_numbers: list[str], draw_number: str) -> bool:
    """
    ✅ Kiểm tra trúng thưởng theo luật "3 số 5 tinh"
    
    Luật: 
    - Người chơi chọn 3 số từ 0-9
    - Thắng nếu 3 số này XUẤT HIỆN trong kết quả 5 chữ số (bất kỳ vị trí nào)
    - Không quan tâm thứ tự, không tính trùng lặp
    
    Args:
        bet_numbers: List 3 số người chơi cược, VD: ["1", "2", "6"]
        draw_number: Chuỗi 5 chữ số kết quả, VD: "12864"
    
    Returns:
        bool: True nếu trúng, False nếu trượt
    """
    if len(bet_numbers) != 3:
        raise ValueError("Bet must contain exactly 3 numbers")
    if len(draw_number) != 5 or not draw_number.isdigit():
        raise ValueError("Draw number must be 5 digits")
    
    # Chuyển kết quả thành set các chữ số duy nhất
    draw_digits = set(draw_number)
    
    # Kiểm tra: cả 3 số cược phải có mặt trong kết quả
    return all(digit in draw_digits for digit in bet_numbers)


def calculate_win_examples():
    """Minh họa logic trúng/thua theo ví dụ trong luật chơi"""
    examples = [
        {"bet": ["1", "2", "6"], "draw": "12864", "expected": True},   # Thắng: có 1,2,6
        {"bet": ["1", "3", "6"], "draw": "12662", "expected": False},  # Thua: thiếu 3
        {"bet": ["2", "6", "8"], "draw": "22668", "expected": True},   # Thắng: có đủ, trùng vẫn tính 1 lần
    ]
    
    results = []
    for ex in examples:
        result = check_win_3so5tinh(ex["bet"], ex["draw"])
        results.append({
            "bet": "".join(ex["bet"]),
            "draw": ex["draw"],
            "predicted": result,
            "expected": ex["expected"],
            "correct": result == ex["expected"]
        })
    return results


def load_database() -> list[str]:
    """Load dữ liệu lịch sử từ JSON file"""
    if DB_FILE.exists():
        try:
            with open(DB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_database(data: list[str], max_entries: int = 5000):
    """Lưu database với giới hạn số lượng entry"""
    # Remove duplicates, keep order, limit size
    unique_data = list(dict.fromkeys(data))[-max_entries:]
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(unique_data, f, ensure_ascii=False, indent=2)


def calculate_frequency_scores(db: list[str], pair_rules: list[str], last_draw: str) -> dict[str, int]:
    """
    Tính điểm tần suất cho từng số 0-9
    - Dựa trên frequency trong 60 kỳ gần nhất
    - Bonus điểm nếu số nằm trong pair_rules khớp với kết quả gần nhất
    """
    all_digits = "".join(db[-60:]) if db else ""
    scores = {str(i): all_digits.count(str(i)) for i in range(10)}
    
    # Bonus từ pair rules
    for rule in pair_rules:
        # Nếu kết quả gần nhất chứa >=2 số trong rule này
        if sum(1 for d in last_draw if d in rule) >= 2:
            for digit in rule:
                scores[digit] = scores.get(digit, 0) + 15
    
    return scores


def generate_fallback_prediction(db: list[str], pair_rules: list[str]) -> dict:
    """
    🔄 Thuật toán dự phòng khi AI fail
    Loại 7 số có điểm cao nhất → chọn 3 số chính + 4 số lót
    """
    if not db:
        return {
            "main": "012",
            "sub": "3456",
            "adv": "DỪNG",
            "logic": "Chưa đủ dữ liệu để phân tích.",
            "conf": 50
        }
    
    scores = calculate_frequency_scores(db, pair_rules, db[-1])
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    main_3 = "".join([x[0] for x in sorted_scores[:3]])
    sub_4 = "".join([x[0] for x in sorted_scores[3:7]])
    
    return {
        "main": main_3,
        "sub": sub_4,
        "adv": "ĐÁNH",
        "logic": "Phân tích tần suất + pair rules cổ điển.",
        "conf": 85
    }


def format_for_ai(db: list[str], pair_rules: list[str], last_n: int = 50) -> str:
    """Format data thành prompt cho AI model"""
    recent = db[-last_n:] if len(db) >= last_n else db
    return f"""
[DATA] {recent}
[PAIR_RULES] {pair_rules}
[TASK] Phân tích xu hướng và dự đoán 3 số chính + 4 số lót cho kỳ tiếp theo.
[OUTPUT_FORMAT] JSON strict: {{"main": "3 digits", "sub": "4 digits", "adv": "ĐÁNH or DỪNG", "logic": "brief explanation", "conf": 0-100}}
[NOTE] Ưu tiên numbers xuất hiện đều, tránh cầu gãy. Confidence >90 mới khuyên ĐÁNH.
"""