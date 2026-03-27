"""
TITAN V27 - Utility Functions [UPDATED: Support 2-so & 3-so]
"""
import re
import json
import itertools
import pandas as pd
from collections import Counter
from pathlib import Path
from config import DB_FILE, PAIR_RULES, LOTTERY_CONFIG


def extract_lottery_numbers(text: str) -> list[str]:
    """Rút trích số 5 chữ số từ input"""
    return re.findall(r"\b\d{5}\b", text)


def check_win_2so5tinh(bet_numbers: list[str], draw_number: str) -> bool:
    """
    ✅ Kiểm tra trúng "2 số 5 tinh"
    Thắng nếu CẢ 2 số cược xuất hiện trong kết quả 5 chữ số
    """
    if len(bet_numbers) != 2:
        raise ValueError("Bet must contain exactly 2 numbers")
    if len(draw_number) != 5 or not draw_number.isdigit():
        raise ValueError("Draw number must be 5 digits")
    
    draw_digits = set(draw_number)
    return all(digit in draw_digits for digit in bet_numbers)


def check_win_3so5tinh(bet_numbers: list[str], draw_number: str) -> bool:
    """
    ✅ Kiểm tra trúng "3 số 5 tinh"
    Thắng nếu CẢ 3 số cược xuất hiện trong kết quả 5 chữ số
    """
    if len(bet_numbers) != 3:
        raise ValueError("Bet must contain exactly 3 numbers")
    if len(draw_number) != 5 or not draw_number.isdigit():
        raise ValueError("Draw number must be 5 digits")
    
    draw_digits = set(draw_number)
    return all(digit in draw_digits for digit in bet_numbers)


def generate_combinations_from_7(numbers_7: str) -> dict:
    """
    🎲 Từ 7 số, sinh tổ hợp 2 số và 3 số
    Returns: {"pairs": [...], "triples": [...]}
    """
    digits = list(set(numbers_7))  # Remove duplicates
    if len(digits) < 7:
        # Pad with remaining digits 0-9 if needed
        remaining = [str(i) for i in range(10) if str(i) not in digits]
        digits += remaining[:7-len(digits)]
    digits = digits[:7]  # Ensure exactly 7
    
    pairs = ["".join(p) for p in itertools.combinations(sorted(digits), 2)]
    triples = ["".join(t) for t in itertools.combinations(sorted(digits), 3)]
    
    return {
        "base_7": "".join(sorted(digits)),
        "pairs": pairs,      # 21 combinations
        "triples": triples,  # 35 combinations
        "total_bets": len(pairs) + len(triples)
    }


def calculate_win_examples():
    """Test cases for both game modes"""
    examples = [
        # 2-so tests
        {"type": "2so", "bet": ["1", "2"], "draw": "12121", "expected": True},
        {"type": "2so", "bet": ["1", "3"], "draw": "12121", "expected": False},
        {"type": "2so", "bet": ["5", "9"], "draw": "59000", "expected": True},
        # 3-so tests
        {"type": "3so", "bet": ["1", "2", "6"], "draw": "12864", "expected": True},
        {"type": "3so", "bet": ["1", "3", "6"], "draw": "12662", "expected": False},
        {"type": "3so", "bet": ["2", "6", "8"], "draw": "22668", "expected": True},
    ]
    
    results = []
    for ex in examples:
        bet = ex["bet"]
        draw = ex["draw"]
        if ex["type"] == "2so":
            result = check_win_2so5tinh(bet, draw)
        else:
            result = check_win_3so5tinh(bet, draw)
        
        results.append({
            "type": ex["type"],
            "bet": "".join(bet),
            "draw": draw,
            "predicted": result,
            "expected": ex["expected"],
            "correct": result == ex["expected"]
        })
    return results


def load_database() -> list[str]:
    if DB_FILE.exists():
        try:
            with open(DB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except:
            return []
    return []


def save_database(data: list[str], max_entries: int = 5000):
    unique_data = list(dict.fromkeys(data))[-max_entries:]
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(unique_data, f, ensure_ascii=False, indent=2)


def calculate_frequency_scores(db: list[str], pair_rules: list[str], last_draw: str) -> dict[str, int]:
    """Tính điểm tần suất + bonus pair rules"""
    all_digits = "".join(db[-60:]) if db else ""
    scores = {str(i): all_digits.count(str(i)) for i in range(10)}
    
    for rule in pair_rules:
        if sum(1 for d in last_draw if d in rule) >= 2:
            for digit in rule:
                scores[digit] = scores.get(digit, 0) + 15
    return scores


def generate_fallback_prediction(db: list[str], pair_rules: list[str]) -> dict:
    """Fallback: Loại 7 số điểm cao → sinh combo 2so/3so"""
    if not db:
        return {
            "base_7": "0123456",
            "main_3": "012",
            "pairs_sample": ["01", "23", "45"],
            "triples_sample": ["012", "234", "456"],
            "adv": "DỪNG",
            "logic": "Chưa đủ dữ liệu.",
            "conf": 50
        }
    
    scores = calculate_frequency_scores(db, pair_rules, db[-1])
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_7 = [x[0] for x in sorted_scores[:7]]
    base_7 = "".join(sorted(top_7))
    
    # Generate combos
    combos = generate_combinations_from_7(base_7)
    
    return {
        "base_7": base_7,
        "main_3": combos["triples"][0] if combos["triples"] else "012",
        "pairs_sample": combos["pairs"][:5],
        "triples_sample": combos["triples"][:5],
        "adv": "ĐÁNH",
        "logic": "Phân tích tần suất + pair rules.",
        "conf": 85,
        "_all_combos": combos  # For internal use
    }


def format_for_ai(db: list[str], pair_rules: list[str], last_n: int = 50) -> str:
    """Format prompt cho AI với yêu cầu output 7 số + combos"""
    recent = db[-last_n:] if len(db) >= last_n else db
    return f"""
[DATA] {recent}
[PAIR_RULES] {pair_rules}
[TASK] Chọn ra 7 số có xác suất cao nhất cho kỳ tiếp theo.
[OUTPUT_FORMAT] JSON strict:
{{
  "base_7": "7 digits sorted",
  "main_3": "3 best digits from base_7",
  "pairs_sample": ["12", "34", "56", "78", "90"],
  "triples_sample": ["123", "345", "567", "789", "012"],
  "adv": "ĐÁNH or DỪNG",
  "logic": "brief explanation in Vietnamese",
  "conf": 0-100
}}
[NOTE] base_7 phải có đúng 7 số khác nhau từ 0-9. Confidence >90 mới khuyên ĐÁNH.
"""