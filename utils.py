"""
TITAN V27 - Utility Functions
Version: 2.8.1
"""
import re
import json
import itertools
from collections import Counter
from pathlib import Path
from config import DB_FILE, PAIR_RULES, LOTTERY_CONFIG


def extract_lottery_numbers(text: str) -> list:
    """
    Rút trích số 5 chữ số từ input người dùng
    Returns: List các số 5 chữ số
    """
    if not text:
        return []
    return re.findall(r"\b\d{5}\b", text)


def check_win_2so5tinh(bet_numbers: list, draw_number: str) -> bool:
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


def check_win_3so5tinh(bet_numbers: list, draw_number: str) -> bool:
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
    Returns: {"base_7": str, "pairs": list, "triples": list, "total_bets": int}
    """
    # Remove duplicates and ensure we have exactly 7 digits
    digits = list(set(numbers_7))
    
    # If less than 7 digits, pad with remaining digits 0-9
    if len(digits) < 7:
        remaining = [str(i) for i in range(10) if str(i) not in digits]
        digits += remaining[:7 - len(digits)]
    
    # Ensure exactly 7 digits
    digits = digits[:7]
    
    # Generate combinations
    pairs = ["".join(p) for p in itertools.combinations(sorted(digits), 2)]
    triples = ["".join(t) for t in itertools.combinations(sorted(digits), 3)]
    
    return {
        "base_7": "".join(sorted(digits)),
        "pairs": pairs,           # 21 combinations
        "triples": triples,       # 35 combinations
        "total_bets": len(pairs) + len(triples)
    }


def calculate_win_examples() -> list:
    """
    Test cases for both game modes
    Returns: List of test results
    """
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


def load_database() -> list:
    """
    Load database from JSON file
    Returns: List of lottery numbers
    """
    try:
        if DB_FILE.exists():
            with open(DB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
    except Exception as e:
        print(f"Error loading database: {e}")
        return []
    return []


def save_database(data: list, max_entries: int = 5000):
    """
    Save database to JSON file with max entries limit
    """
    try:
        # Remove duplicates, keep order, limit size
        unique_data = list(dict.fromkeys(data))[-max_entries:]
        
        # Ensure directory exists
        DB_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump(unique_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving database: {e}")


def calculate_frequency_scores(db: list, pair_rules: list, last_draw: str) -> dict:
    """
    Tính điểm tần suất cho từng số 0-9
    - Dựa trên frequency trong 60 kỳ gần nhất
    - Bonus điểm nếu số nằm trong pair_rules khớp với kết quả gần nhất
    """
    if not db:
        return {str(i): 0 for i in range(10)}
    
    all_digits = "".join(db[-60:]) if db else ""
    scores = {str(i): all_digits.count(str(i)) for i in range(10)}
    
    # Bonus từ pair rules
    if last_draw:
        for rule in pair_rules:
            # Nếu kết quả gần nhất chứa >=2 số trong rule này
            if sum(1 for d in last_draw if d in rule) >= 2:
                for digit in rule:
                    scores[digit] = scores.get(digit, 0) + 15
    
    return scores


def generate_fallback_prediction(db: list, pair_rules: list) -> dict:
    """
    🔄 Thuật toán dự phòng khi AI fail
    Loại 7 số có điểm cao nhất → chọn 3 số chính + 4 số lót + generate combos
    """
    if not db:
        return {
            "base_7": "0123456",
            "main_3": "012",
            "pairs_sample": ["01", "23", "45"],
            "triples_sample": ["012", "234", "456"],
            "adv": "DỪNG",
            "logic": "Chưa đủ dữ liệu để phân tích.",
            "conf": 50
        }
    
    try:
        last_draw = db[-1] if db else ""
        scores = calculate_frequency_scores(db, pair_rules, last_draw)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top 7 digits
        top_7 = [x[0] for x in sorted_scores[:7]]
        base_7 = "".join(sorted(top_7))
        
        # Generate combinations
        combos = generate_combinations_from_7(base_7)
        
        return {
            "base_7": base_7,
            "main_3": combos["triples"][0] if combos["triples"] else "012",
            "pairs_sample": combos["pairs"][:5],
            "triples_sample": combos["triples"][:5],
            "adv": "ĐÁNH",
            "logic": "Phân tích tần suất + pair rules cổ điển.",
            "conf": 85,
            "_all_combos": combos  # For internal use
        }
    except Exception as e:
        print(f"Error in fallback prediction: {e}")
        return {
            "base_7": "0123456",
            "main_3": "012",
            "pairs_sample": ["01", "23", "45"],
            "triples_sample": ["012", "234", "456"],
            "adv": "DỪNG",
            "logic": f"Lỗi tính toán: {str(e)}",
            "conf": 50
        }


def format_for_ai(db: list, pair_rules: list, last_n: int = 50) -> str:
    """
    Format data thành prompt cho AI model
    """
    if not db:
        recent = []
    else:
        recent = db[-last_n:] if len(db) >= last_n else db
    
    prompt = f"""
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
    return prompt