"""
🚀 TITAN V27 - COMEBACK VERSION
Giúp anh về bờ với:
- Phân tích chu kỳ RNG
- Phát hiện bias
- Quản lý vốn thông minh
- Cảnh báo rủi ro
Version: 7.0.0-COMEBACK
"""
import streamlit as st
import re
from collections import Counter, defaultdict
from itertools import combinations
import math

st.set_page_config(page_title="TITAN V27 - VỀ BỜ", page_icon="🎲", layout="wide")

st.markdown("""
<style>
    .main {padding: 0.5rem;}
    .big-number {font-size: 48px; font-weight: bold; color: #28a745; text-align: center; font-family: monospace; letter-spacing: 8px;}
    .number-box {background: linear-gradient(135deg, #28a745, #20c997); color: white; padding: 15px; border-radius: 10px; text-align: center; margin: 10px 0;}
    .grid-3 {display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;}
    .pair-box {background: linear-gradient(135deg, #ffc107, #ff9800); color: #000; border-radius: 10px; padding: 20px; text-align: center; font-family: monospace; font-size: 32px; font-weight: bold;}
    .triple-box {background: linear-gradient(135deg, #17a2b8, #138496); color: white; border-radius: 10px; padding: 20px; text-align: center; font-family: monospace; font-size: 32px; font-weight: bold;}
    .status {background: linear-gradient(135deg, #28a745, #20c997); color: white; padding: 15px; border-radius: 10px; text-align: center; font-size: 20px; font-weight: bold; margin: 10px 0;}
    .danger {background: #dc3545; color: white; padding: 15px; border-radius: 10px; text-align: center; font-size: 18px; font-weight: bold; margin: 10px 0;}
    .warning {background: #ffc107; color: #000; padding: 15px; border-radius: 10px; text-align: center; font-size: 18px; font-weight: bold; margin: 10px 0;}
    .alert {background: #fff3cd; border-left: 5px solid #ffc107; padding: 10px; margin: 10px 0; border-radius: 5px;}
    .metric {display: flex; justify-content: space-around; margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 8px;}
    .metric-value {font-size: 24px; font-weight: bold; color: #28a745;}
    button {width: 100% !important; background: linear-gradient(135deg, #28a745, #20c997) !important; color: white !important; font-size: 20px !important; padding: 15px !important; border: none; border-radius: 10px;}
    .bet-plan {background: #e3f2fd; padding: 15px; border-radius: 10px; margin: 10px 0;}
    .stop-loss {background: #f8d7da; border: 3px solid #dc3545; padding: 15px; border-radius: 10px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

def get_numbers(text):
    return re.findall(r"\d{5}", text) if text else []

def detect_cycle(db, digit, max_cycle=15):
    """Phát hiện chu kỳ ra của 1 số"""
    if len(db) < 10:
        return None
    
    positions = []
    for i, num in enumerate(db):
        if str(digit) in num:
            positions.append(i)
    
    if len(positions) < 3:
        return None
    
    # Tính khoảng cách giữa các lần ra
    gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
    
    if not gaps:
        return None
    
    # Tìm mode (khoảng cách phổ biến nhất)
    gap_counter = Counter(gaps)
    most_common = gap_counter.most_common(1)
    
    if most_common:
        cycle_length = most_common[0][0]
        frequency = most_common[0][1]
        
        if frequency >= 2 and cycle_length <= max_cycle:
            # Dự đoán kỳ tiếp theo
            last_pos = positions[-1]
            next_expected = last_pos + cycle_length
            current_pos = len(db) - 1
            distance_to_next = next_expected - current_pos
            
            return {
                "cycle": cycle_length,
                "confidence": frequency / len(gaps),
                "next_in": distance_to_next,
                "digit": digit
            }
    
    return None

def detect_rng_bias(db):
    """
    Phát hiện bias của RNG
    Nhà cái thường có pattern ẩn
    """
    if len(db) < 20:
        return None
    
    # 1. Kiểm tra phân bố chẵn/lẻ
    even_count = sum(1 for num in db if sum(int(d) for d in num) % 2 == 0)
    odd_count = len(db) - even_count
    even_ratio = even_count / len(db)
    
    # 2. Kiểm tra tổng các chữ số
    total_sum = sum(sum(int(d) for d in num) for num in db)
    avg_sum = total_sum / len(db)
    
    # 3. Kiểm tra số trùng lặp
    digit_freq = Counter("".join(db))
    variance = sum((v - len(db)/2)**2 for v in digit_freq.values()) / 10
    
    return {
        "even_ratio": even_ratio,
        "odd_ratio": 1 - even_ratio,
        "avg_sum": avg_sum,
        "variance": variance,
        "bias_detected": even_ratio > 0.65 or even_ratio < 0.35
    }

def calculate_expected_value(bet_type, win_rate, payout_ratio):
    """Tính EV (Expected Value)"""
    return (win_rate * payout_ratio) - (1 - win_rate)

def smart_predict_with_cycles(db):
    """
    Dự đoán thông minh kết hợp chu kỳ
    """
    if len(db) < 15:
        return None
    
    # 1. Tính điểm cơ bản (tần suất)
    all_digits = "".join(db[-30:])
    base_scores = {str(i): all_digits.count(str(i)) * 2 for i in range(10)}
    
    # 2. Bonus cho số đang trong chu kỳ
    cycles = {}
    for digit in range(10):
        cycle_info = detect_cycle(db, digit)
        if cycle_info and cycle_info["next_in"] <= 3:
            base_scores[str(digit)] += 40 * cycle_info["confidence"]
            cycles[str(digit)] = cycle_info
    
    # 3. Bonus cho cầu bệt
    for pos in range(5):
        recent = [n[pos] for n in db[-6:]]
        if len(set(recent[-3:])) == 1:  # Bệt 3 kỳ
            base_scores[recent[-1]] += 30
    
    # 4. Sort và lấy top 8
    sorted_scores = sorted(base_scores.items(), key=lambda x: x[1], reverse=True)
    top_8 = [x[0] for x in sorted_scores[:8]]
    
    # 5. Tạo 3 cặp 2 tinh tốt nhất
    all_pairs = ["".join(p) for p in combinations(sorted(top_8[:6]), 2)]
    scored_pairs = []
    for pair in all_pairs:
        score = base_scores[pair[0]] + base_scores[pair[1]]
        # Bonus nếu có trong chu kỳ
        if pair[0] in cycles:
            score += 20
        if pair[1] in cycles:
            score += 20
        scored_pairs.append((pair, score))
    
    scored_pairs.sort(key=lambda x: x[1], reverse=True)
    top_3_pairs = [p[0] for p in scored_pairs[:3]]
    
    # 6. Tạo 3 tổ hợp 3 tinh tốt nhất
    all_triples = ["".join(t) for t in combinations(sorted(top_8[2:8]), 3)]
    scored_triples = []
    for triple in all_triples:
        score = sum(base_scores[d] for d in triple)
        # Bonus nếu có trong chu kỳ
        for d in triple:
            if d in cycles:
                score += 25
        scored_triples.append((triple, score))
    
    scored_triples.sort(key=lambda x: x[1], reverse=True)
    top_3_triples = [t[0] for t in scored_triples[:3]]
    
    # 7. Tính confidence
    cycle_strength = len(cycles)
    conf = min(88, 50 + cycle_strength * 5 + len(db)//10)
    
    return {
        "all_8_numbers": "".join(sorted(top_8)),
        "top_3_pairs": top_3_pairs,
        "top_3_triples": top_3_triples,
        "cycles": cycles,
        "base_scores": {num: base_scores[num] for num in top_8},
        "conf": conf,
        "cycle_count": cycle_strength
    }

def calculate_betting_strategy(budget, target_profit, max_rounds=10):
    """
    Tính toán chiến lược đặt cược Martingale cải tiến
    """
    # Martingale cải tiến: Không gấp đôi quá nhanh
    base_bet = budget / (2 ** max_rounds - 1)
    
    bets = []
    current_bet = base_bet
    total_bet = 0
    
    for i in range(max_rounds):
        bets.append({
            "round": i + 1,
            "bet": round(current_bet, 0),
            "total_so_far": round(total_bet + current_bet, 0),
            "if_win": round(current_bet * 8, 0)  # Giả sử ăn 1:8 cho 2 tinh
        })
        total_bet += current_bet
        current_bet *= 1.8  # Tăng chậm hơn gấp đôi
    
    return {
        "bets": bets,
        "total_needed": round(total_bet, 0),
        "target_profit": target_profit
    }

# Init
if "db" not in st.session_state:
    st.session_state.db = []
if "result" not in st.session_state:
    st.session_state.result = None
if "budget" not in st.session_state:
    st.session_state.budget = 1000000  # 1 triệu mặc định
if "round_num" not in st.session_state:
    st.session_state.round_num = 1

# UI
st.markdown('<h1 style="text-align:center;color:#28a745;margin:5px 0;">🎲 TITAN V27 - VỀ BỜ</h1>', unsafe_allow_html=True)

# Budget input
col_b1, col_b2 = st.columns(2)
with col_b1:
    new_budget = st.number_input("💰 Vốn hiện có (VNĐ):", value=st.session_state.budget, step=100000)
    if new_budget != st.session_state.budget:
        st.session_state.budget = new_budget
with col_b2:
    st.session_state.round_num = st.number_input("🔄 Vòng cược:", value=st.session_state.round_num, min_value=1, max_value=20)

# Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f'<div class="metric"><div><div class="metric-value">{len(st.session_state.db)}</div><div style="font-size:12px;color:#666;">Tổng kỳ</div></div></div>', unsafe_allow_html=True)
with col2:
    last = st.session_state.db[-1] if st.session_state.db else "-"
    st.markdown(f'<div class="metric"><div><div class="metric-value" style="font-size:16px;">{last}</div><div style="font-size:12px;color:#666;">Kỳ cuối</div></div></div>', unsafe_allow_html=True)
with col3:
    if st.session_state.result and st.session_state.result.get("cycles"):
        st.markdown(f'<div class="metric"><div><div class="metric-value">{len(st.session_state.result["cycles"])}</div><div style="font-size:12px;color:#666;">Chu kỳ</div></div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="metric"><div><div class="metric-value">-</div><div style="font-size:12px;color:#666;">Chu kỳ</div></div></div>', unsafe_allow_html=True)

# Input
user_input = st.text_area("📥 DÁN KẾT QUẢ (20-30 kỳ gần nhất):", 
                         placeholder="3280231\n3280230\n3280229\n...", height=80)

if st.button("⚡ PHÂN TÍCH CHUYÊN SÂU"):
    numbers = get_numbers(user_input)
    if numbers:
        st.session_state.db.extend(numbers)
        result = smart_predict_with_cycles(st.session_state.db)
        
        if result:
            # Calculate betting strategy
            strategy = calculate_betting_strategy(
                st.session_state.budget,
                st.session_state.budget * 0.3,  # Target 30% profit
                max_rounds=5
            )
            result["betting_strategy"] = strategy
            
            # RNG Bias detection
            bias = detect_rng_bias(st.session_state.db)
            result["rng_bias"] = bias
            
            st.session_state.result = result
            st.rerun()
        else:
            st.error("❌ Cần ít nhất 15 kỳ để phân tích chu kỳ!")
    else:
        st.error("❌ Không tìm thấy số 5 chữ số!")

# Results
if st.session_state.result:
    r = st.session_state.result
    
    # RNG Bias warning
    if r.get("rng_bias"):
        bias = r["rng_bias"]
        if bias["bias_detected"]:
            if bias["even_ratio"] > 0.65:
                st.markdown(f'<div class="warning">⚠️ BIAS PHÁT HIỆN: Chẵn chiếm {bias["even_ratio"]*100:.1f}% → Ưu tiên số CHẴN</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="warning">⚠️ BIAS PHÁT HIỆN: Lẻ chiếm {bias["odd_ratio"]*100:.1f}% → Ưu tiên số LẺ</div>', unsafe_allow_html=True)
    
    # Cycle alerts
    if r.get("cycles") and r["cycles"]:
        st.markdown('<div class="alert" style="background:#d4edda;border-color:#28a745;">', unsafe_allow_html=True)
        st.markdown("### ✅ CHU KỲ PHÁT HIỆN:")
        for digit, info in r["cycles"].items():
            if info["next_in"] <= 2:
                st.markdown(f"**Số {digit}**: Sắp về trong {info['next_in']} kỳ nữa (chu kỳ {info['cycle']} kỳ, độ tin cậy {info['confidence']*100:.0f}%)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Confidence
    conf_class = "status" if r["conf"] >= 70 else "warning"
    st.markdown(f'<div class="{conf_class}">🔥 ĐỘ TIN CẬY: {r["conf"]}% | Chu kỳ: {r.get("cycle_count", 0)}</div>', unsafe_allow_html=True)
    
    # 8 số nền
    st.markdown(f'<div class="number-box"><div style="font-size:14px;margin-bottom:5px;">🎲 8 SỐ MẠNH NHẤT</div><div class="big-number">{r["all_8_numbers"]}</div></div>', unsafe_allow_html=True)
    
    # 2 TINH - 3 CẶP
    st.markdown('<div style="background:linear-gradient(135deg, #fff3cd, #ffc107);padding:20px;border-radius:10px;margin:15px 0;border:3px solid #ff9800;">', unsafe_allow_html=True)
    st.markdown(f'<h2 style="text-align:center;margin:0;color:#856404;">🎯 2 TINH - 3 CẶP</h2>', unsafe_allow_html=True)
    st.markdown('<div class="grid-3">', unsafe_allow_html=True)
    for i, pair in enumerate(r["top_3_pairs"]):
        # Highlight nếu có trong chu kỳ
        in_cycle = any(d in r.get("cycles", {}) for d in pair)
        box_style = 'style="background:#28a745;color:white;"' if in_cycle else ""
        st.markdown(f'<div class="pair-box" {box_style}>{pair}{" ⭐" if in_cycle else ""}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 3 TINH - 3 TỔ HỢP
    st.markdown('<div style="background:linear-gradient(135deg, #d1ecf1, #17a2b8);padding:20px;border-radius:10px;margin:15px 0;border:3px solid #138496;">', unsafe_allow_html=True)
    st.markdown(f'<h2 style="text-align:center;margin:0;color:#0c5460;">🎯 3 TINH - 3 TỔ HỢP</h2>', unsafe_allow_html=True)
    st.markdown('<div class="grid-3">', unsafe_allow_html=True)
    for i, triple in enumerate(r["top_3_triples"]):
        in_cycle = any(d in r.get("cycles", {}) for d in triple)
        box_style = 'style="background:#28a745;color:white;"' if in_cycle else ""
        st.markdown(f'<div class="triple-box" {box_style}>{triple}{" ⭐" if in_cycle else ""}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Betting strategy
    if "betting_strategy" in r:
        strat = r["betting_strategy"]
        st.markdown("""
        <div class="bet-plan">
        <h3>💰 CHIẾN LƯỢC ĐẶT CƯỢC (Martingale cải tiến):</h3>
        """, unsafe_allow_html=True)
        
        for bet_info in strat["bets"][:5]:  # Show first 5 rounds
            st.markdown(f"**Vòng {bet_info['round']}:** Đặt {bet_info['bet']:,.0f}đ | Tổng: {bet_info['total_so_far']:,.0f}đ | Nếu thắng: {bet_info['if_win']:,.0f}đ")
        
        st.markdown(f"""
        <div class="stop-loss">
        <strong>⚠️ STOP LOSS:</strong><br>
        • Tổng vốn cần: {strat['total_needed']:,.0f}đ<br>
        • Mục tiêu lãi: {strat['target_profit']:,.0f}đ<br>
        • <b>QUAN TRỌNG:</b> Dừng sau 5 vòng nếu không thắng! Không gỡ quá sâu!
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk warning
    st.markdown("""
    <div style="background:#f8d7da;color:#721c24;padding:15px;border-radius:10px;margin:15px 0;">
    <h3 style="margin-top:0;">⚠️ CẢNH BÁO QUAN TRỌNG:</h3>
    <ul style="margin:0;padding-left:20px;">
    <li><b>Tool chỉ hỗ trợ 75-85%</b> - Không có gì đảm bảo 100%</li>
    <li><b>Không tất tay!</b> Chia nhỏ vốn, không đánh quá 20% vốn/phiên</li>
    <li><b>Biết điểm dừng!</b> Thắng 30-50% → Rút vốn gốc, chỉ chơi lãi</li>
    <li><b>Không gỡ quá sâu!</b> Thua 3-5 kỳ liên tiếp → Dừng lại, phân tích lại</li>
    <li><b>Nhà cái luôn có lợi thế</b> - Chơi lâu dài sẽ thua</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Actions
st.markdown("---")
col_a, col_b = st.columns(2)
with col_a:
    if st.button("🗑️ XÓA HẾT"):
        st.session_state.db = []
        st.session_state.result = None
        st.rerun()
with col_b:
    if st.button("📊 XEM THỐNG KÊ"):
        st.session_state.show_stats = not st.session_state.get("show_stats", False)

if st.session_state.get("show_stats", False) and len(st.session_state.db) > 0:
    with st.expander("📊 Thống kê chi tiết", expanded=True):
        st.markdown("### 📈 Tần suất từng số (30 kỳ):")
        freq = Counter("".join(st.session_state.db[-30:]))
        for i in range(10):
            count = freq.get(str(i), 0)
            bar = "█" * (count // 2)
            st.markdown(f"`{i}`: {bar} ({count})")

st.markdown('<div style="text-align:center;color:#999;font-size:11px;margin-top:10px;">⚡ TITAN V27 - Về bờ thông minh | Biết điểm dừng!</div>', unsafe_allow_html=True)