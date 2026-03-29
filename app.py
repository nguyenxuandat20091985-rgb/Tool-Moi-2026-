import streamlit as st
import re, json, pandas as pd
from collections import Counter
from itertools import combinations

# --- CẤU HÌNH ---
SHADOW_MAP = {"0":"5", "5":"0", "1":"6", "6":"1", "2":"7", "7":"2", "3":"8", "8":"3", "4":"9", "9":"4"}
LUCKY_OX = [0, 2, 5, 6, 7, 8] # Tuổi Sửu 1985 - Mệnh Kim

st.set_page_config(page_title="TITAN V36 - 5 TINH PRO", page_icon="🐂", layout="centered")

st.markdown("""
<style>
    .stApp {background-color: #050505; color: #FFD700;}
    .big-num {font-size: 45px; font-weight: bold; color: #00FFCC; text-align: center; font-family: monospace;}
    .box {background: linear-gradient(135deg, #1C3A3A, #050505); color: #FFD700; padding: 15px; border-radius: 12px; text-align: center; border: 2px solid #FFD700; margin-bottom: 10px;}
    .item {background: linear-gradient(135deg, #00FFCC, #008B8B); color: #000; padding: 15px; border-radius: 8px; text-align: center; font-size: 30px; font-weight: bold;}
    .item-3 {background: linear-gradient(135deg, #FFD700, #B8860B); color: #000;}
    .status-win {color: #00FF00; font-weight: bold;}
    .status-loss {color: #FF3131; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

def get_nums(text):
    return [n for n in re.findall(r"\d{5}", text) if n]

def predict_5_tinh(db):
    if len(db) < 10: return None
    
    # 1. Phân tích tần suất đơn lẻ
    all_digits = "".join(db[-50:])
    digit_counts = Counter(all_digits)
    
    # 2. Phân tích CẶP (2 số) hay xuất hiện cùng nhau trong 1 kỳ
    pair_counter = Counter()
    triple_counter = Counter()
    
    for num in db[-60:]:
        unique_digits = sorted(list(set(num)))
        if len(unique_digits) >= 2:
            for p in combinations(unique_digits, 2):
                pair_counter["".join(p)] += 1
        if len(unique_digits) >= 3:
            for t in combinations(unique_digits, 3):
                triple_counter["".join(t)] += 1

    # 3. Tính điểm ưu tiên (Scoring)
    # Kết hợp: Tần suất cặp + Cầu rơi kỳ trước + Bóng + Tuổi Sửu
    last_num_set = set(db[-1])
    
    # Lấy top các cặp có tần suất cao nhất
    top_pairs_raw = pair_counter.most_common(15)
    scored_pairs = []
    for p, count in top_pairs_raw:
        score = count * 2.0
        # Thưởng điểm nếu có số vừa nổ kỳ trước (Cầu rơi)
        if any(d in last_num_set for d in p): score += 15
        # Thưởng điểm nếu là số may mắn tuổi Sửu
        if any(int(d) in LUCKY_OX for d in p): score += 10
        scored_pairs.append((p, score))
    
    # Lấy top các bộ 3
    top_triples_raw = triple_counter.most_common(15)
    scored_triples = []
    for t, count in top_triples_raw:
        score = count * 2.5
        if any(d in last_num_set for d in t): score += 20
        if any(int(d) in LUCKY_OX for d in t): score += 10
        scored_triples.append((t, score))

    # Sắp xếp lại theo điểm đã đắp thịt
    final_pairs = [x[0] for x in sorted(scored_pairs, key=lambda x: x[1], reverse=True)][:3]
    final_triples = [x[0] for x in sorted(scored_triples, key=lambda x: x[1], reverse=True)][:3]
    
    return {
        "pairs": final_pairs,
        "triples": final_triples,
        "top_digits": "".join([d for d, c in digit_counts.most_common(8)])
    }

# --- GIAO DIỆN CHÍNH ---
st.markdown('<h1>🐂 TITAN V36 - CHUYÊN GIA 5 TINH</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Luật: Chọn 2 hoặc 3 số xuất hiện bất kỳ trong 5 hàng</p>", unsafe_allow_html=True)

if "history" not in st.session_state: st.session_state.history = []

user_input = st.text_area("📥 Dán kết quả vào đây:", height=150, placeholder="Ví dụ: 12121\n12864...")

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 CHỐT TỔ HỢP 5 TINH"):
        nums = get_nums(user_input)
        if len(nums) >= 10:
            # Check thắng thua kỳ trước
            if "last_res" in st.session_state:
                lr = st.session_state.last_res
                last_actual = nums[-1]
                # Thắng 2 số: Cả 2 số trong cặp đều có mặt trong kết quả
                win_p = any(all(d in last_actual for d in p) for p in lr['pairs'])
                st.session_state.history.insert(0, {
                    "Kỳ": last_actual, 
                    "Cặp soi": lr['pairs'][0], 
                    "KQ": "🔥 WIN" if win_p else "❌"
                })

            st.session_state.last_res = predict_5_tinh(nums)
            st.rerun()
        else:
            st.warning("Dán ít nhất 10 kỳ để phân tích cặp số anh ơi!")

with col2:
    if st.button("🗑️ LÀM MỚI"):
        st.session_state.clear()
        st.rerun()

# --- HIỂN THỊ ---
if "last_res" in st.session_state:
    res = st.session_state.last_res
    
    st.markdown(f"<div class='box'>🎯 8 SỐ XUẤT HIỆN NHIỀU: <span style='font-size:30px; color:#00FFCC;'>{res['top_digits']}</span></div>", unsafe_allow_html=True)

    # 2 SỐ 5 TINH
    st.markdown("<div class='box'>🏆 2 SỐ 5 TINH (Cược 2 số bất kỳ)</div>", unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    for i, pair in enumerate(res['pairs']):
        with [p1, p2, p3][i]: 
            # Hiển thị dạng số cách nhau cho anh dễ nhìn
            st.markdown(f"<div class='item'>{pair[0]} , {pair[1]}</div>", unsafe_allow_html=True)

    # 3 SỐ 5 TINH
    st.markdown("<div class='box' style='border-color:#FFD700;'>🏆 3 SỐ 5 TINH (Cược 3 số bất kỳ)</div>", unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    for i, tri in enumerate(res['triples']):
        with [t1, t2, t3][i]: 
            st.markdown(f"<div class='item item-3'>{tri[0]},{tri[1]},{tri[2]}</div>", unsafe_allow_html=True)

if st.session_state.history:
    st.divider()
    st.subheader("📋 Nhật ký đối soát (2 số)")
    st.table(pd.DataFrame(st.session_state.history).head(10))
