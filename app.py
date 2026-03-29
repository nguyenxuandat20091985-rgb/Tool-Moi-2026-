import streamlit as st
import re, json, os, pandas as pd
from collections import Counter
from itertools import combinations

# --- CẤU HÌNH HỆ THỐNG ---
# Tuổi Sửu 1985 - Mệnh Kim
LUCKY_OX = [0, 2, 5, 6, 7, 8]
SHADOW_MAP = {"0":"5", "5":"0", "1":"6", "6":"1", "2":"7", "7":"2", "3":"8", "8":"3", "4":"9", "9":"4"}

st.set_page_config(page_title="TITAN V38 PRO", page_icon="🐂", layout="centered")

# --- GIAO DIỆN DARK MODE GOLDEN ---
st.markdown("""
<style>
    .stApp {background-color: #050505; color: #FFD700;}
    .big-num {font-size: 42px; font-weight: bold; color: #00FFCC; text-align: center; font-family: monospace; letter-spacing: 4px;}
    .box {background: linear-gradient(135deg, #1C3A3A, #050505); color: #FFD700; padding: 15px; border-radius: 12px; text-align: center; border: 2px solid #FFD700; margin-bottom: 10px;}
    .item {background: linear-gradient(135deg, #00FFCC, #008B8B); color: #000; padding: 12px; border-radius: 8px; text-align: center; font-size: 28px; font-weight: bold;}
    .item-3 {background: linear-gradient(135deg, #FFD700, #FFA500); color: #000;}
    .gan-alert {color: #FF3131; font-size: 14px; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# --- THUẬT TOÁN ĐẮP THỊT V38 ---
def get_nums(text):
    # Lọc chuẩn 5 số, loại bỏ rác văn bản
    return [n for n in re.findall(r"\d{5}", text) if n]

def analyze_gan(db, target_list, size=2):
    """Tính nhịp gan (số kỳ chưa về) của các tổ hợp"""
    gan_data = {}
    for combo in target_list:
        combo_set = set(combo)
        miss_count = 0
        for num in reversed(db):
            if not combo_set.issubset(set(num)):
                miss_count += 1
            else:
                break
        gan_data["".join(combo)] = miss_count
    return gan_data

def predict_v38(db):
    if len(db) < 15: return None
    
    # 1. Thu thập toàn bộ tổ hợp đã nổ
    pair_counts = Counter()
    triple_counts = Counter()
    recent_db = db[-80:]
    
    for num in recent_db:
        unique_digits = sorted(list(set(num)))
        if len(unique_digits) >= 2:
            for p in combinations(unique_digits, 2): pair_counts[tuple(p)] += 1
        if len(unique_digits) >= 3:
            for t in combinations(unique_digits, 3): triple_counts[tuple(t)] += 1

    # 2. Tính toán điểm ELITE MATRIX
    last_num_set = set(db[-1])
    
    # Phân tích Gan cho Top các tổ hợp xuất hiện nhiều
    top_p = [p for p, c in pair_counts.most_common(20)]
    gan_p = analyze_gan(db, top_p, 2)
    
    scored_pairs = []
    for p in top_p:
        p_str = "".join(p)
        # Điểm cơ bản = Tần suất * 3
        score = pair_counts[p] * 3.0
        # Thưởng Cầu Rơi (Nếu có số vừa nổ)
        if any(d in last_num_set for d in p): score += 20
        # Thưởng Mệnh Kim (Bóng)
        if any(SHADOW_MAP.get(d) in last_num_set for d in p): score += 12
        # TRỪ ĐIỂM GAN (Né những cặp đang bị giam quá 15 kỳ - tránh cháy túi)
        if gan_p[p_str] > 15: score -= 30
        # CỘNG ĐIỂM NHỊP (Nếu gan từ 2-5 kỳ là nhịp nổ đẹp nhất)
        if 2 <= gan_p[p_str] <= 5: score += 25
        
        scored_pairs.append((p_str, score, gan_p[p_str]))

    # Sắp xếp lấy 3 cặp tinh túy nhất
    final_pairs = sorted(scored_pairs, key=lambda x: x[1], reverse=True)[:3]

    # Tương tự cho 3 TINH
    top_t = [t for t, c in triple_counts.most_common(20)]
    gan_t = analyze_gan(db, top_t, 3)
    scored_triples = []
    for t in top_t:
        t_str = "".join(t)
        score = triple_counts[t] * 4.0
        if any(d in last_num_set for d in t): score += 25
        if 3 <= gan_t[t_str] <= 8: score += 30 # Nhịp rơi 3 TINH xa hơn chút
        if gan_t[t_str] > 20: score -= 40
        scored_triples.append((t_str, score, gan_t[t_str]))
        
    final_triples = sorted(scored_triples, key=lambda x: x[1], reverse=True)[:3]
    
    # Top 8 số phủ sảnh
    top_8 = "".join([d for d, c in Counter("".join(db[-50:])).most_common(8)])
    
    return {"pairs": final_pairs, "triples": final_triples, "top8": top_8}

# --- GIAO DIỆN CHÍNH ---
st.markdown('<h1>🐂 TITAN V38 - ELITE MATRIX</h1>', unsafe_allow_html=True)

if "history" not in st.session_state: st.session_state.history = []

user_input = st.text_area("📥 Dán bảng kết quả (Kỳ mới nhất nằm dưới cùng):", height=150)

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 PHÂN TÍCH MATRIX"):
        nums = get_nums(user_input)
        if len(nums) >= 10:
            if "last_pred" in st.session_state:
                lp = st.session_state.last_pred
                last_actual = nums[-1]
                # Thắng 5 tinh: tất cả số trong cặp phải nằm trong kết quả
                win_check = all(d in last_actual for d in lp['pairs'][0][0])
                st.session_state.history.insert(0, {
                    "Kỳ": last_actual, 
                    "Cặp": lp['pairs'][0][0], 
                    "Gan": lp['pairs'][0][2],
                    "KQ": "🔥 WIN" if win_check else "❌"
                })

            st.session_state.last_pred = predict_v38(nums)
            st.rerun()
        else:
            st.warning("Dán ít nhất 15 kỳ để soi nhịp gan anh ơi!")

with col2:
    if st.button("🗑️ RESET"):
        st.session_state.clear()
        st.rerun()

# --- HIỂN THỊ ---
if "last_pred" in st.session_state:
    res = st.session_state.last_pred
    
    st.markdown(f"<div class='box'>🎯 PHỦ SẢNH 8 SỐ: <span class='big-num'>{res['top8']}</span></div>", unsafe_allow_html=True)
    
    # 2 TINH
    st.markdown("<div class='box'>🏆 2 SỐ 5 TINH - MATRIX NHỊP RƠI</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for i, (p, score, gan) in enumerate(res['pairs']):
        with [c1, c2, c3][i]: 
            st.markdown(f"<div class='item'>{p[0]},{p[1]}</div>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>Gan: {gan} kỳ</p>", unsafe_allow_html=True)
            
    # 3 TINH
    st.markdown("<div class='box' style='border-color:#FFD700;'>💎 3 SỐ 5 TINH - MATRIX BỘ BA</div>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    for i, (t, score, gan) in enumerate(res['triples']):
        with [d1, d2, d3][i]: 
            st.markdown(f"<div class='item item-3'>{t[0]},{t[1]},{t[2]}</div>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align:center;'>Gan: {gan} kỳ</p>", unsafe_allow_html=True)

if st.session_state.history:
    st.divider()
    st.subheader("📋 Đối soát thắng/thua thực tế")
    st.table(pd.DataFrame(st.session_state.history).head(10))
