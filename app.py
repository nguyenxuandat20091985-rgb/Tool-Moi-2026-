import streamlit as st
import re, pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations

# --- CẤU HÌNH HỆ THỐNG ---
# ⚠️ ĐÃ XÓA API KEY ĐỂ BẢO MẬT (Vì code không dùng đến)
LUCKY_OX = [0, 2, 5, 6, 7, 8] # Tuổi Sửu 1985

st.set_page_config(page_title="TITAN V46 - REAL MONEY", page_icon="🐂", layout="centered")

# --- GIAO DIỆN ---
st.markdown("""
<style>
    .stApp {background-color: #030303; color: #FFD700;}
    .big-num {font-size: 38px; font-weight: bold; color: #00FFCC; text-align: center; font-family: 'Courier New', monospace; letter-spacing: 4px; text-shadow: 0 0 10px #00FFCC;}
    .box {background: linear-gradient(135deg, #0d1b1b, #030303); color: #FFD700; padding: 18px; border-radius: 15px; text-align: center; border: 1px solid #FFD700; margin-bottom: 12px; box-shadow: 0 4px 15px rgba(255, 215, 0, 0.1);}
    .item {background: linear-gradient(135deg, #00FFCC, #008B8B); color: #000; padding: 12px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold; box-shadow: 0 0 10px rgba(0, 255, 204, 0.3);}
    .item-3 {background: linear-gradient(135deg, #FFD700, #FFA500); color: #000; box-shadow: 0 0 10px rgba(255, 215, 0, 0.3);}
    .streak-badge {background-color: #FF3131; color: white; padding: 2px 8px; border-radius: 5px; font-size: 10px; font-weight: bold;}
    .momentum-tag {background-color: #008B8B; color: white; padding: 2px 6px; border-radius: 4px; font-size: 9px; margin-left: 5px;}
    .confidence-high {color: #00FF00; font-weight: bold; font-size: 20px;}
    .confidence-med {color: #FFD700; font-weight: bold; font-size: 20px;}
    .confidence-low {color: #FF3131; font-weight: bold; font-size: 20px;}
    h1 {text-align: center; color: #FFD700; text-shadow: 0 0 20px rgba(255, 215, 0, 0.5);}
</style>
""", unsafe_allow_html=True)

# --- THUẬT TOÁN ---

def get_nums(text):
    # SỬA LỖI: Xóa hết khoảng trắng trước khi tìm số để xử lý lỗi "6 6409"
    clean_text = re.sub(r'\s+', '', text)
    return [n for n in re.findall(r"\d{5}", clean_text) if n]

def check_gan_and_streak(db, combo):
    gan = 0
    streak = 0
    combo_set = set(combo)
    for num in reversed(db):
        if not combo_set.issubset(set(num)): gan += 1
        else: break
    for num in reversed(db):
        if combo_set.issubset(set(num)): streak += 1
        else: break
    return gan, streak

def calculate_momentum(db, combo):
    if len(db) < 60: return 0
    recent_10 = db[-10:]
    past_50 = db[-60:-10]
    count_recent = sum(1 for n in recent_10 if set(combo).issubset(set(n)))
    count_past = sum(1 for n in past_50 if set(combo).issubset(set(n)))
    rate_recent = count_recent / 10
    rate_past = count_past / 50
    if rate_recent > rate_past * 1.5: return 30
    if rate_recent > rate_past: return 15
    return -10

def get_anchor_numbers(db):
    if len(db) < 5: return []
    recent_str = "".join(db[-20:])
    counter = Counter(recent_str)
    top_2 = [item[0] for item in counter.most_common(2)]
    return top_2

def predict_v45_anti_cheat(db):
    if len(db) < 20: return None
    
    recent_db = db[-120:] 
    pair_pool = Counter()
    triple_pool = Counter()
    single_pool = Counter("".join(recent_db))
    
    for num in recent_db:
        u = sorted(list(set(num)))
        if len(u) >= 2:
            for p in combinations(u, 2): pair_pool[p] += 1
        if len(u) >= 3:
            for t in combinations(u, 3): triple_pool[t] += 1

    anchors = get_anchor_numbers(db)
    
    scored_pairs = []
    for p, freq in pair_pool.items():
        gan, streak = check_gan_and_streak(db, p)
        momentum = calculate_momentum(db, p)
        co_occurrence_rate = freq / ((single_pool[p[0]] + single_pool[p[1]]) / 2) if (single_pool[p[0]] + single_pool[p[1]]) > 0 else 0
        
        score = freq * 3.0
        if streak >= 1: score += 50
        if 1 <= gan <= 4: score += 40
        if co_occurrence_rate > 0.6: score += 25
        if any(int(d) in LUCKY_OX for d in p): score += 10
        if gan > 15: score -= 70
        score += momentum
        if p[0] in anchors or p[1] in anchors: score += 20
            
        scored_pairs.append(("".join(p), score, gan, streak, momentum))
        
    scored_triples = []
    for t, freq in triple_pool.items():
        gan, streak = check_gan_and_streak(db, t)
        momentum = calculate_momentum(db, t)
        score = freq * 4.0
        if streak >= 1: score += 60
        if 2 <= gan <= 6: score += 45
        if gan > 20: score -= 80
        score += momentum
        if any(d in anchors for d in t): score += 25
        scored_triples.append(("".join(t), score, gan, streak, momentum))

    res_p = sorted(scored_pairs, key=lambda x: x[1], reverse=True)[:3]
    res_t = sorted(scored_triples, key=lambda x: x[1], reverse=True)[:3]
    
    single_scores = {}
    for d in "0123456789":
        count = single_pool[d]
        bonus = 20 if d in anchors else 0
        single_scores[d] = count + bonus
        
    top_8_list = sorted(single_scores.items(), key=lambda x: x[1], reverse=True)[:8]
    top_8 = "".join([d for d, c in top_8_list])
    
    # Tính điểm tin cậy tổng thể
    confidence_score = 0
    if res_p:
        top_pair_score = res_p[0][1]
        if top_pair_score > 200: confidence_score = 95
        elif top_pair_score > 150: confidence_score = 80
        else: confidence_score = 60
    
    return {"pairs": res_p, "triples": res_t, "top8": top_8, "anchors": anchors, "confidence": confidence_score}

# --- GIAO DIỆN CHÍNH ---
st.markdown('<h1>🐂 TITAN V46 - REAL MONEY MATRIX</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#888;">Đã tối ưu xử lý dữ liệu lỗi & Bảo mật</p>', unsafe_allow_html=True)

if "history" not in st.session_state: st.session_state.history = []

user_input = st.text_area("📥 Dán bảng kết quả thực tế (Kỳ mới nhất ở dưới cùng):", height=150, placeholder="Ví dụ:\n12345\n6 6409\n94453")

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 SOI CẦU THỰC CHIẾN"):
        nums = get_nums(user_input)
        if len(nums) >= 15:
            if "last_pred" in st.session_state:
                lp = st.session_state.last_pred
                last_actual = nums[-1]
                best_pair = lp['pairs'][0][0]
                win_check = all(d in last_actual for d in best_pair)
                st.session_state.history.insert(0, {
                    "Kỳ": last_actual, 
                    "Cặp": best_pair, 
                    "Nhịp": f"G:{lp['pairs'][0][2]} B:{lp['pairs'][0][3]}",
                    "KQ": "🔥 WIN" if win_check else "❌"
                })
            st.session_state.last_pred = predict_v45_anti_cheat(nums)
            st.rerun()
        else:
            st.warning("Dán thêm dữ liệu đi anh Đạt ơi (ít nhất 15 kỳ)!")

with col2:
    if st.button("🗑️ LÀM MỚI"):
        st.session_state.clear()
        st.rerun()

# --- HIỂN THỊ KẾT QUẢ ---
if "last_pred" in st.session_state:
    res = st.session_state.last_pred
    
    # Hiển thị độ tin cậy
    conf_class = "confidence-high" if res['confidence'] >= 90 else ("confidence-med" if res['confidence'] >= 75 else "confidence-low")
    conf_text = "CAO (Vào tiền)" if res['confidence'] >= 90 else ("TB (Thăm dò)" if res['confidence'] >= 75 else "THẤP (Xem thêm)")
    st.markdown(f"<div class='box' style='border-color:#00FFCC;'>🛡 ĐỘ TIN CẬY: <span class='{conf_class}'>{conf_text} ({res['confidence']}%)</span></div>", unsafe_allow_html=True)

    # Hiển thị số Neo
    if res['anchors']:
        anchor_html = f"<span class='big-num' style='color:#FFD700'>{res['anchors'][0]}</span>"
        if len(res['anchors']) > 1:
            anchor_html += f" - <span class='big-num' style='color:#FFD700'>{res['anchors'][1]}</span>"
        st.markdown(f"<div class='box' style='border-color:#00FFCC;'>🔥 SỐ NEO (HOT NUMBERS): {anchor_html}</div>", unsafe_allow_html=True)
    
    st.markdown(f"<div class='box'>🔥 ĐỘ PHỦ SẢNH (8 SỐ): <span class='big-num'>{res['top8']}</span></div>", unsafe_allow_html=True)
    
    # 2 TINH
    st.markdown("<div class='box'>🎯 2 TINH MATRIX (ƯU TIÊN BỆT & NHỊP RƠI)</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for i, (p, score, gan, streak, mom) in enumerate(res['pairs']):
        with [c1, c2, c3][i]: 
            st.markdown(f"<div class='item'>{p[0]} , {p[1]}</div>", unsafe_allow_html=True)
            tags = ""
            if streak > 0: tags += f"<span class='streak-badge'>BỆT {streak}</span> "
            if mom > 20: tags += f"<span class='momentum-tag'>HOT {mom}</span>"
            st.markdown(f"<div style='font-size:12px; margin-top:5px;'>Gan: {gan} kỳ {tags}</div>", unsafe_allow_html=True)
        
    # 3 TINH
    st.markdown("<div class='box' style='border-color:#FFD700;'>💎 3 TINH MATRIX (SIÊU CẤP CHUYÊN BIỆT)</div>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    for i, (t, score, gan, streak, mom) in enumerate(res['triples']):
        with [d1, d2, d3][i]: 
            st.markdown(f"<div class='item item-3'>{t[0]},{t[1]},{t[2]}</div>", unsafe_allow_html=True)
            tags = ""
            if streak > 0: tags += f"<span class='streak-badge'>BỆT {streak}</span> "
            if mom > 20: tags += f"<span class='momentum-tag'>HOT {mom}</span>"
            st.markdown(f"<div style='font-size:12px; margin-top:5px;'>Gan: {gan} kỳ {tags}</div>", unsafe_allow_html=True)

# --- BẢNG ĐỐI SOÁT ---
if st.session_state.history:
    st.divider()
    st.subheader("📋 Nhật ký đối soát (Bao sảnh 5 Tinh)")
    df_history = pd.DataFrame(st.session_state.history).head(15)
    st.table(df_history)