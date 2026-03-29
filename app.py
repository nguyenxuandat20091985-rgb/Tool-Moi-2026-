import streamlit as st
import re, json, os, pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
from openai import OpenAI
import google.generativeai as genai

# --- CẤU HÌNH HỆ THỐNG (GIỮ NGUYÊN BẢO MẬT) ---
# Lưu ý: Anh nên giấu key trong secrets streamlit khi deploy thực tế, ở đây em giữ nguyên để anh test
NVIDIA_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"

# Tuổi Sửu 1985 - Mệnh Kim
LUCKY_OX = [0, 2, 5, 6, 7, 8]

st.set_page_config(page_title="TITAN V45 PRO - ULTIMATE", page_icon="🐂", layout="centered")

# --- GIAO DIỆN DARK MODE PRO (NÂNG CẤP UI) ---
st.markdown("""
<style>
    .stApp {background-color: #030303; color: #FFD700;}
    .big-num {font-size: 38px; font-weight: bold; color: #00FFCC; text-align: center; font-family: 'Courier New', monospace; letter-spacing: 4px; text-shadow: 0 0 10px #00FFCC;}
    .box {background: linear-gradient(135deg, #0d1b1b, #030303); color: #FFD700; padding: 18px; border-radius: 15px; text-align: center; border: 1px solid #FFD700; margin-bottom: 12px; box-shadow: 0 4px 15px rgba(255, 215, 0, 0.1);}
    .item {background: linear-gradient(135deg, #00FFCC, #008B8B); color: #000; padding: 12px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold; box-shadow: 0 0 10px rgba(0, 255, 204, 0.3);}
    .item-3 {background: linear-gradient(135deg, #FFD700, #FFA500); color: #000; box-shadow: 0 0 10px rgba(255, 215, 0, 0.3);}
    .status-win {color: #00FF00; font-weight: bold; text-shadow: 0 0 5px #00FF00;}
    .status-lose {color: #FF3131; font-weight: bold;}
    .streak-badge {background-color: #FF3131; color: white; padding: 2px 8px; border-radius: 5px; font-size: 10px; font-weight: bold;}
    .momentum-tag {background-color: #008B8B; color: white; padding: 2px 6px; border-radius: 4px; font-size: 9px; margin-left: 5px;}
    h1 {text-align: center; color: #FFD700; text-shadow: 0 0 20px rgba(255, 215, 0, 0.5);}
</style>
""", unsafe_allow_html=True)

# --- THUẬT TOÁN PHÂN TÍCH CHUYÊN SÂU ---

def get_nums(text):
    return [n for n in re.findall(r"\d{5}", text) if n]

def check_gan_and_streak(db, combo):
    """Tính toán nhịp Gan và nhịp Bệt của tổ hợp"""
    gan = 0
    streak = 0
    combo_set = set(combo)
    
    # Tính Gan (đếm ngược từ dưới lên)
    for num in reversed(db):
        if not combo_set.issubset(set(num)): gan += 1
        else: break
            
    # Tính Bệt (Streak) - Kiểm tra 5 kỳ gần nhất xem nổ mấy kỳ liên tục
    for num in reversed(db):
        if combo_set.issubset(set(num)): streak += 1
        else: break
            
    return gan, streak

def calculate_momentum(db, combo):
    """
    Tính điểm đà (Momentum): 
    So sánh tần suất xuất hiện của cặp trong 10 kỳ gần nhất vs 50 kỳ trước đó.
    Nếu tần suất gần đây cao hơn -> Cộng điểm mạnh.
    """
    if len(db) < 60: return 0
    
    recent_10 = db[-10:]
    past_50 = db[-60:-10]
    
    count_recent = sum(1 for n in recent_10 if set(combo).issubset(set(n)))
    count_past = sum(1 for n in past_50 if set(combo).issubset(set(n)))
    
    # Chuẩn hóa tần suất (trên mỗi kỳ)
    rate_recent = count_recent / 10
    rate_past = count_past / 50
    
    if rate_recent > rate_past * 1.5: return 30 # Đang vào cầu nóng
    if rate_recent > rate_past: return 15      # Ổn định
    return -10                                 # Đang nguội

def get_anchor_numbers(db):
    """Tìm ra các số 'Neo' (Hot numbers) trong 20 kỳ gần nhất"""
    if len(db) < 5: return []
    recent_str = "".join(db[-20:])
    counter = Counter(recent_str)
    # Lấy 2 số xuất hiện nhiều nhất
    top_2 = [item[0] for item in counter.most_common(2)]
    return top_2

def predict_v45_anti_cheat(db):
    if len(db) < 20: return None
    
    recent_db = db[-120:] 
    pair_pool = Counter()
    triple_pool = Counter()
    single_pool = Counter("".join(recent_db))
    
    # 1. Thu thập dữ liệu tần suất
    for num in recent_db:
        u = sorted(list(set(num)))
        if len(u) >= 2:
            for p in combinations(u, 2): pair_pool[p] += 1
        if len(u) >= 3:
            for t in combinations(u, 3): triple_pool[t] += 1

    # Lấy số Neo (Anchor) để tối ưu hóa
    anchors = get_anchor_numbers(db)
    
    # 2. Xử lý 2 TINH - Điểm Bayes & Chống Lừa & Momentum
    scored_pairs = []
    for p, freq in pair_pool.items():
        gan, streak = check_gan_and_streak(db, p)
        momentum = calculate_momentum(db, p)
        
        # Chỉ số "Chống Lừa": Nếu 2 số đơn nổ nhiều nhưng ít đi cùng nhau -> Giảm điểm
        co_occurrence_rate = freq / ((single_pool[p[0]] + single_pool[p[1]]) / 2) if (single_pool[p[0]] + single_pool[p[1]]) > 0 else 0
        
        score = freq * 3.0
        
        # --- CÁC YẾU TỐ NÂNG CẤP ---
        if streak >= 1: score += 50  # Thưởng nhịp Bệt
        if 1 <= gan <= 4: score += 40 # Thưởng nhịp rơi chuẩn
        if co_occurrence_rate > 0.6: score += 25 # Thưởng cặp số "trung thành"
        if any(int(d) in LUCKY_OX for d in p): score += 10 # Tuổi Sửu 1985
        if gan > 15: score -= 70 # Loại bỏ số bị nhà cái giam
        
        # Yếu tố Momentum (Mới)
        score += momentum 
        
        # Yếu tố Anchor (Mới): Ưu tiên cặp có chứa số đang Hot
        if p[0] in anchors or p[1] in anchors:
            score += 20
            
        scored_pairs.append(("".join(p), score, gan, streak, momentum))
        
    # 3. Xử lý 3 TINH
    scored_triples = []
    for t, freq in triple_pool.items():
        gan, streak = check_gan_and_streak(db, t)
        momentum = calculate_momentum(db, t)
        
        score = freq * 4.0
        if streak >= 1: score += 60
        if 2 <= gan <= 6: score += 45
        if gan > 20: score -= 80
        
        # Yếu tố Momentum cho 3 số
        score += momentum
        
        # Ưu tiên 3 số có chứa ít nhất 1 số Neo
        if any(d in anchors for d in t):
            score += 25
            
        scored_triples.append(("".join(t), score, gan, streak, momentum))

    res_p = sorted(scored_pairs, key=lambda x: x[1], reverse=True)[:3]
    res_t = sorted(scored_triples, key=lambda x: x[1], reverse=True)[:3]
    
    # Phủ sảnh 8 số dùng Bayes đơn lẻ + Momentum
    # Tính điểm cho từng số đơn dựa trên tần suất gần đây
    single_scores = {}
    for d in "0123456789":
        count = single_pool[d]
        # Cộng thêm trọng số nếu là Anchor
        bonus = 20 if d in anchors else 0
        single_scores[d] = count + bonus
        
    top_8_list = sorted(single_scores.items(), key=lambda x: x[1], reverse=True)[:8]
    top_8 = "".join([d for d, c in top_8_list])
    
    return {"pairs": res_p, "triples": res_t, "top8": top_8, "anchors": anchors}

# --- GIAO DIỆN CHÍNH ---
st.markdown('<h1>🐂 TITAN V45 PRO - ANTI-CHEAT MATRIX</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#888;">Thuật toán Momentum + Anchor System + Bayes</p>', unsafe_allow_html=True)

if "history" not in st.session_state: st.session_state.history = []

user_input = st.text_area("📥 Dán bảng kết quả thực tế (Kỳ mới nhất ở dưới cùng):", height=150, placeholder="Ví dụ:\n12345\n67890\n...")

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 SOI CẦU CHỐNG LỪA"):
        nums = get_nums(user_input)
        if len(nums) >= 15:
            if "last_pred" in st.session_state:
                lp = st.session_state.last_pred
                last_actual = nums[-1]
                # Kiểm tra thắng cho cặp VIP nhất
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

# --- HIỂN THỊ KẾT QUẢ TỐI ƯU ---
if "last_pred" in st.session_state:
    res = st.session_state.last_pred
    
    # Hiển thị số Neo (Anchor)
    if res['anchors']:
        st.markdown(f"<div class='box' style='border-color:#00FFCC;'>🔥 SỐ NEO (HOT NUMBERS): <span class='big-num' style='color:#FFD700'>{res['anchors'][0]}</span> {f'- <span class=\"big-num\" style=\"color:#FFD700\">{res[\"anchors\"][1]}</span>' if len(res['anchors'])>1 else ''}</div>", unsafe_allow_html=True)
    
    st.markdown(f"<div class='box'>🔥 ĐỘ PHỦ SẢNH (8 SỐ): <span class='big-num'>{res['top8']}</span></div>", unsafe_allow_html=True)
    
    # 2 TINH
    st.markdown("<div class='box'>🎯 2 TINH MATRIX (ƯU TIÊN BỆT & NHỊP RƠI)</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for i, (p, score, gan, streak, mom) in enumerate(res['pairs']):
        with [c1, c2, c3][i]: 
            # Highlight nếu có chứa số Neo
            is_anchor = p[0] in res['anchors'] or p[1] in res['anchors']
            style_class = "item" if not is_anchor else "item" # Giữ nguyên style nhưng có thể thêm logic màu nếu muốn
            
            st.markdown(f"<div class='{style_class}'>{p[0]} , {p[1]}</div>", unsafe_allow_html=True)
            
            tags = ""
            if streak > 0: tags += f"<span class='streak-badge'>BỆT {streak}</span> "
            if mom > 20: tags += f"<span class='momentum-tag'>HOT {mom}</span>"
            
            st.markdown(f"<div style='font-size:12px; margin-top:5px;'>Gan: {gan} kỳ {tags}</div>", unsafe_allow_html=True)
        
    # 3 TINH
    st.markdown("<div class='box' style='border-color:#FFD700;'>💎 3 TINH MATRIX (SIÊU CẤP CHUYÊN BIỆT)</div>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    for i, (t, score, gan, streak, mom) in enumerate(res['triples']):
        with [d1, d2, d3][i]: 
            is_anchor = any(d in res['anchors'] for d in t)
            st.markdown(f"<div class='item item-3'>{t[0]},{t[1]},{t[2]}</div>", unsafe_allow_html=True)
            
            tags = ""
            if streak > 0: tags += f"<span class='streak-badge'>BỆT {streak}</span> "
            if mom > 20: tags += f"<span class='momentum-tag'>HOT {mom}</span>"
            
            st.markdown(f"<div style='font-size:12px; margin-top:5px;'>Gan: {gan} kỳ {tags}</div>", unsafe_allow_html=True)

# --- BẢNG ĐỐI SOÁT THỰC CHIẾN ---
if st.session_state.history:
    st.divider()
    st.subheader("📋 Nhật ký đối soát (Bao sảnh 5 Tinh)")
    df_history = pd.DataFrame(st.session_state.history).head(15)
    # Tô màu cho cột KQ
    def color_val(val):
        color = '#00FF00' if 'WIN' in val else '#FF3131'
        return f'color: {color}; font-weight: bold'
    
    st.table(df_history.style.applymap(color_val, subset=['KQ']))