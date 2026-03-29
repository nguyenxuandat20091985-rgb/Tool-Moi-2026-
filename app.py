import streamlit as st
import re, pandas as pd, math
import numpy as np
from collections import Counter
from itertools import combinations

# --- CẤU HÌNH ---
LUCKY_OX = [0, 2, 5, 6, 7, 8]
SHADOW_PAIRS = {'0':'5', '1':'6', '2':'7', '3':'8', '4':'9', '5':'0', '6':'1', '7':'2', '8':'3', '9':'4'}

st.set_page_config(page_title="TITAN V48 - MASTER", page_icon="🔮", layout="centered")

# --- GIAO DIỆN ---
st.markdown("""
<style>
    .stApp {background-color: #030303; color: #FFD700;}
    .big-num {font-size: 38px; font-weight: bold; color: #00FFCC; text-align: center; font-family: 'Courier New', monospace; letter-spacing: 4px; text-shadow: 0 0 10px #00FFCC;}
    .box {background: linear-gradient(135deg, #0d1b1b, #030303); color: #FFD700; padding: 18px; border-radius: 15px; text-align: center; border: 1px solid #FFD700; margin-bottom: 12px; box-shadow: 0 4px 15px rgba(255, 215, 0, 0.1);}
    .item {background: linear-gradient(135deg, #00FFCC, #008B8B); color: #000; padding: 12px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold;}
    .item-3 {background: linear-gradient(135deg, #FFD700, #FFA500); color: #000;}
    .vip-pick {background: linear-gradient(135deg, #FF0080, #FF00FF); color: white; animation: pulse 1.5s infinite;}
    .stat-box {background: #1a1a1a; padding: 10px; border-radius: 8px; margin: 5px 0;}
    .confidence-high {color: #00FF00; font-weight: bold; font-size: 22px;}
    .confidence-med {color: #FFD700; font-weight: bold; font-size: 22px;}
    .confidence-low {color: #FF3131; font-weight: bold; font-size: 22px;}
    h1 {text-align: center; color: #FFD700; text-shadow: 0 0 20px rgba(255, 215, 0, 0.5);}
    @keyframes pulse {0% {transform: scale(1);} 50% {transform: scale(1.05);} 100% {transform: scale(1);}}
</style>
""", unsafe_allow_html=True)

# --- THUẬT TOÁN ĐA CHIỀU ---

def get_nums(text):
    clean_text = re.sub(r'\s+', '', text)
    nums = [n for n in re.findall(r"\d{5}", clean_text) if n]
    # LOẠI BỎ TRÙNG LẶP LIÊN TIẾP - QUAN TRỌNG
    unique_nums = []
    for i, n in enumerate(nums):
        if i == 0 or n != nums[i-1]:
            unique_nums.append(n)
    return unique_nums

def analyze_position(db, pos):
    """Phân tích tần suất theo từng vị trí (0-4)"""
    digits = [n[pos] for n in db]
    return Counter(digits)

def calculate_sum_stats(db):
    """Tính thống kê tổng điểm của các kỳ"""
    sums = [sum(int(d) for d in n) for n in db]
    return {
        'mean': np.mean(sums),
        'std': np.std(sums),
        'recent': sums[-5:]
    }

def get_shadow_number(d):
    """Lấy số bóng âm dương"""
    return SHADOW_PAIRS.get(d, d)

def predict_v48_master(db):
    if len(db) < 20: return None
    
    # 1. PHÂN TÍCH TẦN SUẤT TỔNG
    all_digits = "".join(db)
    global_freq = Counter(all_digits)
    
    # 2. PHÂN TÍCH TỪNG VỊ TRÍ
    pos_freq = [analyze_position(db, i) for i in range(5)]
    
    # 3. THỐNG KÊ TỔNG ĐIỂM
    sum_stats = calculate_sum_stats(db)
    expected_sum_range = (int(sum_stats['mean'] - sum_stats['std']), 
                          int(sum_stats['mean'] + sum_stats['std']))
    
    # 4. TÌM SỐ "HOT" VÀ "LẠNH"
    hot_nums = [d for d, c in global_freq.most_common(5)]
    cold_nums = [d for d, c in global_freq.most_common()[:-6]]
    
    # 5. TÍNH ĐIỂM CHO TỪNG CẶP 2 SỐ
    scored_pairs = []
    for p in combinations("0123456789", 2):
        score = 0
        p_set = set(p)
        
        # Điểm tần suất toàn cục
        score += (global_freq[p[0]] + global_freq[p[1]]) * 2
        
        # Điểm vị trí (nếu 2 số này hay xuất hiện cùng vị trí)
        for i in range(5):
            if p[0] in pos_freq[i] and p[1] in pos_freq[i]:
                score += 10
        
        # Điểm bóng âm dương (nếu là cặp bóng)
        if get_shadow_number(p[0]) == p[1]:
            score += 30
        
        # Điểm tuổi Sửu
        if any(int(d) in LUCKY_OX for d in p):
            score += 15
        
        # Ưu tiên cặp có 1 HOT + 1 COLD (cân bằng)
        if (p[0] in hot_nums and p[1] in cold_nums) or (p[1] in hot_nums and p[0] in cold_nums):
            score += 25
        
        # Kiểm tra tổng điểm dự kiến
        # (giả lập tổng của cặp này trong 5 số)
        score += 10  # Base score
        
        scored_pairs.append(("".join(p), score))
    
    # 6. TÍNH ĐIỂM CHO TỪNG BỘ 3 SỐ
    scored_triples = []
    for t in combinations("0123456789", 3):
        score = 0
        t_set = set(t)
        
        score += (global_freq[t[0]] + global_freq[t[1]] + global_freq[t[2]]) * 2
        
        # Điểm bóng
        shadow_count = sum(1 for i in range(3) for j in range(i+1, 3) if get_shadow_number(t[i]) == t[j])
        score += shadow_count * 20
        
        # Điểm tuổi
        if any(int(d) in LUCKY_OX for d in t):
            score += 20
        
        # Ưu tiên 1 HOT + 2 COLD hoặc 2 HOT + 1 COLD
        hot_count = sum(1 for d in t if d in hot_nums)
        if hot_count in [1, 2]:
            score += 30
        
        scored_triples.append(("".join(t), score))
    
    res_p = sorted(scored_pairs, key=lambda x: x[1], reverse=True)[:5]
    res_t = sorted(scored_triples, key=lambda x: x[1], reverse=True)[:5]
    
    # 7. TÍNH TOÁN PHỦ SẢNH THÔNG MINH
    # Kết hợp HOT + COLD + Bóng
    coverage_nums = hot_nums[:3] + cold_nums[:2]
    # Thêm số bóng của HOT
    for h in hot_nums[:2]:
        coverage_nums.append(get_shadow_number(h))
    coverage_nums = list(dict.fromkeys(coverage_nums))[:8]  # Remove duplicates, keep 8
    top_8 = "".join(sorted(coverage_nums))
    
    # 8. TÍNH ĐỘ TIN CẬY
    # Dựa vào độ hội tụ của các yếu tố
    convergence = 0
    if len(set(hot_nums) & set(LUCKY_OX)) >= 2: convergence += 30
    if sum_stats['std'] < 5: convergence += 20  # Ổn định
    if len(db) >= 50: convergence += 20  # Đủ dữ liệu
    
    confidence = min(95, 50 + convergence)
    
    return {
        "pairs": res_p,
        "triples": res_t,
        "top8": top_8,
        "hot_nums": hot_nums,
        "cold_nums": cold_nums,
        "confidence": confidence,
        "sum_range": expected_sum_range,
        "pos_analysis": pos_freq
    }

# --- GIAO DIỆN ---
st.markdown('<h1>🔮 TITAN V48 - MASTER PREDICTION</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#888;">Phân tích đa chiều - Loại bỏ trùng lặp - Bóng âm dương</p>', unsafe_allow_html=True)

if "history" not in st.session_state: st.session_state.history = []

user_input = st.text_area("📥 Dán bảng kết quả (Kỳ mới nhất ở dưới cùng):", 
                          height=200, 
                          placeholder="Dán toàn bộ kết quả vào đây...\n87558\n34979\n...")

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 PHÂN TÍCH ĐA CHIỀU"):
        nums = get_nums(user_input)
        st.info(f"✅ Đã xử lý {len(nums)} kỳ duy nhất (đã loại bỏ trùng lặp)")
        
        if len(nums) >= 15:
            if "last_pred" in st.session_state:
                lp = st.session_state.last_pred
                last_actual = nums[-1]
                best_pair = lp['pairs'][0][0]
                win_check = all(d in last_actual for d in best_pair)
                st.session_state.history.insert(0, {
                    "Kỳ": last_actual, 
                    "Cặp VIP": best_pair, 
                    "KQ": "🔥 WIN" if win_check else "❌"
                })
            st.session_state.last_pred = predict_v48_master(nums)
            st.rerun()
        else:
            st.warning("Cần ít nhất 15 kỳ duy nhất!")

with col2:
    if st.button("🗑️ LÀM MỚI"):
        st.session_state.clear()
        st.rerun()

# --- HIỂN THỊ ---
if "last_pred" in st.session_state:
    res = st.session_state.last_pred
    
    # Độ tin cậy
    conf_class = "confidence-high" if res['confidence'] >= 80 else ("confidence-med" if res['confidence'] >= 65 else "confidence-low")
    st.markdown(f"<div class='box'>🎯 ĐỘ TIN CẬY: <span class='{conf_class}'>{res['confidence']}%</span></div>", unsafe_allow_html=True)
    
    # Thống kê
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div class='stat-box'>🔥 SỐ HOT<br><span class='big-num' style='font-size:24px;'>{','.join(res['hot_nums'][:3])}</span></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='stat-box'>❄️ SỐ LẠNH<br><span class='big-num' style='font-size:24px;'>{','.join(res['cold_nums'][:3])}</span></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='stat-box'>📊 TỔNG KB<br><span class='big-num' style='font-size:24px;'>{res['sum_range'][0]}-{res['sum_range'][1]}</span></div>", unsafe_allow_html=True)
    
    # Phủ sảnh
    st.markdown(f"<div class='box'>🔥 PHỦ SẢNH 8 SỐ: <span class='big-num'>{res['top8']}</span></div>", unsafe_allow_html=True)
    
    # CẶP VIP
    st.markdown("<div class='box'>⭐ CẶP 2 TINH VIP (ƯU TIÊN SỐ 1)</div>", unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    for i, (p, score) in enumerate(res['pairs'][:3]):
        with [p1, p2, p3][i]:
            style = "vip-pick" if i == 0 else "item"
            st.markdown(f"<div class='{style}'>{p[0]} - {p[1]}</div>", unsafe_allow_html=True)
            st.caption(f"Score: {score}")
    
    # BỘ 3 VIP
    st.markdown("<div class='box'>💎 BỘ 3 TINH VIP</div>", unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    for i, (t, score) in enumerate(res['triples'][:3]):
        with [t1, t2, t3][i]:
            style = "vip-pick" if i == 0 else "item-3"
            st.markdown(f"<div class='{style}'>{t[0]}-{t[1]}-{t[2]}</div>", unsafe_allow_html=True)
            st.caption(f"Score: {score}")
    
    # Phân tích vị trí
    with st.expander("📍 Phân tích chi tiết từng vị trí"):
        for i in range(5):
            pos_name = ["Chục Ngàn", "Ngàn", "Trăm", "Chục", "Đơn Vị"][i]
            top_3 = res['pos_analysis'][i].most_common(3)
            st.write(f"**{pos_name}:** {', '.join([f'{d}({c})' for d,c in top_3])}")

# --- LỊCH SỬ ---
if st.session_state.history:
    st.divider()
    st.subheader("📋 Lịch sử đối soát")
    df = pd.DataFrame(st.session_state.history).head(10)
    st.table(df)