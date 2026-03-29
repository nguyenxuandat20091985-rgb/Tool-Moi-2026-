import streamlit as st
import re, pandas as pd, math
import numpy as np
from collections import Counter
from itertools import combinations

# --- CẤU HÌNH HỆ THỐNG ---
LUCKY_OX = [0, 2, 5, 6, 7, 8]

st.set_page_config(page_title="TITAN V48 - QUANTUM GOD MODE", page_icon="⚡", layout="centered")

# --- GIAO DIỆN ---
st.markdown("""
<style>
    .stApp {background-color: #000000; color: #FFD700;}
    .big-num {font-size: 42px; font-weight: bold; color: #00FFCC; text-align: center; font-family: 'Courier New', monospace; letter-spacing: 6px; text-shadow: 0 0 20px #00FFCC;}
    .box {background: linear-gradient(135deg, #0d1b1b, #000000); color: #FFD700; padding: 20px; border-radius: 15px; text-align: center; border: 2px solid #FFD700; margin-bottom: 15px; box-shadow: 0 0 30px rgba(255, 215, 0, 0.3);}
    .item {background: linear-gradient(135deg, #00FFCC, #008B8B); color: #000; padding: 15px; border-radius: 10px; text-align: center; font-size: 28px; font-weight: bold; box-shadow: 0 0 15px rgba(0, 255, 204, 0.5);}
    .item-3 {background: linear-gradient(135deg, #FFD700, #FFA500); color: #000; box-shadow: 0 0 15px rgba(255, 215, 0, 0.5);}
    .quantum-tag {background: linear-gradient(90deg, #FF00FF, #00FFFF); color: white; padding: 3px 8px; border-radius: 5px; font-size: 10px; font-weight: bold;}
    .shadow-tag {background: linear-gradient(90deg, #8B00FF, #FF0080); color: white; padding: 3px 8px; border-radius: 5px; font-size: 10px; font-weight: bold;}
    .trap-warning {background-color: #FF0000; color: white; padding: 3px 8px; border-radius: 5px; font-size: 10px; font-weight: bold; animation: pulse 0.5s infinite;}
    .confidence-critical {color: #00FF00; font-weight: bold; font-size: 24px; text-shadow: 0 0 10px #00FF00;}
    .confidence-warning {color: #FFD700; font-weight: bold; font-size: 24px;}
    .confidence-danger {color: #FF0000; font-weight: bold; font-size: 24px;}
    .quantum-box {background: linear-gradient(135deg, #1a0a2e, #000000); border: 2px solid #FF00FF; color: #FF00FF; padding: 15px; border-radius: 10px; margin-bottom: 10px;}
    h1 {text-align: center; color: #FFD700; text-shadow: 0 0 30px rgba(255, 215, 0, 0.8); font-size: 36px;}
    @keyframes pulse {0% {opacity: 1;} 50% {opacity: 0.5;} 100% {opacity: 1;}}
    .matrix-rain {color: #00FF00; font-family: 'Courier New', monospace; font-size: 12px;}
</style>
""", unsafe_allow_html=True)

# --- THUẬT TOÁN QUANTUM ---

def get_nums(text):
    clean_text = re.sub(r'\s+', '', text)
    nums = [n for n in re.findall(r"\d{5}", clean_text) if n]
    # LỌC TRÙNG LẶP - Chỉ giữ kỳ duy nhất
    seen = set()
    unique_nums = []
    for n in nums:
        if n not in seen:
            seen.add(n)
            unique_nums.append(n)
    return unique_nums

def digital_root(num_str):
    """Tính digital root - tổng các số đến khi còn 1 chữ số"""
    total = sum(int(d) for d in num_str)
    while total > 9:
        total = sum(int(d) for d in str(total))
    return total

def position_analysis(db):
    """Phân tích trọng số từng vị trí"""
    positions = {i: Counter() for i in range(5)}
    for num in db:
        for i, d in enumerate(num):
            positions[i][d] += 1
    return positions

def get_shadow_numbers(db, window=20):
    """
    Shadow Number Theory:
    Số nào thường xuất hiện NGAY SAU khi số X xuất hiện?
    """
    if len(db) < 10: return {}
    shadow_map = {str(i): Counter() for i in range(10)}
    
    for idx in range(len(db) - 1):
        current = db[idx]
        next_num = db[idx + 1]
        for d in current:
            for nd in next_num:
                shadow_map[d][nd] += 1
    
    # Lấy top 2 shadow cho mỗi số
    result = {}
    for d in shadow_map:
        top_shadows = shadow_map[d].most_common(2)
        result[d] = [s[0] for s in top_shadows] if top_shadows else []
    return result

def get_complement_pairs(db):
    """
    Complement Theory:
    Số X và số (9-X) thường có mối quan hệ bù trừ
    """
    complements = {}
    for i in range(10):
        complements[str(i)] = str(9 - i)
    return complements

def calculate_entropy(db, window=10):
    """Tính entropy - độ hỗn loạn của dữ liệu"""
    if len(db) < window: return 0
    recent = db[-window:]
    all_digits = "".join(recent)
    counter = Counter(all_digits)
    total = len(all_digits)
    entropy = 0
    for count in counter.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy

def detect_cycle(db):
    """Phát hiện chu kỳ lặp trong dữ liệu"""
    if len(db) < 20: return None
    # Tìm xem có kỳ nào lặp lại không
    for i in range(len(db) - 1):
        for j in range(i + 1, len(db)):
            if db[i] == db[j]:
                cycle_length = j - i
                return cycle_length
    return None

def quantum_predict(db):
    if len(db) < 15: return None
    
    # === LỚP 1: DIGITAL ROOT ANALYSIS ===
    dr_counts = Counter(digital_root(n) for n in db[-30:])
    dr_mode = dr_counts.most_common(1)[0][0] if dr_counts else 5
    
    # === LỚP 2: POSITION WEIGHT ===
    positions = position_analysis(db[-30:])
    hot_positions = {}
    for pos in range(5):
        if positions[pos]:
            hot_positions[pos] = positions[pos].most_common(2)
    
    # === LỚP 3: SHADOW NUMBERS ===
    shadows = get_shadow_numbers(db)
    
    # === LỚP 4: COMPLEMENT ===
    complements = get_complement_pairs(db)
    
    # === LỚP 5: ENTROPY & CYCLE ===
    entropy = calculate_entropy(db)
    cycle = detect_cycle(db)
    
    # === TỔNG HỢP DỰ ĐOÁN ===
    pair_pool = Counter()
    triple_pool = Counter()
    single_pool = Counter("".join(db[-50:]))
    
    for num in db[-50:]:
        u = sorted(list(set(num)))
        if len(u) >= 2:
            for p in combinations(u, 2): pair_pool[p] += 1
        if len(u) >= 3:
            for t in combinations(u, 3): triple_pool[t] += 1
    
    # Tìm số hot nhất 20 kỳ gần
    recent_str = "".join(db[-20:])
    anchors = [item[0] for item in Counter(recent_str).most_common(3)]
    
    scored_pairs = []
    for p, freq in pair_pool.items():
        gan = 0
        for num in reversed(db):
            if not set(p).issubset(set(num)): gan += 1
            else: break
        
        streak = 0
        for num in reversed(db):
            if set(p).issubset(set(num)): streak += 1
            else: break
        
        score = freq * 5.0
        
        # Quantum Boost Factors
        # 1. Digital Root Match
        dr_pair = (int(p[0]) + int(p[1])) % 9
        if dr_pair == dr_mode or dr_pair == 0: score += 40
        
        # 2. Shadow Connection
        if p[1] in shadows.get(p[0], []) or p[0] in shadows.get(p[1], []):
            score += 50  # Cặp số có quan hệ shadow
        
        # 3. Complement Pair
        if complements[p[0]] == p[1] or complements[p[1]] == p[0]:
            score += 35  # Cặp bù
        
        # 4. Anchor Boost
        if p[0] in anchors or p[1] in anchors: score += 25
        
        # 5. Gan Zone (Vùng vàng 4-10 kỳ)
        if 4 <= gan <= 10: score += 60
        elif 1 <= gan <= 3: score += 30
        elif gan > 18: score -= 50
        
        # 6. Anti-Bet (Tránh bệt quá 3)
        if streak >= 3: score -= 80
        
        # 7. Tuổi Sửu
        if any(int(d) in LUCKY_OX for d in p): score += 15
        
        # 8. Entropy Adjustment
        if entropy < 2.5:  # Dữ liệu đang ổn định -> Sắp biến động
            score += 20  # Ưu tiên số ít xuất hiện
        
        scored_pairs.append(("".join(p), score, gan, streak))
    
    scored_triples = []
    for t, freq in triple_pool.items():
        gan = 0
        for num in reversed(db):
            if not set(t).issubset(set(num)): gan += 1
            else: break
        
        streak = 0
        for num in reversed(db):
            if set(t).issubset(set(num)): streak += 1
            else: break
        
        score = freq * 6.0
        
        # Quantum Boost cho 3 số
        dr_triple = (int(t[0]) + int(t[1]) + int(t[2])) % 9
        if dr_triple == dr_mode: score += 45
        
        # Shadow connection cho 3 số
        shadow_match = 0
        for d in t:
            for other in t:
                if other != d and other in shadows.get(d, []):
                    shadow_match += 1
        if shadow_match >= 2: score += 55
        
        if 4 <= gan <= 10: score += 65
        elif gan > 18: score -= 60
        if streak >= 3: score -= 90
        if any(d in anchors for d in t): score += 30
        
        scored_triples.append(("".join(t), score, gan, streak))
    
    res_p = sorted(scored_pairs, key=lambda x: x[1], reverse=True)[:3]
    res_t = sorted(scored_triples, key=lambda x: x[1], reverse=True)[:3]
    
    # Top 8 thông minh
    single_scores = {}
    for d in "0123456789":
        base = single_pool[d]
        # Boost nếu là shadow của số hot
        for anchor in anchors:
            if d in shadows.get(anchor, []): base += 15
        # Boost nếu là complement của số hot
        for anchor in anchors:
            if complements[anchor] == d: base += 10
        single_scores[d] = base
    
    top_8_list = sorted(single_scores.items(), key=lambda x: x[1], reverse=True)[:8]
    top_8 = "".join([d for d, c in top_8_list])
    
    # === TÍNH ĐỘ TIN CẬY QUANTUM ===
    confidence = 50
    if cycle: confidence += 20  # Phát hiện được chu kỳ
    if entropy < 2.8: confidence += 15  # Sắp có biến động lớn
    if res_p and res_p[0][1] > 200: confidence += 25
    if len(anchors) >= 2: confidence += 10
    
    confidence = min(confidence, 98)
    
    return {
        "pairs": res_p,
        "triples": res_t,
        "top8": top_8,
        "anchors": anchors,
        "confidence": confidence,
        "entropy": entropy,
        "cycle": cycle,
        "digital_root": dr_mode,
        "shadows": shadows
    }

# --- GIAO DIỆN CHÍNH ---
st.markdown('<h1>⚡ TITAN V48 - QUANTUM GOD MODE</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#FF00FF;">5 Lớp Thuật Toán Lượng Tử | Phát Hiện Chu Kỳ Ẩn | Shadow Number Theory</p>', unsafe_allow_html=True)

if "history" not in st.session_state: st.session_state.history = []

user_input = st.text_area("📥 Dán bảng kết quả thực tế (Kỳ mới nhất ở dưới cùng):", 
                          height=200, 
                          placeholder="Ví dụ:\n87558\n34979\n16695")

col1, col2 = st.columns(2)
with col1:
    if st.button("⚡ KÍCH HOẠT QUANTUM"):
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
            st.session_state.last_pred = quantum_predict(nums)
            st.rerun()
        else:
            st.warning("Anh Đạt ơi, dán ít nhất 15 kỳ để Quantum hoạt động!")

with col2:
    if st.button("🗑️ RESET HỆ THỐNG"):
        st.session_state.clear()
        st.rerun()

# --- HIỂN THỊ KẾT QUẢ ---
if "last_pred" in st.session_state:
    res = st.session_state.last_pred
    
    # === HIỂN THỊ THÔNG TIN QUANTUM ===
    st.markdown(f"""
    <div class='quantum-box'>
        <div style='display:flex; justify-content:space-around;'>
            <div>🌀 <b>Entropy:</b> {res['entropy']:.2f}</div>
            <div>🔁 <b>Chu Kỳ:</b> {res['cycle'] if res['cycle'] else 'Không phát hiện'}</div>
            <div>🔢 <b>Digital Root:</b> {res['digital_root']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # === ĐỘ TIN CẬY ===
    conf_class = "confidence-critical" if res['confidence'] >= 80 else ("confidence-warning" if res['confidence'] >= 60 else "confidence-danger")
    conf_text = "🟢 CAO - VÀO TIỀN" if res['confidence'] >= 80 else ("🟡 TB - THĂM DÒ" if res['confidence'] >= 60 else "🔴 THẤP - XEM THÊM")
    st.markdown(f"<div class='box'>🛡 ĐỘ TIN CẬY QUANTUM: <span class='{conf_class}'>{conf_text} ({res['confidence']}%)</span></div>", unsafe_allow_html=True)
    
    # === SỐ NEO ===
    if res['anchors']:
        anchor_html = " | ".join([f"<span class='big-num' style='color:#FFD700'>{a}</span>" for a in res['anchors']])
        st.markdown(f"<div class='box'>🔥 SỐ NEO (HOT NUMBERS): {anchor_html}</div>", unsafe_allow_html=True)
    
    # === ĐỘ PHỦ SẢNH ===
    st.markdown(f"<div class='box'>🎯 ĐỘ PHỦ SẢNH (8 SỐ): <span class='big-num'>{res['top8']}</span></div>", unsafe_allow_html=True)
    
    # === 2 TINH ===
    st.markdown("<div class='box'>🎯 2 TINH QUANTUM MATRIX</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for i, (p, score, gan, streak) in enumerate(res['pairs']):
        with [c1, c2, c3][i]:
            # Check shadow connection
            is_shadow = p[1] in res['shadows'].get(p[0], []) or p[0] in res['shadows'].get(p[1], [])
            st.markdown(f"<div class='item'>{p[0]} , {p[1]}</div>", unsafe_allow_html=True)
            tags = ""
            if is_shadow: tags += f"<span class='shadow-tag'>👤 SHADOW</span> "
            if streak >= 3: tags += f"<span class='trap-warning'>⚠️ BẪY</span> "
            if 4 <= gan <= 10: tags += f"<span class='quantum-tag'>GAN VÀNG {gan}</span>"
            st.markdown(f"<div style='font-size:12px; margin-top:5px;'>Score: {score:.0f} {tags}</div>", unsafe_allow_html=True)
    
    # === 3 TINH ===
    st.markdown("<div class='box' style='border-color:#FF00FF;'>💎 3 TINH QUANTUM MATRIX</div>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    for i, (t, score, gan, streak) in enumerate(res['triples']):
        with [d1, d2, d3][i]:
            st.markdown(f"<div class='item item-3'>{t[0]},{t[1]},{t[2]}</div>", unsafe_allow_html=True)
            tags = ""
            if streak >= 3: tags += f"<span class='trap-warning'>⚠️ BẪY</span> "
            if 4 <= gan <= 10: tags += f"<span class='quantum-tag'>GAN VÀNG {gan}</span>"
            st.markdown(f"<div style='font-size:12px; margin-top:5px;'>Score: {score:.0f} {tags}</div>", unsafe_allow_html=True)
    
    # === SHADOW NUMBER TABLE ===
    st.divider()
    st.markdown("<div class='box'>👤 BẢNG SHADOW NUMBER (Số thường về sau số X)</div>", unsafe_allow_html=True)
    shadow_cols = st.columns(5)
    for i, d in enumerate(["0", "1", "2", "3", "4"]):
        with shadow_cols[i]:
            shadows = res['shadows'].get(d, [])
            shadow_str = ", ".join(shadows) if shadows else "-"
            st.markdown(f"<div style='text-align:center; padding:10px; background:#1a1a1a; border-radius:8px;'>Số <b>{d}</b><br>👤 <span style='color:#FF00FF'>{shadow_str}</span></div>", unsafe_allow_html=True)
    
    shadow_cols2 = st.columns(5)
    for i, d in enumerate(["5", "6", "7", "8", "9"]):
        with shadow_cols2[i]:
            shadows = res['shadows'].get(d, [])
            shadow_str = ", ".join(shadows) if shadows else "-"
            st.markdown(f"<div style='text-align:center; padding:10px; background:#1a1a1a; border-radius:8px;'>Số <b>{d}</b><br>👤 <span style='color:#FF00FF'>{shadow_str}</span></div>", unsafe_allow_html=True)

# === NHẬT KÝ ===
if st.session_state.history:
    st.divider()
    st.subheader("📋 Nhật ký đối soát thực chiến")
    df_history = pd.DataFrame(st.session_state.history).head(15)
    st.table(df_history)