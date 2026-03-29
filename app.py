import streamlit as st
import re, pandas as pd, math
import numpy as np
from collections import Counter
from itertools import combinations

# --- CẤU HÌNH HỆ THỐNG ---
LUCKY_OX = [0, 2, 5, 6, 7, 8] # Tuổi Sửu 1985

st.set_page_config(page_title="TITAN V47 - ANTI-REVERSAL", page_icon="🛡️", layout="centered")

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
    .trap-warning {background-color: #FF0000; color: white; padding: 2px 6px; border-radius: 4px; font-size: 9px; margin-left: 5px; animation: blink 1s infinite;}
    .confidence-high {color: #00FF00; font-weight: bold; font-size: 20px;}
    .confidence-med {color: #FFD700; font-weight: bold; font-size: 20px;}
    .confidence-low {color: #FF3131; font-weight: bold; font-size: 20px;}
    .risk-box {background: rgba(255, 0, 0, 0.2); border: 1px solid #FF0000; color: #FF6666; padding: 10px; border-radius: 8px; margin-bottom: 10px;}
    h1 {text-align: center; color: #FFD700; text-shadow: 0 0 20px rgba(255, 215, 0, 0.5);}
    @keyframes blink {50% {opacity: 0.5;}}
</style>
""", unsafe_allow_html=True)

# --- THUẬT TOÁN CHỐNG ĐẢO CẦU ---

def get_nums(text):
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

def calculate_volatility(db):
    """Tính độ biến động của dữ liệu để phát hiện nguy cơ đảo cầu"""
    if len(db) < 10: return 0
    # Tính tần suất xuất hiện của từng số trong 10 kỳ gần nhất
    recent_str = "".join(db[-10:])
    counts = [recent_str.count(str(i)) for i in range(10)]
    mean = sum(counts) / 10
    variance = sum((x - mean) ** 2 for x in counts) / 10
    std_dev = math.sqrt(variance)
    # Nếu độ lệch chuẩn quá thấp -> Dữ liệu đang quá đều -> Sắp đảo cầu
    # Nếu độ lệch chuẩn quá cao -> Dữ liệu đang mất cân bằng -> Sắp cân bằng lại
    return std_dev

def is_trap_pair(db, pair, recent_window=10):
    """Kiểm tra xem cặp số có đang bị 'bẫy' không (về quá nhiều trong thời gian ngắn)"""
    if len(db) < recent_window: return False
    recent_db = db[-recent_window:]
    count = sum(1 for n in recent_db if set(pair).issubset(set(n)))
    # Nếu về quá 3 lần trong 10 kỳ -> Coi là bẫy
    return count >= 3

def predict_v47_anti_reversal(db):
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

    anchors = [item[0] for item in Counter("".join(db[-20:])).most_common(2)]
    volatility = calculate_volatility(db)
    
    scored_pairs = []
    for p, freq in pair_pool.items():
        gan, streak = check_gan_and_streak(db, p)
        is_trap = is_trap_pair(db, p)
        
        # --- LOGIC CHỐNG ĐẢO CẦU ---
        score = freq * 3.0
        
        # 1. Xử lý nhịp Bệt (Streak) - Giảm thưởng nếu bệt quá dài
        if streak == 1: score += 40 # Rơi lại kỳ đầu tiên an toàn
        elif streak == 2: score += 20 # Kỳ thứ 2 bắt đầu nghi ngờ
        elif streak >= 3: score -= 50 # Kỳ thứ 3 trở đi: Coi là bẫy, trừ điểm mạnh
        
        # 2. Xử lý nhịp Gan (Ưu tiên vùng chín muồi 5-12 kỳ)
        if 5 <= gan <= 12: score += 60 # Vùng an toàn nhất để rơi
        elif 1 <= gan <= 4: score += 30 # Rơi rải rác
        elif gan > 20: score -= 70 # Gan quá sâu, khó về
        
        # 3. Phát hiện bẫy cầu
        if is_trap:
            score -= 100 # Trừ điểm cực mạnh nếu detect là bẫy
        
        # 4. Yếu tố tuổi & Neo
        if any(int(d) in LUCKY_OX for d in p): score += 10
        if p[0] in anchors or p[1] in anchors: score += 15 # Giảm trọng số neo để tránh đám đông
            
        scored_pairs.append(("".join(p), score, gan, streak, is_trap))
        
    scored_triples = []
    for t, freq in triple_pool.items():
        gan, streak = check_gan_and_streak(db, t)
        is_trap = is_trap_pair(db, t) # Áp dụng logic bẫy cho cả 3 số
        
        score = freq * 4.0
        if streak >= 1: score += 30 # Thưởng bệt ít hơn để an toàn
        if 5 <= gan <= 12: score += 50
        if gan > 20: score -= 80
        if is_trap: score -= 100
        
        scored_triples.append(("".join(t), score, gan, streak, is_trap))

    res_p = sorted(scored_pairs, key=lambda x: x[1], reverse=True)[:3]
    res_t = sorted(scored_triples, key=lambda x: x[1], reverse=True)[:3]
    
    # Tính độ phủ sảnh thông minh (Tránh các số đang bị bẫy)
    safe_singles = []
    for d in "0123456789":
        # Kiểm tra xem số đơn này có nằm trong nhiều cặp bẫy không
        trap_count = sum(1 for p, _, _, _, is_trap in scored_pairs if is_trap and d in p)
        if trap_count == 0:
            safe_singles.append((d, single_pool[d]))
    
    safe_singles.sort(key=lambda x: x[1], reverse=True)
    top_8 = "".join([d for d, c in safe_singles[:8]])
    if len(top_8) < 8: # Fallback nếu ít số an toàn
        top_8 = "".join([d for d, c in single_pool.most_common(8)])
    
    # Tính rủi ro đảo cầu
    reversal_risk = "THẤP"
    risk_color = "#00FF00"
    if volatility < 1.5: 
        reversal_risk = "CAO (Sắp đảo cầu)"
        risk_color = "#FF0000"
    elif volatility < 2.5:
        reversal_risk = "TRUNG BÌNH"
        risk_color = "#FFD700"
        
    # Tính điểm tin cậy
    confidence_score = 0
    if res_p:
        top_pair_score = res_p[0][1]
        if top_pair_score > 150 and not res_p[0][4]: confidence_score = 90
        elif top_pair_score > 100: confidence_score = 75
        else: confidence_score = 50
    
    return {
        "pairs": res_p, 
        "triples": res_t, 
        "top8": top_8, 
        "anchors": anchors, 
        "confidence": confidence_score,
        "reversal_risk": reversal_risk,
        "risk_color": risk_color,
        "volatility": volatility
    }

# --- GIAO DIỆN CHÍNH ---
st.markdown('<h1>🛡️ TITAN V47 - ANTI-REVERSAL MATRIX</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#888;">Chuyên sâu chống bẫy cầu & đảo chiều</p>', unsafe_allow_html=True)

if "history" not in st.session_state: st.session_state.history = []

user_input = st.text_area("📥 Dán bảng kết quả thực tế (Kỳ mới nhất ở dưới cùng):", height=150, placeholder="Ví dụ:\n43060\n16695")

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 PHÂN TÍCH CHỐNG LỪA"):
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
            st.session_state.last_pred = predict_v47_anti_reversal(nums)
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
    
    # Cảnh báo rủi ro đảo cầu
    st.markdown(f"<div class='box' style='border-color:{res['risk_color']};'>⚠️ RỦI RO ĐẢO CẦU: <span style='color:{res['risk_color']}; font-weight:bold; font-size:20px;'>{res['reversal_risk']}</span></div>", unsafe_allow_html=True)
    
    if res['reversal_risk'] == "CAO (Sắp đảo cầu)":
        st.markdown(f"<div class='risk-box'>🚨 <b>Cảnh báo:</b> Dữ liệu đang quá ổn định, nhà cái có thể sẽ đảo cầu ngay kỳ tới. Ưu tiên đánh ngược lại các số đang Bệt!</div>", unsafe_allow_html=True)

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
    
    st.markdown(f"<div class='box'>🔥 ĐỘ PHỦ SẢNH (8 SỐ AN TOÀN): <span class='big-num'>{res['top8']}</span></div>", unsafe_allow_html=True)
    
    # 2 TINH
    st.markdown("<div class='box'>🎯 2 TINH MATRIX (ƯU TIÊN GAN 5-12 & TRÁNH BẸT)</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for i, (p, score, gan, streak, is_trap) in enumerate(res['pairs']):
        with [c1, c2, c3][i]: 
            st.markdown(f"<div class='item'>{p[0]} , {p[1]}</div>", unsafe_allow_html=True)
            tags = ""
            if is_trap: tags += f"<span class='trap-warning'>⚠️ BẪY</span>"
            elif streak > 0: tags += f"<span class='streak-badge'>BỆT {streak}</span>"
            if 5 <= gan <= 12: tags += f"<span class='momentum-tag'>GAN CHÍN {gan}</span>"
            st.markdown(f"<div style='font-size:12px; margin-top:5px;'>Score: {score} {tags}</div>", unsafe_allow_html=True)
        
    # 3 TINH
    st.markdown("<div class='box' style='border-color:#FFD700;'>💎 3 TINH MATRIX (SIÊU CẤP CHUYÊN BIỆT)</div>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    for i, (t, score, gan, streak, is_trap) in enumerate(res['triples']):
        with [d1, d2, d3][i]: 
            st.markdown(f"<div class='item item-3'>{t[0]},{t[1]},{t[2]}</div>", unsafe_allow_html=True)
            tags = ""
            if is_trap: tags += f"<span class='trap-warning'>⚠️ BẪY</span>"
            elif streak > 0: tags += f"<span class='streak-badge'>BỆT {streak}</span>"
            st.markdown(f"<div style='font-size:12px; margin-top:5px;'>Score: {score} {tags}</div>", unsafe_allow_html=True)

# --- BẢNG ĐỐI SOÁT ---
if st.session_state.history:
    st.divider()
    st.subheader("📋 Nhật ký đối soát (Bao sảnh 5 Tinh)")
    df_history = pd.DataFrame(st.session_state.history).head(15)
    st.table(df_history)