import streamlit as st
import re, json, pandas as pd, numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from openai import OpenAI
import google.generativeai as genai
import math

# --- CẤU HÌNH ---
NVIDIA_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"

# Tuổi Sửu 1985 - Mệnh Kim
LUCKY_OX = {0, 2, 5, 6, 7, 8}
SHADOW_MAP = {"0":"5", "5":"0", "1":"6", "6":"1", "2":"7", "7":"2", "3":"8", "8":"3", "4":"9", "9":"4"}

st.set_page_config(page_title="TITAN V38 PRO - CẦU KÈO", page_icon="🐂📊", layout="wide")

st.markdown("""
<style>
    .stApp {background-color: #050505; color: #FFD700;}
    .big-num {font-size: 48px; font-weight: bold; color: #00FFCC; text-align: center; font-family: monospace; letter-spacing: 8px; text-shadow: 0 0 20px #00FFCC;}
    .box {background: linear-gradient(135deg, #1C3A3A, #050505); color: #FFD700; padding: 15px; border-radius: 12px; text-align: center; border: 2px solid #FFD700; margin: 10px 0; box-shadow: 0 4px 15px rgba(255,215,0,0.3);}
    .item {background: linear-gradient(135deg, #00FFCC, #008B8B); color: #000; padding: 15px; border-radius: 10px; text-align: center; font-size: 28px; font-weight: bold; box-shadow: 0 4px 10px rgba(0,255,204,0.4);}
    .item-3 {background: linear-gradient(135deg, #FFD700, #FFA500); color: #000;}
    .win {background: rgba(0,255,0,0.2); border: 2px solid #00FF00;}
    .lose {background: rgba(255,0,0,0.2); border: 2px solid #FF0000;}
    .cau-box {background: rgba(255,105,180,0.1); border-left: 4px solid #FF69B4; padding: 10px; margin: 5px 0;}
    .gan-box {background: rgba(255,165,0,0.1); border-left: 4px solid #FFA500; padding: 10px; margin: 5px 0;}
    table {width: 100%; border-collapse: collapse; margin: 10px 0;}
    th, td {border: 1px solid #FFD700; padding: 8px; text-align: center;}
    th {background: rgba(255,215,0,0.3);}
</style>
""", unsafe_allow_html=True)

def get_nums(text):
    return [n for n in re.findall(r"\d{5}", text) if n and len(n)==5]

# ============= 🎯 THUẬT TOÁN CẦU KÈO CHUYÊN SÂU =============

def analyze_cau_keo(db):
    """
    Phân tích cầu kèo thực tế từ bảng kết quả
    Trả về: dict các cầu đang chạy
    """
    if len(db) < 20:
        return {}
    
    cau = {
        'cau_rung': [],      # Cầu rụng (số vừa về lại)
        'cau_chay': [],      # Cầu chạy (xu hướng tăng)
        'cau_gan': [],       # Cầu gan (sắp về)
        'vi_tri_bias': {},   # Bias theo vị trí
        'cap_hot': [],       # Cặp hay về cùng
        'nhip_3': [],        # Nhịp 3 kỳ
    }
    
    # 1. Cầu rụng: Số kỳ trước về, kỳ này lại về
    last_10 = db[-10:]
    for i in range(1, len(last_10)):
        prev_set = set(last_10[i-1])
        curr_set = set(last_10[i])
        common = prev_set.intersection(curr_set)
        if len(common) >= 2:
            cau['cau_rung'].extend(list(common))
    
    # 2. Cầu chạy: Số xuất hiện tăng dần trong 5 kỳ
    for d in range(10):
        ds = str(d)
        counts = [sum(1 for num in db[-(i+1)*5:-i*5] if ds in num) for i in range(4)]
        if len(counts) >= 3 and counts[-1] > counts[0]:
            cau['cau_chay'].append(ds)
    
    # 3. Số gan: Không về >= 7 kỳ nhưng có dấu hiệu
    for d in range(10):
        ds = str(d)
        gan_count = 0
        for num in reversed(db):
            if ds not in num:
                gan_count += 1
            else:
                break
        if gan_count >= 5:
            # Check dấu hiệu: có trong 2 kỳ gần nhất không?
            if any(ds in num for num in db[-3:-1]):
                cau['cau_gan'].append((ds, gan_count))
    
    # 4. Bias vị trí
    for pos in range(5):
        pos_digits = [n[pos] for n in db[-50:] if len(n) > pos]
        freq = Counter(pos_digits)
        top_3 = [d for d, c in freq.most_common(3)]
        cau['vi_tri_bias'][pos] = top_3
    
    # 5. Cặp hot (hay về cùng >= 3 lần trong 20 kỳ)
    pair_counts = Counter()
    for num in db[-20:]:
        unique = sorted(set(num))
        for p in combinations(unique, 2):
            pair_counts["".join(p)] += 1
    
    cau['cap_hot'] = [p for p, c in pair_counts.most_common(5) if c >= 3]
    
    # 6. Nhịp 3 kỳ (pattern lặp mỗi 3 kỳ)
    for d in range(10):
        ds = str(d)
        pattern = [1 if ds in num else 0 for num in db[-15:]]
        # Check pattern 100, 010, 001
        for i in range(len(pattern)-2):
            if pattern[i:i+3] == [1,0,0] or pattern[i:i+3] == [0,1,0]:
                cau['nhip_3'].append(ds)
                break
    
    return cau

def shadow_prediction(db, cau):
    """
    Dự đoán theo bóng dương/âm (dân gian)
    """
    if not db:
        return []
    
    last = db[-1]
    shadows = []
    
    # Bóng dương: 0-5, 1-6, 2-7, 3-8, 4-9
    for d in last:
        if d in SHADOW_MAP:
            shadows.append(SHADOW_MAP[d])
    
    return list(set(shadows))

def markov_predict(db, order=2):
    """
    Markov Chain: Dự đoán dựa trên chuỗi
    """
    if len(db) < 30:
        return []
    
    # Build transition matrix cho từng vị trí
    transitions = {pos: defaultdict(lambda: defaultdict(int)) for pos in range(5)}
    
    for num in db[:-1]:
        for pos in range(5):
            if pos < len(num):
                state = num[max(0,pos-order+1):pos+1]
                next_num = db[db.index(num)+1] if db.index(num)+1 < len(db) else ""
                if pos < len(next_num):
                    transitions[pos][state][next_num[pos]] += 1
    
    # Predict từ kỳ cuối
    last = db[-1]
    predictions = []
    
    for pos in range(5):
        state = last[max(0,pos-order+1):pos+1]
        if state in transitions[pos]:
            next_probs = transitions[pos][state]
            top = max(next_probs.items(), key=lambda x: x[1])[0]
            predictions.append(top)
    
    return predictions

def rhythm_analysis(db):
    """
    Phân tích nhịp điệu (rhythm pattern)
    """
    if len(db) < 40:
        return {}
    
    rhythm = {}
    
    # Tính tổng từng kỳ
    sums = [sum(int(d) for d in num) for num in db[-30:]]
    
    # Xu hướng tổng: tăng/giảm/đi ngang
    if len(sums) >= 5:
        recent_avg = sum(sums[-5:]) / 5
        prev_avg = sum(sums[-10:-5]) / 5
        
        if recent_avg > prev_avg + 2:
            rhythm['trend'] = 'tang'
        elif recent_avg < prev_avg - 2:
            rhythm['trend'] = 'giam'
        else:
            rhythm['trend'] = 'ong_dinh'
        
        rhythm['target_sum'] = int(recent_avg)
    
    # Chẵn/lẻ trend
    even_counts = [sum(1 for d in num if int(d)%2==0) for num in db[-20:]]
    recent_even = sum(even_counts[-5:]) / 5
    
    rhythm['even_odd'] = 'chan' if recent_even >= 3 else 'le'
    
    return rhythm

def predict_v38_enhanced(db):
    """
    Thuật toán V38: Kết hợp TẤT CẢ phương pháp
    """
    if len(db) < 15:
        return None
    
    # 1. Phân tích cầu kèo
    cau = analyze_cau_keo(db)
    
    # 2. Shadow prediction
    shadows = shadow_prediction(db, cau)
    
    # 3. Markov prediction
    markov_preds = markov_predict(db)
    
    # 4. Rhythm analysis
    rhythm = rhythm_analysis(db)
    
    # 5. Frequency với weight động
    all_digits = "".join(db[-50:])
    base_scores = {str(i): all_digits.count(str(i)) * 1.5 for i in range(10)}
    
    # 6. Bonus theo cầu
    for d in cau.get('cau_rung', []):
        base_scores[d] += 20  # Cầu rụng ưu tiên cao
    
    for d in cau.get('cau_chay', []):
        base_scores[d] += 15  # Cầu chạy
    
    for d, gan_count in cau.get('cau_gan', []):
        base_scores[d] += gan_count * 3  # Số gan
    
    # 7. Bonus shadow
    for d in shadows:
        base_scores[d] += 10
    
    # 8. Bonus Markov
    for pos, d in enumerate(markov_preds):
        base_scores[d] += 12
    
    # 9. Bonus tuổi Sửu
    for d in LUCKY_OX:
        ds = str(d)
        if ds in base_scores:
            base_scores[ds] += 8
    
    # 10. Bonus bias vị trí
    for pos, top_digits in cau.get('vi_tri_bias', {}).items():
        for d in top_digits:
            base_scores[d] += 8
    
    # Sort và chọn top 8
    sorted_scores = sorted(base_scores.items(), key=lambda x: x[1], reverse=True)
    top_8 = [d for d, s in sorted_scores[:8]]
    
    # Tạo cặp 2-tinh với scoring nâng cao
    pool_2 = sorted(top_8[:6])
    all_pairs = ["".join(p) for p in combinations(pool_2, 2)]
    
    scored_pairs = []
    for p in all_pairs:
        score = base_scores[p[0]] + base_scores[p[1]]
        
        # Bonus nếu là cặp hot
        if p in cau.get('cap_hot', []):
            score += 30
        
        # Bonus nếu cùng xuất hiện trong 5 kỳ gần
        for num in db[-5:]:
            if p[0] in num and p[1] in num:
                score += 25
                break
        
        scored_pairs.append((p, score))
    
    scored_pairs.sort(key=lambda x: x[1], reverse=True)
    final_pairs = [p for p, s in scored_pairs[:3]]
    
    # Tạo bộ 3-tinh
    pool_3 = sorted(top_8[2:8])
    all_triples = ["".join(t) for t in combinations(pool_3, 3)]
    
    scored_triples = []
    for t in all_triples:
        score = sum(base_scores[d] for d in t)
        
        # Bonus nếu có trong nhịp 3
        for d in t:
            if d in cau.get('nhip_3', []):
                score += 15
        
        scored_triples.append((t, score))
    
    scored_triples.sort(key=lambda x: x[1], reverse=True)
    final_triples = [t for t, s in scored_triples[:3]]
    
    # Confidence calculation
    cau_strength = len(cau.get('cau_rung', [])) + len(cau.get('cau_chay', []))
    conf = min(95, 60 + cau_strength * 5)
    
    return {
        "pairs": final_pairs,
        "triples": final_triples,
        "top8": "".join(sorted(top_8)),
        "cau": cau,
        "shadows": shadows,
        "markov": markov_preds,
        "rhythm": rhythm,
        "confidence": conf,
        "scores": {d: f"{base_scores[d]:.1f}" for d in top_8}
    }

# ============= GIAO DIỆN =============

st.markdown('<h1>🐂📊 TITAN V38 PRO - CẦU KÈO MASTER</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#00FFCC;">Phân tích cầu chuyên sâu • Markov • Shadow • Rhythm</p>', unsafe_allow_html=True)

if "db" not in st.session_state:
    st.session_state.db = []
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None

col1, col2 = st.columns(2)
with col1:
    st.metric("📊 Tổng kỳ", len(st.session_state.db))
with col2:
    last = st.session_state.db[-1] if st.session_state.db else "-"
    st.metric("🎯 Kỳ cuối", last)

user_input = st.text_area("📥 Dán kết quả 50-100 kỳ gần nhất:", height=120, 
                         placeholder="83337\n16568\n81262\n...")

col_btn1, col_btn2, col_btn3 = st.columns(3)

with col_btn1:
    if st.button("🔍 PHÂN TÍCH CẦU"):
        nums = get_nums(user_input)
        if len(nums) >= 30:
            st.session_state.db = nums[-100:]
            st.session_state.last_pred = predict_v38_enhanced(st.session_state.db)
            st.rerun()
        else:
            st.error("Cần ít nhất 30 kỳ!")

with col_btn2:
    if st.button("🎯 CHỐT SỐ"):
        if st.session_state.last_pred:
            # Lưu prediction để đối soát
            st.session_state.predictions.append({
                "ky": len(st.session_state.predictions) + 1,
                "pairs": st.session_state.last_pred["pairs"],
                "triples": st.session_state.last_pred["triples"],
                "timestamp": pd.Timestamp.now()
            })
            st.success("✅ Đã chốt số!")
        else:
            st.warning("Phân tích cầu trước!")

with col_btn3:
    if st.button("🗑️ RESET"):
        st.session_state.clear()
        st.rerun()

# ============= HIỂN THỊ KẾT QUẢ =============

if st.session_state.last_pred:
    res = st.session_state.last_pred
    cau = res.get("cau", {})
    
    st.markdown("---")
    st.markdown(f"<div class='box'>🔥 ĐỘ TIN CẬY: {res['confidence']:.0f}% | Top 8: <span class='big-num'>{res['top8']}</span></div>", unsafe_allow_html=True)
    
    # Hiển thị cầu đang chạy
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        if cau.get('cau_rung'):
            st.markdown(f"<div class='cau-box'><b>🍎 Cầu rụng:</b> {', '.join(set(cau['cau_rung']))}</div>", unsafe_allow_html=True)
        if cau.get('cau_chay'):
            st.markdown(f"<div class='cau-box'><b>🏃 Cầu chạy:</b> {', '.join(cau['cau_chay'])}</div>", unsafe_allow_html=True)
    
    with col_c2:
        if cau.get('cau_gan'):
            gan_str = ", ".join([f"{d}({c}kỳ)" for d,c in cau['cau_gan']])
            st.markdown(f"<div class='gan-box'><b>⏰ Số gan:</b> {gan_str}</div>", unsafe_allow_html=True)
        if cau.get('cap_hot'):
            st.markdown(f"<div class='cau-box'><b>🔥 Cặp hot:</b> {', '.join(cau['cap_hot'])}</div>", unsafe_allow_html=True)
    
    # Shadow & Markov
    if res.get('shadows'):
        st.markdown(f"<div class='box'>🌓 Bóng dương/âm: <b>{', '.join(res['shadows'])}</b></div>", unsafe_allow_html=True)
    
    if res.get('markov'):
        st.markdown(f"<div class='box'>🔮 Markov dự đoán: <b>{''.join(res['markov'])}</b></div>", unsafe_allow_html=True)
    
    # Rhythm
    if res.get('rhythm'):
        rhythm = res['rhythm']
        st.markdown(f"<div class='box'>📈 Nhịp điệu: Xu hướng <b>{rhythm.get('trend','ong_dinh')}</b> | Chẵn/Lẻ: <b>{rhythm.get('even_odd','')}</b></div>", unsafe_allow_html=True)
    
    st.divider()
    
    # 2 TINH
    st.markdown("<div class='box'>🎯 2 SỐ 5 TINH (Top 3 cặp)</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for i, p in enumerate(res['pairs']):
        with [c1, c2, c3][i]:
            st.markdown(f"<div class='item'>{p[0]}-{p[1]}</div>", unsafe_allow_html=True)
    
    # 3 TINH
    st.markdown("<div class='box' style='border-color:#FFD700;'>💎 3 SỐ 5 TINH (Top 3 bộ)</div>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    for i, t in enumerate(res['triples']):
        with [d1, d2, d3][i]:
            st.markdown(f"<div class='item item-3'>{t[0]}-{t[1]}-{t[2]}</div>", unsafe_allow_html=True)
    
    # Scores chi tiết
    with st.expander("📊 Điểm số chi tiết từng số"):
        st.json(res.get("scores", {}))

# ============= BẢNG ĐỐI SOÁT =============

if st.session_state.predictions:
    st.divider()
    st.subheader("📋 Lịch sử chốt số")
    
    df_preds = pd.DataFrame(st.session_state.predictions)
    st.dataframe(df_preds.tail(10), use_container_width=True)

st.markdown('<div style="text-align:center;color:#666;font-size:12px;margin-top:20px;">🐂📊 TITAN V38 PRO - Cầu kèo Master</div>', unsafe_allow_html=True)