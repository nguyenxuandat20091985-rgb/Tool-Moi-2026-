import streamlit as st
import re, pandas as pd, math
import numpy as np
from collections import Counter
from itertools import combinations

# --- CẤU HÌNH ---
LUCKY_OX = [0, 2, 5, 6, 7, 8]

st.set_page_config(page_title="TITAN V48 - ENSEMBLE MASTER", page_icon="🎯", layout="centered")

# --- GIAO DIỆN ---
st.markdown("""
<style>
    .stApp {background-color: #030303; color: #FFD700;}
    .big-num {font-size: 38px; font-weight: bold; color: #00FFCC; text-align: center; font-family: 'Courier New', monospace; letter-spacing: 4px; text-shadow: 0 0 10px #00FFCC;}
    .box {background: linear-gradient(135deg, #0d1b1b, #030303); color: #FFD700; padding: 18px; border-radius: 15px; text-align: center; border: 1px solid #FFD700; margin-bottom: 12px; box-shadow: 0 4px 15px rgba(255, 215, 0, 0.1);}
    .item {background: linear-gradient(135deg, #00FFCC, #008B8B); color: #000; padding: 12px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold; box-shadow: 0 0 10px rgba(0, 255, 204, 0.3);}
    .item-3 {background: linear-gradient(135deg, #FFD700, #FFA500); color: #000; box-shadow: 0 0 10px rgba(255, 215, 0, 0.3);}
    .item-vote {background: linear-gradient(135deg, #FF00FF, #8B008B); color: #fff; padding: 12px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold;}
    .streak-badge {background-color: #FF3131; color: white; padding: 2px 8px; border-radius: 5px; font-size: 10px; font-weight: bold;}
    .vote-badge {background-color: #00FF00; color: black; padding: 2px 8px; border-radius: 5px; font-size: 10px; font-weight: bold;}
    .safe-zone {border: 2px solid #00FF00; background: rgba(0, 255, 0, 0.1);}
    .risk-zone {border: 2px solid #FF0000; background: rgba(255, 0, 0, 0.1);}
    .confidence-high {color: #00FF00; font-weight: bold; font-size: 20px;}
    .confidence-med {color: #FFD700; font-weight: bold; font-size: 20px;}
    h1 {text-align: center; color: #FFD700; text-shadow: 0 0 20px rgba(255, 215, 0, 0.5);}
</style>
""", unsafe_allow_html=True)

# --- 3 BỘ MÁY DỰ ĐOÁN ĐỘC LẬP ---

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

# === MÁY 1: THỐNG KÊ THUẦN (FREQUENCY) ===
def engine_frequency(db, combo_type='pair'):
    if len(db) < 20: return []
    recent_db = db[-100:]
    pool = Counter()
    
    for num in recent_db:
        u = sorted(list(set(num)))
        if combo_type == 'pair' and len(u) >= 2:
            for p in combinations(u, 2): pool[p] += 1
        elif combo_type == 'triple' and len(u) >= 3:
            for t in combinations(u, 3): pool[t] += 1
    
    scored = []
    for combo, freq in pool.items():
        gan, streak = check_gan_and_streak(db, combo)
        score = freq * 5
        if 3 <= gan <= 10: score += 50  # Vùng rơi đẹp
        if streak == 1: score += 30
        scored.append(("".join(combo), score, gan, streak))
    
    return sorted(scored, key=lambda x: x[1], reverse=True)[:5]

# === MÁY 2: MOMENTUM (ĐÀ TĂNG TRƯỞNG) ===
def engine_momentum(db, combo_type='pair'):
    if len(db) < 50: return []
    recent_15 = db[-15:]
    past_35 = db[-50:-15]
    pool = Counter()
    
    for num in recent_15:
        u = sorted(list(set(num)))
        if combo_type == 'pair' and len(u) >= 2:
            for p in combinations(u, 2): pool[p] += 1
        elif combo_type == 'triple' and len(u) >= 3:
            for t in combinations(u, 3): pool[t] += 1
    
    scored = []
    for combo, freq_recent in pool.items():
        freq_past = sum(1 for n in past_35 if set(combo).issubset(set(n)))
        gan, streak = check_gan_and_streak(db, combo)
        
        # Tính tỷ lệ tăng trưởng
        growth_rate = (freq_recent / 15) / ((freq_past / 35) + 0.01)
        score = growth_rate * 40 + freq_recent * 10
        
        if growth_rate > 2.0: score += 50  # Đang vào cầu mạnh
        if streak >= 1: score += 30
        scored.append(("".join(combo), score, gan, streak, growth_rate))
    
    return sorted(scored, key=lambda x: x[1], reverse=True)[:5]

# === MÁY 3: PATTERN RECOGNITION (MẪU HÌNH) ===
def engine_pattern(db, combo_type='pair'):
    if len(db) < 30: return []
    
    # Tìm các mẫu hình lặp lại
    patterns = {}
    for i in range(len(db) - 3):
        curr_set = set(db[i])
        for j in range(i+1, min(i+10, len(db))):
            next_set = set(db[j])
            overlap = curr_set.intersection(next_set)
            if len(overlap) >= 2:
                for p in combinations(sorted(overlap), 2):
                    if p not in patterns: patterns[p] = []
                    patterns[p].append(j - i)
    
    scored = []
    for combo, gaps in patterns.items():
        gan, streak = check_gan_and_streak(db, combo)
        avg_gap = sum(gaps) / len(gaps) if gaps else 999
        score = 100 / (avg_gap + 1)  # Khoảng cách càng ngắn càng tốt
        
        # Kiểm tra xem có đúng nhịp không
        if gan <= avg_gap + 2 and gan >= avg_gap - 2:
            score += 60  # Đúng nhịp mẫu
        
        if streak >= 1: score += 40
        scored.append(("".join(combo), score, gan, streak, avg_gap))
    
    return sorted(scored, key=lambda x: x[1], reverse=True)[:5]

# === HỆ THỐNG BẦU CHỌN ENSEMBLE ===
def ensemble_voting(db, combo_type='pair'):
    results_1 = engine_frequency(db, combo_type)
    results_2 = engine_momentum(db, combo_type)
    results_3 = engine_pattern(db, combo_type)
    
    # Gộp tất cả và đếm vote
    all_candidates = {}
    
    for source, results in [("FREQ", results_1), ("MOM", results_2), ("PAT", results_3)]:
        for i, (combo, score, gan, streak, *rest) in enumerate(results):
            if combo not in all_candidates:
                all_candidates[combo] = {"votes": 0, "scores": [], "gan": gan, "streak": streak, "sources": []}
            all_candidates[combo]["votes"] += 1
            all_candidates[combo]["scores"].append(score)
            all_candidates[combo]["sources"].append(source)
    
    # Tính điểm tổng hợp
    final_list = []
    for combo, data in all_candidates.items():
        avg_score = sum(data["scores"]) / len(data["scores"])
        total_score = avg_score * data["votes"]  # Vote càng nhiều càng tốt
        
        # Bonus nếu được 2/3 máy chọn
        if data["votes"] >= 2:
            total_score += 100
        
        final_list.append((combo, total_score, data["gan"], data["streak"], data["votes"], data["sources"]))
    
    return sorted(final_list, key=lambda x: x[1], reverse=True)[:5]

def predict_v48_ensemble(db):
    if len(db) < 20: return None
    
    # Chạy ensemble cho 2 tinh và 3 tinh
    pairs_vote = ensemble_voting(db, 'pair')
    triples_vote = ensemble_voting(db, 'triple')
    
    # Tính top 8 số đơn dựa trên voting
    single_votes = Counter()
    for combo, score, gan, streak, votes, sources in pairs_vote:
        for d in combo:
            single_votes[d] += votes * 2
    for combo, score, gan, streak, votes, sources in triples_vote:
        for d in combo:
            single_votes[d] += votes * 3
    
    top_8 = "".join([d for d, c in single_votes.most_common(8)])
    
    # Tính độ tin cậy dựa trên sự đồng thuận
    if pairs_vote and pairs_vote[0][4] >= 2:
        confidence = 85 + (pairs_vote[0][4] - 2) * 5
    else:
        confidence = 50 + pairs_vote[0][4] * 10 if pairs_vote else 50
    
    confidence = min(confidence, 95)
    
    # Phân vùng an toàn/rủi ro
    safe_pairs = [p for p in pairs_vote if p[4] >= 2]  # Được 2+ máy chọn
    risk_pairs = [p for p in pairs_vote if p[4] == 1]  # Chỉ 1 máy chọn
    
    return {
        "pairs": pairs_vote,
        "triples": triples_vote,
        "top8": top_8,
        "confidence": confidence,
        "safe_pairs": safe_pairs,
        "risk_pairs": risk_pairs
    }

# --- GIAO DIỆN CHÍNH ---
st.markdown('<h1>🎯 TITAN V48 - ENSEMBLE MASTER</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#888;">3 Thuật Toán Độc Lập + Hệ Thống Bầu Chọn</p>', unsafe_allow_html=True)

if "history" not in st.session_state: st.session_state.history = []

user_input = st.text_area("📥 Dán bảng kết quả (Kỳ mới nhất ở dưới cùng):", height=150, placeholder="87558\n34979\n...")

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 PHÂN TÍCH ĐA THUẬT TOÁN"):
        nums = get_nums(user_input)
        if len(nums) >= 15:
            if "last_pred" in st.session_state:
                lp = st.session_state.last_pred
                last_actual = nums[-1]
                best_pair = lp['pairs'][0][0] if lp['pairs'] else ""
                win_check = all(d in last_actual for d in best_pair) if best_pair else False
                st.session_state.history.insert(0, {
                    "Kỳ": last_actual, 
                    "Cặp": best_pair, 
                    "Vote": f"{lp['pairs'][0][4]}/3" if lp['pairs'] else "N/A",
                    "KQ": "🔥 WIN" if win_check else "❌"
                })
            st.session_state.last_pred = predict_v48_ensemble(nums)
            st.rerun()
        else:
            st.warning("Cần ít nhất 15 kỳ dữ liệu!")

with col2:
    if st.button("🗑️ LÀM MỚI"):
        st.session_state.clear()
        st.rerun()

# --- HIỂN THỊ KẾT QUẢ ---
if "last_pred" in st.session_state:
    res = st.session_state.last_pred
    
    # Độ tin cậy
    conf_class = "confidence-high" if res['confidence'] >= 80 else "confidence-med"
    st.markdown(f"<div class='box'>🎯 ĐỘ ĐỒNG THUẬN: <span class='{conf_class}'>{res['confidence']}%</span></div>", unsafe_allow_html=True)
    
    # VÙNG AN TOÀN (2+ máy chọn)
    if res['safe_pairs']:
        st.markdown("<div class='box safe-zone'>✅ VÙNG AN TOÀN (2+ Máy Chọn)</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for i, (p, score, gan, streak, votes, sources) in enumerate(res['safe_pairs'][:3]):
            with [c1, c2, c3][i]: 
                st.markdown(f"<div class='item-vote'>{p[0]} , {p[1]}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:12px; margin-top:5px;'>🗳 Vote: {votes}/3 | Nguồn: {', '.join(sources)}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:12px;'>Gan: {gan} | Bệt: {streak}</div>", unsafe_allow_html=True)
    
    # VÙNG RỦI RO (1 máy chọn)
    if res['risk_pairs']:
        st.markdown("<div class='box risk-zone'>⚠️ VÙNG RỦI RO (1 Máy Chọn - Cân Nhắc)</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for i, (p, score, gan, streak, votes, sources) in enumerate(res['risk_pairs'][:3]):
            with [c1, c2, c3][i]: 
                st.markdown(f"<div class='item'>{p[0]} , {p[1]}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:12px; margin-top:5px;'>🗳 Vote: {votes}/3 | Nguồn: {', '.join(sources)}</div>", unsafe_allow_html=True)
    
    # 3 TINH
    st.markdown(f"<div class='box'>💎 3 TINH (Top {len(res['triples'])} Mã):</div>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    for i, (t, score, gan, streak, votes, sources) in enumerate(res['triples'][:3]):
        with [d1, d2, d3][i]: 
            st.markdown(f"<div class='item-3'>{t[0]},{t[1]},{t[2]}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:12px; margin-top:5px;'>🗳 Vote: {votes}/3</div>", unsafe_allow_html=True)
    
    # Độ phủ sảnh
    st.markdown(f"<div class='box'>🔥 ĐỘ PHỦ SẢNH (8 SỐ): <span class='big-num'>{res['top8']}</span></div>", unsafe_allow_html=True)
    
    # Gợi ý quản lý vốn
    st.markdown("<div class='box'>💰 GỢI Ý QUẢN LÝ VỐN</div>", unsafe_allow_html=True)
    if res['confidence'] >= 80:
        st.info("✅ Độ tin cậy cao: Có thể đánh 50-70% vốn cho vùng an toàn")
    elif res['confidence'] >= 60:
        st.warning("⚠️ Độ tin cậy trung bình: Đánh 30-50% vốn, ưu tiên vùng an toàn")
    else:
        st.error("❌ Độ tin cậy thấp: Nên xem thêm hoặc đánh nhỏ 10-20%")

# --- BẢNG ĐỐI SOÁT ---
if st.session_state.history:
    st.divider()
    st.subheader("📋 Nhật ký đối soát")
    df_history = pd.DataFrame(st.session_state.history).head(15)
    st.table(df_history)