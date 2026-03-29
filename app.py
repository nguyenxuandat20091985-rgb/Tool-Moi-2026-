import streamlit as st
import re, json, os, pandas as pd, numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from openai import OpenAI
import google.generativeai as genai
import math

# --- CẤU HÌNH HỆ THỐNG ---
NVIDIA_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"

# Tuổi Sửu 1985 - Mệnh Kim
LUCKY_OX = [0, 2, 5, 6, 7, 8]
SHADOW_MAP = {"0":"5", "5":"0", "1":"6", "6":"1", "2":"7", "7":"2", "3":"8", "8":"3", "4":"9", "9":"4"}

# --- BỔ SUNG: BÓNG ÂM DƯƠNG ---
AM_DUONG_MAP = {
    "0": ["0", "5"], "1": ["1", "6"], "2": ["2", "7"], 
    "3": ["3", "8"], "4": ["4", "9"], "5": ["5", "0"],
    "6": ["6", "1"], "7": ["7", "2"], "8": ["8", "3"], "9": ["9", "4"]
}

st.set_page_config(page_title="TITAN V38 PRO - BẮT CẦU SIÊU TỐC", page_icon="🐂⚡", layout="centered")

# --- GIAO DIỆN ---
st.markdown("""
<style>
    .stApp {background-color: #050505; color: #FFD700;}
    .big-num {font-size: 48px; font-weight: bold; color: #00FFCC; text-align: center; font-family: monospace; letter-spacing: 8px;}
    .box {background: linear-gradient(135deg, #1C3A3A, #050505); color: #FFD700; padding: 15px; border-radius: 12px; text-align: center; border: 2px solid #FFD700; margin-bottom: 10px;}
    .item {background: linear-gradient(135deg, #00FFCC, #008B8B); color: #000; padding: 15px; border-radius: 8px; text-align: center; font-size: 28px; font-weight: bold;}
    .item-3 {background: linear-gradient(135deg, #FFD700, #FFA500); color: #000;}
    .alert {background: rgba(255,0,0,0.2); border-left: 4px solid #FF0000; padding: 10px; margin: 10px 0; border-radius: 5px;}
    .success {background: rgba(0,255,0,0.2); border-left: 4px solid #00FF00;}
    .cau-tag {background: #FF6B6B; color: white; padding: 3px 8px; border-radius: 5px; font-size: 11px; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# --- LOGIC THUẬT TOÁN ---
def get_nums(text):
    return [n for n in re.findall(r"\d{5}", text) if n]

def analyze_position_patterns(db):
    """PHÂN TÍCH TỪNG VỊ TRÍ (chục ngàn, ngàn, trăm, chục, đơn vị)"""
    if len(db) < 20:
        return {}
    
    position_analysis = {}
    for pos in range(5):
        pos_digits = [num[pos] for num in db[-50:] if len(num) > pos]
        freq = Counter(pos_digits)
        
        # Tìm số hot nhất vị trí này
        most_common = freq.most_common(3)
        
        # Tìm số gan vị trí này
        all_digits_in_pos = set(pos_digits)
        cold_digits = [str(i) for i in range(10) if str(i) not in all_digits_in_pos]
        
        position_analysis[f"pos_{pos}"] = {
            "hot": [d for d, c in most_common],
            "cold": cold_digits[:2],
            "freq": dict(freq)
        }
    
    return position_analysis

def detect_cau_running(db):
    """PHÁT HIỆN CẦU ĐANG CHẠY"""
    if len(db) < 10:
        return []
    
    cau_signals = []
    
    # 1. Cầu rơi (số kỳ trước về lại)
    last_num = db[-1]
    for i in range(2, min(10, len(db))):
        prev_num = db[-i]
        # Kiểm tra trùng 2+ số
        common = set(last_num).intersection(set(prev_num))
        if len(common) >= 2:
            cau_signals.append({"type": "cau_roi", "digits": list(common), "strength": len(common)})
    
    # 2. Cầu bóng (âm/dương)
    for digit in last_num:
        shadow = SHADOW_MAP.get(digit)
        if shadow:
            # Kiểm tra xem bóng có về trong 3 kỳ gần không
            for recent in db[-4:-1]:
                if shadow in recent:
                    cau_signals.append({"type": "cau_bong", "digits": [digit, shadow], "strength": 2})
                    break
    
    # 3. Cầu tổng (tổng các số)
    last_sum = sum(int(d) for d in last_num) % 10
    for recent in db[-5:-1]:
        recent_sum = sum(int(d) for d in recent) % 10
        if recent_sum == last_sum:
            cau_signals.append({"type": "cau_tong", "digits": [str(last_sum)], "strength": 2})
            break
    
    return cau_signals

def markov_chain_predict(db, order=2):
    """MARKOV CHAIN - DỰ ĐOÁN CHUỖI"""
    if len(db) < 30:
        return {}
    
    transitions = defaultdict(lambda: defaultdict(int))
    
    # Học transition matrix
    for num in db:
        digits = list(num)
        for i in range(len(digits) - order):
            state = "".join(digits[i:i+order])
            next_digit = digits[i+order]
            transitions[state][next_digit] += 1
    
    # Dự đoán từ số cuối
    last_num = db[-1]
    predictions = {}
    
    for pos in range(len(last_num) - order):
        state = last_num[pos:pos+order]
        if state in transitions:
            next_probs = transitions[state]
            total = sum(next_probs.values())
            predictions[f"pos_{pos+order}"] = {
                digit: count/total for digit, count in next_probs.items()
            }
    
    return predictions

def neural_weight_prediction(db):
    """MÔ PHỎNG NEURAL WEIGHT (đơn giản hóa)"""
    if len(db) < 20:
        return {}
    
    weights = {str(i): 0 for i in range(10)}
    
    # 1. Recency weight (số gần đây)
    for i, num in enumerate(reversed(db[-15:])):
        decay = 0.9 ** i  # Exponential decay
        for digit in set(num):
            weights[digit] += decay * 2
    
    # 2. Frequency weight
    all_digits = "".join(db[-40:])
    freq = Counter(all_digits)
    max_freq = max(freq.values()) if freq else 1
    for digit, count in freq.items():
        weights[digit] += (count / max_freq) * 3
    
    # 3. Position weight
    for pos in range(5):
        pos_digits = [num[pos] for num in db[-30:] if len(num) > pos]
        pos_freq = Counter(pos_digits)
        for digit, count in pos_freq.items():
            if count > len(pos_digits) * 0.2:  # >20% frequency
                weights[digit] += 2
    
    # 4. Zodiac boost
    for digit in map(str, LUCKY_OX):
        weights[digit] += 3
    
    return weights

def predict_5tinh_v38_enhanced(db):
    """HỆ THỐNG DỰ ĐOÁN NÂNG CAO V38"""
    if len(db) < 10:
        return None
    
    # 1. Phân tích vị trí
    pos_analysis = analyze_position_patterns(db)
    
    # 2. Phát hiện cầu đang chạy
    cau_signals = detect_cau_running(db)
    
    # 3. Markov prediction
    markov_preds = markov_chain_predict(db)
    
    # 4. Neural weight
    neural_weights = neural_weight_prediction(db)
    
    # 5. Kết hợp tất cả
    final_scores = {str(i): 0 for i in range(10)}
    
    # Aggregate scores
    for digit in map(str, range(10)):
        # Position hot
        for pos_key, pos_data in pos_analysis.items():
            if digit in pos_data.get("hot", []):
                final_scores[digit] += 5
        
        # Neural weight
        final_scores[digit] += neural_weights.get(digit, 0)
        
        # Cau signals
        for signal in cau_signals:
            if digit in signal.get("digits", []):
                final_scores[digit] += signal.get("strength", 1) * 3
        
        # Markov
        for markov_key, markov_data in markov_preds.items():
            if digit in markov_data:
                final_scores[digit] += markov_data[digit] * 4
    
    # Sort và lấy top 8
    sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    top_8_digits = [d for d, s in sorted_scores[:8]]
    
    # Generate pairs (2-tinh)
    scored_pairs = []
    for p in combinations(top_8_digits[:6], 2):
        pair = "".join(p)
        score = final_scores[p[0]] + final_scores[p[1]]
        
        # Bonus if in cau signals
        for signal in cau_signals:
            if all(d in signal.get("digits", []) for d in pair):
                score += 15
        
        scored_pairs.append((pair, score))
    
    scored_pairs.sort(key=lambda x: x[1], reverse=True)
    top_3_pairs = [p for p, s in scored_pairs[:3]]
    
    # Generate triples (3-tinh)
    scored_triples = []
    for t in combinations(top_8_digits[2:8], 3):
        triple = "".join(t)
        score = sum(final_scores[d] for d in triple)
        
        for signal in cau_signals:
            if all(d in signal.get("digits", []) for d in triple):
                score += 20
        
        scored_triples.append((triple, score))
    
    scored_triples.sort(key=lambda x: x[1], reverse=True)
    top_3_triples = [t for t, s in scored_triples[:3]]
    
    return {
        "pairs": top_3_pairs,
        "triples": top_3_triples,
        "top8": "".join(sorted(top_8_digits)),
        "cau_signals": cau_signals,
        "scores": {d: f"{s:.1f}" for d, s in sorted_scores[:8]}
    }

# --- GIAO DIỆN CHÍNH ---
st.markdown('<h1>🐂⚡ TITAN V38 PRO - BẮT CẦU SIÊU TỐC</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#888;font-size:12px;">Position Analysis + Markov + Neural Weight + Cầu Detection</p>', unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None

user_input = st.text_area("📥 Dán bảng kết quả (tối thiểu 30 kỳ):", height=150, 
                          placeholder="83337\n16568\n81262\n...")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🚀 CHỐT SỐ"):
        nums = get_nums(user_input)
        if len(nums) >= 30:
            # Đối soát
            if st.session_state.last_pred:
                lp = st.session_state.last_pred
                last_actual = nums[-1]
                win_2tinh = any(all(d in last_actual for d in pair) for pair in lp.get('pairs', []))
                win_3tinh = any(all(d in last_actual for d in triple) for triple in lp.get('triples', []))
                
                st.session_state.history.insert(0, {
                    "Kỳ": last_actual, 
                    "2-tinh": "✅" if win_2tinh else "❌",
                    "3-tinh": "✅" if win_3tinh else "❌"
                })
            
            st.session_state.last_pred = predict_5tinh_v38_enhanced(nums)
            st.rerun()
        else:
            st.warning(f"Chỉ có {len(nums)} kỳ. Cần 30+ kỳ!")

with col2:
    if st.button("📊 THỐNG KÊ"):
        nums = get_nums(user_input)
        if nums:
            st.session_state.show_stats = not st.session_state.get("show_stats", False)

with col3:
    if st.button("🗑️ RESET"):
        st.session_state.clear()
        st.rerun()

# --- HIỂN THỊ KẾT QUẢ ---
if st.session_state.last_pred:
    res = st.session_state.last_pred
    
    # Cảnh báo cầu
    if res.get("cau_signals"):
        st.markdown('<div class="alert success">', unsafe_allow_html=True)
        st.markdown("<b>🔥 CẦU ĐANG CHẠY:</b>", unsafe_allow_html=True)
        for signal in res["cau_signals"]:
            st.write(f"• {signal['type'].upper()}: {', '.join(signal['digits'])} (độ mạnh: {signal['strength']})")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown(f"<div class='box'>🔥 TOP 8 SỐ MẠNH: <span class='big-num'>{res['top8']}</span></div>", unsafe_allow_html=True)
    
    # 2 TINH
    st.markdown("<div class='box'>🎯 2 SỐ 5 TINH</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for i, p in enumerate(res['pairs']):
        with [c1, c2, c3][i]:
            st.markdown(f"<div class='item'>{p[0]}-{p[1]}</div>", unsafe_allow_html=True)
    
    # 3 TINH
    st.markdown("<div class='box' style='border-color:#FFD700;'>💎 3 SỐ 5 TINH</div>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    for i, t in enumerate(res['triples']):
        with [d1, d2, d3][i]:
            st.markdown(f"<div class='item item-3'>{t[0]}-{t[1]}-{t[2]}</div>", unsafe_allow_html=True)
    
    # Điểm số chi tiết
    with st.expander("📊 Điểm số chi tiết từng số"):
        for digit, score in res.get("scores", {}).items():
            bar_width = min(100, float(score) * 10)
            st.markdown(f"**Số {digit}**: {'█' * int(bar_width/10)} ({score})")

# --- THỐNG KÊ ---
if st.session_state.get("show_stats", False):
    nums = get_nums(user_input)
    if nums:
        st.divider()
        st.subheader("📊 THỐNG KÊ CHI TIẾT")
        
        # Tần suất
        all_digits = "".join(nums)
        freq = Counter(all_digits)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**🔥 Số nóng:**")
            for d, c in freq.most_common(5):
                st.write(f"• Số {d}: {c} lần")
        
        with col_b:
            st.markdown("**❄️ Số lạnh:**")
            all_present = set(str(i) for i in range(10))
            cold = all_present - set(freq.keys())
            for d in cold:
                st.write(f"• Số {d}: 0 lần")
        
        # Position analysis
        pos_analysis = analyze_position_patterns(nums)
        st.markdown("**📍 Phân tích vị trí:**")
        pos_cols = st.columns(5)
        for pos in range(5):
            with pos_cols[pos]:
                pos_data = pos_analysis.get(f"pos_{pos}", {})
                st.markdown(f"**Vị trí {pos}**")
                st.write(f"Hot: {', '.join(pos_data.get('hot', []))}")

# --- BẢNG ĐỐI SOÁT ---
if st.session_state.history:
    st.divider()
    st.subheader("📋 LỊCH SỬ ĐỐI SOÁT")
    win_rate_2tinh = sum(1 for h in st.session_state.history if h.get("2-tinh") == "✅") / len(st.session_state.history) * 100 if st.session_state.history else 0
    win_rate_3tinh = sum(1 for h in st.session_state.history if h.get("3-tinh") == "✅") / len(st.session_state.history) * 100 if st.session_state.history else 0
    
    st.markdown(f"**Tỷ lệ thắng 2-tinh:** {win_rate_2tinh:.1f}% | **3-tinh:** {win_rate_3tinh:.1f}%")
    st.table(pd.DataFrame(st.session_state.history).head(15))

st.markdown('<div style="text-align:center;color:#666;font-size:10px;margin-top:20px;">🐂⚡ TITAN V38 PRO - Bắt cầu chuyên sâu</div>', unsafe_allow_html=True)