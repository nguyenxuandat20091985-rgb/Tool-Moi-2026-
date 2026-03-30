import streamlit as st
import json, os, re, math
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import google.generativeai as genai

# === CẤU HÌNH ===
DB_FILE = "titan_v53_data.json"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

st.set_page_config(page_title="TITAN V53 - ONE PAGE", page_icon="⚡", layout="centered")

# === CSS OPTIMIZED ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&display=swap');
    .stApp {
        background: linear-gradient(180deg, #000000 0%, #0a0e27 100%);
        color: #00ffff;
        font-family: 'Orbitron', sans-serif;
    }
    .main-header {
        font-size: 28px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #00ffff, #ff00ff, #00ffff);
        background-size: 200% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 3s ease infinite;
    }
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    .vip-pair {
        background: linear-gradient(135deg, #1a1f3a, #0f1428);
        border: 3px solid #FFD700;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 0 30px rgba(255, 215, 0, 0.4);
    }
    .big-number {
        font-size: 56px;
        font-weight: 900;
        color: #00ffff;
        text-shadow: 0 0 20px #00ffff;
        letter-spacing: 12px;
    }
    .score-badge {
        background: linear-gradient(90deg, #00ffff, #0080ff);
        color: #000;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
        font-size: 12px;
    }
    .quick-input {
        background: linear-gradient(135deg, #0f1428, #1a1f3a);
        border: 2px solid #00ff40;
        border-radius: 15px;
        padding: 20px;
        margin-top: 30px;
    }
    .quick-input-title {
        color: #00ff40;
        font-size: 18px;
        font-weight: 900;
        margin-bottom: 10px;
        text-align: center;
    }
    .stat-box {
        background: #0f1428;
        border: 1px solid #00ffff;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        margin: 5px;
    }
    .metric-row {
        display: flex;
        justify-content: space-around;
        margin: 10px 0;
    }
    .win { color: #00ff40; font-weight: 900; }
    .lose { color: #ff0040; font-weight: 900; }
    .streak-hot {
        background: linear-gradient(90deg, #ff0040, #ff8000);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 3px;
    }
    .gan-cold {
        background: linear-gradient(90deg, #8000ff, #ff00ff);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 3px;
    }
    textarea {
        background: #0a0e27 !important;
        border: 1px solid #00ffff !important;
        color: #00ffff !important;
    }
    .confidence-bar {
        height: 20px;
        background: #0a0e27;
        border-radius: 10px;
        overflow: hidden;
        margin: 10px 0;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff0000, #ffff00, #00ff00);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: #000;
        font-size: 12px;
    }
    .tab-btn {
        background: linear-gradient(135deg, #1a1f3a, #0f1428);
        border: 2px solid #00ffff;
        color: #00ffff;
        padding: 8px 15px;
        border-radius: 10px;
        margin: 3px;
        font-weight: bold;
        cursor: pointer;
        font-size: 12px;
    }
    .tab-btn:hover, .tab-btn.active {
        background: #00ffff;
        color: #000;
    }
</style>
""", unsafe_allow_html=True)

# === DATA MANAGEMENT ===
def load_data():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {"results": [], "predictions": [], "weights": {"freq": 1.0, "gap": 1.0, "streak": 1.0}}

def save_data(data):
    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def parse_numbers(text):
    clean = re.sub(r'\s+', '', text)
    return [n for n in re.findall(r'\d{5}', clean) if n]

# === STATISTICAL ENGINE ===
class StatisticalEngine:
    def __init__(self, results):
        self.results = results
        self.digits = "".join(results) if results else ""
        
    def frequency_analysis(self, window=None):
        data = self.digits if window is None else "".join(self.results[-window:])
        counter = Counter(data)
        total = len(data) or 1
        return {d: {"count": counter.get(d, 0), "freq": counter.get(d, 0)/total*100} 
                for d in "0123456789"}
    
    def calculate_gap(self):
        gaps = {}
        for d in "0123456789":
            gap = 0
            for r in reversed(self.results):
                if d in r:
                    break
                gap += 1
            gaps[d] = gap
        return gaps
    
    def detect_streaks(self):
        streaks = {}
        for d in "0123456789":
            count = 0
            for r in reversed(self.results):
                if d in r:
                    count += 1
                else:
                    break
            streaks[d] = count
        return streaks
    
    def bayesian_probability(self, pair):
        prior = 0.01
        likes = sum(1 for r in self.results[-50:] if set(pair).issubset(set(r)))
        total = len(self.results[-50:]) or 1
        likelihood = likes / total
        posterior = (likelihood * prior) / (likelihood * prior + 0.01 * (1-prior))
        return posterior * 100
    
    def calculate_pair_score(self, pair):
        score = 0
        reasons = []
        
        freq = self.frequency_analysis(window=30)
        gaps = self.calculate_gap()
        streaks = self.detect_streaks()
        
        p1, p2 = pair[0], pair[1]
        
        # Frequency
        if freq[p1]["freq"] > 8 and freq[p2]["freq"] > 8:
            score += 30
            reasons.append("Tần suất cao")
        
        # Gap (golden zone 3-10)
        if 3 <= gaps[p1] <= 10:
            score += 25
            reasons.append("Độ gan vàng")
        if 3 <= gaps[p2] <= 10:
            score += 25
            reasons.append("Độ gan vàng")
        
        # Streak
        if streaks[p1] >= 2:
            score += 20
            reasons.append("Đang bệt")
        if streaks[p2] >= 2:
            score += 20
            reasons.append("Đang bệt")
        if streaks[p1] >= 4 or streaks[p2] >= 4:
            score -= 30
        
        # Bayes
        bayes = self.bayesian_probability(pair)
        score += bayes * 0.5
        
        return min(100, score), reasons

# === PREDICTION SYSTEM ===
class PredictionSystem:
    def __init__(self, results):
        self.results = results
        self.stats = StatisticalEngine(results)
    
    def generate_predictions(self):
        all_pairs = list(combinations("0123456789", 2))
        scored = []
        
        for pair in all_pairs:
            score, reasons = self.stats.calculate_pair_score(pair)
            bayes = self.stats.bayesian_probability(pair)
            scored.append({
                "pair": "".join(pair),
                "score": score,
                "bayes": bayes,
                "reasons": reasons
            })
        
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:5]
    
    def get_hot_numbers(self):
        streaks = self.stats.detect_streaks()
        return [(d, s) for d, s in streaks.items() if s >= 2]
    
    def get_cold_numbers(self):
        gaps = self.stats.calculate_gap()
        return sorted(gaps.items(), key=lambda x: x[1], reverse=True)[:5]

# === MAIN APP - ONE PAGE ===
def main():
    data = load_data()
    
    st.markdown('<div class="main-header">⚡ TITAN V53 - ONE PAGE</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center; color:#888; margin-bottom:15px; font-size:11px;">Nhập → Hiện số ngay | Không cần chuyển tab</div>', unsafe_allow_html=True)
    
    # === QUICK INPUT AT BOTTOM ===
    st.markdown("---")
    st.markdown("""
    <div class="quick-input">
        <div class="quick-input-title">📥 NHẬP KẾT QUẢ MỚI (5 số)</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        new_result = st.text_input("", placeholder="Ví dụ: 12345", label_visibility="collapsed")
    with col2:
        if st.button("💾 LƯU & TÍNH", type="primary", use_container_width=True):
            if new_result and len(new_result) == 5 and new_result.isdigit():
                data["results"].append(new_result)
                data["results"] = data["results"][-200:]
                save_data(data)
                st.rerun()
            else:
                st.error("Nhập đúng 5 số!")
    
    if st.button("🗑️ XÓA DATA", use_container_width=True):
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
            st.session_state.clear()
            st.success("Đã xóa")
            st.rerun()
    
    # === SHOW PREDICTIONS IF ENOUGH DATA ===
    if len(data["results"]) >= 10:
        st.markdown("---")
        st.markdown("### 🎯 DỰ ĐOÁN KỲ TIẾP THEO")
        
        pred_sys = PredictionSystem(data["results"])
        predictions = pred_sys.generate_predictions()
        
        if predictions:
            # TOP 1 VIP
            top = predictions[0]
            confidence = min(95, max(40, top["score"]))
            
            st.markdown(f"""
            <div class="vip-pair">
                <div style="font-size:16px; color:#FFD700; margin-bottom:10px;"> CẶP VIP</div>
                <div class="big-number">{top['pair'][0]} - {top['pair'][1]}</div>
                <div style="margin:15px 0;">
                    <span class="score-badge">Score: {top['score']:.1f}</span>
                    <span class="score-badge" style="background:linear-gradient(90deg,#00ff40,#80ff80);color:#000">
                        Bayes: {top['bayes']:.1f}%
                    </span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width:{confidence}%; background:{'#00ff40' if confidence >= 70 else '#ffff00' if confidence >= 50 else '#ff0040'};">
                        {confidence:.0f}% TIN CẬY
                    </div>
                </div>
                <div style="font-size:11px; color:#888; margin-top:10px;">
                    {', '.join(top['reasons']) if top['reasons'] else 'Phân tích thống kê'}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # TOP 2-5
            st.markdown("#### 📊 BACKUP:")
            cols = st.columns(2)
            for i, pred in enumerate(predictions[1:5], 1):
                with cols[(i-1) % 2]:
                    st.markdown(f"""
                    <div class="stat-box">
                        <div style="font-size:24px; font-weight:900; color:#00ffff;">{pred['pair']}</div>
                        <div style="font-size:12px; color:#888;">Score: {pred['score']:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # === STATS ROW ===
        st.markdown("---")
        st.markdown("### 📊 THỐNG KÊ NHANH")
        
        hot = pred_sys.get_hot_numbers()
        cold = pred_sys.get_cold_numbers()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔥 ĐANG BỆT:**")
            if hot:
                for d, s in hot[:3]:
                    st.markdown(f'<span class="streak-hot">Số {d}: {s} kỳ</span>', unsafe_allow_html=True)
            else:
                st.info("Không có")
        
        with col2:
            st.markdown("**❄️ SỐ GAN:**")
            if cold:
                for d, g in cold[:3]:
                    st.markdown(f'<span class="gan-cold">Số {d}: {g} kỳ</span>', unsafe_allow_html=True)
            else:
                st.info("Không có")
        
        # === RECENT HISTORY ===
        st.markdown("---")
        st.markdown("### 📋 5 KỲ GẦN NHẤT")
        if data["results"]:
            recent = data["results"][-5:]
            st.markdown(f"**{' → '.join(recent)}**")
            
            # Check predictions history
            if data.get("predictions"):
                recent_preds = data["predictions"][-5:]
                df = pd.DataFrame(recent_preds)
                
                def color_result(val):
                    return "color: #00ff40; font-weight: 900" if val == "WIN" else "color: #ff0040; font-weight: 900"
                
                st.dataframe(df.style.applymap(color_result, subset=['result']), 
                            use_container_width=True, hide_index=True)
                
                # Win rate
                wins = sum(1 for p in data["predictions"] if p.get("result") == "WIN")
                total = len(data["predictions"])
                if total > 0:
                    rate = wins/total*100
                    st.markdown(f"**Tỷ lệ thắng:** {rate:.1f}% ({wins}/{total})")
    
    else:
        st.warning(f"⚠️ Cần ít nhất 10 kỳ dữ liệu (hiện có: {len(data['results'])})")
        if data["results"]:
            st.info(f"Đã có: {', '.join(data['results'][-5:])}")

if __name__ == "__main__":
    main()