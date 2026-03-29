import streamlit as st
import json, os, re, math
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import google.generativeai as genai
from openai import OpenAI

# === CẤU HÌNH ===
DB_FILE = "titan_v32_data.json"
NVIDIA_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

st.set_page_config(page_title="TITAN V32 PRO", page_icon="🎯", layout="centered")

# === CSS DARK NEON ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    .stApp {
        background: linear-gradient(180deg, #0a0e27 0%, #1a1f3a 100%);
        color: #00ffff;
        font-family: 'Orbitron', sans-serif;
    }
    .main-header {
        font-size: 28px;
        font-weight: 900;
        text-align: center;
        color: #00ffff;
        text-shadow: 0 0 20px #00ffff;
        margin-bottom: 15px;
    }
    .tab-button {
        background: linear-gradient(135deg, #1a1f3a, #0f1428);
        border: 2px solid #00ffff;
        color: #00ffff;
        padding: 12px;
        border-radius: 10px;
        margin: 5px 0;
        font-weight: bold;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s;
    }
    .tab-button:hover {
        background: #00ffff;
        color: #000;
        box-shadow: 0 0 20px #00ffff;
    }
    .prediction-box {
        background: linear-gradient(135deg, #0f1428, #1a1f3a);
        border: 2px solid #00ffff;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
    }
    .big-number {
        font-size: 42px;
        font-weight: 900;
        color: #00ffff;
        text-shadow: 0 0 15px #00ffff;
        letter-spacing: 8px;
    }
    .score-badge {
        background: linear-gradient(90deg, #00ffff, #0080ff);
        color: #000;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
    }
    .streak-hot {
        background: linear-gradient(90deg, #ff0040, #ff8000);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .gan-cold {
        background: linear-gradient(90deg, #8000ff, #ff00ff);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    .win { color: #00ff40; font-weight: 900; }
    .lose { color: #ff0040; font-weight: 900; }
    .metric-card {
        background: #0f1428;
        border: 1px solid #00ffff;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        margin: 5px;
    }
    textarea {
        background: #0a0e27 !important;
        border: 1px solid #00ffff !important;
        color: #00ffff !important;
    }
    .stTextInput>div>div>input {
        background: #0a0e27;
        border: 1px solid #00ffff;
        color: #00ffff;
    }
</style>
""", unsafe_allow_html=True)

# === DATA MANAGEMENT ===
def load_data():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"results": [], "predictions": [], "stats": {}}

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
        
    def frequency_analysis(self):
        counter = Counter(self.digits)
        total = len(self.digits) or 1
        return {d: {"count": counter.get(d, 0), "freq": counter.get(d, 0)/total*100} 
                for d in "0123456789"}
    
    def markov_chain(self, order=1):
        transitions = defaultdict(lambda: defaultdict(int))
        for i in range(len(self.digits) - order):
            state = self.digits[i:i+order]
            next_digit = self.digits[i+order]
            transitions[state][next_digit] += 1
        
        probs = {}
        for state, next_counts in transitions.items():
            total = sum(next_counts.values())
            probs[state] = {d: c/total for d, c in next_counts.items()}
        return probs
    
    def bayesian_probability(self, pair):
        prior = 0.01
        likes = sum(1 for r in self.results[-50:] if set(pair).issubset(set(r)))
        total = len(self.results[-50:]) or 1
        likelihood = likes / total
        posterior = (likelihood * prior) / (likelihood * prior + 0.01 * (1-prior))
        return posterior * 100
    
    def moving_average(self, window=10):
        if len(self.results) < window:
            return {}
        recent = self.results[-window:]
        counter = Counter("".join(recent))
        return {d: counter.get(d, 0)/window for d in "0123456789"}
    
    def detect_streaks(self):
        streaks = {}
        for d in "0123456789":
            count = 0
            max_streak = 0
            for r in reversed(self.results):
                if d in r:
                    count += 1
                    max_streak = max(max_streak, count)
                else:
                    break
            streaks[d] = {"current": count, "max": max_streak}
        return streaks
    
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

# === AI ANALYSIS ===
class AIAnalyzer:
    def __init__(self, results, stats_engine):
        self.results = results
        self.stats = stats_engine
        
    def analyze_with_gemini(self):
        if len(self.results) < 10:
            return None
        
        freq = self.stats.frequency_analysis()
        gaps = self.stats.calculate_gap()
        streaks = self.stats.detect_streaks()
        
        prompt = f"""
        Phân tích xổ số 5D. Dữ liệu 20 kỳ gần nhất:
        {', '.join(self.results[-20:])}
        
        Tần suất: {json.dumps(freq)}
        Độ gan: {json.dumps(gaps)}
        Bệt: {json.dumps(streaks)}
        
        Đề xuất TOP 3 cặp 2 số mạnh nhất cho kỳ tiếp theo.
        Trả về JSON: {{"predictions": [["0","1"], ["2","3"]], "confidence": [80, 70, 65], "reasons": ["lý do"]}}
        """
        
        try:
            response = gemini_model.generate_content(prompt)
            text = response.text
            if "{" in text and "}" in text:
                start = text.index("{")
                end = text.rindex("}") + 1
                return json.loads(text[start:end])
        except:
            pass
        return None
    
    def hybrid_score(self, pair):
        score = 0
        reasons = []
        
        freq = self.stats.frequency_analysis()
        gaps = self.stats.calculate_gap()
        streaks = self.stats.detect_streaks()
        ma = self.stats.moving_average()
        
        p1, p2 = pair[0], pair[1]
        
        if freq[p1]["freq"] > 8 and freq[p2]["freq"] > 8:
            score += 30
            reasons.append("Tần suất cao")
        
        if 3 <= gaps[p1] <= 8 or 3 <= gaps[p2] <= 8:
            score += 25
            reasons.append("Độ gan vàng")
        
        if streaks[p1]["current"] >= 2 or streaks[p2]["current"] >= 2:
            score += 20
            reasons.append("Đang bệt")
        
        if ma.get(p1, 0) > 0.3 or ma.get(p2, 0) > 0.3:
            score += 15
            reasons.append("MA cao")
        
        bayes = self.stats.bayesian_probability(pair)
        score += bayes * 0.5
        
        return min(100, score), reasons

# === PREDICTION SYSTEM ===
class PredictionSystem:
    def __init__(self, results):
        self.results = results
        self.stats = StatisticalEngine(results)
        self.ai = AIAnalyzer(results, self.stats)
    
    def generate_predictions(self, mode="safe"):
        all_pairs = list(combinations("0123456789", 2))
        scored = []
        
        for pair in all_pairs:
            score, reasons = self.ai.hybrid_score(pair)
            bayes = self.stats.bayesian_probability(pair)
            scored.append({
                "pair": "".join(pair),
                "score": score,
                "bayes": bayes,
                "reasons": reasons
            })
        
        scored.sort(key=lambda x: x["score"], reverse=True)
        
        if mode == "risky":
            return scored[:8]
        return scored[:5]
    
    def detect_patterns(self):
        patterns = {
            "streaks": [],
            "gaps": [],
            "cycles": []
        }
        
        streaks = self.stats.detect_streaks()
        gaps = self.stats.calculate_gap()
        
        for d, s in streaks.items():
            if s["current"] >= 3:
                patterns["streaks"].append({"digit": d, "length": s["current"]})
        
        for d, g in gaps.items():
            if g >= 10:
                patterns["gaps"].append({"digit": d, "gap": g})
        
        return patterns

# === UI COMPONENTS ===
def render_input_tab(data):
    st.markdown("### 📥 NHẬP KẾT QUẢ")
    
    new_input = st.text_area("Dán kết quả (mỗi kỳ 1 dòng):", height=150, 
                             placeholder="12345\n67890\n...")
    
    if st.button("💾 LƯU DỮ LIỆU", type="primary"):
        nums = parse_numbers(new_input)
        if nums:
            data["results"].extend(nums)
            data["results"] = data["results"][-200:]
            save_data(data)
            st.success(f"✅ Đã lưu {len(nums)} kỳ")
            st.rerun()
    
    if data["results"]:
        st.markdown(f"**Tổng số kỳ:** {len(data['results'])}")
        st.markdown("**10 kỳ gần nhất:**")
        st.write(", ".join(data["results"][-10:]))

def render_bach_thu_tab(data):
    st.markdown("### 🎯 BẠCH THỦ 2 SỐ")
    
    if len(data["results"]) < 10:
        st.warning("Cần ít nhất 10 kỳ dữ liệu")
        return
    
    mode = st.radio("Chế độ:", ["An toàn", "Mạo hiểm"], horizontal=True)
    mode_str = "risky" if "Mạo hiểm" in mode else "safe"
    
    pred_sys = PredictionSystem(data["results"])
    predictions = pred_sys.generate_predictions(mode_str)
    
    st.markdown("#### TOP 3 CẶP VIP:")
    
    for i, pred in enumerate(predictions[:3]):
        emoji = "🥇" if i == 0 else ("🥈" if i == 1 else "🥉")
        st.markdown(f"""
        <div class="prediction-box">
            <div style="font-size:20px; margin-bottom:10px;">{emoji} CẶP {pred['pair'][0]} - {pred['pair'][1]}</div>
            <div class="big-number" style="font-size:36px;">{pred['pair']}</div>
            <div>
                <span class="score-badge">Score: {pred['score']:.1f}</span>
                <span class="score-badge" style="background:linear-gradient(90deg,#00ff40,#80ff80);color:#000">Bayes: {pred['bayes']:.1f}%</span>
            </div>
            <div style="margin-top:10px; font-size:12px; color:#888;">
                {', '.join(pred['reasons']) if pred['reasons'] else 'Phân tích thống kê'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if len(predictions) > 3:
        with st.expander("📊 TOP 5 BACKUP"):
            for pred in predictions[3:]:
                st.markdown(f"**{pred['pair']}** - Score: {pred['score']:.1f}")

def render_bet_tab(data):
    st.markdown("### 🔥 SỐ BỆT (LẶP)")
    
    if len(data["results"]) < 5:
        st.warning("Cần ít nhất 5 kỳ dữ liệu")
        return
    
    pred_sys = PredictionSystem(data["results"])
    patterns = pred_sys.detect_patterns()
    
    if patterns["streaks"]:
        st.markdown("#### ⚡ ĐANG BỆT:")
        for s in patterns["streaks"]:
            st.markdown(f"""
            <div class="prediction-box">
                <div class="streak-hot">Số {s['digit']} - Bệt {s['length']} kỳ</div>
                <div style="margin-top:10px;">
                    {'✅ NÊN THEO' if s['length'] <= 4 else '⚠️ CẨN THẬN'}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Không có số bệt đáng chú ý")

def render_gan_tab(data):
    st.markdown("### ❄️ SỐ GAN (LÂU CHƯA VỀ)")
    
    if len(data["results"]) < 10:
        st.warning("Cần ít nhất 10 kỳ dữ liệu")
        return
    
    stats = StatisticalEngine(data["results"])
    gaps = stats.calculate_gap()
    
    sorted_gaps = sorted(gaps.items(), key=lambda x: x[1], reverse=True)
    
    st.markdown("#### TOP 5 SỐ GAN NHẤT:")
    
    for digit, gap in sorted_gaps[:5]:
        status = "🔥 SẮP VỀ" if gap >= 8 else "⏳ THEO DÕI"
        st.markdown(f"""
        <div class="prediction-box">
            <div class="gan-cold" style="font-size:28px;">Số {digit}</div>
            <div style="margin:10px 0;">Đã {gap} kỳ chưa về</div>
            <div>{status}</div>
        </div>
        """, unsafe_allow_html=True)

def render_history_tab(data):
    st.markdown("### 📊 LỊCH SỬ DỰ ĐOÁN")
    
    if not data.get("predictions"):
        st.info("Chưa có lịch sử dự đoán")
        return
    
    df = pd.DataFrame(data["predictions"][-20:])
    
    def color_result(val):
        if val == "WIN":
            return "color: #00ff40; font-weight: 900"
        return "color: #ff0040; font-weight: 900"
    
    st.dataframe(df.style.applymap(color_result, subset=['Kết quả']), 
                 use_container_width=True, hide_index=True)
    
    if len(data["predictions"]) > 0:
        wins = sum(1 for p in data["predictions"] if p.get("ket_qua") == "WIN")
        total = len(data["predictions"])
        rate = wins/total*100 if total > 0 else 0
        
        st.markdown(f"""
        <div class="prediction-box" style="margin-top:20px; border-color:{'#00ff40' if rate >= 40 else '#ff0040'};">
            <div style="font-size:16px;">TỶ LỆ THẮNG</div>
            <div style="font-size:36px; font-weight:900; color:{'#00ff40' if rate >= 40 else '#ff0040'};">
                {rate:.1f}% ({wins}/{total})
            </div>
        </div>
        """, unsafe_allow_html=True)

# === MAIN APP ===
def main():
    data = load_data()
    
    st.markdown('<div class="main-header">🎯 TITAN V32 PRO</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center; color:#888; margin-bottom:20px;">AI-Powered 5D Prediction System</div>', unsafe_allow_html=True)
    
    tabs = ["📥 Nhập Kết Quả", "🎯 Bạch Thủ", "🔥 Số Bệt", "❄️ Số Gan", "📊 Lịch Sử"]
    
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = tabs[0]
    
    cols = st.columns(len(tabs))
    for i, tab in enumerate(tabs):
        with cols[i]:
            if st.button(tab, key=f"tab_{i}", use_container_width=True):
                st.session_state.active_tab = tab
    
    st.markdown("---")
    
    if st.session_state.active_tab == tabs[0]:
        render_input_tab(data)
    elif st.session_state.active_tab == tabs[1]:
        render_bach_thu_tab(data)
        
        if st.button("🔮 PHÂN TÍCH AI NÂNG CAO"):
            if len(data["results"]) >= 15:
                with st.spinner("Đang phân tích với AI..."):
                    ai_result = AIAnalyzer(data["results"], StatisticalEngine(data["results"])).analyze_with_gemini()
                    if ai_result:
                        st.success("✅ AI đề xuất:")
                        st.json(ai_result)
                    else:
                        st.warning("AI không đưa ra kết quả")
            else:
                st.warning("Cần ít nhất 15 kỳ để AI phân tích")
    
    elif st.session_state.active_tab == tabs[2]:
        render_bet_tab(data)
    elif st.session_state.active_tab == tabs[3]:
        render_gan_tab(data)
    elif st.session_state.active_tab == tabs[4]:
        render_history_tab(data)
    
    st.markdown("---")
    st.markdown('<div style="text-align:center; color:#444; font-size:11px; margin-top:20px;">TITAN V32 PRO | AI-Powered Prediction</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()