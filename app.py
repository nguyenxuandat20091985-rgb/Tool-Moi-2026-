import streamlit as st
import json, os, re, math
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import google.generativeai as genai

# === CẤU HÌNH ===
DB_FILE = "titan_v52_data.json"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

st.set_page_config(page_title="TITAN V52 - AI MASTER", page_icon="🧠", layout="centered")

# === CSS DARK NEON ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    .stApp {
        background: linear-gradient(180deg, #000000 0%, #0a0e27 100%);
        color: #00ffff;
        font-family: 'Orbitron', sans-serif;
    }
    .main-header {
        font-size: 32px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #00ffff, #ff00ff, #00ffff);
        background-size: 200% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 3s ease infinite;
        margin-bottom: 10px;
    }
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .tab-btn {
        background: linear-gradient(135deg, #1a1f3a, #0f1428);
        border: 2px solid #00ffff;
        color: #00ffff;
        padding: 10px 15px;
        border-radius: 10px;
        margin: 3px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s;
        font-size: 12px;
    }
    .tab-btn:hover, .tab-btn.active {
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
        font-size: 48px;
        font-weight: 900;
        color: #00ffff;
        text-shadow: 0 0 20px #00ffff;
        letter-spacing: 10px;
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
    .streak-hot {
        background: linear-gradient(90deg, #ff0040, #ff8000);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        animation: pulse 1.5s infinite;
        display: inline-block;
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
        display: inline-block;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin: 15px 0;
    }
    .metric-cell {
        background: #0f1428;
        border: 1px solid #00ffff;
        border-radius: 10px;
        padding: 15px 10px;
        text-align: center;
    }
    .win { color: #00ff40; font-weight: 900; }
    .lose { color: #ff0040; font-weight: 900; }
    textarea {
        background: #0a0e27 !important;
        border: 1px solid #00ffff !important;
        color: #00ffff !important;
    }
    .confidence-bar {
        height: 25px;
        background: #0a0e27;
        border-radius: 12px;
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
        transition: width 0.5s;
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

# === ADVANCED STATISTICAL ENGINE ===
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
    
    def markov_probability(self, order=1):
        if len(self.digits) < order + 1:
            return {}
        
        transitions = defaultdict(lambda: defaultdict(int))
        for i in range(len(self.digits) - order):
            state = self.digits[i:i+order]
            next_digit = self.digits[i+order]
            transitions[state][next_digit] += 1
        
        probs = {}
        for state, next_counts in transitions.items():
            total = sum(next_counts.values())
            probs[state] = {d: c/total*100 for d, c in next_counts.items()}
        
        return probs
    
    def calculate_entropy(self):
        if not self.digits:
            return 0
        counter = Counter(self.digits)
        total = len(self.digits)
        entropy = -sum((c/total) * math.log2(c/total) for c in counter.values() if c > 0)
        return entropy
    
    def pattern_matching(self, target_pair):
        matches = 0
        for i in range(len(self.results) - 1):
            if set(target_pair).issubset(set(self.results[i])):
                next_result = self.results[i + 1]
                if set(target_pair).issubset(set(next_result)):
                    matches += 1
        return matches
    
    def calculate_pair_score(self, pair, data_weights):
        score = 0
        details = {}
        
        freq = self.frequency_analysis(window=30)
        gaps = self.calculate_gap()
        streaks = self.detect_streaks()
        
        p1, p2 = pair[0], pair[1]
        
        freq_score = (freq[p1]["freq"] + freq[p2]["freq"]) / 2 * data_weights["freq"]
        score += freq_score
        details["freq"] = freq_score
        
        gap_bonus = 0
        if 3 <= gaps[p1] <= 10:
            gap_bonus += 30 * data_weights["gap"]
        if 3 <= gaps[p2] <= 10:
            gap_bonus += 30 * data_weights["gap"]
        score += gap_bonus
        details["gap"] = gap_bonus
        
        streak_bonus = 0
        if streaks[p1] >= 2:
            streak_bonus += 20 * data_weights["streak"]
        if streaks[p2] >= 2:
            streak_bonus += 20 * data_weights["streak"]
        if streaks[p1] >= 4 or streaks[p2] >= 4:
            streak_bonus -= 30
        score += streak_bonus
        details["streak"] = streak_bonus
        
        pattern_match = self.pattern_matching(pair)
        pattern_score = pattern_match * 15
        score += pattern_score
        details["pattern"] = pattern_score
        
        entropy = self.calculate_entropy()
        if entropy < 3.0:
            score += 20
            details["entropy"] = 20
        else:
            details["entropy"] = 0
        
        return score, details

# === AI ENHANCED PREDICTION ===
class AIPredictor:
    def __init__(self, results, data):
        self.results = results
        self.data = data
        self.stats = StatisticalEngine(results)
    
    def get_gemini_analysis(self):
        if len(self.results) < 15:
            return None
        
        freq = self.stats.frequency_analysis(window=30)
        gaps = self.stats.calculate_gap()
        streaks = self.stats.detect_streaks()
        
        prompt = f"""
        Bạn là chuyên gia phân tích xổ số 5D. Hãy phân tích:
        
        20 kỳ gần nhất: {', '.join(self.results[-20:])}
        
        Thống kê:
        - Tần suất 30 kỳ: {json.dumps(freq)}
        - Độ gan: {json.dumps(gaps)}
        - Số bệt: {json.dumps(streaks)}
        
        Đề xuất TOP 3 cặp 2 số (0-9) có khả năng ra cao nhất kỳ tiếp theo.
        Trả về JSON format:
        {{
            "predictions": [
                {{"pair": "12", "confidence": 85, "reason": "lý do"}},
                {{"pair": "34", "confidence": 75, "reason": "lý do"}},
                {{"pair": "56", "confidence": 68, "reason": "lý do"}}
            ]
        }}
        """
        
        try:
            response = gemini_model.generate_content(prompt)
            text = response.text
            
            if "{" in text and "}" in text:
                start = text.index("{")
                end = text.rindex("}") + 1
                result = json.loads(text[start:end])
                return result.get("predictions", [])
        except Exception as e:
            pass
        
        return None
    
    def calculate_adaptive_weights(self):
        if len(self.data.get("predictions", [])) < 5:
            return {"freq": 1.0, "gap": 1.0, "streak": 1.0}
        
        recent_preds = self.data["predictions"][-10:]
        wins = [p for p in recent_preds if p.get("result") == "WIN"]
        
        if len(wins) < 2:
            return self.data.get("weights", {"freq": 1.0, "gap": 1.0, "streak": 1.0})
        
        freq_success = sum(1 for w in wins if "Tần suất" in w.get("reason", ""))
        gap_success = sum(1 for w in wins if "gan" in w.get("reason", "").lower())
        streak_success = sum(1 for w in wins if "bệt" in w.get("reason", "").lower())
        
        total_wins = len(wins) or 1
        
        return {
            "freq": 0.5 + (freq_success / total_wins),
            "gap": 0.5 + (gap_success / total_wins),
            "streak": 0.5 + (streak_success / total_wins)
        }
    
    def generate_predictions(self):
        weights = self.calculate_adaptive_weights()
        all_pairs = list(combinations("0123456789", 2))
        
        scored_pairs = []
        for pair in all_pairs:
            pair_str = "".join(pair)
            score, details = self.stats.calculate_pair_score(pair, weights)
            
            reasons = []
            if details["freq"] > 10:
                reasons.append("Tần suất cao")
            if details["gap"] > 20:
                reasons.append("Độ gan vàng")
            if details["streak"] > 15:
                reasons.append("Đang bệt")
            if details["pattern"] > 15:
                reasons.append("Pattern lặp")
            
            scored_pairs.append({
                "pair": pair_str,
                "score": score,
                "details": details,
                "reasons": reasons
            })
        
        scored_pairs.sort(key=lambda x: x["score"], reverse=True)
        
        ai_suggestions = self.get_gemini_analysis()
        if ai_suggestions:
            for ai_pred in ai_suggestions[:3]:
                pair = ai_pred.get("pair", "")
                if pair and len(pair) == 2:
                    for sp in scored_pairs:
                        if sp["pair"] == pair:
                            sp["score"] += ai_pred.get("confidence", 0) * 0.5
                            sp["reasons"].append(f"AI: {ai_pred.get('reason', '')}")
                            break
        
        scored_pairs.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_pairs[:5], weights

# === UI COMPONENTS ===
def render_input_tab(data):
    st.markdown("### 📥 NHẬP KẾT QUẢ")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        new_input = st.text_area("Dán kết quả (mỗi kỳ 1 dòng):", height=120, 
                                 placeholder="12345\n67890\n13579\n...",
                                 label_visibility="collapsed")
    
    with col2:
        if st.button("💾 LƯU", type="primary", use_container_width=True):
            nums = parse_numbers(new_input)
            if nums:
                data["results"].extend(nums)
                data["results"] = data["results"][-200:]
                save_data(data)
                st.success(f"✅ {len(nums)} kỳ")
                st.rerun()
    
    if st.button("🗑️ XÓA DATA", use_container_width=True):
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
            st.session_state.clear()
            st.success("Đã xóa toàn bộ data")
            st.rerun()
    
    if data["results"]:
        st.markdown(f"**📊 Tổng: {len(data['results'])} kỳ**")
        st.markdown(f"**Mới nhất:** {' → '.join(data['results'][-5:])}")

def render_prediction_tab(data):
    st.markdown("### 🎯 DỰ ĐOÁN 2 SỐ")
    
    if len(data["results"]) < 10:
        st.warning("⚠️ Cần ít nhất 10 kỳ dữ liệu")
        return
    
    predictor = AIPredictor(data["results"], data)
    predictions, weights = predictor.generate_predictions()
    
    st.markdown("#### 📊 TRỌNG SỐ HIỆN TẠI:")
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-cell">
            <div style="font-size:10px; color:#888;">TẦN SUẤT</div>
            <div style="font-size:20px; font-weight:900; color:#00ffff;">{weights['freq']:.2f}x</div>
        </div>
        <div class="metric-cell">
            <div style="font-size:10px; color:#888;">ĐỘ GAN</div>
            <div style="font-size:20px; font-weight:900; color:#ff00ff;">{weights['gap']:.2f}x</div>
        </div>
        <div class="metric-cell">
            <div style="font-size:10px; color:#888;">BỆT</div>
            <div style="font-size:20px; font-weight:900; color:#00ff40;">{weights['streak']:.2f}x</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if predictions:
        st.markdown("### 🏆 TOP 3 CẶP VIP:")
        
        for i, pred in enumerate(predictions[:3]):
            medal = "🥇" if i == 0 else ("🥈" if i == 1 else "🥉")
            confidence = min(95, max(40, pred["score"] / 3))
            
            reasons_str = ", ".join(pred["reasons"][:3]) if pred["reasons"] else "Phân tích thống kê"
            
            st.markdown(f"""
            <div class="prediction-box" style="border-color: {'#FFD700' if i == 0 else '#C0C0C0' if i == 1 else '#CD7F32'};">
                <div style="font-size:18px; margin-bottom:10px;">{medal} CẶP {pred['pair'][0]} - {pred['pair'][1]}</div>
                <div class="big-number">{pred['pair']}</div>
                <div style="margin:10px 0;">
                    <span class="score-badge">Score: {pred['score']:.1f}</span>
                    <span class="score-badge" style="background:linear-gradient(90deg,#00ff40,#80ff80);color:#000">
                        Tin cậy: {confidence:.0f}%
                    </span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width:{confidence}%; background:{'#00ff40' if confidence >= 70 else '#ffff00' if confidence >= 50 else '#ff0040'};">
                        {confidence:.0f}%
                    </div>
                </div>
                <div style="font-size:11px; color:#888; margin-top:10px;">{reasons_str}</div>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🔮 PHÂN TÍCH AI CHI TIẾT"):
            with st.spinner("Đang phân tích với Gemini AI..."):
                ai_result = predictor.get_gemini_analysis()
                if ai_result:
                    st.success("✅ Gemini AI đề xuất:")
                    for ai_pred in ai_result:
                        st.markdown(f"**{ai_pred['pair']}** - {ai_pred['confidence']}%: {ai_pred.get('reason', 'N/A')}")
                else:
                    st.warning("AI chưa đưa ra kết quả")

def render_streak_tab(data):
    st.markdown("### 🔥 SỐ BỆT")
    
    if len(data["results"]) < 5:
        st.warning("Cần ít nhất 5 kỳ")
        return
    
    stats = StatisticalEngine(data["results"])
    streaks = stats.detect_streaks()
    
    active_streaks = [(d, s) for d, s in streaks.items() if s >= 2]
    active_streaks.sort(key=lambda x: x[1], reverse=True)
    
    if active_streaks:
        for digit, streak in active_streaks:
            status = "✅ THEO" if streak <= 3 else "⚠️ CẨN THẬN" if streak <= 5 else "❌ TRÁNH"
            st.markdown(f"""
            <div class="prediction-box">
                <span class="streak-hot">Số {digit} - Bệt {streak} kỳ</span>
                <div style="margin-top:10px; font-size:14px;">{status}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Không có số bệt đáng chú ý")

def render_gap_tab(data):
    st.markdown("### ❄️ SỐ GAN")
    
    if len(data["results"]) < 10:
        st.warning("Cần ít nhất 10 kỳ")
        return
    
    stats = StatisticalEngine(data["results"])
    gaps = stats.calculate_gap()
    
    sorted_gaps = sorted(gaps.items(), key=lambda x: x[1], reverse=True)
    
    for digit, gap in sorted_gaps[:5]:
        status = "🔥 SẮP VỀ" if gap >= 8 else "⏳ THEO DÕI" if gap >= 5 else "❄️ LẠNH"
        st.markdown(f"""
        <div class="prediction-box">
            <span class="gan-cold" style="font-size:24px;">Số {digit}</span>
            <div style="margin:10px 0; font-size:16px;">{gap} kỳ chưa về</div>
            <div style="font-size:14px;">{status}</div>
        </div>
        """, unsafe_allow_html=True)

def render_history_tab(data):
    st.markdown("### 📊 LỊCH SỬ")
    
    if not data.get("predictions"):
        st.info("Chưa có lịch sử")
        return
    
    df = pd.DataFrame(data["predictions"][-20:])
    
    def color_result(val):
        if val == "WIN":
            return "color: #00ff40; font-weight: 900"
        return "color: #ff0040; font-weight: 900"
    
    st.dataframe(df.style.applymap(color_result, subset=['result']), 
                 use_container_width=True, hide_index=True)
    
    if data["predictions"]:
        wins = sum(1 for p in data["predictions"] if p.get("result") == "WIN")
        total = len(data["predictions"])
        rate = wins/total*100 if total > 0 else 0
        
        color = "#00ff40" if rate >= 40 else "#ffff00" if rate >= 30 else "#ff0040"
        
        st.markdown(f"""
        <div class="prediction-box" style="border-color:{color};">
            <div style="font-size:16px;">TỶ LỆ THẮNG</div>
            <div style="font-size:36px; font-weight:900; color:{color};">
                {rate:.1f}% ({wins}/{total})
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width:{rate}%; background:{color};">{rate:.0f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# === MAIN APP ===
def main():
    data = load_data()
    
    st.markdown('<div class="main-header">🧠 TITAN V52 - AI MASTER</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center; color:#888; margin-bottom:15px; font-size:12px;">Statistical + Gemini AI | Auto-Learning</div>', unsafe_allow_html=True)
    
    tabs = ["📥 Nhập", "🎯 Dự đoán", "🔥 Bệt", "❄️ Gan", "📊 Lịch sử"]
    
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = tabs[1]
    
    cols = st.columns(len(tabs))
    for i, tab in enumerate(tabs):
        with cols[i]:
            if st.button(tab, key=f"tab_{i}", use_container_width=True):
                st.session_state.active_tab = tab
    
    st.markdown("---")
    
    if st.session_state.active_tab == tabs[0]:
        render_input_tab(data)
    elif st.session_state.active_tab == tabs[1]:
        render_prediction_tab(data)
        
        if data["results"] and st.button("✅ ĐÁNH DẤU KẾT QUẢ"):
            latest = data["results"][-1]
            pred_col = st.columns(3)
            with pred_col[0]:
                num1 = st.selectbox("Số 1", list("0123456789"), key="check1")
            with pred_col[1]:
                num2 = st.selectbox("Số 2", list("0123456789"), key="check2")
            with pred_col[2]:
                if st.button("Lưu KQ", type="primary"):
                    pair = "".join(sorted([num1, num2]))
                    win = all(d in latest for d in pair)
                    
                    data["predictions"].append({
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "result_number": latest,
                        "predicted": pair,
                        "result": "WIN" if win else "LOSE",
                        "reason": "Manual check"
                    })
                    data["predictions"] = data["predictions"][-100:]
                    save_data(data)
                    st.success("✅ WIN!" if win else "❌ LOSE")
                    st.rerun()
    
    elif st.session_state.active_tab == tabs[2]:
        render_streak_tab(data)
    elif st.session_state.active_tab == tabs[3]:
        render_gap_tab(data)
    elif st.session_state.active_tab == tabs[4]:
        render_history_tab(data)
    
    st.markdown("---")
    st.markdown('<div style="text-align:center; color:#444; font-size:10px; margin-top:20px;">TITAN V52 | No Sklearn | Pure Stats + AI</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()