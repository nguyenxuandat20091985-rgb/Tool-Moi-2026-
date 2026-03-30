import streamlit as st
import json, os, re, math
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import google.generativeai as genai

# === CẤU HÌNH ===
DB_FILE = "titan_v32_data.json"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

st.set_page_config(page_title="TITAN V32 PRO", page_icon="🎯", layout="centered")

# === CSS MOBILE SCROLL ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    .stApp {
        background: linear-gradient(180deg, #0a0e27 0%, #1a1f3a 100%);
        color: #00ffff;
        font-family: 'Orbitron', sans-serif;
    }
    .main-header {
        font-size: 24px;
        font-weight: 900;
        text-align: center;
        color: #00ffff;
        text-shadow: 0 0 20px #00ffff;
        margin: 10px 0;
        position: sticky;
        top: 0;
        background: linear-gradient(180deg, #0a0e27, transparent);
        padding: 10px 0;
        z-index: 100;
    }
    .section {
        background: linear-gradient(135deg, #0f1428, #1a1f3a);
        border: 2px solid #00ffff;
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
    }
    .section-title {
        font-size: 18px;
        font-weight: 900;
        color: #FFD700;
        margin-bottom: 10px;
        border-bottom: 1px solid #00ffff;
        padding-bottom: 5px;
    }
    .prediction-box {
        background: linear-gradient(135deg, #1a1f3a, #0f1428);
        border-radius: 12px;
        padding: 15px;
        margin: 8px 0;
        text-align: center;
        border-left: 4px solid #00ffff;
    }
    .big-number {
        font-size: 36px;
        font-weight: 900;
        color: #00ffff;
        letter-spacing: 6px;
        margin: 5px 0;
    }
    .score-badge {
        background: linear-gradient(90deg, #00ffff, #0080ff);
        color: #000;
        padding: 4px 12px;
        border-radius: 15px;
        font-weight: bold;
        font-size: 12px;
        display: inline-block;
        margin: 3px;
    }
    .streak-hot {
        background: linear-gradient(90deg, #ff0040, #ff8000);
        color: white;
        padding: 4px 12px;
        border-radius: 15px;
        font-weight: bold;
        font-size: 12px;
        display: inline-block;
        margin: 3px;
    }
    .gan-cold {
        background: linear-gradient(90deg, #8000ff, #ff00ff);
        color: white;
        padding: 4px 12px;
        border-radius: 15px;
        font-weight: bold;
        font-size: 12px;
        display: inline-block;
        margin: 3px;
    }
    .win { color: #00ff40; font-weight: 900; }
    .lose { color: #ff0040; font-weight: 900; }
    .heatmap-row {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 5px;
        margin: 10px 0;
    }
    .heatmap-cell {
        width: 35px;
        height: 35px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 12px;
        flex-direction: column;
    }
    .heatmap-num { font-size: 14px; }
    .heatmap-count { font-size: 9px; color: #888; }
    textarea {
        background: #0a0e27 !important;
        border: 1px solid #00ffff !important;
        color: #00ffff !important;
        border-radius: 10px;
        font-size: 14px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        font-weight: bold;
    }
    .metric-inline {
        display: flex;
        justify-content: space-around;
        margin: 10px 0;
    }
    .metric-item {
        text-align: center;
        padding: 8px;
        background: #0a0e27;
        border-radius: 8px;
        min-width: 70px;
    }
    .metric-val { font-size: 20px; font-weight: 900; color: #00ffff; }
    .metric-lbl { font-size: 10px; color: #888; }
</style>
""", unsafe_allow_html=True)

# === DATA MANAGEMENT ===
def load_data():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except: pass
    return {"results": [], "predictions": [], "stats": {}}

def save_data(data):
    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def parse_numbers(text):
    clean = re.sub(r'\s+', '', text)
    return [n for n in re.findall(r'\d{5}', clean) if n]

# === STATISTICAL ENGINE (GIỮ NGUYÊN) ===
class StatisticalEngine:
    def __init__(self, results):
        self.results = results
        self.digits = "".join(results) if results else ""
        
    def frequency_analysis(self):
        counter = Counter(self.digits)
        total = len(self.digits) or 1
        return {d: {"count": counter.get(d, 0), "freq": counter.get(d, 0)/total*100} for d in "0123456789"}
    
    def bayesian_probability(self, pair):
        prior = 0.01
        likes = sum(1 for r in self.results[-50:] if set(pair).issubset(set(r)))
        total = len(self.results[-50:]) or 1
        likelihood = likes / total
        posterior = (likelihood * prior) / (likelihood * prior + 0.01 * (1-prior))
        return posterior * 100
    
    def moving_average(self, window=10):
        if len(self.results) < window: return {}
        recent = self.results[-window:]
        counter = Counter("".join(recent))
        return {d: counter.get(d, 0)/window for d in "0123456789"}
    
    def detect_streaks(self):
        streaks = {}
        for d in "0123456789":
            count = 0
            for r in reversed(self.results):
                if d in r: count += 1
                else: break
            streaks[d] = {"current": count, "max": count}
        return streaks
    
    def calculate_gap(self):
        gaps = {}
        for d in "0123456789":
            gap = 0
            for r in reversed(self.results):
                if d in r: break
                gap += 1
            gaps[d] = gap
        return gaps

# === AI ANALYSIS (GIỮ NGUYÊN) ===
class AIAnalyzer:
    def __init__(self, results, stats_engine):
        self.results = results
        self.stats = stats_engine
        
    def analyze_with_gemini(self):
        if len(self.results) < 10: return None
        freq = self.stats.frequency_analysis()
        gaps = self.stats.calculate_gap()
        streaks = self.stats.detect_streaks()
        prompt = f"""Phân tích xổ số 5D. Dữ liệu 20 kỳ: {', '.join(self.results[-20:])}
        Tần suất: {json.dumps(freq)}, Gan: {json.dumps(gaps)}, Bệt: {json.dumps(streaks)}
        Đề xuất TOP 3 cặp 2 số. JSON: {{"predictions": [["0","1"]], "confidence": [80], "reasons": ["lý do"]}}"""
        try:
            response = gemini_model.generate_content(prompt)
            text = response.text
            if "{" in text and "}" in text:
                return json.loads(text[text.index("{"):text.rindex("}")+1])
        except: pass
        return None
    
    def hybrid_score(self, pair):
        score, reasons = 0, []
        freq = self.stats.frequency_analysis()
        gaps = self.stats.calculate_gap()
        streaks = self.stats.detect_streaks()
        ma = self.stats.moving_average()
        p1, p2 = pair[0], pair[1]
        if freq[p1]["freq"] > 8 and freq[p2]["freq"] > 8: score += 30; reasons.append("Tần suất cao")
        if 3 <= gaps[p1] <= 8 or 3 <= gaps[p2] <= 8: score += 25; reasons.append("Độ gan vàng")
        if streaks[p1]["current"] >= 2 or streaks[p2]["current"] >= 2: score += 20; reasons.append("Đang bệt")
        if ma.get(p1, 0) > 0.3 or ma.get(p2, 0) > 0.3: score += 15; reasons.append("MA cao")
        score += self.stats.bayesian_probability(pair) * 0.5
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
            scored.append({"pair": "".join(pair), "score": score, "bayes": bayes, "reasons": reasons})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:8] if mode == "risky" else scored[:5]
    
    def detect_patterns(self):
        streaks = self.stats.detect_streaks()
        gaps = self.stats.calculate_gap()
        return {
            "streaks": [{"digit": d, "length": s["current"]} for d, s in streaks.items() if s["current"] >= 3],
            "gaps": [{"digit": d, "gap": g} for d, g in gaps.items() if g >= 10]
        }

# === MAIN APP - SINGLE SCROLL ===
def main():
    data = load_data()
    
    st.markdown('<div class="main-header">🎯 TITAN V32 PRO</div>', unsafe_allow_html=True)
    
    # === SECTION 1: NHẬP KẾT QUẢ ===
    st.markdown('<div class="section"><div class="section-title">📥 NHẬP KẾT QUẢ</div></div>', unsafe_allow_html=True)
    
    new_input = st.text_area("Dán kết quả (mỗi kỳ 1 dòng, kỳ mới ở dưới):", height=100, 
                             placeholder="12345\n67890\n...", key="input_area")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 LƯU & PHÂN TÍCH", type="primary"):
            nums = parse_numbers(new_input)
            if nums:
                # Auto check WIN
                if "last_pair" in st.session_state and data.get("predictions"):
                    last_actual = nums[-1]
                    win = all(d in last_actual for d in st.session_state.last_pair)
                    data["predictions"].insert(0, {"pair": st.session_state.last_pair, "result": last_actual, "win": win})
                    st.session_state.last_result = "WIN" if win else "LOSE"
                
                data["results"].extend(nums)
                data["results"] = data["results"][-200:]
                save_data(data)
                st.success(f"✅ Đã lưu {len(nums)} kỳ")
                st.rerun()
            else:
                st.warning("⚠️ Không tìm thấy số hợp lệ")
    
    with col2:
        if st.button("🗑️ XÓA DATA"):
            data["results"] = []
            data["predictions"] = []
            save_data(data)
            st.session_state.clear()
            st.rerun()
    
    if data["results"]:
        st.markdown(f"**📊 Tổng:** {len(data['results'])} kỳ | **Mới nhất:** {data['results'][-1]}")
    
    # === AUTO ANALYZE IF ENOUGH DATA ===
    if len(data["results"]) >= 15:
        pred_sys = PredictionSystem(data["results"])
        predictions = pred_sys.generate_predictions()
        patterns = pred_sys.detect_patterns()
        stats = pred_sys.stats
        
        # === SECTION 2: BẠCH THỦ 2 SỐ ===
        st.markdown('<div class="section"><div class="section-title">🎯 BẠCH THỦ 2 SỐ</div></div>', unsafe_allow_html=True)
        
        mode = st.radio("Chế độ:", ["An toàn", "Mạo hiểm"], horizontal=True, key="mode_radio")
        if "Mạo hiểm" in mode:
            predictions = pred_sys.generate_predictions("risky")
        
        # TOP 1 VIP
        if predictions:
            top = predictions[0]
            st.session_state.last_pair = top["pair"]
            st.markdown(f"""
            <div class="prediction-box" style="border-color:#00ff40; border-left-width:6px;">
                <div style="font-size:14px; color:#FFD700;">🥇 CẶP VIP</div>
                <div class="big-number">{top['pair'][0]} - {top['pair'][1]}</div>
                <div>
                    <span class="score-badge">Score: {top['score']:.0f}</span>
                    <span class="score-badge" style="background:linear-gradient(90deg,#00ff40,#80ff80);">Bayes: {top['bayes']:.0f}%</span>
                </div>
                <div style="font-size:11px; color:#888; margin-top:5px;">{', '.join(top['reasons'])[:50]}...</div>
            </div>
            """, unsafe_allow_html=True)
            
            # TOP 2-5
            st.markdown("**📋 Dự phòng:**")
            for p in predictions[1:5]:
                st.markdown(f"""
                <div class="prediction-box" style="padding:10px; margin:5px 0;">
                    <span style="font-size:22px; font-weight:900;">{p['pair']}</span>
                    <span style="margin-left:10px; color:#00ff40;">{p['score']:.0f}</span>
                </div>
                """, unsafe_allow_html=True)
    
    # === SECTION 3: SỐ BỆT ===
    st.markdown('<div class="section"><div class="section-title">🔥 SỐ BỆT</div></div>', unsafe_allow_html=True)
    
    if len(data["results"]) >= 5:
        streaks = StatisticalEngine(data["results"]).detect_streaks()
        active_streaks = [(d, s["current"]) for d, s in streaks.items() if s["current"] > 0]
        if active_streaks:
            for d, s in sorted(active_streaks, key=lambda x: x[1], reverse=True)[:5]:
                st.markdown(f"""
                <div class="prediction-box" style="padding:10px;">
                    <span class="streak-hot">Số {d}</span>
                    <span style="margin-left:10px;">Bệt {s} kỳ</span>
                    <span style="font-size:11px; color:#888; display:block; margin-top:3px;">{'✅ Theo' if s <= 3 else '⚠️ Cẩn thận'}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align:center; color:#888;">ℹ️ Không có số bệt</div>', unsafe_allow_html=True)
    
    # === SECTION 4: SỐ GAN ===
    st.markdown('<div class="section"><div class="section-title">❄️ SỐ GAN</div></div>', unsafe_allow_html=True)
    
    if len(data["results"]) >= 10:
        gaps = StatisticalEngine(data["results"]).calculate_gap()
        for d, g in sorted(gaps.items(), key=lambda x: x[1], reverse=True)[:5]:
            status = "🔥 Sắp về" if g >= 8 else "⏳ Theo dõi"
            st.markdown(f"""
            <div class="prediction-box" style="padding:10px;">
                <span class="gan-cold">Số {d}</span>
                <span style="margin-left:10px;">Gan {g} kỳ</span>
                <span style="font-size:11px; color:#888; display:block; margin-top:3px;">{status}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # === SECTION 5: HEATMAP ===
    st.markdown('<div class="section"><div class="section-title">🔥 HEATMAP (30 kỳ)</div></div>', unsafe_allow_html=True)
    
    if len(data["results"]) >= 30:
        recent = "".join(data["results"][-30:])
        counter = Counter(recent)
        max_c = max(counter.values()) if counter else 1
        cells = ""
        for d in "0123456789":
            c = counter.get(d, 0)
            intensity = c / max_c
            if intensity > 0.7: bg = "#ff0040"
            elif intensity > 0.4: bg = "#ff8000"
            elif intensity > 0.2: bg = "#00ffff"
            else: bg = "#333"
            cells += f'<div class="heatmap-cell" style="background:{bg};"><span class="heatmap-num">{d}</span><span class="heatmap-count">{c}</span></div>'
        st.markdown(f'<div class="heatmap-row">{cells}</div>', unsafe_allow_html=True)
    
    # === SECTION 6: LỊCH SỬ ===
    if data.get("predictions"):
        st.markdown('<div class="section"><div class="section-title">📊 LỊCH SỬ</div></div>', unsafe_allow_html=True)
        
        df_data = []
        for p in data["predictions"][:10]:
            df_data.append({
                "Cặp": p.get("pair", ""),
                "Kết quả": p.get("result", ""),
                "KQ": "🔥" if p.get("win") else "❌"
            })
        
        df = pd.DataFrame(df_data)
        def color_kq(val): return "color:#00ff40;font-weight:900" if val=="🔥" else "color:#ff0040;font-weight:900"
        st.dataframe(df.style.applymap(color_kq, subset=['KQ']), use_container_width=True, hide_index=True)
        
        # Win rate
        wins = sum(1 for p in data["predictions"] if p.get("win"))
        total = len(data["predictions"])
        rate = (wins/total*100) if total > 0 else 0
        st.markdown(f"""
        <div class="metric-inline">
            <div class="metric-item">
                <div class="metric-val" style="color:{'#00ff40' if rate>=40 else '#ff0040'};">{rate:.0f}%</div>
                <div class="metric-lbl">Win Rate</div>
            </div>
            <div class="metric-item">
                <div class="metric-val">{wins}</div>
                <div class="metric-lbl">Thắng</div>
            </div>
            <div class="metric-item">
                <div class="metric-val">{total}</div>
                <div class="metric-lbl">Tổng</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # === FOOTER ===
    st.markdown('<div style="text-align:center; color:#444; font-size:10px; margin:20px 0;">TITAN V32 PRO | Mobile Scroll | AI-Powered</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()