import streamlit as st
import pandas as pd
import re
import random
import math
from collections import Counter, defaultdict

# --- BỘ NÃO AI SUPREME ---
class LegendEngine:
    def __init__(self):
        self.weights = {'markov': 30, 'monte_carlo': 40, 'pattern': 30}
        self.win_history = []

    def predict_logic(self, history):
        if len(history) < 2: return None
        # Markov Chain
        nodes = defaultdict(Counter)
        for i in range(len(history)-1):
            nodes[history[i][-1]][history[i+1][-1]] += 1
        last = history[-1][-1]
        mk_res = [x[0] for x in nodes[last].most_common(3)]
        # Monte Carlo 100k
        pool = list("".join(history[-60:]))
        sim = Counter()
        for _ in range(100000):
            sample = random.choices(pool, k=3)
            for n in sample: sim[n] += 1
        mc_res = [x[0] for x in sim.most_common(3)]
        # Pattern
        p_res = [history[-1][0], history[-1][2], history[-1][4]]
        # Ensemble
        votes = Counter()
        for n in mk_res: votes[n] += self.weights['markov']
        for n in mc_res: votes[n] += self.weights['monte_carlo']
        for n in p_res: votes[n] += self.weights['pattern']
        final = votes.most_common(7)
        # Risk Entropy
        data = "".join(history[-40:])
        counts = Counter(data)
        probs = [c/len(data) for c in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs)
        risk_score = int(max(0, min(100, (3.32 - entropy) * 160)))
        
        return {
            'main': [x[0] for x in final[:3]],
            'support': [x[0] for x in final[3:7]],
            'risk_val': risk_score,
            'risk_lvl': "HIGH" if risk_score > 60 else "MEDIUM" if risk_score > 35 else "LOW"
        }

# --- GIAO DIỆN LEGENDARY ---
st.set_page_config(page_title="TITAN LEGENDARY", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&display=swap');
    .stApp { background: #050505; color: #ffffff; }
    
    /* Title Header */
    .header { font-family: 'Orbitron', sans-serif; text-align: center; color: #00f2ff; font-size: 32px; font-weight: 900; 
                text-shadow: 0 0 20px #00f2ff; margin: 10px 0; letter-spacing: 2px; }
    
    /* Risk Section */
    .risk-card { border-radius: 12px; padding: 15px; text-align: center; font-weight: bold; margin-bottom: 20px; 
                  border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(5px); }
    .LOW { background: rgba(0, 255, 127, 0.15); color: #00ff7f; border-color: #00ff7f; box-shadow: 0 0 15px rgba(0,255,127,0.2); }
    .MEDIUM { background: rgba(255, 165, 0, 0.15); color: #ffa500; border-color: #ffa500; }
    .HIGH { background: rgba(255, 50, 50, 0.15); color: #ff3232; border-color: #ff3232; box-shadow: 0 0 15px rgba(255,50,50,0.3); }

    /* Number Display */
    .main-grid { display: flex; justify-content: center; gap: 15px; margin: 20px 0; }
    .number-card { background: linear-gradient(145deg, #111, #222); border: 2px solid #00f2ff; border-radius: 15px; 
                    width: 100px; height: 130px; display: flex; flex-direction: column; align-items: center; justify-content: center;
                    box-shadow: 0 10px 20px rgba(0,0,0,0.5), inset 0 0 10px rgba(0,242,255,0.2); }
    .number-val { font-family: 'Orbitron', sans-serif; font-size: 55px; color: #fff; line-height: 1; }
    .number-tag { font-size: 10px; color: #00f2ff; margin-top: 8px; font-weight: bold; }

    /* Support Section */
    .support-container { display: flex; justify-content: center; gap: 10px; margin-top: 10px; }
    .sup-item { background: #1a1a1a; border: 1px solid #58a6ff; border-radius: 8px; padding: 10px 18px; 
                color: #58a6ff; font-size: 24px; font-weight: 900; box-shadow: 0 4px 8px rgba(0,0,0,0.3); }
</style>
""", unsafe_allow_html=True)

if "engine" not in st.session_state: st.session_state.engine = LegendEngine()
if "db" not in st.session_state: st.session_state.db = []
if "res" not in st.session_state: st.session_state.res = None

def main():
    st.markdown("<div class='header'>TITAN LEGENDARY v42</div>", unsafe_allow_html=True)

    # NHẬP DỮ LIỆU CẦU
    with st.expander("📥 DÁN KẾT QUẢ KỲ TRƯỚC", expanded=True):
        raw = st.text_area("", height=70, placeholder="Nhập 5 số mỗi kỳ (ví dụ: 12345)")
        if st.button("🚀 PHÂN TÍCH NHỊP CẦU", use_container_width=True):
            nums = re.findall(r'\d{5}', raw)
            if nums:
                st.session_state.db = nums
                st.session_state.res = st.session_state.engine.predict_logic(nums)
                st.rerun()

    # HIỂN THỊ KẾT QUẢ SUPREME
    if st.session_state.res:
        r = st.session_state.res
        
        # Risk Banner
        st.markdown(f"<div class='risk-card {r['risk_lvl']}'>RỦI RO: {r['risk_val']}/100 - {r['risk_lvl']}</div>", unsafe_allow_html=True)

        # Main Numbers
        st.markdown("<div style='text-align:center; color:#888; font-size:12px; font-weight:bold;'>🔮 3 SỐ CHÍNH (TỶ LỆ NỔ CAO)</div>", unsafe_allow_html=True)
        main_html = "".join([f'<div class="number-card"><div class="number-val">{n}</div><div class="number-tag">LEGEND</div></div>' for n in r['main']])
        st.markdown(f'<div class="main-grid">{main_html}</div>', unsafe_allow_html=True)

        # Support Numbers
        st.markdown("<div style='text-align:center; color:#888; font-size:12px; font-weight:bold; margin-top:20px;'>🎲 4 SỐ LÓT (GIỮ VỐN)</div>", unsafe_allow_html=True)
        sup_html = "".join([f'<div class="sup-item">{n}</div>' for n in r['support']])
        st.markdown(f'<div class="support-container">{sup_html}</div>', unsafe_allow_html=True)

        # Footer info
        st.markdown(f"<div style='text-align:center; margin-top:30px; font-size:11px; color:#444;'>Engine: Monte Carlo 100k + Markov v2 | Data: {len(st.session_state.db)} kỳ</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
