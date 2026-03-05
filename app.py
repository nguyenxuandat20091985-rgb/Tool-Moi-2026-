# ==============================================================================
# TITAN AI ENGINE v2.1 - FIXED TEXT VISIBILITY
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import random
import re
import math
import time
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. FIXED CSS - Visible Text Input
# ==============================================================================

st.set_page_config(
    page_title="🎯 TITAN AI ENGINE",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        color: #e6edf3;
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    }
    #MainMenu, footer, header { visibility: hidden; }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 10px 40px rgba(102,126,234,0.3);
    }
    .header-title {
        font-size: 32px;
        font-weight: 900;
        color: white;
        margin: 0;
    }
    .header-subtitle {
        font-size: 14px;
        color: rgba(255,255,255,0.9);
        margin-top: 8px;
    }
    
    /* Main Numbers - Horizontal Grid */
    .main-numbers-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 15px;
        margin: 25px 0;
    }
    .main-number-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 20px;
        padding: 30px 20px;
        text-align: center;
        box-shadow: 0 8px 30px rgba(245,87,108,0.4);
    }
    .main-number {
        font-size: 64px;
        font-weight: 900;
        color: white;
        line-height: 1;
    }
    .number-label {
        font-size: 12px;
        color: rgba(255,255,255,0.9);
        margin-top: 10px;
        text-transform: uppercase;
    }
    
    /* Support Numbers */
    .support-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 25px 0;
    }
    .support-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 15px;
        padding: 22px 15px;
        text-align: center;
    }
    .support-number {
        font-size: 38px;
        font-weight: 800;
        color: white;
    }
    .support-label {
        font-size: 11px;
        color: rgba(255,255,255,0.9);
        margin-top: 8px;
    }
    
    /* Status Banner */
    .status-banner {
        padding: 15px 25px;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        font-size: 15px;
        margin: 20px 0;
    }
    .status-ok {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: #065f46;
    }
    .status-warn {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: #7c2d12;
    }
    .status-stop {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
    }
    
    /* Info Card */
    .info-card {
        background: rgba(255,255,255,0.05);
        border-left: 4px solid #667eea;
        border-radius: 10px;
        padding: 15px 20px;
        margin: 15px 0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        padding: 14px 32px;
        font-size: 15px;
    }
    
    /* FIXED: Text Area - Visible Text */
    .stTextArea textarea {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 2px solid #667eea !important;
        border-radius: 12px;
        font-size: 16px;
    }
    .stTextArea textarea::placeholder {
        color: #94a3b8 !important;
        opacity: 0.7;
    }
    
    /* FIXED: Text Input - Visible Text */
    .stTextInput input {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 2px solid #667eea !important;
    }
    .stTextInput input::placeholder {
        color: #94a3b8 !important;
    }
    
    /* Mobile Responsive */
    @media (max-width: 600px) {
        .main-numbers-container {
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
        }
        .main-number {
            font-size: 48px;
        }
        .support-container {
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
        }
        .support-number {
            font-size: 28px;
        }
        .header-title {
            font-size: 24px;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. AI ENGINE (Keep same as before)
# ==============================================================================

class TitanAI:
    def __init__(self):
        self.weights = {
            'frequency': 25, 'gap': 20, 'markov': 20,
            'monte_carlo': 15, 'pattern': 12, 'hot_cold': 8
        }
    
    def analyze(self, history, max_simulations=2000):
        if not history or len(history) < 15:
            return self._fallback()
        clean_data = self._clean_history(history)
        if len(clean_data) < 15:
            return self._fallback("Cần ít nhất 15 kỳ")
        results = {}
        results['frequency'] = self._analyze_frequency(clean_data)
        results['gap'] = self._analyze_gap(clean_data)
        results['markov'] = self._analyze_markov(clean_data)
        results['monte_carlo'] = self._analyze_monte_carlo(clean_data, max_simulations)
        results['pattern'] = self._analyze_pattern(clean_data)
        results['hot_cold'] = self._analyze_hot_cold(clean_data)
        ensemble = self._ensemble_vote(results)
        stats_df = self._build_stats_df(clean_data, results)
        risk = self._calculate_risk(clean_data)
        return {
            'main_3': ensemble['main_3'],
            'support_4': ensemble['support_4'],
            'stats_df': stats_df,
            'risk': risk,
            'confidence': ensemble['confidence'],
            'logic': self._build_logic(results, ensemble)
        }
    
    def _clean_history(self, history):
        cleaned = []
        for item in history:
            s = str(item).strip()
            match = re.search(r'\d{5}', s)
            if match:
                cleaned.append(match.group())
        return cleaned
    
    def _analyze_frequency(self, data):
        weighted = Counter()
        n = len(data)
        for idx, num in enumerate(data):
            weight = 3.0 - 2.0 * (idx / max(n, 1))
            for d in num:
                weighted[d] += weight
        scores = {d: weighted.get(d, 0) for d in '0123456789'}
        top_3 = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        return {'scores': scores, 'top_3': top_3}
    
    def _analyze_gap(self, data):
        last_seen = {d: -1 for d in '0123456789'}
        for idx, num in enumerate(data):
            for d in num:
                if last_seen[d] == -1:
                    last_seen[d] = idx
        scores = {}
        for d in '0123456789':
            gap = last_seen[d] if last_seen[d] >= 0 else len(data)
            scores[d] = gap * 2.5 if gap <= 15 else max(0, 37.5 - (gap - 15) * 0.5)
        top_3 = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        return {'scores': scores, 'top_3': top_3}
    
    def _analyze_markov(self, data):
        if len(data) < 20:
            return {'scores': {d: 10 for d in '0123456789'}, 'top_3': ['1','5','9']}
        transitions = defaultdict(Counter)
        for i in range(len(data) - 1):
            curr, next_num = data[i], data[i + 1]
            for pos in range(5):
                if pos < len(curr) and pos < len(next_num):
                    transitions[curr[pos]][next_num[pos]] += 1
        last_num = data[0] if data else '00000'
        next_prob = Counter()
        for pos, last_d in enumerate(last_num[:5]):
            if last_d in transitions and transitions[last_d]:
                total = sum(transitions[last_d].values())
                for next_d, count in transitions[last_d].items():
                    pos_weight = 1.0 + 0.25 * (2 - abs(pos - 2))
                    next_prob[next_d] += (count / total) * pos_weight
        scores = {d: next_prob.get(d, 0) * 10 for d in '0123456789'}
        top_3 = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        while len(top_3) < 3:
            for d in '0123456789':
                if d not in top_3:
                    top_3.append(d)
                    break
        return {'scores': scores, 'top_3': top_3}
    
    def _analyze_monte_carlo(self, data, n_simulations):
        if len(data) < 20:
            return {'scores': {d: 10 for d in '0123456789'}, 'top_3': ['2','4','6']}
        recent = data[:80] if len(data) >= 80 else data
        pool = []
        for idx, num in enumerate(recent):
            weight = max(1, 4 - idx // 20)
            for d in num:
                pool.extend([d] * weight)
        if not pool:
            return {'scores': {d: 10 for d in '0123456789'}, 'top_3': ['0','1','2']}
        sim_count = Counter()
        for _ in range(n_simulations):
            sample = random.choices(pool, k=3)
            for d in sample:
                sim_count[d] += 1
        total = sum(sim_count.values()) or 1
        scores = {d: sim_count.get(d, 0) / total * 100 for d in '0123456789'}
        top_3 = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        return {'scores': scores, 'top_3': top_3}
    
    def _analyze_pattern(self, data):
        if len(data) < 25:
            return {'scores': {d: 10 for d in '0123456789'}, 'top_3': ['3','5','7']}
        recent = data[:40] if len(data) >= 40 else data
        candidates = Counter()
        for pos in range(5):
            seq = [n[pos] if len(n) > pos else '0' for n in recent]
            for i in range(len(seq) - 2):
                if seq[i] == seq[i+1] == seq[i+2]:
                    d = seq[i]
                    streak = 3
                    while i + streak < len(seq) and seq[i + streak] == d:
                        streak += 1
                    candidates[d] += 4 if streak == 3 else -2
        for pos in range(5):
            seq = [n[pos] if len(n) > pos else '0' for n in recent]
            for i in range(len(seq) - 4):
                if seq[i] == seq[i+2] == seq[i+4] and seq[i] != seq[i+1]:
                    candidates[seq[i]] += 3
        if not candidates:
            all_digits = ''.join(recent)
            freq = Counter(all_digits)
            for d, c in freq.most_common(3):
                candidates[d] += 3
        scores = {d: candidates.get(d, 0) * 2 for d in '0123456789'}
        top_3 = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        while len(top_3) < 3:
            for d in '0123456789':
                if d not in top_3:
                    top_3.append(d)
                    break
        return {'scores': scores, 'top_3': top_3}
    
    def _analyze_hot_cold(self, data):
        recent = data[:15] if len(data) >= 15 else data
        older = data[15:45] if len(data) >= 45 else data
        recent_count = Counter(''.join(recent))
        older_count = Counter(''.join(older)) if older else Counter()
        scores = {}
        for d in '0123456789':
            r = recent_count.get(d, 0)
            o = older_count.get(d, 0)
            if r >= 4:
                scores[d] = 25 + r * 3
            elif r == 0 and o >= 3:
                scores[d] = 20 + o * 2
            elif r >= 2:
                scores[d] = 15 + r * 2
            else:
                scores[d] = 10 + r
        top_3 = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])[:3]]
        return {'scores': scores, 'top_3': top_3}
    
    def _ensemble_vote(self, results):
        votes = Counter()
        for algo_name, result in results.items():
            weight = self.weights.get(algo_name, 10)
            for d in result.get('top_3', []):
                votes[d] += weight
        main_3 = [d for d, _ in votes.most_common(3)]
        while len(main_3) < 3:
            for d in '0123456789':
                if d not in main_3:
                    main_3.append(d)
                    break
        remaining = [d for d, _ in votes.most_common(10) if d not in main_3]
        support_4 = remaining[:4]
        while len(support_4) < 4:
            for d in '0123456789':
                if d not in main_3 and d not in support_4:
                    support_4.append(d)
                    break
        if votes:
            top_votes = [c for _, c in votes.most_common(3)]
            confidence = min(95, 55 + sum(top_votes) / 3)
        else:
            confidence = 50
        return {'main_3': main_3, 'support_4': support_4, 'confidence': int(confidence)}
    
    def _build_stats_df(self, data, results):
        rows = []
        for d in '0123456789':
            row = {'Digit': d}
            row['Frequency'] = results['frequency']['scores'].get(d, 0)
            row['Gap'] = results['gap']['scores'].get(d, 0)
            row['Markov'] = results['markov']['scores'].get(d, 0)
            row['Monte_Carlo'] = results['monte_carlo']['scores'].get(d, 0)
            row['Pattern'] = results['pattern']['scores'].get(d, 0)
            row['Hot_Cold'] = results['hot_cold']['scores'].get(d, 0)
            ai_score = (row['Frequency'] * 0.25 + row['Gap'] * 0.20 + 
                       row['Markov'] * 0.20 + row['Monte_Carlo'] * 0.15 + 
                       row['Pattern'] * 0.12 + row['Hot_Cold'] * 0.08)
            row['AI_Score'] = round(ai_score, 1)
            rows.append(row)
        df = pd.DataFrame(rows)
        return df.sort_values('AI_Score', ascending=False).reset_index(drop=True)
    
    def _calculate_risk(self, data):
        if len(data) < 20:
            return {'score': 30, 'level': 'MEDIUM', 'reason': 'Dữ liệu ít'}
        all_digits = ''.join(data[:50])
        counts = Counter(all_digits)
        total = len(all_digits)
        score = 0
        reasons = []
        entropy = sum(- (c/total) * math.log2(c/total) for c in counts.values() if c > 0)
        if entropy < 2.8:
            score += 25
            reasons.append('Kết quả quá đều')
        elif entropy > 3.4:
            score += 15
            reasons.append('Biến động mạnh')
        for pos in range(5):
            seq = [n[pos] if len(n) > pos else '0' for n in data[:30]]
            max_streak, curr = 1, 1
            for i in range(1, len(seq)):
                if seq[i] == seq[i-1]:
                    curr += 1
                    max_streak = max(max_streak, curr)
                else:
                    curr = 1
            if max_streak >= 5:
                score += 30
                reasons.append(f'Cầu bệt {max_streak} kỳ')
                break
        score = min(100, score)
        level = 'HIGH' if score >= 50 else 'MEDIUM' if score >= 25 else 'OK'
        return {'score': score, 'level': level, 'reason': '; '.join(reasons) if reasons else 'Ổn định'}
    
    def _build_logic(self, results, ensemble):
        parts = []
        freq_top = [d for d, _ in sorted(results['frequency']['scores'].items(), key=lambda x: -x[1])[:2]]
        if freq_top:
            parts.append(f"Tần suất: {','.join(freq_top)}")
        if ensemble['confidence'] >= 75:
            parts.append('Đồng thuận cao')
        return ' | '.join(parts) if parts else 'Phân tích AI'
    
    def _fallback(self, msg="Chưa đủ dữ liệu"):
        return {
            'main_3': ['?', '?', '?'],
            'support_4': ['0', '0', '0', '0'],
            'stats_df': pd.DataFrame({'Digit': list('0123456789'), 'AI_Score': [0]*10}),
            'risk': {'score': 0, 'level': 'LOW', 'reason': msg},
            'confidence': 0,
            'logic': msg
        }

# ==============================================================================
# 3. MAIN APPLICATION
# ==============================================================================

def main():
    if 'db' not in st.session_state:
        st.session_state.db = []
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'ai' not in st.session_state:
        st.session_state.ai = TitanAI()
    
    # Header
    st.markdown("""
    <div class="header-card">
        <div class="header-title">🎯 TITAN AI ENGINE</div>
        <div class="header-subtitle">Multi-Algorithm Prediction System</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.metric("📦 Tổng kỳ", len(st.session_state.db))
        if st.button("🗑️ Xóa dữ liệu"):
            st.session_state.db = []
            st.session_state.result = None
            st.success("✅ Đã xóa!")
            time.sleep(0.5)
            st.rerun()
    
    # Input Section
    st.markdown("### 📥 Nhập kết quả lịch sử")
    raw_input = st.text_area(
        "Dán kết quả tại đây (mỗi kỳ 1 dòng, 5 chữ số):",
        height=120,
        placeholder="12345\n67890\n54321\n...",
        key="data_input"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        analyze_btn = st.button("🚀 PHÂN TÍCH", type="primary", use_container_width=True)
    with col2:
        if st.button("📋 Demo", use_container_width=True):
            demo = "\n".join(["87746", "56421", "69137", "00443", "04475",
                            "64472", "16755", "58569", "62640", "99723"] * 5)
            st.session_state.data_input = demo
            st.rerun()
    with col3:
        if st.button("🔄 Mới", use_container_width=True):
            st.rerun()
    
    # Process
    if analyze_btn and raw_input.strip():
        with st.spinner("🧠 Đang phân tích..."):
            numbers = re.findall(r'\d{5}', raw_input)
            if not numbers:
                st.error("❌ Không tìm thấy số 5 chữ số!")
            else:
                existing = set(st.session_state.db)
                added = sum(1 for n in numbers if n not in existing and not existing.add(n))
                st.session_state.db = [n for n in numbers if n not in existing] + st.session_state.db
                if len(st.session_state.db) > 500:
                    st.session_state.db = st.session_state.db[:500]
                if added > 0:
                    st.success(f"✅ Đã thêm {added} số mới")
                if len(st.session_state.db) >= 15:
                    st.session_state.result = st.session_state.ai.analyze(st.session_state.db)
                    st.rerun()
                else:
                    st.warning(f"⚠️ Cần ít nhất 15 kỳ (hiện có: {len(st.session_state.db)})")
    
    # Display Results
    if st.session_state.result:
        res = st.session_state.result
        risk = res['risk']
        
        if risk['level'] == 'OK':
            status_class, status_text = 'status-ok', '✅ CÓ THỂ ĐÁNH'
        elif risk['level'] == 'MEDIUM':
            status_class, status_text = 'status-warn', '⚠️ THEO DÕI'
        else:
            status_class, status_text = 'status-stop', '🛑 NÊN DỪNG'
        
        st.markdown(f"""
        <div class="status-banner {status_class}">
            {status_text} | Risk: {risk['score']}/100 | {risk['reason']}
        </div>
        """, unsafe_allow_html=True)
        
        # 3 Main Numbers
        st.markdown("### 🔮 3 SỐ CHÍNH")
        main_3 = res['main_3']
        st.markdown(f"""
        <div class="main-numbers-container">
            <div class="main-number-card"><div class="main-number">{main_3[0]}</div><div class="number-label">SỐ 1</div></div>
            <div class="main-number-card"><div class="main-number">{main_3[1]}</div><div class="number-label">SỐ 2</div></div>
            <div class="main-number-card"><div class="main-number">{main_3[2]}</div><div class="number-label">SỐ 3</div></div>
        </div>
        """, unsafe_allow_html=True)
        
        # 4 Support Numbers
        st.markdown("### 🎲 4 SỐ LÓT")
        support_4 = res['support_4']
        st.markdown(f"""
        <div class="support-container">
            <div class="support-card"><div class="support-number">{support_4[0]}</div><div class="support-label">Lót 1</div></div>
            <div class="support-card"><div class="support-number">{support_4[1]}</div><div class="support-label">Lót 2</div></div>
            <div class="support-card"><div class="support-number">{support_4[2]}</div><div class="support-label">Lót 3</div></div>
            <div class="support-card"><div class="support-number">{support_4[3]}</div><div class="support-label">Lót 4</div></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.code(','.join(main_3 + support_4), language=None)
        
        if res['logic']:
            st.markdown(f'<div class="info-card"><strong>💡 Logic:</strong> {res["logic"]}</div>', unsafe_allow_html=True)
        
        # Verification
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            actual = st.text_input("Kết quả thực tế:", key="actual_result", placeholder="12864")
        with col2:
            if st.button("✅ Kiểm tra", type="primary", use_container_width=True):
                if actual and len(actual) == 5 and actual.isdigit():
                    is_win = set(main_3).issubset(set(actual))
                    st.success("🎉 TRÚNG!" if is_win else "❌ Trượt")
    
    st.markdown('<div style="text-align:center;color:#8b949e;padding:20px;">TITAN AI v2.1 | Fixed Text Visibility</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()