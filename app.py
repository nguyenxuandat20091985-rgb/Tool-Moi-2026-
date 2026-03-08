# ==============================================================================
# TITAN AI ENGINE v3.0 - RESPONSIBLE GAMBLING VERSION
# Phân tích thông minh + Quản lý vốn + Cảnh báo rủi ro
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import random
import re
import math
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CSS STYLING - Professional & Clean
# ==============================================================================

st.set_page_config(
    page_title="🎯 TITAN AI v3.0 | Responsible",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Global */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
        color: #e6edf3;
        font-family: 'Segoe UI', system-ui, sans-serif;
    }
    #MainMenu, footer, header { visibility: hidden; }
    
    /* Header */
    .header-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 100%);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-bottom: 25px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .header-title {
        font-size: 32px;
        font-weight: 900;
        color: white;
        margin: 0;
    }
    .header-subtitle {
        font-size: 14px;
        color: rgba(255,255,255,0.8);
        margin-top: 8px;
    }
    
    /* Warning Banner */
    .warning-banner {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: white;
        padding: 15px 25px;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        margin: 20px 0;
        border: 2px solid #fca5a5;
    }
    
    /* Status Cards */
    .status-card {
        padding: 15px 25px;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        font-size: 15px;
        margin: 20px 0;
    }
    .status-ok {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        border: 2px solid #34d399;
    }
    .status-warn {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        color: white;
        border: 2px solid #fbbf24;
    }
    .status-stop {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: white;
        border: 2px solid #f87171;
    }
    
    /* Number Cards - Horizontal */
    .numbers-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin: 20px 0;
    }
    .num-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 3px solid #60a5fa;
        border-radius: 16px;
        padding: 25px 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(96,165,250,0.3);
    }
    .num-value {
        font-size: 56px;
        font-weight: 900;
        color: #60a5fa;
        line-height: 1;
    }
    .num-label {
        font-size: 12px;
        color: #94a3b8;
        margin-top: 10px;
        text-transform: uppercase;
    }
    
    /* Support Numbers */
    .support-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin: 20px 0;
    }
    .support-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 2px solid #34d399;
        border-radius: 12px;
        padding: 18px 12px;
        text-align: center;
    }
    .support-value {
        font-size: 36px;
        font-weight: 800;
        color: #34d399;
    }
    
    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 15px;
        margin: 20px 0;
    }
    .stat-box {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .stat-value {
        font-size: 32px;
        font-weight: 800;
        color: #60a5fa;
    }
    .stat-label {
        font-size: 12px;
        color: #94a3b8;
        margin-top: 8px;
        text-transform: uppercase;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        padding: 14px 32px;
        font-size: 15px;
    }
    
    /* Text Input */
    .stTextArea textarea, .stTextInput input {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 2px solid #475569 !important;
        border-radius: 12px;
    }
    
    /* Info Box */
    .info-box {
        background: rgba(96,165,250,0.1);
        border-left: 4px solid #60a5fa;
        border-radius: 10px;
        padding: 15px 20px;
        margin: 15px 0;
    }
    
    /* Mobile */
    @media (max-width: 600px) {
        .num-value { font-size: 42px; }
        .support-value { font-size: 28px; }
        .numbers-grid, .support-grid { gap: 8px; }
        .header-title { font-size: 24px; }
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. AI ENGINE - With Reality Check
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
        
        # Add reality check
        ensemble['reality_check'] = self._reality_check()
        
        return {
            'main_3': ensemble['main_3'],
            'support_4': ensemble['support_4'],
            'stats_df': stats_df,
            'risk': risk,
            'confidence': ensemble['confidence'],
            'logic': self._build_logic(results, ensemble),
            'reality_check': ensemble['reality_check']
        }
    
    def _reality_check(self):
        """Important reality check about lottery odds."""
        return {
            'win_probability': '0.1% - 0.5%',  # Realistic odds for 3-of-5
            'house_edge': '30% - 50%',  # Typical for online lottery
            'recommendation': 'Chỉ chơi giải trí với tiền có thể mất',
            'warning': 'Không có hệ thống nào thắng lâu dài'
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
            'logic': msg,
            'reality_check': self._reality_check()
        }

# ==============================================================================
# 3. RESPONSIBLE GAMBLING FEATURES
# ==============================================================================

class ResponsibleGaming:
    """Responsible gambling tracking and warnings."""
    
    def __init__(self):
        self.session_start = datetime.now()
        self.bets = []
        self.daily_limit = 500000  # Default daily loss limit
        self.session_limit = 100000  # Default session loss limit
        self.max_bets_per_session = 20
    
    def add_bet(self, amount, won=False):
        """Record a bet."""
        self.bets.append({
            'timestamp': datetime.now(),
            'amount': amount,
            'won': won
        })
    
    def get_session_stats(self):
        """Get current session statistics."""
        if not self.bets:
            return {
                'total_bet': 0,
                'total_won': 0,
                'net': 0,
                'bet_count': 0,
                'win_rate': 0
            }
        
        total_bet = sum(b['amount'] for b in self.bets)
        total_won = sum(b['amount'] * 1.9 for b in self.bets if b['won'])
        net = total_won - total_bet
        wins = sum(1 for b in self.bets if b['won'])
        
        return {
            'total_bet': total_bet,
            'total_won': total_won,
            'net': net,
            'bet_count': len(self.bets),
            'win_rate': wins / len(self.bets) * 100 if self.bets else 0
        }
    
    def should_stop(self):
        """Check if user should stop playing."""
        stats = self.get_session_stats()
        warnings = []
        
        # Check session loss limit
        if stats['net'] < -self.session_limit:
            warnings.append(f"⚠️已达 session loss limit (-{abs(stats['net']):,.0f})")
        
        # Check bet count
        if stats['bet_count'] >= self.max_bets_per_session:
            warnings.append("⚠️ Đã đủ số ván cược tối đa")
        
        # Check win rate (if too low, suggest stop)
        if stats['bet_count'] >= 10 and stats['win_rate'] < 30:
            warnings.append("⚠️ Win rate quá thấp - Nên dừng")
        
        # Check time
        session_duration = datetime.now() - self.session_start
        if session_duration > timedelta(hours=1):
            warnings.append("⚠️ Đã chơi quá 1 tiếng - Nên nghỉ")
        
        return len(warnings) > 0, warnings, stats

# ==============================================================================
# 4. MAIN APPLICATION
# ==============================================================================

def main():
    # Initialize session state
    if 'db' not in st.session_state:
        st.session_state.db = []
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'ai' not in st.session_state:
        st.session_state.ai = TitanAI()
    if 'gaming' not in st.session_state:
        st.session_state.gaming = ResponsibleGaming()
    if 'bankroll' not in st.session_state:
        st.session_state.bankroll = {
            'initial': 1000000,
            'current': 1000000,
            'bet_per_round': 10000
        }
    
    # Header with WARNING
    st.markdown("""
    <div class="header-card">
        <div class="header-title">🎯 TITAN AI v3.0</div>
        <div class="header-subtitle">Phân tích thông minh | Chơi có trách nhiệm</div>
    </div>
    """, unsafe_allow_html=True)
    
    # IMPORTANT WARNING
    st.markdown("""
    <div class="warning-banner">
        ⚠️ CẢNH BÁO: Không có tool nào đảm bảo thắng. Chỉ chơi với tiền có thể mất.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar: Bankroll & Limits
    with st.sidebar:
        st.markdown("### 💰 Quản lý vốn")
        
        bankroll = st.session_state.bankroll
        st.metric("Vốn ban đầu", f"₫{bankroll['initial']:,.0f}")
        st.metric("Vốn hiện tại", f"₫{bankroll['current']:,.0f}")
        
        profit = bankroll['current'] - bankroll['initial']
        color = "🟢" if profit >= 0 else "🔴"
        st.metric("Lợi nhuận", f"{color} ₫{profit:,.0f}")
        
        st.markdown("---")
        st.markdown("### ⚙️ Giới hạn")
        
        new_limit = st.number_input(
            "Giới hạn thua/ngày:",
            value=500000,
            step=100000,
            key="daily_limit_input"
        )
        st.session_state.gaming.daily_limit = new_limit
        
        st.markdown("---")
        
        if st.button("🗑️ Reset tất cả"):
            st.session_state.db = []
            st.session_state.result = None
            st.session_state.bankroll['current'] = st.session_state.bankroll['initial']
            st.session_state.gaming = ResponsibleGaming()
            st.success("✅ Đã reset!")
            time.sleep(0.5)
            st.rerun()
    
    # Check if should stop
    should_stop, stop_warnings, session_stats = st.session_state.gaming.should_stop()
    
    if should_stop:
        st.markdown("""
        <div class="status-card status-stop">
            🛑 KHUYẾN NGHỊ: DỪNG CHƠI<br>
            <small style="font-size: 12px; margin-top: 10px;">
        """, unsafe_allow_html=True)
        for warning in stop_warnings:
            st.markdown(f"- {warning}")
        st.markdown("</small></div>", unsafe_allow_html=True)
    
    # Session Stats
    st.markdown("### 📊 Thống kê phiên")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tổng cược", f"₫{session_stats['total_bet']:,.0f}")
    with col2:
        st.metric("Thắng", f"₫{session_stats['total_won']:,.0f}")
    with col3:
        net_color = "🟢" if session_stats['net'] >= 0 else "🔴"
        st.metric("Lời/Lỗ", f"{net_color} ₫{session_stats['net']:,.0f}")
    with col4:
        st.metric("Win Rate", f"{session_stats['win_rate']:.1f}%")
    
    # Input Section
    st.markdown("### 📥 Nhập kết quả lịch sử")
    raw_input = st.text_area(
        "Dán kết quả (mỗi kỳ 1 dòng, 5 chữ số):",
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
    
    # Process Analysis
    if analyze_btn and raw_input.strip():
        with st.spinner("🧠 Đang phân tích..."):
            numbers = re.findall(r'\d{5}', raw_input)
            
            if not numbers:
                st.error("❌ Không tìm thấy số 5 chữ số!")
            else:
                existing_set = set(st.session_state.db)
                new_numbers = [n for n in numbers if n not in existing_set]
                
                if new_numbers:
                    st.session_state.db = new_numbers + st.session_state.db
                    if len(st.session_state.db) > 500:
                        st.session_state.db = st.session_state.db[:500]
                    st.success(f"✅ Đã thêm {len(new_numbers)} số mới")
                
                if len(st.session_state.db) >= 15:
                    st.session_state.result = st.session_state.ai.analyze(st.session_state.db)
                    st.rerun()
                else:
                    st.warning(f"⚠️ Cần ít nhất 15 kỳ (hiện có: {len(st.session_state.db)})")
    
    # Display Results
    if st.session_state.result:
        res = st.session_state.result
        risk = res['risk']
        
        # Status
        if risk['level'] == 'OK':
            status_class, status_text = 'status-ok', '✅ CÓ THỂ THAM KHẢO'
        elif risk['level'] == 'MEDIUM':
            status_class, status_text = 'status-warn', '⚠️ CẨN THẬN'
        else:
            status_class, status_text = 'status-stop', '🛑 NÊN DỪNG'
        
        st.markdown(f"""
        <div class="status-card {status_class}">
            {status_text} | Risk: {risk['score']}/100 | {risk['reason']}
        </div>
        """, unsafe_allow_html=True)
        
        # Reality Check
        rc = res.get('reality_check', {})
        st.markdown(f"""
        <div class="info-box">
            <strong>📊 Thực tế về xác suất:</strong><br>
            • Xác suất trúng: {rc.get('win_probability', 'N/A')}<br>
            • Nhà cái luôn có lợi thế: {rc.get('house_edge', 'N/A')}<br>
            • {rc.get('recommendation', '')}
        </div>
        """, unsafe_allow_html=True)
        
        # 3 Main Numbers
        st.markdown("### 🔮 3 SỐ PHÂN TÍCH")
        main_3 = res['main_3']
        st.markdown(f"""
        <div class="numbers-grid">
            <div class="num-card"><div class="num-value">{main_3[0]}</div><div class="num-label">Số 1</div></div>
            <div class="num-card"><div class="num-value">{main_3[1]}</div><div class="num-label">Số 2</div></div>
            <div class="num-card"><div class="num-value">{main_3[2]}</div><div class="num-label">Số 3</div></div>
        </div>
        """, unsafe_allow_html=True)
        
        # 4 Support Numbers
        st.markdown("### 🎲 4 SỐ THAM KHẢO")
        support_4 = res['support_4']
        st.markdown(f"""
        <div class="support-grid">
            <div class="support-card"><div class="support-value">{support_4[0]}</div></div>
            <div class="support-card"><div class="support-value">{support_4[1]}</div></div>
            <div class="support-card"><div class="support-value">{support_4[2]}</div></div>
            <div class="support-card"><div class="support-value">{support_4[3]}</div></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.code(','.join(main_3 + support_4), language=None)
        
        if res['logic']:
            st.markdown(f'<div class="info-box">💡 <strong>Logic:</strong> {res["logic"]}</div>', unsafe_allow_html=True)
        
        # Verification with Bankroll Update
        st.markdown("---")
        st.markdown("### ✅ Xác nhận kết quả")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            actual = st.text_input("Kết quả thực tế:", key="actual_result", placeholder="12864")
        with col2:
            if st.button("✅ Xác nhận", type="primary", use_container_width=True):
                if actual and len(actual) == 5 and actual.isdigit():
                    is_win = set(main_3).issubset(set(actual))
                    
                    # Update bankroll
                    bet = st.session_state.bankroll['bet_per_round']
                    if is_win:
                        profit = bet * 1.9
                        st.session_state.bankroll['current'] += profit
                        st.success(f"🎉 TRÚNG! +₫{profit:,.0f}")
                    else:
                        st.session_state.bankroll['current'] -= bet
                        st.warning(f"❌ Trượt! -₫{bet:,.0f}")
                    
                    # Record bet for responsible gaming
                    st.session_state.gaming.add_bet(bet, is_win)
                    
                    st.rerun()
    
    # Footer with Responsible Gaming Message
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #94a3b8; padding: 20px; font-size: 12px;">
        🎯 TITAN AI v3.0 | Responsible Gaming Version<br>
        ⚠️ Công cụ phân tích - Không đảm bảo thắng<br>
        🛑 Biết điểm dừng - Chơi có trách nhiệm<br>
        💡 Nếu thua liên tiếp, hãy dừng lại và nghỉ ngơi
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()