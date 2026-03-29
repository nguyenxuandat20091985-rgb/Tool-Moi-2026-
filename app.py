import streamlit as st
import re, pandas as pd, numpy as np, math
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import hashlib

# === CẤU HÌNH ===
LUCKY_OX = [0, 2, 5, 6, 7, 8]
st.set_page_config(page_title="TITAN V52 - PHOENIX", page_icon="🔥", layout="centered")

# === CSS PHOENIX ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&display=swap');
    
    .stApp {
        background: linear-gradient(180deg, #0a0000 0%, #1a0a0a 50%, #0a0000 100%);
        color: #FFD700;
        font-family: 'Orbitron', monospace;
    }
    
    .phoenix-header {
        font-size: 38px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #FF0000, #FFD700, #FF0000);
        background-size: 200% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: phoenix-glow 2s ease infinite;
    }
    
    @keyframes phoenix-glow {
        0%, 100% { filter: brightness(1); }
        50% { filter: brightness(1.5); }
    }
    
    .signal-box {
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
        border: 3px solid;
    }
    
    .signal-green {
        background: linear-gradient(135deg, #1a3a1a, #0a2a0a);
        border-color: #00ff40;
        color: #00ff40;
        animation: pulse-green 2s infinite;
    }
    
    .signal-yellow {
        background: linear-gradient(135deg, #3a3a1a, #2a2a0a);
        border-color: #ffff00;
        color: #ffff00;
    }
    
    .signal-red {
        background: linear-gradient(135deg, #3a1a1a, #2a0a0a);
        border-color: #ff0040;
        color: #ff0040;
        animation: pulse-red 1s infinite;
    }
    
    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 0 20px #00ff40; }
        50% { box-shadow: 0 0 40px #00ff40; }
    }
    
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 20px #ff0040; }
        50% { box-shadow: 0 0 40px #ff0040; }
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #1a1a2e, #0f0f1a);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 2px solid #FFD700;
        text-align: center;
    }
    
    .big-number {
        font-size: 56px;
        font-weight: 900;
        letter-spacing: 12px;
        color: #00FFFF;
        text-shadow: 0 0 20px #00FFFF;
    }
    
    .ensemble-meter {
        display: flex;
        justify-content: space-around;
        margin: 15px 0;
    }
    
    .ensemble-cell {
        text-align: center;
        padding: 10px;
        background: #1a1a2e;
        border-radius: 10px;
        min-width: 70px;
    }
    
    .tag {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 10px;
        font-weight: bold;
        margin: 3px;
    }
    
    .tag-green { background: #00ff40; color: #000; }
    .tag-red { background: #ff0040; color: #fff; }
    .tag-yellow { background: #ffff00; color: #000; }
    .tag-blue { background: #0066ff; color: #fff; }
    .tag-purple { background: #9900ff; color: #fff; }
    
    .stop-loss-box {
        background: linear-gradient(135deg, #3a0a0a, #2a0000);
        border: 3px solid #ff0040;
        color: #ff6680;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 18px;
        font-weight: 900;
        animation: blink-fast 0.5s infinite;
    }
    
    @keyframes blink-fast {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    
    .bet-size-indicator {
        font-size: 32px;
        font-weight: 900;
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .bet-max { background: #00ff40; color: #000; }
    .bet-mid { background: #ffff00; color: #000; }
    .bet-min { background: #ff6600; color: #000; }
    .bet-none { background: #ff0040; color: #fff; }
    
    .history-win { color: #00ff40; font-weight: 900; }
    .history-lose { color: #ff0040; font-weight: 900; }
    
    .progress-bar {
        height: 25px;
        background: #1a1a1a;
        border-radius: 12px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff0000, #ffff00, #00ff00);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        font-weight: bold;
        color: #000;
        transition: width 0.5s;
    }
    
    .stDataFrame {
        background: #0a0a0a !important;
    }
</style>
""", unsafe_allow_html=True)

# === 5-LỚP ENSEMBLE ENGINE ===

class EnsemblePredictor:
    """Kết hợp 5 thuật toán độc lập, chỉ bet khi đồng thuận"""
    
    def __init__(self, db, history=None):
        self.db = db
        self.history = history or []
        self.signals = {}
        self._run_all_models()
    
    def _run_all_models(self):
        """Chạy 5 model độc lập"""
        self.signals['frequency'] = self._frequency_model()
        self.signals['pattern'] = self._pattern_model()
        self.signals['entropy'] = self._entropy_model()
        self.signals['cycle'] = self._cycle_model()
        self.signals['behavioral'] = self._behavioral_model()
    
    def _frequency_model(self):
        """Model 1: Phân tích tần suất thuần túy"""
        if len(self.db) < 20:
            return {'score': 50, 'pairs': [], 'signal': 'NEUTRAL'}
        
        pair_pool = Counter()
        for num in self.db[-50:]:
            for p in combinations(sorted(set(num)), 2):
                pair_pool[p] += 1
        
        top_pairs = []
        for p, freq in pair_pool.most_common(10):
            gan = 0
            for num in reversed(self.db):
                if not set(p).issubset(set(num)):
                    gan += 1
                else:
                    break
            
            score = freq * 5
            if 4 <= gan <= 10:
                score += 40
            elif gan > 15:
                score -= 30
            
            top_pairs.append({'pair': "".join(p), 'score': score, 'gan': gan})
        
        top_pairs.sort(key=lambda x: x['score'], reverse=True)
        
        if top_pairs and top_pairs[0]['score'] > 100:
            signal = 'BUY'
        elif top_pairs and top_pairs[0]['score'] < 50:
            signal = 'SELL'
        else:
            signal = 'NEUTRAL'
        
        return {'score': top_pairs[0]['score'] if top_pairs else 50, 
                'pairs': top_pairs[:5], 'signal': signal}
    
    def _pattern_model(self):
        """Model 2: Nhận diện pattern lặp"""
        if len(self.db) < 30:
            return {'score': 50, 'pairs': [], 'signal': 'NEUTRAL'}
        
        # Tìm số xuất hiện theo pattern
        pattern_score = defaultdict(int)
        
        for i in range(len(self.db) - 2):
            curr = self.db[i]
            next_num = self.db[i + 1]
            
            # Pattern: số nào thường đi cùng nhau
            for d in curr:
                for nd in next_num:
                    pattern_score[(d, nd)] += 10
        
        # Tìm cặp có pattern mạnh nhất
        top_patterns = sorted(pattern_score.items(), key=lambda x: x[1], reverse=True)[:10]
        
        pairs = [{'pair': f"{p[0]}{p[1]}", 'score': s, 'gan': 0} for p, s in top_patterns]
        
        if top_patterns and top_patterns[0][1] > 30:
            signal = 'BUY'
        else:
            signal = 'NEUTRAL'
        
        return {'score': top_patterns[0][1] if top_patterns else 50,
                'pairs': pairs, 'signal': signal}
    
    def _entropy_model(self):
        """Model 3: Phân tích độ hỗn loạn"""
        if len(self.db) < 10:
            return {'score': 50, 'pairs': [], 'signal': 'NEUTRAL'}
        
        # Tính entropy
        recent = "".join(self.db[-15:])
        counter = Counter(recent)
        total = len(recent)
        
        entropy = 0
        for count in counter.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        # Entropy thấp -> sắp có biến động -> BUY
        # Entropy cao -> quá hỗn loạn -> SELL
        if entropy < 2.8:
            signal = 'BUY'
            score = 80 + (2.8 - entropy) * 50
        elif entropy > 3.3:
            signal = 'SELL'
            score = 50 - (entropy - 3.3) * 50
        else:
            signal = 'NEUTRAL'
            score = 60
        
        return {'score': max(0, min(100, score)), 'pairs': [], 
                'signal': signal, 'entropy': entropy}
    
    def _cycle_model(self):
        """Model 4: Phát hiện chu kỳ nhà cái"""
        if len(self.history) < 5:
            return {'score': 50, 'pairs': [], 'signal': 'NEUTRAL'}
        
        # Phân tích win/loss pattern trong history
        wins = [1 if '🔥' in h.get('KQ', '') else 0 for h in self.history[:15]]
        
        if len(wins) < 5:
            return {'score': 50, 'pairs': [], 'signal': 'NEUTRAL'}
        
        # Tìm pattern win/loss
        win_rate = sum(wins) / len(wins)
        
        # Nếu đang thua nhiều -> sắp đến chu kỳ trả -> BUY
        # Nếu đang thắng nhiều -> sắp đến chu kỳ thu -> SELL
        recent_5 = wins[:5]
        recent_win_rate = sum(recent_5) / len(recent_5) if recent_5 else 0.5
        
        if recent_win_rate < 0.3:
            signal = 'BUY'  # Sắp trả
            score = 70 + (0.3 - recent_win_rate) * 100
        elif recent_win_rate > 0.6:
            signal = 'SELL'  # Sắp thu
            score = 50 - (recent_win_rate - 0.6) * 100
        else:
            signal = 'NEUTRAL'
            score = 60
        
        return {'score': max(0, min(100, score)), 'pairs': [], 
                'signal': signal, 'win_rate': recent_win_rate}
    
    def _behavioral_model(self):
        """Model 5: Phân tích hành vi nhà cái"""
        if len(self.db) < 20:
            return {'score': 50, 'pairs': [], 'signal': 'NEUTRAL'}
        
        # Tìm số "hot" và "cold"
        recent_str = "".join(self.db[-20:])
        counter = Counter(recent_str)
        
        hot_numbers = [d for d, c in counter.most_common(3)]
        cold_numbers = [d for d, c in counter.most_common()[:-4]]
        
        # Tạo cặp từ hot numbers
        pairs = []
        for p in combinations(hot_numbers, 2):
            pairs.append({'pair': "".join(p), 'score': 70, 'gan': 0})
        
        # Thêm cặp hot-cold (số lạnh sắp nổ)
        for hot in hot_numbers[:2]:
            for cold in cold_numbers[:2]:
                pairs.append({'pair': "".join(sorted(p)), 'score': 60, 'gan': 0})
        
        if hot_numbers:
            signal = 'BUY'
        else:
            signal = 'NEUTRAL'
        
        return {'score': 70 if hot_numbers else 50, 'pairs': pairs, 
                'signal': signal, 'hot': hot_numbers}
    
    def get_ensemble_decision(self):
        """
        QUYẾT ĐỊNH ENSEMBLE:
        Chỉ bet khi 3/5 model đồng thuận BUY
        """
        buy_count = sum(1 for s in self.signals.values() if s.get('signal') == 'BUY')
        sell_count = sum(1 for s in self.signals.values() if s.get('signal') == 'SELL')
        
        # Tính score trung bình
        avg_score = np.mean([s['score'] for s in self.signals.values()])
        
        if buy_count >= 3:
            decision = 'BET'
            confidence = min(95, 50 + buy_count * 15)
        elif sell_count >= 3:
            decision = 'SKIP'
            confidence = min(95, 50 + sell_count * 15)
        else:
            decision = 'CAUTION'
            confidence = 50
        
        return {
            'decision': decision,
            'confidence': confidence,
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'avg_score': avg_score,
            'signals': self.signals
        }

# === QUẢN LÝ VỐN THÔNG MINH ===

class SmartBankroll:
    """Quản lý vốn theo Kelly Criterion + Stop Loss"""
    
    def __init__(self, history=None, base_bankroll=100):
        self.history = history or []
        self.base_bankroll = base_bankroll
        self.current_streak = self._calculate_streak()
        self.win_rate = self._calculate_win_rate()
    
    def _calculate_streak(self):
        """Tính streak hiện tại"""
        if not self.history:
            return 0
        
        streak = 0
        for h in self.history:
            if '🔥' in h.get('KQ', ''):
                streak += 1
            else:
                break
        return streak
    
    def _calculate_win_rate(self, window=20):
        """Tính win rate trong window kỳ"""
        if not self.history:
            return 0.5
        
        recent = self.history[:window]
        wins = sum(1 for h in recent if '🔥' in h.get('KQ', ''))
        return wins / len(recent) if recent else 0.5
    
    def get_bet_size(self, confidence, ensemble_decision):
        """
        Tính kích thước bet:
        - MAX: Confidence > 80% + ensemble BUY
        - MID: Confidence 60-80% + ensemble CAUTION
        - MIN: Confidence < 60% hoặc ensemble SKIP
        - NONE: Stop loss triggered
        """
        # Stop loss: thua 5 kỳ liên tiếp
        losses = 0
        for h in self.history:
            if '❌' in h.get('KQ', ''):
                losses += 1
            else:
                break
        
        if losses >= 5:
            return 'NONE', 0, "STOP LOSS - THUA 5 KỲ LIÊN TIẾP"
        
        # Kelly Criterion điều chỉnh
        kelly = (confidence / 100 * 1.85 - (1 - confidence / 100)) / 0.85
        kelly = max(0, min(kelly, 0.25))  # 0-25%
        
        if ensemble_decision == 'BET' and confidence >= 80:
            return 'MAX', min(kelly * 2, 0.25), "TIN CẬY CAO"
        elif ensemble_decision == 'BET' and confidence >= 60:
            return 'MID', kelly, "TIN CẬY TB"
        elif ensemble_decision == 'CAUTION':
            return 'MIN', kelly * 0.5, "THẬN TRỌNG"
        else:
            return 'NONE', 0, "KHÔNG NÊN BET"
    
    def get_recommendation(self, confidence, ensemble_decision):
        """Khuyến nghị hành động"""
        bet_size, percentage, reason = self.get_bet_size(confidence, ensemble_decision)
        
        bet_class = f"bet-{bet_size.lower()}"
        bet_text = {
            'MAX': 'VÀO TIỀN MẠNH',
            'MID': 'VÀO TIỀN TB',
            'MIN': 'VÀO TIỀN NHỎ',
            'NONE': 'KHÔNG BET'
        }
        
        return bet_class, bet_text.get(bet_size, ''), percentage, reason

# === XỬ LÝ DỮ LIỆU ===

def get_nums(text):
    clean_text = re.sub(r'\s+', '', text)
    nums = [n for n in re.findall(r"\d{5}", clean_text) if n]
    seen = set()
    return [n for n in nums if not (n in seen or seen.add(n))]

# === GIAO DIỆN ===

st.markdown('<h1 class="phoenix-header">🔥 TITAN V52 - PHOENIX RISING</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#888; font-size:11px;">5-Lớp Ensemble | Smart Bankroll | Stop Loss | Adaptive Learning</p>', unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

if "bankroll" not in st.session_state:
    st.session_state.bankroll = 100

user_input = st.text_area("📥 Kết quả (kỳ mới nhất ở dưới):", height=100, 
                          placeholder="99180\n50655\n06213")

col1, col2 = st.columns(2)
with col1:
    if st.button("🔥 KÍCH HOẠT PHOENIX", type="primary", use_container_width=True):
        nums = get_nums(user_input)
        if len(nums) >= 15:
            # Check kết quả kỳ trước
            if "last_pred" in st.session_state and nums:
                lp = st.session_state.last_pred
                last = nums[-1]
                if lp and lp.get('decision') == 'BET' and lp['pairs']:
                    best = lp['pairs'][0]['pair']
                    win = all(d in last for d in best)
                    st.session_state.history.insert(0, {
                        'Kỳ': last,
                        'Dự đoán': best,
                        'KQ': '🔥' if win else '❌'
                    })
            
            ensemble = EnsemblePredictor(nums, st.session_state.history)
            decision = ensemble.get_ensemble_decision()
            
            bankroll = SmartBankroll(st.session_state.history)
            bet_class, bet_text, percentage, reason = bankroll.get_recommendation(
                decision['confidence'], 
                decision['decision']
            )
            
            st.session_state.last_pred = {
                **decision,
                'bet_class': bet_class,
                'bet_text': bet_text,
                'bet_percentage': percentage,
                'bet_reason': reason
            }
            st.rerun()
        else:
            st.warning(f"Cần 15+ kỳ (có {len(nums)})")

with col2:
    if st.button("🗑️ Reset", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# === HIỂN THỊ KẾT QUẢ ===

if "last_pred" in st.session_state:
    res = st.session_state.last_pred
    
    # === SIGNAL BOX ===
    if res['decision'] == 'BET':
        signal_class = "signal-green"
        signal_text = "✅ TÍN HIỆU BET"
        signal_detail = f"{res['buy_signals']}/5 model đồng thuận"
    elif res['decision'] == 'SKIP':
        signal_class = "signal-red"
        signal_text = "⚠️ KHÔNG NÊN BET"
        signal_detail = f"{res['sell_signals']}/5 model báo rủi ro"
    else:
        signal_class = "signal-yellow"
        signal_text = "⚡ THẬN TRỌNG"
        signal_detail = "Không đủ đồng thuận"
    
    st.markdown(f"""
    <div class="signal-box {signal_class}">
        <div style="font-size:24px; font-weight:900;">{signal_text}</div>
        <div style="font-size:14px; margin-top:5px;">{signal_detail}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # === BET SIZE INDICATOR ===
    if res['bet_class'] != 'bet-none':
        st.markdown(f"""
        <div class="bet-size-indicator {res['bet_class']}">
            {res['bet_text']} ({res['bet_percentage']*100:.1f}% vốn)<br>
            <span style="font-size:14px; font-weight:500;">{res['bet_reason']}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="stop-loss-box">
            🛑 {res['bet_text']}<br>
            <span style="font-size:14px; font-weight:500;">{res['bet_reason']}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # === ENSEMBLE METER ===
    st.markdown("""<div style="text-align:center; margin:15px 0; font-size:14px;">📊 5-LỚP MODEL</div>""", unsafe_allow_html=True)
    
    signals = res.get('signals', {})
    st.markdown(f"""
    <div class="ensemble-meter">
        <div class="ensemble-cell">
            <div style="font-size:10px; color:#888;">FREQ</div>
            <div style="font-size:16px; font-weight:900; color:{'#00ff40' if signals.get('frequency', {}).get('signal') == 'BUY' else '#ff0040'};">
                {signals.get('frequency', {}).get('signal', '-')}
            </div>
        </div>
        <div class="ensemble-cell">
            <div style="font-size:10px; color:#888;">PATTERN</div>
            <div style="font-size:16px; font-weight:900; color:{'#00ff40' if signals.get('pattern', {}).get('signal') == 'BUY' else '#ff0040'};">
                {signals.get('pattern', {}).get('signal', '-')}
            </div>
        </div>
        <div class="ensemble-cell">
            <div style="font-size:10px; color:#888;">ENTROPY</div>
            <div style="font-size:16px; font-weight:900; color:{'#00ff40' if signals.get('entropy', {}).get('signal') == 'BUY' else '#ff0040'};">
                {signals.get('entropy', {}).get('signal', '-')}
            </div>
        </div>
        <div class="ensemble-cell">
            <div style="font-size:10px; color:#888;">CYCLE</div>
            <div style="font-size:16px; font-weight:900; color:{'#00ff40' if signals.get('cycle', {}).get('signal') == 'BUY' else '#ff0040'};">
                {signals.get('cycle', {}).get('signal', '-')}
            </div>
        </div>
        <div class="ensemble-cell">
            <div style="font-size:10px; color:#888;">BEHAV</div>
            <div style="font-size:16px; font-weight:900; color:{'#00ff40' if signals.get('behavioral', {}).get('signal') == 'BUY' else '#ff0040'};">
                {signals.get('behavioral', {}).get('signal', '-')}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # === CONFIDENCE ===
    st.markdown(f"""
    <div class="prediction-card">
        <div style="font-size:14px; color:#888;">ĐỘ TIN CẬY ENSEMBLE</div>
        <div style="font-size:42px; font-weight:900; color:{'#00ff40' if res['confidence'] >= 70 else '#ffff00' if res['confidence'] >= 50 else '#ff0040'};">
            {res['confidence']:.0f}%
        </div>
        <div class="progress-bar">
            <div class="progress-fill" style="width:{res['confidence']}%;">{res['confidence']:.0f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # === TOP PAIRS ===
    if res['decision'] == 'BET' and 'pairs' in res.get('signals', {}).get('frequency', {}):
        st.markdown("""<div style="text-align:center; margin:15px 0; font-size:14px;">🎯 TOP 5 PAIRS</div>""", unsafe_allow_html=True)
        
        freq_pairs = res['signals']['frequency'].get('pairs', [])
        for i, p in enumerate(freq_pairs[:5]):
            tags = ""
            if p.get('gan', 0) >= 4 and p.get('gan', 0) <= 10:
                tags += '<span class="tag tag-green">GAN VÀNG</span>'
            
            st.markdown(f"""
            <div style="background:#1a1a2e; border-radius:10px; padding:12px; margin:5px 0; 
                        border-left: 4px solid {'#00ff40' if i == 0 else '#444'};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:26px; font-weight:900; color:#FFD700;">{p['pair'][0]}-{p['pair'][1]}</span>
                    <span style="font-size:18px; color:#00ff40;">{p['score']:.0f}</span>
                </div>
                <div style="margin-top:5px;">{tags}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # === HISTORY ===
    if st.session_state.history:
        st.markdown("""<div style="text-align:center; margin:20px 0; font-size:16px;">📋 LỊCH SỬ</div>""", unsafe_allow_html=True)
        
        df = pd.DataFrame(st.session_state.history[:15])
        
        def color_kq(val):
            return 'color: #00ff40; font-weight: 900' if '🔥' in val else 'color: #ff0040; font-weight: 900'
        
        st.dataframe(df.style.applymap(color_kq, subset=['KQ']), 
                     use_container_width=True, hide_index=True)
        
        # Win rate
        wins = sum(1 for h in st.session_state.history if '🔥' in h['KQ'])
        total = len(st.session_state.history)
        rate = (wins/total*100) if total > 0 else 0
        
        color_rate = "#00ff40" if rate >= 40 else ("#ffff00" if rate >= 30 else "#ff0040")
        
        st.markdown(f"""
        <div class="signal-box" style="border-color:{color_rate}; color:{color_rate}; margin-top:15px;">
            <div style="font-size:14px;">TỶ LỆ THẮNG</div>
            <div style="font-size:36px; font-weight:900;">{rate:.1f}% ({wins}/{total})</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width:{rate}%; background:{color_rate};">{rate:.0f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Phân tích xu hướng
        if len(st.session_state.history) >= 5:
            recent_5 = st.session_state.history[:5]
            recent_wins = sum(1 for h in recent_5 if '🔥' in h['KQ'])
            trend = "📈 TĂNG" if recent_wins >= 3 else ("📉 GIẢM" if recent_wins <= 1 else "➡️ ỔN ĐỊNH")
            st.markdown(f"""
            <div style="text-align:center; margin-top:10px; font-size:12px; color:#888;">
                Xu hướng 5 kỳ gần: {trend} ({recent_wins}/5 thắng)
            </div>
            """, unsafe_allow_html=True)

# === FOOTER ===
st.markdown("""
<div style="text-align:center; color:#444; font-size:10px; margin-top:30px; padding-top:15px; border-top:1px solid #333;">
    🔥 TITAN V52 - PHOENIX RISING | 5-Lớp Ensemble | Smart Bankroll | Stop Loss<br>
    <i>Tool hỗ trợ phân tích - Không đảm bảo 100% - Quản lý vốn thông minh</i>
</div>
""", unsafe_allow_html=True)