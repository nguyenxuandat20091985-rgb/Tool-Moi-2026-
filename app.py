import streamlit as st
import re, pandas as pd, numpy as np, math
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime

# === CẤU HÌNH ===
LUCKY_OX = [0, 2, 5, 6, 7, 8]
st.set_page_config(page_title="TITAN V51 - NEURAL GENESIS", page_icon="🧬", layout="centered")

# === CSS GOD MODE ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&display=swap');
    
    .stApp {
        background: linear-gradient(180deg, #000000 0%, #0a0a1a 50%, #000000 100%);
        color: #FFD700;
        font-family: 'Orbitron', monospace;
    }
    
    .god-header {
        font-size: 36px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #FFD700, #FF00FF, #00FFFF, #FFD700);
        background-size: 300% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 3s ease infinite;
        text-shadow: 0 0 30px rgba(255, 215, 0, 0.5);
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .status-box {
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
        border: 3px solid;
        animation: pulse-glow 2s infinite;
    }
    
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 20px currentColor; }
        50% { box-shadow: 0 0 40px currentColor; }
    }
    
    .status-pay {
        background: linear-gradient(135deg, #1a3a1a, #0a2a0a);
        border-color: #00ff40;
        color: #00ff40;
    }
    
    .status-take {
        background: linear-gradient(135deg, #3a1a1a, #2a0a0a);
        border-color: #ff0040;
        color: #ff0040;
    }
    
    .status-chaos {
        background: linear-gradient(135deg, #3a3a1a, #2a2a0a);
        border-color: #ffff00;
        color: #ffff00;
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
    
    .skip-warning {
        background: linear-gradient(135deg, #3a1a3a, #2a0a2a);
        border: 3px solid #FF00FF;
        color: #FF00FF;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: 900;
        animation: blink-slow 2s infinite;
    }
    
    @keyframes blink-slow {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin: 15px 0;
    }
    
    .metric-cell {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 15px 10px;
        text-align: center;
        border: 1px solid #333;
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
    .tag-purple { background: #9900ff; color: #fff; }
    
    .history-win { color: #00ff40; font-weight: 900; }
    .history-lose { color: #ff0040; font-weight: 900; }
    
    .progress-bar {
        height: 20px;
        background: #1a1a1a;
        border-radius: 10px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff0000, #ffff00, #00ff00);
        transition: width 0.5s;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: bold;
        color: #000;
    }
</style>
""", unsafe_allow_html=True)

# === NEURAL GENESIS ENGINE ===

class NeuralGenesis:
    """Engine học từ sai lầm và phát hiện chu kỳ nhà cái"""
    
    def __init__(self, db, history=None):
        self.db = db
        self.history = history or []
        self.mode = self._detect_house_mode()
        self.learning_rate = self._calculate_learning_rate()
        
    def _detect_house_mode(self):
        """
        Phát hiện chế độ nhà cái:
        - PAY: Đang trả thưởng (win rate cao)
        - TAKE: Đang thu (win rate thấp)
        - CHAOS: Không rõ pattern
        """
        if len(self.history) < 5:
            return "CHAOS"
        
        recent_wins = sum(1 for h in self.history[:10] if '🔥' in h.get('KQ', ''))
        win_rate = recent_wins / min(len(self.history), 10)
        
        if win_rate >= 0.5:
            return "PAY"
        elif win_rate <= 0.3:
            return "TAKE"
        return "CHAOS"
    
    def _calculate_learning_rate(self):
        """Điều chỉnh độ tin cậy dựa trên lịch sử thua"""
        if len(self.history) < 3:
            return 1.0
        
        recent_losses = sum(1 for h in self.history[:5] if '❌' in h.get('KQ', ''))
        
        # Nếu thua nhiều -> giảm confidence threshold
        if recent_losses >= 4:
            return 0.6  # Cần confidence cao hơn mới đánh
        elif recent_losses >= 2:
            return 0.8
        return 1.0
    
    def _calculate_entropy(self, window=10):
        """Độ hỗn loạn - quyết định có nên đánh không"""
        if len(self.db) < window:
            return 3.5
        
        recent = "".join(self.db[-window:])
        counter = Counter(recent)
        total = len(recent)
        
        entropy = 0
        for count in counter.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _detect_cycle(self):
        """Phát hiện chu kỳ lặp"""
        if len(self.db) < 20:
            return None
        
        # Tìm pattern lặp
        for cycle_len in range(5, 15):
            matches = 0
            for i in range(len(self.db) - cycle_len - 1):
                if self.db[i] == self.db[i + cycle_len]:
                    matches += 1
            
            if matches >= 3:
                return cycle_len
        
        return None
    
    def _analyze_position_patterns(self):
        """Phân tích pattern từng vị trí"""
        positions = {i: defaultdict(int) for i in range(5)}
        
        for num in self.db[-30:]:
            for i, d in enumerate(num):
                positions[i][d] += 1
        
        # Tìm số hot từng vị trí
        hot_by_pos = {}
        for pos in positions:
            if positions[pos]:
                hot_by_pos[pos] = max(positions[pos].items(), key=lambda x: x[1])[0]
        
        return hot_by_pos
    
    def _calculate_pair_intelligence(self, pair):
        """
        Tính điểm thông minh cho cặp số
        Học từ lịch sử thua để điều chỉnh
        """
        score = 0
        reasons = []
        
        # 1. Frequency analysis
        freq = sum(1 for n in self.db[-30:] if set(pair).issubset(set(n)))
        if 3 <= freq <= 8:
            score += 30
            reasons.append("FREQ_OK")
        elif freq > 10:
            score -= 40  # Quá nhiều -> bẫy
            reasons.append("FREQ_TRAP")
        
        # 2. Gan analysis (vùng vàng 5-12)
        gan = 0
        for num in reversed(self.db):
            if not set(pair).issubset(set(num)):
                gan += 1
            else:
                break
        
        if 5 <= gan <= 12:
            score += 50
            reasons.append("GAN_VANG")
        elif 2 <= gan <= 4:
            score += 25
        elif gan > 18:
            score -= 30
        
        # 3. Streak analysis (tránh bệt >= 3)
        streak = 0
        for num in reversed(self.db):
            if set(pair).issubset(set(num)):
                streak += 1
            else:
                break
        
        if streak >= 3:
            score -= 60
            reasons.append("BET_TRAP")
        elif streak == 1:
            score += 30
        
        # 4. Learning from history
        if self.history:
            # Check nếu cặp này từng thua nhiều
            pair_losses = sum(1 for h in self.history if h.get('Dự đoán') == pair and '❌' in h.get('KQ', ''))
            if pair_losses >= 2:
                score -= 50
                reasons.append("HISTORY_LOSE")
        
        # 5. Position pattern match
        hot_pos = self._analyze_position_patterns()
        pos_match = 0
        for pos, hot_num in hot_pos.items():
            if hot_num in pair:
                pos_match += 1
        score += pos_match * 15
        
        # 6. House mode adjustment
        if self.mode == "TAKE":
            score *= 0.7  # Giảm score khi nhà cái đang thu
            reasons.append("HOUSE_TAKE")
        elif self.mode == "PAY":
            score *= 1.2
            reasons.append("HOUSE_PAY")
        
        # 7. Lucky ox bonus
        if any(int(d) in LUCKY_OX for d in pair):
            score += 10
        
        # 8. Entropy adjustment
        entropy = self._calculate_entropy()
        if entropy < 2.8:
            score += 20  # Sắp có biến động
            reasons.append("LOW_ENTROPY")
        elif entropy > 3.3:
            score -= 20  # Quá hỗn loạn
            reasons.append("HIGH_ENTROPY")
        
        return score, reasons
    
    def should_skip_bet(self, top_score, confidence):
        """
        QUYẾT ĐỊNH QUAN TRỌNG: Có nên đánh kỳ này không?
        """
        # Skip nếu house đang TAKE mode
        if self.mode == "TAKE" and confidence < 75:
            return True, "NHÀ CÁI ĐANG THU"
        
        # Skip nếu entropy quá cao
        entropy = self._calculate_entropy()
        if entropy > 3.3:
            return True, "QUÁ HỖN LOẠN"
        
        # Skip nếu score quá thấp
        if top_score < 80:
            return True, "ĐỘ TIN CẬY THẤP"
        
        # Skip nếu thua 3 kỳ liên tiếp
        if len(self.history) >= 3:
            recent_losses = sum(1 for h in self.history[:3] if '❌' in h.get('KQ', ''))
            if recent_losses == 3:
                return True, "THUA 3 KỲ LIÊN TIẾP - NGHỈ"
        
        return False, "OK"
    
    def predict(self):
        """Dự đoán thông minh"""
        if len(self.db) < 15:
            return None
        
        # Generate tất cả cặp
        all_pairs = []
        for p in combinations("0123456789", 2):
            pair = "".join(p)
            score, reasons = self._calculate_pair_intelligence(pair)
            all_pairs.append({
                'pair': pair,
                'score': score,
                'reasons': reasons
            })
        
        all_pairs.sort(key=lambda x: x['score'], reverse=True)
        top_pairs = all_pairs[:5]
        
        # Tính confidence
        if top_pairs:
            base_confidence = 40 + (top_pairs[0]['score'] / 5)
            base_confidence = min(95, max(30, base_confidence))
            
            # Adjust theo learning rate
            confidence = base_confidence * self.learning_rate
        else:
            confidence = 50
        
        # Quyết định có đánh không
        skip, skip_reason = self.should_skip_bet(
            top_pairs[0]['score'] if top_pairs else 0,
            confidence
        )
        
        # Top 3 triples
        triples = []
        for t in combinations("0123456789", 3):
            score = sum(
                next((p['score'] for p in top_pairs if p['pair'] == ''.join(c)), 0)
                for c in combinations(t, 2)
            )
            triples.append((''.join(t), score))
        triples.sort(key=lambda x: x[1], reverse=True)
        
        # Top 8
        single_pool = Counter("".join(self.db[-50:]))
        top8 = "".join([d for d, _ in single_pool.most_common(8)])
        
        return {
            'pairs': top_pairs,
            'triples': triples[:3],
            'top8': top8,
            'confidence': confidence,
            'skip': skip,
            'skip_reason': skip_reason,
            'house_mode': self.mode,
            'entropy': self._calculate_entropy(),
            'cycle': self._detect_cycle(),
            'learning_rate': self.learning_rate
        }

# === XỬ LÝ DỮ LIỆU ===

def get_nums(text):
    clean_text = re.sub(r'\s+', '', text)
    nums = [n for n in re.findall(r"\d{5}", clean_text) if n]
    seen = set()
    return [n for n in nums if not (n in seen or seen.add(n))]

# === GIAO DIỆN ===

st.markdown('<h1 class="god-header">🧬 TITAN V51 - NEURAL GENESIS</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#888; font-size:11px;">Học Từ Sai Lầm | Phát Hiện Chu Kỳ | Biết Khi Nào Nên Nghỉ</p>', unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_area("📥 Kết quả (kỳ mới nhất ở dưới):", height=100, 
                          placeholder="07988\n35782\n01053")

col1, col2 = st.columns(2)
with col1:
    if st.button("🧬 KÍCH HOẠT", type="primary", use_container_width=True):
        nums = get_nums(user_input)
        if len(nums) >= 15:
            # Check kết quả kỳ trước
            if "last_pred" in st.session_state and nums:
                lp = st.session_state.last_pred
                last = nums[-1]
                if lp and lp['pairs'] and not lp.get('skip', False):
                    best = lp['pairs'][0]['pair']
                    win = all(d in last for d in best)
                    st.session_state.history.insert(0, {
                        'Kỳ': last,
                        'Dự đoán': best,
                        'KQ': '🔥' if win else '❌'
                    })
            
            engine = NeuralGenesis(nums, st.session_state.history)
            st.session_state.last_pred = engine.predict()
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
    
    # === HOUSE MODE STATUS ===
    mode_class = "status-pay" if res['house_mode'] == "PAY" else ("status-take" if res['house_mode'] == "TAKE" else "status-chaos")
    mode_icon = "💰" if res['house_mode'] == "PAY" else ("🦈" if res['house_mode'] == "TAKE" else "🌪️")
    mode_text = "TRẢ THƯỞNG" if res['house_mode'] == "PAY" else ("ĐANG THU" if res['house_mode'] == "TAKE" else "HỖN LOẠN")
    
    st.markdown(f"""
    <div class="status-box {mode_class}">
        <div style="font-size:14px;">CHẾ ĐỘ NHÀ CÁI</div>
        <div style="font-size:28px; font-weight:900;">{mode_icon} {mode_text}</div>
        <div style="font-size:11px; margin-top:5px;">Entropy: {res['entropy']:.2f} | Learning Rate: {res['learning_rate']:.1f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # === SKIP WARNING ===
    if res['skip']:
        st.markdown(f"""
        <div class="skip-warning">
            ⚠️ KHÔNG NÊN ĐÁNH KỲ NÀY<br>
            <span style="font-size:16px; font-weight:700;">{res['skip_reason']}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        # === METRICS GRID ===
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-cell">
                <div style="font-size:10px; color:#888;">TIN CẬY</div>
                <div style="font-size:24px; font-weight:900; color:#00ffff;">{res['confidence']:.0f}%</div>
            </div>
            <div class="metric-cell">
                <div style="font-size:10px; color:#888;">TOP PAIR</div>
                <div style="font-size:24px; font-weight:900; color:#FFD700;">{res['pairs'][0]['pair']}</div>
            </div>
            <div class="metric-cell">
                <div style="font-size:10px; color:#888;">SCORE</div>
                <div style="font-size:24px; font-weight:900; color:#00ff40;">{res['pairs'][0]['score']:.0f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # === TOP PAIR ===
        st.markdown("""<div style="text-align:center; margin:15px 0; font-size:14px;">🎯 CẶP VIP</div>""", unsafe_allow_html=True)
        
        reasons_tags = "".join([f'<span class="tag tag-purple">{r}</span>' for r in res['pairs'][0]['reasons'][:3]])
        
        st.markdown(f"""
        <div class="prediction-card">
            <div class="big-number">{res['pairs'][0]['pair'][0]} - {res['pairs'][0]['pair'][1]}</div>
            <div style="margin-top:10px;">{reasons_tags}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # === TOP 5 PAIRS ===
        st.markdown("""<div style="text-align:center; margin:15px 0; font-size:14px;">🎯 TOP 5</div>""", unsafe_allow_html=True)
        
        for i, p in enumerate(res['pairs'][:5]):
            tags = ""
            if "GAN_VANG" in p['reasons']:
                tags += '<span class="tag tag-green">GAN VÀNG</span>'
            if "BET_TRAP" in p['reasons']:
                tags += '<span class="tag tag-red">BẪY</span>'
            if "HOUSE_PAY" in p['reasons']:
                tags += '<span class="tag tag-yellow">NHÀ CÁI TRẢ</span>'
            
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
        
        # === TOP 8 ===
        st.markdown(f"""
        <div class="prediction-card" style="margin-top:15px;">
            <div style="font-size:12px; color:#888;">ĐỘ PHỦ SẢNH</div>
            <div style="font-size:32px; font-weight:900; letter-spacing:8px; color:#00ffff;">{res['top8']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # === HISTORY ===
    if st.session_state.history:
        st.markdown("""<div style="text-align:center; margin:20px 0; font-size:16px;">📋 LỊCH SỬ ĐỐI SOÁT</div>""", unsafe_allow_html=True)
        
        df = pd.DataFrame(st.session_state.history[:15])
        
        def color_kq(val):
            return 'color: #00ff40; font-weight: 900' if '🔥' in val else 'color: #ff0040; font-weight: 900'
        
        st.dataframe(df.style.applymap(color_kq, subset=['KQ']), 
                     use_container_width=True, hide_index=True)
        
        # Win rate với progress bar
        wins = sum(1 for h in st.session_state.history if '🔥' in h['KQ'])
        total = len(st.session_state.history)
        rate = (wins/total*100) if total > 0 else 0
        
        color_rate = "#00ff40" if rate >= 40 else ("#ffff00" if rate >= 30 else "#ff0040")
        
        st.markdown(f"""
        <div class="status-box" style="border-color:{color_rate}; color:{color_rate}; margin-top:15px;">
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
    🧬 TITAN V51 - NEURAL GENESIS | Học Từ Sai Lầm | Phát Hiện Chu Kỳ | Biết Khi Nên Nghỉ<br>
    <i>Tool hỗ trợ phân tích - Không đảm bảo 100% - Quản lý vốn thông minh</i>
</div>
""", unsafe_allow_html=True)