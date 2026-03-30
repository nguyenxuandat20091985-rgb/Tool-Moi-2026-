import streamlit as st
import re, pandas as pd, numpy as np, math, json, os
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime

# === CẤU HÌNH ===
DB_FILE = "titan_v60_data.json"
LUCKY_OX = [0, 2, 5, 6, 7, 8]

st.set_page_config(page_title="TITAN V60", page_icon="🧠", layout="centered")

# === CSS MOBILE-FIRST ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    .stApp {
        background: linear-gradient(180deg, #0a0a0a 0%, #1a1a2e 100%);
        color: #FFD700;
        font-family: 'Orbitron', sans-serif;
    }
    .main-header {
        font-size: 28px;
        font-weight: 900;
        text-align: center;
        color: #FFD700;
        text-shadow: 0 0 20px rgba(255, 215, 0, 0.5);
        margin-bottom: 15px;
    }
    .tab-btn {
        width: 100%;
        padding: 12px;
        margin: 5px 0;
        background: linear-gradient(135deg, #1a1a2e, #0f3460);
        border: 2px solid #FFD700;
        border-radius: 10px;
        color: #FFD700;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s;
    }
    .tab-btn:hover, .tab-btn.active {
        background: #FFD700;
        color: #000;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.5);
    }
    .card {
        background: linear-gradient(135deg, #1a1a2e, #0f3460);
        border: 2px solid #FFD700;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
    }
    .big-num {
        font-size: 42px;
        font-weight: 900;
        color: #00ffff;
        letter-spacing: 8px;
        text-shadow: 0 0 15px #00ffff;
    }
    .score {
        font-size: 20px;
        font-weight: bold;
        color: #00ff40;
    }
    .tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 10px;
        font-weight: bold;
        margin: 3px;
    }
    .tag-hot { background: #ff0040; color: white; }
    .tag-cold { background: #0066ff; color: white; }
    .tag-trap { background: #ff0000; color: white; animation: blink 0.5s infinite; }
    .tag-quantum { background: #9900ff; color: white; }
    @keyframes blink { 50% { opacity: 0.5; } }
    .heatmap-cell {
        display: inline-block;
        width: 30px;
        height: 30px;
        margin: 2px;
        border-radius: 5px;
        text-align: center;
        line-height: 30px;
        font-weight: bold;
        font-size: 12px;
    }
    .win { color: #00ff40; font-weight: 900; }
    .lose { color: #ff0040; font-weight: 900; }
    .confidence-bar {
        height: 20px;
        background: #1a1a1a;
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
        font-size: 12px;
        font-weight: bold;
        color: #000;
    }
    textarea {
        background: #0a0a0a !important;
        border: 1px solid #FFD700 !important;
        color: #FFD700 !important;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# === DATA MANAGEMENT ===
def load_data():
    default = {"results": [], "predictions": [], "weights": {"quantum": 1.0, "behavior": 1.0, "frequency": 1.0, "neural": 1.0}, "stats": {"total": 0, "wins": 0}}
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for k in default:
                    if k not in data: data[k] = default[k]
                return data
        except: pass
    return default

def save_data(data):
    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def parse_nums(text):
    clean = re.sub(r'\s+', '', text)
    return [n for n in re.findall(r'\d{5}', clean) if n]

# === ALGORITHM CLASSES (GIỮ NGUYÊN) ===
class QuantumAnalyzer:
    @staticmethod
    def digital_root_sequence(nums):
        roots = [sum(int(d) for d in n) % 9 for n in nums]
        return Counter(roots).most_common(3)
    
    @staticmethod
    def fibonacci_weight(nums):
        fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
        weights = {}
        for idx, num in enumerate(reversed(nums[-10:])):
            for d in num:
                if d not in weights: weights[d] = 0
                weights[d] += fib[idx] if idx < len(fib) else fib[-1]
        return weights
    
    @staticmethod
    def prime_factor_analysis(nums):
        primes = [2, 3, 5, 7]
        prime_counts = {p: 0 for p in primes}
        for num in nums:
            for d in num:
                digit = int(d)
                if digit in primes: prime_counts[digit] += 1
        return prime_counts

class AntiPatternDetector:
    def __init__(self, db):
        self.db = db
        self.trap_pairs = set()
        self.detect_traps()
    
    def detect_traps(self):
        if len(self.db) < 20: return
        pair_counts = Counter()
        for num in self.db[-30:]:
            for p in combinations(sorted(set(num)), 2):
                pair_counts[p] += 1
        for pair, count in pair_counts.items():
            if count >= 8: self.trap_pairs.add(pair)
        for i in range(len(self.db) - 2):
            curr_set = set(self.db[i])
            next_set = set(self.db[i+1])
            if len(curr_set & next_set) >= 3:
                for p in combinations(sorted(curr_set & next_set), 2):
                    self.trap_pairs.add(p)
    
    def is_trap(self, pair):
        return tuple(sorted(pair)) in self.trap_pairs

class BehavioralAnalyzer:
    @staticmethod
    def reverse_psychology_score(db, pair):
        if len(db) < 15: return 0
        recent_count = sum(1 for n in db[-15:] if set(pair).issubset(set(n)))
        if recent_count >= 4: return -50
        elif recent_count >= 2: return -20
        return 10
    
    @staticmethod
    def crowd_behavior(db):
        recent_str = "".join(db[-10:])
        hot_numbers = Counter(recent_str).most_common(3)
        return [n[0] for n in hot_numbers]

class RiskManager:
    @staticmethod
    def calculate_kelly_fraction(win_rate, odds=1.85):
        b = odds - 1
        p = win_rate / 100
        q = 1 - p
        kelly = (b * p - q) / b
        return max(0, min(kelly, 0.25))
    
    @staticmethod
    def diversification_score(predictions):
        if len(predictions) < 3: return 0
        all_numbers = set()
        for pred in predictions: all_numbers.update(pred)
        return len(all_numbers) / 10

class NeuralPatternMatcher:
    def __init__(self, db):
        self.db = db
        self.pattern_memory = defaultdict(list)
        self.build_pattern_memory()
    
    def build_pattern_memory(self):
        for idx in range(len(self.db) - 1):
            current = self.db[idx]
            next_num = self.db[idx + 1]
            for d in current:
                for nd in next_num:
                    self.pattern_memory[d].append(nd)
    
    def get_next_probability(self, last_num):
        if not last_num: return {}
        prob = Counter()
        for d in last_num:
            if d in self.pattern_memory:
                for next_d in self.pattern_memory[d]:
                    prob[next_d] += 1
        total = sum(prob.values())
        if total == 0: return {}
        return {k: v/total * 100 for k, v in prob.items()}

class TITANV60Engine:
    def __init__(self, db, weights=None):
        self.db = db
        self.weights = weights or {"quantum": 1.0, "behavior": 1.0, "frequency": 1.0, "neural": 1.0}
        self.quantum = QuantumAnalyzer()
        self.anti_pattern = AntiPatternDetector(db)
        self.behavioral = BehavioralAnalyzer()
        self.risk_manager = RiskManager()
        self.neural = NeuralPatternMatcher(db)
    
    def calculate_composite_score(self, pair):
        score = 0
        details = {}
        
        dr_score = self._quantum_score(pair) * self.weights["quantum"]
        score += dr_score
        details['quantum'] = dr_score
        
        if self.anti_pattern.is_trap(pair):
            score -= 100
            details['trap'] = -100
        else:
            score += 20
            details['trap'] = 20
        
        behavior_score = self.behavioral.reverse_psychology_score(self.db, pair) * self.weights["behavior"]
        score += behavior_score
        details['behavior'] = behavior_score
        
        freq_score = self._frequency_score(pair) * self.weights["frequency"]
        score += freq_score
        details['frequency'] = freq_score
        
        neural_score = self._neural_score(pair) * self.weights["neural"]
        score += neural_score
        details['neural'] = neural_score
        
        if any(int(d) in LUCKY_OX for d in pair):
            score += 15
            details['lucky'] = 15
        
        return score, details
    
    def _quantum_score(self, pair):
        score = 0
        dr_mode = self.quantum.digital_root_sequence(self.db)
        if dr_mode:
            pair_dr = (int(pair[0]) + int(pair[1])) % 9
            if pair_dr == dr_mode[0][0]: score += 30
        fib_weights = self.quantum.fibonacci_weight(self.db)
        for d in pair:
            if d in fib_weights:
                score += min(fib_weights[d] / 10, 25)
        return score
    
    def _frequency_score(self, pair):
        score = 0
        gan = 0
        for num in reversed(self.db):
            if not set(pair).issubset(set(num)): gan += 1
            else: break
        streak = 0
        for num in reversed(self.db):
            if set(pair).issubset(set(num)): streak += 1
            else: break
        if 5 <= gan <= 12: score += 60
        elif 2 <= gan <= 4: score += 30
        elif gan > 18: score -= 40
        if streak >= 3: score -= 80
        elif streak == 1: score += 40
        return score
    
    def _neural_score(self, pair):
        if not self.db: return 0
        last_num = self.db[-1]
        probs = self.neural.get_next_probability(last_num)
        score = 0
        for d in pair:
            if d in probs: score += probs[d] * 0.5
        return score
    
    def predict(self):
        if len(self.db) < 15: return None
        all_pairs = []
        recent_str = "".join(self.db[-50:])
        single_pool = Counter(recent_str)
        
        for p in combinations("0123456789", 2):
            pair = "".join(p)
            score, details = self.calculate_composite_score(p)
            freq = sum(1 for n in self.db[-30:] if set(p).issubset(set(n)))
            all_pairs.append({'pair': pair, 'score': score, 'details': details, 'frequency': freq})
        
        all_pairs.sort(key=lambda x: x['score'], reverse=True)
        top_pairs = all_pairs[:5]
        
        if top_pairs:
            top_score = top_pairs[0]['score']
            confidence = min(95, max(30, (top_score + 100) / 3))
        else:
            confidence = 50
        
        all_triples = []
        for t in combinations("0123456789", 3):
            score, _ = self.calculate_composite_score(t)
            all_triples.append((''.join(t), score))
        all_triples.sort(key=lambda x: x[1], reverse=True)
        top_triples = all_triples[:3]
        
        top_8 = "".join([d for d, _ in single_pool.most_common(8)])
        crowd = self.behavioral.crowd_behavior(self.db)
        
        return {'pairs': top_pairs, 'triples': top_triples, 'top8': top_8, 'confidence': confidence, 'crowd_numbers': crowd, 'single_pool': single_pool}
    
    def update_weights(self, pair_won, details):
        """Tự học: WIN tăng weight, LOSE giảm weight"""
        for algo in ['quantum', 'behavior', 'frequency', 'neural']:
            val = details.get(algo, 0)
            if pair_won and val > 20:
                self.weights[algo] = min(2.0, self.weights[algo] * 1.1)
            elif not pair_won and val > 20:
                self.weights[algo] = max(0.5, self.weights[algo] * 0.9)

# === UI FUNCTIONS ===
def render_input_tab(data):
    st.markdown("### 📥 NHẬP KẾT QUẢ")
    new_input = st.text_area("Dán kết quả (mỗi kỳ 1 dòng, kỳ mới nhất ở dưới):", height=120, placeholder="12345\n67890\n...")
    
    if st.button("💾 LƯU & PHÂN TÍCH", type="primary", use_container_width=True):
        nums = parse_nums(new_input)
        if nums:
            # Auto check WIN với dự đoán trước
            if "last_pred" in st.session_state and data.get("predictions"):
                lp = st.session_state.last_pred
                last_actual = nums[-1]
                if lp and lp.get('pairs'):
                    best_pair = lp['pairs'][0]['pair']
                    win = all(d in last_actual for d in best_pair)
                    # Update weights
                    engine = TITANV60Engine(data["results"], data.get("weights"))
                    if lp['pairs'][0].get('details'):
                        engine.update_weights(win, lp['pairs'][0]['details'])
                        data["weights"] = engine.weights
                    # Save history
                    data["predictions"].insert(0, {"date": datetime.now().isoformat(), "pair": best_pair, "result": last_actual, "win": win})
                    data["stats"]["total"] += 1
                    if win: data["stats"]["wins"] += 1
            
            data["results"].extend(nums)
            data["results"] = data["results"][-200:]
            save_data(data)
            st.success(f"✅ Đã lưu {len(nums)} kỳ")
            st.rerun()
    
    if data["results"]:
        st.markdown(f"**Tổng kỳ:** {len(data['results'])} | **10 kỳ gần:** {', '.join(data['results'][-10:])}")

def render_bachthu_tab(data):
    st.markdown("### 🎯 BẠCH THỦ 2 SỐ")
    if len(data["results"]) < 15:
        st.warning("⚠️ Cần ít nhất 15 kỳ dữ liệu")
        return
    
    if st.button("🔮 PHÂN TÍCH NGAY", type="primary", use_container_width=True):
        engine = TITANV60Engine(data["results"], data.get("weights"))
        res = engine.predict()
        if res:
            st.session_state.last_pred = res
            st.rerun()
    
    if "last_pred" in st.session_state:
        res = st.session_state.last_pred
        
        # Confidence
        st.markdown(f"""
        <div class="card">
            <div style="font-size:12px; color:#888;">ĐỘ TIN CẬY</div>
            <div class="confidence-bar"><div class="confidence-fill" style="width:{res['confidence']}%;">{res['confidence']:.0f}%</div></div>
        </div>
        """, unsafe_allow_html=True)
        
        # TOP 1
        top1 = res['pairs'][0]
        tags = ""
        if top1['details'].get('trap', 0) < 0: tags += '<span class="tag tag-trap">⚠️ BẪY</span>'
        if any(d in res['crowd_numbers'] for d in top1['pair']): tags += '<span class="tag tag-hot">HOT</span>'
        if top1['details'].get('quantum', 0) > 30: tags += '<span class="tag tag-quantum">QUANTUM</span>'
        
        st.markdown(f"""
        <div class="card" style="border-color:#00ff40;">
            <div style="font-size:14px; color:#888;">🥇 TOP 1</div>
            <div class="big-num">{top1['pair'][0]} - {top1['pair'][1]}</div>
            <div class="score">Score: {top1['score']:.0f}</div>
            <div>{tags}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # TOP 2-5
        st.markdown("**📋 TOP 2-5:**")
        for p in res['pairs'][1:]:
            t = ""
            if p['details'].get('trap', 0) < 0: t += '<span class="tag tag-trap">BẪY</span>'
            if any(d in res['crowd_numbers'] for d in p['pair']): t += '<span class="tag tag-hot">HOT</span>'
            st.markdown(f"""
            <div class="card" style="padding:10px; margin:5px 0;">
                <span style="font-size:24px; font-weight:900;">{p['pair']}</span>
                <span style="margin-left:10px; color:#00ff40;">{p['score']:.0f}</span>
                <div style="margin-top:5px;">{t}</div>
            </div>
            """, unsafe_allow_html=True)

def render_bet_tab(data):
    st.markdown("### 🔁 SỐ BỆT")
    if len(data["results"]) < 5:
        st.warning("⚠️ Cần ít nhất 5 kỳ")
        return
    
    streaks = {}
    for d in "0123456789":
        count = 0
        for r in reversed(data["results"]):
            if d in r: count += 1
            else: break
        if count > 0: streaks[d] = count
    
    if streaks:
        for d, s in sorted(streaks.items(), key=lambda x: x[1], reverse=True):
            st.markdown(f"""
            <div class="card">
                <span style="font-size:32px; font-weight:900; color:#ff0040;">{d}</span>
                <div style="margin-top:5px;">Bệt {s} kỳ</div>
                <div style="font-size:12px; color:#888;">{'✅ Theo' if s <= 3 else '⚠️ Cẩn thận'}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("ℹ️ Không có số bệt")

def render_gan_tab(data):
    st.markdown("### ⏳ SỐ GAN")
    if len(data["results"]) < 10:
        st.warning("⚠️ Cần ít nhất 10 kỳ")
        return
    
    gaps = {}
    for d in "0123456789":
        gap = 0
        for r in reversed(data["results"]):
            if d in r: break
            gap += 1
        gaps[d] = gap
    
    for d, g in sorted(gaps.items(), key=lambda x: x[1], reverse=True):
        status = "🔥 Sắp về" if g >= 8 else "⏳ Theo dõi"
        st.markdown(f"""
        <div class="card">
            <span style="font-size:32px; font-weight:900; color:#0066ff;">{d}</span>
            <div style="margin-top:5px;">Gan {g} kỳ</div>
            <div style="font-size:12px; color:#888;">{status}</div>
        </div>
        """, unsafe_allow_html=True)

def render_history_tab(data):
    st.markdown("### 📊 LỊCH SỬ")
    
    # Heatmap
    st.markdown("**🔥 HEATMAP (30 kỳ gần)**")
    if len(data["results"]) >= 30:
        recent = "".join(data["results"][-30:])
        counter = Counter(recent)
        max_c = max(counter.values()) if counter else 1
        cells = ""
        for d in "0123456789":
            c = counter.get(d, 0)
            intensity = int(c / max_c * 100)
            bg = f"linear-gradient(135deg, #ff0040, #ff0040{int(intensity/2.55):02x})" if c > 0 else "#333"
            cells += f'<div class="heatmap-cell" style="background:{bg};">{d}<br>{c}</div>'
        st.markdown(f'<div style="text-align:center;">{cells}</div>', unsafe_allow_html=True)
    
    # History table
    if data.get("predictions"):
        st.markdown("**📋 Kết quả dự đoán:**")
        df_data = []
        for p in data["predictions"][:15]:
            df_data.append({"Date": p.get("date", "")[:16].replace("T"," "), "Pair": p.get("pair",""), "Result": p.get("result",""), "Win": "🔥" if p.get("win") else "❌"})
        df = pd.DataFrame(df_data)
        def color_win(val): return "color:#00ff40;font-weight:900" if val=="🔥" else "color:#ff0040;font-weight:900"
        st.dataframe(df.style.applymap(color_win, subset=['Win']), use_container_width=True, hide_index=True)
        
        # Win rate
        total = data["stats"].get("total", 0)
        wins = data["stats"].get("wins", 0)
        rate = (wins/total*100) if total > 0 else 0
        st.markdown(f"""
        <div class="card">
            <div style="font-size:14px;">Tỷ lệ thắng</div>
            <div style="font-size:32px; font-weight:900; color:{'#00ff40' if rate>=40 else '#ff0040'};">{rate:.1f}%</div>
            <div style="font-size:12px; color:#888;">({wins}/{total} kỳ)</div>
        </div>
        """, unsafe_allow_html=True)

# === MAIN APP ===
def main():
    data = load_data()
    
    st.markdown('<h1 class="main-header">🧠 TITAN V60</h1>', unsafe_allow_html=True)
    
    # Vertical tabs
    tabs = [
        ("📥 Nhập dữ liệu", "input"),
        ("🎯 Bạch thủ 2 số", "bachthu"),
        ("🔁 Số bệt", "bet"),
        ("⏳ Số gan", "gan"),
        ("📊 Lịch sử", "history")
    ]
    
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "input"
    
    for label, key in tabs:
        btn_class = "tab-btn active" if st.session_state.active_tab == key else "tab-btn"
        if st.button(label, key=f"btn_{key}", use_container_width=True):
            st.session_state.active_tab = key
    
    st.markdown("---")
    
    if st.session_state.active_tab == "input":
        render_input_tab(data)
    elif st.session_state.active_tab == "bachthu":
        render_bachthu_tab(data)
    elif st.session_state.active_tab == "bet":
        render_bet_tab(data)
    elif st.session_state.active_tab == "gan":
        render_gan_tab(data)
    elif st.session_state.active_tab == "history":
        render_history_tab(data)
    
    # Footer
    st.markdown("---")
    st.markdown('<div style="text-align:center; color:#666; font-size:10px;">TITAN V60 | Mobile-First | Self-Learning</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()