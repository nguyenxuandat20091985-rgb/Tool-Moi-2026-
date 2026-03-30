import streamlit as st
import re, pandas as pd, numpy as np, math, json, os
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime

# === CẤU HÌNH ===
DB_FILE = "titan_v60_data.json"
LUCKY_OX = [0, 2, 5, 6, 7, 8]

st.set_page_config(page_title="TITAN V60", page_icon="🧠", layout="centered")

# === CSS MOBILE SCROLL ===
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
        position: sticky;
        top: 0;
        background: #0a0a0a;
        z-index: 100;
        padding: 10px 0;
    }
    .section {
        margin: 20px 0;
        padding: 15px;
        background: linear-gradient(135deg, #1a1a2e, #0f3460);
        border-radius: 15px;
        border: 2px solid #FFD700;
    }
    .section-title {
        font-size: 20px;
        font-weight: 900;
        color: #FFD700;
        margin-bottom: 15px;
        border-bottom: 2px solid #FFD700;
        padding-bottom: 10px;
    }
    .card {
        background: linear-gradient(135deg, #0f3460, #16213e);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
        border: 1px solid #00ffff;
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
    .tag-safe { background: #00ff40; color: black; }
    @keyframes blink { 50% { opacity: 0.5; } }
    .heatmap-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 5px;
        margin: 10px 0;
    }
    .heatmap-cell {
        aspect-ratio: 1;
        border-radius: 8px;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        font-weight: bold;
        font-size: 14px;
        border: 2px solid rgba(255,255,255,0.2);
    }
    .win { color: #00ff40; font-weight: 900; }
    .lose { color: #ff0040; font-weight: 900; }
    .confidence-bar {
        height: 25px;
        background: #1a1a1a;
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
        font-size: 14px;
        font-weight: bold;
        color: #000;
    }
    .prediction-box {
        background: linear-gradient(135deg, #1a1a2e, #0f3460);
        border: 3px solid #FFD700;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        text-align: center;
    }
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin: 10px 0;
    }
    .stat-box {
        background: #0f3460;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
    textarea {
        background: #0a0a0a !important;
        border: 2px solid #FFD700 !important;
        color: #FFD700 !important;
        border-radius: 10px;
        font-family: 'Orbitron', monospace;
    }
    .stButton>button {
        background: linear-gradient(135deg, #FFD700, #FFA500);
        color: #000;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: bold;
        font-family: 'Orbitron', sans-serif;
        width: 100%;
    }
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #FFD700, transparent);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# === DATA MANAGEMENT ===
def load_data():
    default = {
        "results": [], 
        "predictions": [], 
        "weights": {"quantum": 1.0, "behavior": 1.0, "frequency": 1.0, "neural": 1.0}, 
        "stats": {"total": 0, "wins": 0}
    }
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for k in default:
                    if k not in data:
                        data[k] = default[k]
                return data
        except: 
            pass
    return default

def save_data(data):
    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def parse_nums(text):
    clean = re.sub(r'\s+', '', text)
    return [n for n in re.findall(r'\d{5}', clean) if n]

# === ALGORITHM CLASSES ===
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
        return [n[0] for n in Counter(recent_str).most_common(3)]

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
        
        return {
            'pairs': top_pairs, 
            'triples': top_triples, 
            'top8': top_8, 
            'confidence': confidence, 
            'crowd_numbers': crowd, 
            'single_pool': single_pool
        }

# === MAIN APP SCROLL ===
def main():
    data = load_data()
    
    st.markdown('<h1 class="main-header">🧠 TITAN V60</h1>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center; color:#888; margin-bottom:20px; font-size:12px;">Mobile Scroll | AI Prediction</div>', unsafe_allow_html=True)
    
    # === SECTION 1: INPUT ===
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📥 NHẬP KẾT QUẢ</div>', unsafe_allow_html=True)
    
    new_input = st.text_area(
        "Dán kết quả (mỗi kỳ 1 dòng, kỳ mới nhất ở dưới):", 
        height=100, 
        placeholder="12345\n67890\n...",
        key="input_text"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 LƯU & PHÂN TÍCH", type="primary"):
            nums = parse_nums(new_input)
            if nums:
                # Auto check WIN
                if "last_pred" in st.session_state and data.get("predictions"):
                    lp = st.session_state.last_pred
                    last_actual = nums[-1]
                    if lp and lp.get('pairs'):
                        best_pair = lp['pairs'][0]['pair']
                        win = all(d in last_actual for d in best_pair)
                        data["predictions"].insert(0, {
                            "date": datetime.now().isoformat(), 
                            "pair": best_pair, 
                            "result": last_actual, 
                            "win": win
                        })
                        data["stats"]["total"] += 1
                        if win: 
                            data["stats"]["wins"] += 1
                
                data["results"].extend(nums)
                data["results"] = data["results"][-200:]
                save_data(data)
                st.success(f"✅ Đã lưu {len(nums)} kỳ")
                st.rerun()
    
    with col2:
        if st.button("🗑️ XÓA DATA"):
            data["results"] = []
            data["predictions"] = []
            data["stats"] = {"total": 0, "wins": 0}
            save_data(data)
            st.warning("🗑️ Đã xóa")
            st.rerun()
    
    if data["results"]:
        st.markdown(f"""
        <div style="margin-top:10px; padding:10px; background:#0f3460; border-radius:8px;">
            <b>Tổng kỳ:</b> {len(data['results'])}<br>
            <b>10 kỳ gần:</b> {', '.join(data['results'][-10:])}
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # === SECTION 2: STATS & HEATMAP ===
    if len(data["results"]) >= 15:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 THỐNG KÊ & HEATMAP</div>', unsafe_allow_html=True)
        
        # Stats
        total = data["stats"].get("total", 0)
        wins = data["stats"].get("wins", 0)
        rate = (wins/total*100) if total > 0 else 0
        
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-box">
                <div style="font-size:10px; color:#888;">Tổng cược</div>
                <div style="font-size:24px; font-weight:900; color:#00ffff;">{total}</div>
            </div>
            <div class="stat-box">
                <div style="font-size:10px; color:#888;">Thắng</div>
                <div style="font-size:24px; font-weight:900; color:#00ff40;">{wins}</div>
            </div>
            <div class="stat-box">
                <div style="font-size:10px; color:#888;">Win Rate</div>
                <div style="font-size:24px; font-weight:900; color:{'#00ff40' if rate>=40 else '#ff0040'};">{rate:.0f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Heatmap
        if len(data["results"]) >= 30:
            recent = "".join(data["results"][-30:])
            counter = Counter(recent)
            max_c = max(counter.values()) if counter else 1
            
            st.markdown("**🔥 Heatmap 30 kỳ gần:**")
            cells_html = ""
            for d in "0123456789":
                c = counter.get(d, 0)
                intensity = c / max_c
                if intensity > 0.7:
                    bg = "linear-gradient(135deg, #ff0040, #ff4040)"
                elif intensity > 0.4:
                    bg = "linear-gradient(135deg, #ffa500, #ffcc00)"
                elif intensity > 0.2:
                    bg = "linear-gradient(135deg, #0066ff, #00ccff)"
                else:
                    bg = "#333"
                cells_html += f'<div class="heatmap-cell" style="background:{bg}; color:white;">{d}<br><small>{c}</small></div>'
            
            st.markdown(f'<div class="heatmap-grid">{cells_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # === SECTION 3: PREDICTIONS ===
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🎯 DỰ ĐOÁN 2 SỐ</div>', unsafe_allow_html=True)
        
        if st.button("🔮 PHÂN TÍCH NGAY", type="primary"):
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
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width:{res['confidence']}%;">{res['confidence']:.0f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # TOP 1
            top1 = res['pairs'][0]
            tags = ""
            if top1['details'].get('trap', 0) < 0: 
                tags += '<span class="tag tag-trap">⚠️ BẪY</span>'
            if any(d in res['crowd_numbers'] for d in top1['pair']): 
                tags += '<span class="tag tag-hot">HOT</span>'
            if top1['details'].get('quantum', 0) > 30: 
                tags += '<span class="tag tag-quantum">QUANTUM</span>'
            if 5 <= top1['details'].get('frequency', 0) <= 12:
                tags += '<span class="tag tag-safe">GAN VÀNG</span>'
            
            st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size:16px; color:#FFD700; margin-bottom:10px;">🥇 TOP 1 VIP</div>
                <div class="big-num">{top1['pair'][0]} - {top1['pair'][1]}</div>
                <div class="score">Score: {top1['score']:.0f}</div>
                <div style="margin-top:10px;">{tags}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # TOP 2-5
            st.markdown("**📋 BACKUP:**")
            for i, p in enumerate(res['pairs'][1:], 2):
                t = ""
                if p['details'].get('trap', 0) < 0: t += '<span class="tag tag-trap">BẪY</span>'
                if any(d in res['crowd_numbers'] for d in p['pair']): t += '<span class="tag tag-hot">HOT</span>'
                st.markdown(f"""
                <div class="card" style="padding:12px; margin:8px 0;">
                    <span style="font-size:28px; font-weight:900; color:#FFD700;">#{i} {p['pair']}</span>
                    <span style="margin-left:15px; color:#00ff40; font-size:20px;">{p['score']:.0f}</span>
                    <div style="margin-top:5px;">{t}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Top 8
            st.markdown(f"""
            <div class="card" style="margin-top:15px;">
                <div style="font-size:12px; color:#888;">ĐỘ PHỦ SẢNH (8 SỐ)</div>
                <div style="font-size:32px; font-weight:900; color:#00ffff; letter-spacing:6px;">{res['top8']}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # === SECTION 4: STREAKS & GAPS ===
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔥 SỐ BỆT & ❄️ SỐ GAN</div>', unsafe_allow_html=True)
        
        # Streaks
        streaks = {}
        for d in "0123456789":
            count = 0
            for r in reversed(data["results"]):
                if d in r: count += 1
                else: break
            if count > 0: streaks[d] = count
        
        if streaks:
            st.markdown("**🔥 Đang bệt:**")
            for d, s in sorted(streaks.items(), key=lambda x: x[1], reverse=True)[:3]:
                st.markdown(f"""
                <div class="card" style="padding:10px; margin:5px 0;">
                    <span style="font-size:28px; font-weight:900; color:#ff0040;">{d}</span>
                    <span style="margin-left:15px;">Bệt {s} kỳ</span>
                    <span style="float:right;">{'✅' if s <= 3 else '⚠️'}</span>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Gaps
        gaps = {}
        for d in "0123456789":
            gap = 0
            for r in reversed(data["results"]):
                if d in r: break
                gap += 1
            if gap > 0: gaps[d] = gap
        
        if gaps:
            st.markdown("**❄️ Số gan:**")
            for d, g in sorted(gaps.items(), key=lambda x: x[1], reverse=True)[:3]:
                st.markdown(f"""
                <div class="card" style="padding:10px; margin:5px 0;">
                    <span style="font-size:28px; font-weight:900; color:#0066ff;">{d}</span>
                    <span style="margin-left:15px;">Gan {g} kỳ</span>
                    <span style="float:right;">{'🔥' if g >= 8 else '⏳'}</span>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # === SECTION 5: HISTORY ===
        if data.get("predictions"):
            st.markdown('<div class="section">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📋 LỊCH SỬ DỰ ĐOÁN</div>', unsafe_allow_html=True)
            
            df_data = []
            for p in data["predictions"][:15]:
                df_data.append({
                    "Date": p.get("date", "")[:16].replace("T"," "), 
                    "Pair": p.get("pair",""), 
                    "Result": p.get("result",""), 
                    "Win": "🔥" if p.get("win") else "❌"
                })
            df = pd.DataFrame(df_data)
            
            def color_win(val): 
                return "color:#00ff40;font-weight:900" if val=="🔥" else "color:#ff0040;font-weight:900"
            
            st.dataframe(
                df.style.applymap(color_win, subset=['Win']), 
                use_container_width=True, 
                hide_index=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.warning("⚠️ Cần ít nhất 15 kỳ để phân tích. Vui lòng nhập thêm dữ liệu ở trên.")
    
    # Footer
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center; color:#666; font-size:10px; padding:20px;">TITAN V60 | Mobile Scroll | AI Self-Learning</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()