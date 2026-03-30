import streamlit as st
import json, os, re, math, random
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import google.generativeai as genai
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# === CẤU HÌNH ===
DB_FILE = "titan_v52_data.json"
NVIDIA_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

st.set_page_config(page_title="TITAN V52 - AI NEURAL", page_icon="🧠", layout="centered")

# === CSS CAO CẤP ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .stApp {
        background: linear-gradient(180deg, #000000 0%, #0a0a1a 50%, #000000 100%);
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
        text-shadow: 0 0 30px rgba(0, 255, 255, 0.5);
    }
    
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .tab-btn {
        background: linear-gradient(135deg, #1a1a2e, #0f0f1a);
        border: 2px solid #00ffff;
        color: #00ffff;
        padding: 10px;
        border-radius: 10px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s;
        text-align: center;
    }
    
    .tab-btn:hover {
        background: #00ffff;
        color: #000;
        box-shadow: 0 0 20px #00ffff;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #0f0f1a, #1a1a2e);
        border: 2px solid #00ffff;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.2);
    }
    
    .big-number {
        font-size: 48px;
        font-weight: 900;
        letter-spacing: 10px;
        color: #00ffff;
        text-shadow: 0 0 20px #00ffff;
    }
    
    .confidence-box {
        background: linear-gradient(90deg, #ff0040, #ffff00, #00ff40);
        padding: 3px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .confidence-fill {
        background: #000;
        border-radius: 8px;
        padding: 8px;
        text-align: center;
        font-weight: bold;
        color: #fff;
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
    .tag-gan { background: #9900ff; color: white; }
    .tag-safe { background: #00ff40; color: black; }
    .tag-ai { background: #ff00ff; color: white; }
    
    .skip-box {
        background: linear-gradient(135deg, #2a0a2a, #1a0a1a);
        border: 3px solid #ff00ff;
        color: #ff00ff;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: 900;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 20px #ff00ff; }
        50% { box-shadow: 0 0 40px #ff00ff; }
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        margin: 15px 0;
    }
    
    .metric-cell {
        background: #0f0f1a;
        border: 1px solid #00ffff;
        border-radius: 10px;
        padding: 12px 8px;
        text-align: center;
    }
    
    .win { color: #00ff40; font-weight: 900; }
    .lose { color: #ff0040; font-weight: 900; }
    
    textarea {
        background: #0a0a1a !important;
        border: 1px solid #00ffff !important;
        color: #00ffff !important;
    }
</style>
""", unsafe_allow_html=True)

# === DATA MANAGEMENT ===
def load_data():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"results": [], "predictions": [], "model_weights": {}, "strategies": {}}

def save_data(data):
    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def parse_numbers(text):
    clean = re.sub(r'\s+', '', text)
    return [n for n in re.findall(r'\d{5}', clean) if n]

# === ADVANCED STATISTICAL ENGINE ===
class AdvancedStatsEngine:
    def __init__(self, results):
        self.results = results
        self.digits = "".join(results) if results else ""
        
    def frequency_analysis(self, window=None):
        data = self.results[-window:] if window else self.results
        counter = Counter("".join(data))
        total = len(counter) or 1
        return {d: {"count": counter.get(d, 0), "freq": counter.get(d, 0)/total*100} 
                for d in "0123456789"}
    
    def markov_chain_advanced(self, order=2):
        """Markov chain với order linh hoạt"""
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
    
    def calculate_digit_probability(self, digit, position=None):
        """Tính xác suất xuất hiện của từng số"""
        if not self.results:
            return 0.0
        
        if position is not None:
            # Xác suất theo vị trí
            count = sum(1 for r in self.results if r[position] == digit)
        else:
            # Xác suất tổng quát
            count = sum(1 for r in self.results if digit in r)
        
        return count / len(self.results) * 100
    
    def detect_cycle(self, digit):
        """Phát hiện chu kỳ xuất hiện của số"""
        positions = [i for i, r in enumerate(self.results) if digit in r]
        if len(positions) < 3:
            return None
        
        gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
        if not gaps:
            return None
        
        avg_gap = sum(gaps) / len(gaps)
        return avg_gap
    
    def entropy_analysis(self, window=20):
        """Tính entropy - độ hỗn loạn"""
        if len(self.results) < window:
            return 3.5
        
        recent = "".join(self.results[-window:])
        counter = Counter(recent)
        total = len(recent)
        
        entropy = 0
        for count in counter.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    def get_position_hot_numbers(self):
        """Số hot theo từng vị trí"""
        positions = {i: Counter() for i in range(5)}
        for num in self.results[-30:]:
            for i, d in enumerate(num):
                positions[i][d] += 1
        
        hot_by_pos = {}
        for pos in positions:
            if positions[pos]:
                hot_by_pos[pos] = [d for d, _ in positions[pos].most_common(2)]
        
        return hot_by_pos

# === MACHINE LEARNING ENGINE ===
class MLEngine:
    def __init__(self, results):
        self.results = results
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def prepare_features(self):
        """Chuẩn bị features cho ML"""
        if len(self.results) < 20:
            return None, None
        
        X = []
        y = []
        
        for i in range(10, len(self.results)):
            # Features: tần suất 10 kỳ gần
            recent = self.results[i-10:i]
            freq = Counter("".join(recent))
            features = [freq.get(str(d), 0) for d in range(10)]
            
            # Gap features
            for d in "0123456789":
                gap = 0
                for r in reversed(recent):
                    if d in r:
                        break
                    gap += 1
                features.append(gap)
            
            # Label: số nào xuất hiện trong kỳ tiếp theo
            next_num = self.results[i]
            labels = [1 if str(d) in next_num else 0 for d in range(10)]
            
            X.append(features)
            y.append(labels)
        
        return np.array(X), np.array(y)
    
    def train(self):
        """Huấn luyện model"""
        X, y = self.prepare_features()
        if X is None:
            return False
        
        self.model.fit(X, y)
        self.is_trained = True
        return True
    
    def predict_next(self, recent_results):
        """Dự đoán kỳ tiếp theo"""
        if not self.is_trained or len(recent_results) < 10:
            return None
        
        # Chuẩn bị features từ 10 kỳ gần nhất
        freq = Counter("".join(recent_results[-10:]))
        features = [freq.get(str(d), 0) for d in range(10)]
        
        for d in "0123456789":
            gap = 0
            for r in reversed(recent_results[-10:]):
                if d in r:
                    break
                gap += 1
            features.append(gap)
        
        probs = self.model.predict_proba([features])[0]
        return {str(i): prob for i, prob in enumerate(probs)}

# === AI ANALYZER ===
class AIAnalyzer:
    def __init__(self, results):
        self.results = results
        self.stats = AdvancedStatsEngine(results)
        self.ml = MLEngine(results)
        
    def analyze_with_gemini(self):
        """Sử dụng Gemini AI để phân tích"""
        if len(self.results) < 15:
            return None
        
        freq = self.stats.frequency_analysis(30)
        entropy = self.stats.entropy_analysis()
        
        prompt = f"""
        Phân tích xổ số 5D chuyên sâu. 20 kỳ gần nhất:
        {', '.join(self.results[-20:])}
        
        Thống kê:
        - Tần suất: {json.dumps(freq)}
        - Entropy: {entropy:.2f}
        
        Đề xuất TOP 3 cặp 2 số mạnh nhất.
        Trả về JSON: {{"predictions": [["0","1"], ["2","3"]], "confidence": [85, 75, 65], "reasons": ["lý do 1", "lý do 2"]}}
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
        """Tính điểm tổng hợp"""
        score = 0
        reasons = []
        
        p1, p2 = pair[0], pair[1]
        
        # 1. Frequency score
        freq = self.stats.frequency_analysis(30)
        if freq[p1]["freq"] > 7 and freq[p2]["freq"] > 7:
            score += 30
            reasons.append("TẦN SUẤT CAO")
        
        # 2. Gap score (vùng vàng 4-10)
        for p in [p1, p2]:
            gap = 0
            for r in reversed(self.results):
                if p in r:
                    break
                gap += 1
            
            if 4 <= gap <= 10:
                score += 25
                reasons.append("GAN VÀNG")
            elif gap > 15:
                score -= 20
        
        # 3. Streak score
        for p in [p1, p2]:
            streak = 0
            for r in reversed(self.results):
                if p in r:
                    streak += 1
                else:
                    break
            
            if streak >= 3:
                score -= 40
                reasons.append("BỆT QUÁ DÀI")
            elif streak == 1:
                score += 20
        
        # 4. Position match
        hot_pos = self.stats.get_position_hot_numbers()
        pos_match = 0
        for pos, hot_nums in hot_pos.items():
            if p1 in hot_nums or p2 in hot_nums:
                pos_match += 1
        score += pos_match * 10
        
        # 5. ML prediction
        if self.ml.is_trained:
            ml_probs = self.ml.predict_next(self.results[-10:])
            if ml_probs:
                ml_score = (float(ml_probs.get(p1, 0)) + float(ml_probs.get(p2, 0))) * 20
                score += ml_score
        
        # 6. Cycle detection
        for p in [p1, p2]:
            cycle = self.stats.detect_cycle(p)
            if cycle and abs(len(self.results) % cycle - cycle) <= 2:
                score += 15
                reasons.append("THEO CHU KỲ")
        
        return min(100, score), reasons

# === PREDICTION SYSTEM ===
class PredictionSystem:
    def __init__(self, results, history=None):
        self.results = results
        self.history = history or []
        self.stats = AdvancedStatsEngine(results)
        self.ai = AIAnalyzer(results)
        
    def detect_house_mode(self):
        """Phát hiện chế độ nhà cái"""
        if len(self.history) < 5:
            return "CHAOS", 0.5
        
        recent_wins = sum(1 for h in self.history[:10] if h.get('result') == 'WIN')
        win_rate = recent_wins / min(len(self.history), 10)
        
        if win_rate >= 0.5:
            return "PAY", win_rate
        elif win_rate <= 0.25:
            return "TAKE", win_rate
        return "CHAOS", win_rate
    
    def should_skip(self, confidence):
        """Quyết định có nên đánh không"""
        mode, win_rate = self.detect_house_mode()
        entropy = self.stats.entropy_analysis()
        
        if mode == "TAKE" and confidence < 75:
            return True, "NHÀ CÁI ĐANG THU"
        
        if entropy > 3.3:
            return True, "QUÁ HỖN LOẠN"
        
        if len(self.history) >= 3:
            recent_losses = sum(1 for h in self.history[:3] if h.get('result') == 'LOSE')
            if recent_losses == 3:
                return True, "THUA 3 KỲ - NGHỈ"
        
        return False, "OK"
    
    def generate_predictions(self):
        """Tạo dự đoán"""
        if len(self.results) < 15:
            return None
        
        # Train ML model
        self.ai.ml.train()
        
        # Generate all pairs
        all_pairs = list(combinations("0123456789", 2))
        scored = []
        
        for pair in all_pairs:
            score, reasons = self.ai.hybrid_score(pair)
            scored.append({
                "pair": "".join(pair),
                "score": score,
                "reasons": reasons
            })
        
        scored.sort(key=lambda x: x["score"], reverse=True)
        top_pairs = scored[:5]
        
        # Calculate confidence
        if top_pairs:
            confidence = min(95, max(30, 40 + top_pairs[0]["score"] / 3))
        else:
            confidence = 50
        
        # Check if should skip
        skip, skip_reason = self.should_skip(confidence)
        
        # House mode
        house_mode, win_rate = self.detect_house_mode()
        
        # Top 8 numbers
        freq = self.stats.frequency_analysis(50)
        top8 = "".join([d for d, _ in sorted(freq.items(), key=lambda x: x[1]["count"], reverse=True)[:8]])
        
        return {
            "pairs": top_pairs,
            "confidence": confidence,
            "skip": skip,
            "skip_reason": skip_reason,
            "house_mode": house_mode,
            "win_rate": win_rate,
            "entropy": self.stats.entropy_analysis(),
            "top8": top8
        }

# === UI COMPONENTS ===
def render_input_tab(data):
    st.markdown("### 📥 NHẬP KẾT QUẢ")
    
    new_input = st.text_area("Dán kết quả (mỗi kỳ 1 dòng):", height=120, 
                             placeholder="12345\n67890\n...", key="input_area")
    
    if st.button("💾 LƯU & PHÂN TÍCH", type="primary", use_container_width=True):
        nums = parse_numbers(new_input)
        if nums:
            data["results"].extend(nums)
            data["results"] = data["results"][-200:]
            save_data(data)
            st.success(f"✅ Đã lưu {len(nums)} kỳ")
            st.rerun()
        else:
            st.error("❌ Không tìm thấy số hợp lệ")
    
    if data["results"]:
        st.markdown(f"**📊 Tổng số kỳ:** {len(data['results'])}")
        st.markdown(f"**📌 10 kỳ gần nhất:**")
        st.write(", ".join(data["results"][-10:]))

def render_prediction_tab(data):
    st.markdown("### 🎯 DỰ ĐOÁN THÔNG MINH")
    
    if len(data["results"]) < 15:
        st.warning("⚠️ Cần ít nhất 15 kỳ dữ liệu")
        return
    
    pred_sys = PredictionSystem(data["results"], data.get("predictions", []))
    result = pred_sys.generate_predictions()
    
    if not result:
        st.error("Không thể tạo dự đoán")
        return
    
    # House mode indicator
    mode_colors = {
        "PAY": ("#00ff40", "💰 ĐANG TRẢ"),
        "TAKE": ("#ff0040", "🦈 ĐANG THU"),
        "CHAOS": ("#ffff00", "🌪️ HỖN LOẠN")
    }
    
    color, text = mode_colors.get(result["house_mode"], ("#888", "❓"))
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e, #0f0f1a); 
                border: 2px solid {color}; 
                border-radius: 15px; 
                padding: 20px; 
                text-align: center; 
                margin: 10px 0;">
        <div style="font-size: 28px; color: {color}; font-weight: 900;">{text}</div>
        <div style="margin-top: 10px; color: #888;">
            Win Rate: {result['win_rate']*100:.1f}% | Entropy: {result['entropy']:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Skip warning
    if result["skip"]:
        st.markdown(f"""
        <div class="skip-box">
            ⚠️ {result['skip_reason']}<br>
            <span style="font-size: 16px;">KHÔNG NÊN ĐÁNH KỲ NÀY</span>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Metrics
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-cell">
            <div style="font-size:10px; color:#888;">TIN CẬY</div>
            <div style="font-size:24px; font-weight:900; color:#00ffff;">{result['confidence']:.0f}%</div>
        </div>
        <div class="metric-cell">
            <div style="font-size:10px; color:#888;">TOP PAIR</div>
            <div style="font-size:20px; font-weight:900; color:#FFD700;">{result['pairs'][0]['pair']}</div>
        </div>
        <div class="metric-cell">
            <div style="font-size:10px; color:#888;">SCORE</div>
            <div style="font-size:20px; font-weight:900; color:#00ff40;">{result['pairs'][0]['score']:.0f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Top pair
    st.markdown("#### 🎯 CẶP VIP:")
    top = result["pairs"][0]
    tags = "".join([f'<span class="tag tag-safe">{r}</span>' for r in top["reasons"][:3]])
    
    st.markdown(f"""
    <div class="prediction-card">
        <div class="big-number">{top['pair'][0]} - {top['pair'][1]}</div>
        <div style="margin: 10px 0;">{tags}</div>
        <div style="font-size: 18px; color: #00ff40;">Score: {top['score']:.0f}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Top 5
    st.markdown("#### 📊 TOP 5 PAIRS:")
    for i, pair_data in enumerate(result["pairs"][1:5], 1):
        tags = "".join([f'<span class="tag tag-gan">{r}</span>' for r in pair_data["reasons"][:2]])
        st.markdown(f"""
        <div style="background: #0f0f1a; border-left: 4px solid #00ffff; 
                    padding: 12px; margin: 5px 0; border-radius: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 24px; font-weight: 900; color: #FFD700;">
                    {pair_data['pair'][0]} - {pair_data['pair'][1]}
                </span>
                <span style="font-size: 16px; color: #00ff40;">{pair_data['score']:.0f}</span>
            </div>
            <div style="margin-top: 5px;">{tags}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Top 8
    st.markdown(f"""
    <div class="prediction-card" style="margin-top: 15px;">
        <div style="font-size: 12px; color: #888;">ĐỘ PHỦ SẢNH (8 SỐ)</div>
        <div style="font-size: 32px; font-weight: 900; letter-spacing: 6px; color: #00ffff;">
            {result['top8']}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Save prediction
    if st.button("💾 LƯU DỰ ĐOÁN", use_container_width=True):
        prediction_record = {
            "date": datetime.now().isoformat(),
            "pair": result["pairs"][0]["pair"],
            "confidence": result["confidence"],
            "result": None
        }
        data["predictions"].insert(0, prediction_record)
        data["predictions"] = data["predictions"][-50:]
        save_data(data)
        st.success("✅ Đã lưu dự đoán")
        st.rerun()

def render_stats_tab(data):
    st.markdown("### 📊 THỐNG KÊ CHI TIẾT")
    
    if len(data["results"]) < 10:
        st.warning("Cần ít nhất 10 kỳ dữ liệu")
        return
    
    stats = AdvancedStatsEngine(data["results"])
    
    # Frequency
    st.markdown("#### 🔥 TẦN SUẤT (30 kỳ gần)")
    freq = stats.frequency_analysis(30)
    
    cols = st.columns(5)
    for i, (digit, info) in enumerate(sorted(freq.items(), key=lambda x: x[1]["count"], reverse=True)):
        with cols[i % 5]:
            color = "#00ff40" if info["count"] > 15 else ("#ff0040" if info["count"] < 5 else "#ffff00")
            st.markdown(f"""
            <div style="background: #0f0f1a; border: 1px solid {color}; 
                        border-radius: 10px; padding: 10px; text-align: center; margin: 5px 0;">
                <div style="font-size: 24px; font-weight: 900; color: {color};">{digit}</div>
                <div style="font-size: 12px; color: #888;">{info['count']} lần</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Gaps
    st.markdown("#### ❄️ SỐ GAN")
    gaps = {}
    for d in "0123456789":
        gap = 0
        for r in reversed(data["results"]):
            if d in r:
                break
            gap += 1
        gaps[d] = gap
    
    sorted_gaps = sorted(gaps.items(), key=lambda x: x[1], reverse=True)
    
    for digit, gap in sorted_gaps[:5]:
        status = "🔥 SẮP VỀ" if gap >= 8 else "⏳ THEO DÕI"
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; 
                    background: #0f0f1a; padding: 10px; margin: 5px 0; border-radius: 8px;">
            <span style="font-size: 20px; font-weight: 900;">Số {digit}</span>
            <span style="color: #ff00ff;">{gap} kỳ</span>
            <span>{status}</span>
        </div>
        """, unsafe_allow_html=True)

def render_history_tab(data):
    st.markdown("### 📋 LỊCH SỬ DỰ ĐOÁN")
    
    if not data.get("predictions"):
        st.info("Chưa có lịch sử dự đoán")
        return
    
    # Input result for last prediction
    if data["predictions"] and data["predictions"][0].get("result") is None:
        st.markdown("#### ✅ CẬP NHẬT KẾT QUẢ")
        last_result = st.text_input("Nhập kết quả kỳ gần nhất:", placeholder="12345")
        
        if st.button("XÁC NHẬN KẾT QUẢ", use_container_width=True):
            if len(last_result) == 5 and last_result.isdigit():
                pair = data["predictions"][0]["pair"]
                is_win = all(d in last_result for d in pair)
                data["predictions"][0]["result"] = "WIN" if is_win else "LOSE"
                data["predictions"][0]["actual"] = last_result
                save_data(data)
                st.success(f"✅ {'THẮNG! 🎉' if is_win else 'THUA ❌'}")
                st.rerun()
    
    # Display history
    df_data = []
    for pred in data["predictions"][:20]:
        df_data.append({
            "Ngày": pred.get("date", "")[:16].replace("T", " "),
            "Dự đoán": pred.get("pair", ""),
            "Confidence": f"{pred.get('confidence', 0):.0f}%",
            "Kết quả": pred.get("result", "Chờ"),
            "Thực tế": pred.get("actual", "-")
        })
    
    df = pd.DataFrame(df_data)
    
    def color_result(val):
        if val == "WIN":
            return "color: #00ff40; font-weight: 900"
        elif val == "LOSE":
            return "color: #ff0040; font-weight: 900"
        return "color: #888"
    
    st.dataframe(df.style.applymap(color_result, subset=['Kết quả']), 
                 use_container_width=True, hide_index=True)
    
    # Win rate
    if data["predictions"]:
        wins = sum(1 for p in data["predictions"] if p.get("result") == "WIN")
        total_with_result = sum(1 for p in data["predictions"] if p.get("result"))
        
        if total_with_result > 0:
            rate = wins / total_with_result * 100
            
            st.markdown(f"""
            <div class="prediction-card" style="margin-top: 20px; 
                        border-color: {'#00ff40' if rate >= 40 else '#ff0040'};">
                <div style="font-size: 16px;">TỶ LỆ THẮNG</div>
                <div style="font-size: 36px; font-weight: 900; 
                            color: {'#00ff40' if rate >= 40 else '#ff0040'};">
                    {rate:.1f}% ({wins}/{total_with_result})
                </div>
            </div>
            """, unsafe_allow_html=True)

# === MAIN APP ===
def main():
    data = load_data()
    
    st.markdown('<div class="main-header">🧠 TITAN V52 - AI NEURAL</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center; color:#888; margin-bottom:15px; font-size:11px;">ML + Markov + Gemini AI | Self-Learning System</div>', unsafe_allow_html=True)
    
    # Tabs
    tabs = ["📥 Nhập", "🎯 Dự Đoán", "📊 Thống Kê", "📋 Lịch Sử"]
    
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = tabs[0]
    
    cols = st.columns(4)
    for i, tab in enumerate(tabs):
        with cols[i]:
            if st.button(tab, key=f"tab_{i}", use_container_width=True):
                st.session_state.active_tab = tab
    
    st.markdown("---")
    
    if st.session_state.active_tab == tabs[0]:
        render_input_tab(data)
    elif st.session_state.active_tab == tabs[1]:
        render_prediction_tab(data)
    elif st.session_state.active_tab == tabs[2]:
        render_stats_tab(data)
    elif st.session_state.active_tab == tabs[3]:
        render_history_tab(data)
    
    st.markdown("---")
    st.markdown('<div style="text-align:center; color:#444; font-size:10px; margin-top:20px;">TITAN V52 | AI-Powered | Self-Learning</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()