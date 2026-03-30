import streamlit as st
import re, pandas as pd, numpy as np, math, json, os
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime
import google.generativeai as genai

# === CẤU HÌNH ===
DB_FILE = "titan_v53_data.json"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
LUCKY_OX = [0, 2, 5, 6, 7, 8]

genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="TITAN V53 - ULTIMATE", page_icon="🛡️", layout="centered")

# === CSS NÂNG CẤP ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .stApp {
        background: linear-gradient(180deg, #0a0000 0%, #1a0a0a 50%, #0a0a0a 100%);
        color: #ff6600;
        font-family: 'Orbitron', sans-serif;
    }
    
    .main-header {
        font-size: 36px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #ff6600, #ff0000, #ff6600);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: glow 2s ease-in-out infinite;
        text-shadow: 0 0 30px rgba(255, 102, 0, 0.5);
    }
    
    @keyframes glow {
        0%, 100% { filter: brightness(1); }
        50% { filter: brightness(1.5); }
    }
    
    .metric-box {
        background: linear-gradient(135deg, #1a0a0a, #2a1a1a);
        border: 2px solid #ff6600;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 0 20px rgba(255, 102, 0, 0.3);
        margin: 10px 0;
        min-height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-label {
        font-size: 12px;
        color: #888;
        margin-bottom: 5px;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 900;
        color: #ff6600;
    }
    
    .danger-box {
        background: linear-gradient(135deg, #2a0a0a, #1a0505);
        border: 3px solid #ff0000;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 0 30px rgba(255, 0, 0, 0.5);
        animation: pulse-danger 1.5s infinite;
    }
    
    @keyframes pulse-danger {
        0%, 100% { box-shadow: 0 0 20px rgba(255, 0, 0, 0.5); }
        50% { box-shadow: 0 0 40px rgba(255, 0, 0, 0.8); }
    }
    
    .warning-text {
        color: #ff0000;
        font-weight: 900;
        font-size: 20px;
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .safe-box {
        background: linear-gradient(135deg, #0a2a0a, #0a1a0a);
        border: 3px solid #00ff00;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        text-align: center;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #1a0a0a, #2a1a1a);
        border: 2px solid #ff6600;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        text-align: center;
    }
    
    .big-number {
        font-size: 48px;
        font-weight: 900;
        color: #ff6600;
        letter-spacing: 12px;
        text-shadow: 0 0 20px rgba(255, 102, 0, 0.5);
    }
    
    .tag {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 10px;
        font-weight: bold;
        margin: 3px;
    }
    
    .tag-gan { background: #ff6600; color: #000; }
    .tag-trap { background: #ff0000; color: #fff; animation: blink 0.5s infinite; }
    .tag-safe { background: #00ff00; color: #000; }
    
    textarea {
        background: #0a0000 !important;
        border: 2px solid #ff6600 !important;
        color: #ff6600 !important;
        border-radius: 10px;
        padding: 10px;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #ff6600, #ff0000);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: bold;
        font-family: 'Orbitron', sans-serif;
    }
    
    .history-win { color: #00ff00; font-weight: 900; }
    .history-lose { color: #ff0000; font-weight: 900; }
</style>
""", unsafe_allow_html=True)

# === DATA MANAGEMENT (FIXED) ===
def load_data():
    """Load data với error handling đầy đủ"""
    default_data = {
        "results": [],
        "predictions": [],
        "stats": {
            "total_bets": 0,
            "wins": 0,
            "lose_streak": 0,
            "last_updated": None
        }
    }
    
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Ensure all keys exist
                for key in default_data:
                    if key not in data:
                        data[key] = default_data[key]
                if "stats" in data:
                    for key in default_data["stats"]:
                        if key not in data["stats"]:
                            data["stats"][key] = default_data["stats"][key]
                return data
        except Exception as e:
            st.error(f"⚠️ Lỗi load data: {e}")
            return default_data
    return default_data

def save_data(data):
    """Save data với error handling"""
    try:
        data["stats"]["last_updated"] = datetime.now().isoformat()
        with open(DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"⚠️ Lỗi save data: {e}")
        return False

def parse_numbers(text):
    clean = re.sub(r'\s+', '', text)
    return [n for n in re.findall(r'\d{5}', clean) if n]

# === SURVIVAL ANALYZER (NÂNG CẤP) ===
class SurvivalAnalyzer:
    def __init__(self, results, predictions_history):
        self.results = results
        self.history = predictions_history
        self.lose_streak = self._calculate_lose_streak()
        self.trap_pairs = self._identify_trap_pairs()
        
    def _calculate_lose_streak(self):
        if not self.history:
            return 0
        streak = 0
        for pred in self.history:
            if pred.get("ket_qua") == "LOSE":
                streak += 1
            else:
                break
        return streak
    
    def _identify_trap_pairs(self):
        pair_losses = defaultdict(int)
        for pred in self.history:
            if pred.get("ket_qua") == "LOSE" and "du_doan" in pred:
                pair_losses[pred["du_doan"]] += 1
        return {pair for pair, count in pair_losses.items() if count >= 2}
    
    def should_skip_bet(self):
        if self.lose_streak >= 3:
            return True, f"ĐANG THUA {self.lose_streak} KỲ LIÊN TIẾP - BẮT BUỘC NGHỈ"
        if self.lose_streak >= 2:
            return True, "ĐANG THUA 2 KỲ - CỰC KỲ THẬN TRỌNG"
        if len(self.history) >= 5:
            recent_losses = sum(1 for h in self.history[:5] if h.get("ket_qua") == "LOSE")
            if recent_losses >= 4:
                return True, "4/5 KỲ GẦN THUA - NHÀ CÁI ĐANG THU"
        return False, "OK"
    
    def analyze_pattern(self):
        if len(self.results) < 10:
            return {"hot_numbers": [], "hot_pairs": []}
        
        recent = self.results[-10:]
        all_digits = "".join(recent)
        hot_numbers = Counter(all_digits).most_common(5)
        
        pair_freq = Counter()
        for num in recent:
            for pair in combinations(sorted(set(num)), 2):
                pair_freq["".join(pair)] += 1
        
        return {
            "hot_numbers": [n[0] for n in hot_numbers],
            "hot_pairs": [p for p, c in pair_freq.most_common(5)]
        }
    
    def get_survival_prediction(self):
        if len(self.results) < 15:
            return None, "Cần ít nhất 15 kỳ dữ liệu", 0
        
        should_skip, reason = self.should_skip_bet()
        if should_skip:
            return None, reason, 0
        
        pattern = self.analyze_pattern()
        all_pairs = list(combinations("0123456789", 2))
        candidates = []
        
        for pair in all_pairs:
            pair_str = "".join(pair)
            
            if pair_str in self.trap_pairs:
                continue
            if pair_str in pattern.get("hot_pairs", []):
                continue
            
            gan = 0
            for num in reversed(self.results):
                if set(pair).issubset(set(num)):
                    break
                gan += 1
            
            if 4 <= gan <= 10:
                score = 80 + (10 - abs(gan - 7)) * 2
                candidates.append((pair_str, score, gan))
            elif 2 <= gan <= 12:
                score = 50 + gan
                candidates.append((pair_str, score, gan))
        
        if not candidates:
            return None, "KHÔNG CÓ CẶP NÀO AN TOÀN - NÊN NGHỈ", 0
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        best = candidates[0]
        confidence = min(75, best[1])
        
        return best[0], f"Gan {best[2]} kỳ - Score {best[1]}", confidence

# === UI MAIN ===
def main():
    st.markdown('<h1 class="main-header">🛡️ TITAN V53 - ULTIMATE SURVIVOR</h1>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center; color:#ff6600; margin-bottom:20px; font-size:14px;">CHẾ ĐỘ SINH TỒN - ƯU TIÊN KHÔNG THUA</div>', unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    
    # Initialize session state
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
    
    # Stats display
    stats = data.get("stats", {"total_bets": 0, "wins": 0, "lose_streak": 0})
    total_bets = stats.get("total_bets", 0)
    wins = stats.get("wins", 0)
    lose_streak = stats.get("lose_streak", 0)
    win_rate = (wins/total_bets*100) if total_bets > 0 else 0
    
    st.markdown("### 📊 THỐNG KÊ")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">TỔNG CƯỢC</div>
            <div class="metric-value">{total_bets}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">THẮNG</div>
            <div class="metric-value" style="color:#00ff00;">{wins}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        color = "#ff0000" if lose_streak >= 3 else "#ff6600"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">THUA LIÊN TIẾP</div>
            <div class="metric-value" style="color:{color};">{lose_streak}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Warning if lose streak high
    if lose_streak >= 3:
        st.markdown(f"""
        <div class="danger-box">
            <div class="warning-text">⚠️ CẢNH BÁO ĐỎ</div>
            <div style="font-size:18px; margin-top:10px; color:#ff6680;">
                ĐANG THUA {lose_streak} KỲ LIÊN TIẾP<br>
                <span style="color:#ff0000; font-weight:900; font-size:20px;">BẮT BUỘC NGHỈ ÍT NHẤT 5 KỲ</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Input section
    st.markdown("### 📥 NHẬP KẾT QUẢ MỚI")
    new_input = st.text_area("Dán kết quả kỳ vừa rồi (5 số):", height=80, 
                             placeholder="Ví dụ:\n12345\n67890")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 LƯU KẾT QUẢ", use_container_width=True):
            nums = parse_numbers(new_input)
            if nums:
                data["results"].extend(nums)
                data["results"] = data["results"][-100:]
                
                # Update last prediction result
                if data.get("predictions"):
                    last_pred = data["predictions"][0] if data["predictions"] else None
                    if last_pred and last_pred.get("ket_qua") is None:
                        last_actual = nums[-1]
                        pred_pair = last_pred.get("du_doan", "")
                        if pred_pair and all(d in last_actual for d in pred_pair):
                            last_pred["ket_qua"] = "WIN"
                            stats["wins"] += 1
                            stats["lose_streak"] = 0
                        else:
                            last_pred["ket_qua"] = "LOSE"
                            stats["lose_streak"] += 1
                        stats["total_bets"] += 1
                
                if save_data(data):
                    st.success(f"✅ Đã lưu {len(nums)} kỳ")
                    st.rerun()
                else:
                    st.error("❌ Không thể lưu dữ liệu")
            else:
                st.warning("⚠️ Không tìm thấy số hợp lệ")
    
    with col2:
        if st.button("🎯 DỰ ĐOÁN KỲ TIẾP", use_container_width=True, type="primary"):
            if len(data["results"]) >= 15:
                analyzer = SurvivalAnalyzer(data["results"], data.get("predictions", []))
                prediction, reason, confidence = analyzer.get_survival_prediction()
                
                if prediction:
                    pred_record = {
                        "thoi_gian": datetime.now().isoformat(),
                        "du_doan": prediction,
                        "ly_do": reason,
                        "confidence": confidence,
                        "ket_qua": None
                    }
                    data["predictions"].insert(0, pred_record)
                    data["predictions"] = data["predictions"][:50]
                    
                    if save_data(data):
                        st.markdown(f"""
                        <div class="safe-box">
                            <div style="font-size:20px; color:#00ff00;">✅ DỰ ĐOÁN:</div>
                            <div class="big-number">{prediction[0]} - {prediction[1]}</div>
                            <div style="margin-top:10px; color:#ffcc00;">{reason}</div>
                            <div style="margin-top:10px; font-size:24px; font-weight:900;">
                                Confidence: <span style="color:#00ff00;">{confidence:.0f}%</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="danger-box">
                        <div class="warning-text">⛔ KHÔNG NÊN ĐÁNH</div>
                        <div style="margin-top:10px; color:#ff6680;">{reason}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ Cần ít nhất 15 kỳ (hiện có: {len(data['results'])})")
    
    with col3:
        if st.button("🗑️ XÓA DATA", use_container_width=True):
            data["results"] = []
            data["predictions"] = []
            data["stats"] = {"total_bets": 0, "wins": 0, "lose_streak": 0, "last_updated": None}
            if save_data(data):
                st.warning("🗑️ Đã xóa toàn bộ dữ liệu")
                st.rerun()
    
    # Show recent results
    if data["results"]:
        st.markdown(f"### 📋 **10 kỳ gần nhất:**")
        st.write(", ".join(data["results"][-10:]))
    
    # History
    if data.get("predictions"):
        st.markdown("### 📊 LỊCH SỬ DỰ ĐOÁN (10 kỳ gần nhất)")
        
        df_data = []
        for pred in data["predictions"][:10]:
            df_data.append({
                "Thời gian": pred.get("thoi_gian", "")[:16].replace("T", " "),
                "Dự đoán": pred.get("du_doan", ""),
                "Lý do": pred.get("ly_do", "")[:30],
                "Confidence": f"{pred.get('confidence', 0):.0f}%",
                "Kết quả": pred.get("ket_qua", "Chờ") or "Chờ"
            })
        
        df = pd.DataFrame(df_data)
        
        def color_kq(val):
            if val == "WIN":
                return "color: #00ff00; font-weight: 900"
            elif val == "LOSE":
                return "color: #ff0000; font-weight: 900"
            return "color: #ffaa00"
        
        st.dataframe(df.style.applymap(color_kq, subset=['Kết quả']), 
                     use_container_width=True, hide_index=True)
        
        # Win rate
        if total_bets > 0:
            st.markdown(f"""
            <div class="prediction-card" style="margin-top:20px; border-color:{'#00ff00' if win_rate >= 40 else '#ff0000'};">
                <div style="font-size:18px; margin-bottom:10px;">📈 TỶ LỆ THẮNG</div>
                <div style="font-size:42px; font-weight:900; color:{'#00ff00' if win_rate >= 40 else '#ff0000'};">
                    {win_rate:.1f}%
                </div>
                <div style="font-size:16px; color:#888;">({wins}/{total_bets} kỳ)</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown('<div style="text-align:center; color:#666; font-size:10px; margin-top:20px;">TITAN V53 ULTIMATE SURVIVOR | Sinh tồn là ưu tiên số 1</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()