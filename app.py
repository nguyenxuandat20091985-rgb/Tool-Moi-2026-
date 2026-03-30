import streamlit as st
import json, os, re, math
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import google.generativeai as genai
from datetime import datetime, timedelta

# === CẤU HÌNH ===
DB_FILE = "titan_v52_data.json"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"

genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="TITAN V52 - SURVIVOR", page_icon="🛡️", layout="centered")

# === CSS ===
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&display=swap');
    .stApp {
        background: linear-gradient(180deg, #0a0000 0%, #1a0a0a 100%);
        color: #ff6600;
        font-family: 'Orbitron', sans-serif;
    }
    .main-header {
        font-size: 32px;
        font-weight: 900;
        text-align: center;
        color: #ff6600;
        text-shadow: 0 0 30px #ff0000;
        margin-bottom: 20px;
        animation: flicker 2s infinite;
    }
    @keyframes flicker {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    .danger-box {
        background: linear-gradient(135deg, #2a0a0a, #1a0505);
        border: 3px solid #ff0000;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 0 30px rgba(255, 0, 0, 0.5);
    }
    .warning-text {
        color: #ff0000;
        font-weight: 900;
        font-size: 18px;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .safe-box {
        background: linear-gradient(135deg, #0a2a0a, #051a05);
        border: 3px solid #00ff00;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
    }
    .prediction-card {
        background: #1a0a0a;
        border: 2px solid #ff6600;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
    }
    .big-number {
        font-size: 48px;
        font-weight: 900;
        color: #ff6600;
        letter-spacing: 10px;
    }
    .lose-streak {
        background: #ff0000;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    textarea {
        background: #0a0000 !important;
        border: 2px solid #ff6600 !important;
        color: #ff6600 !important;
    }
</style>
""", unsafe_allow_html=True)

# === DATA MANAGEMENT ===
def load_data():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {"results": [], "predictions": [], "stats": {"lose_streak": 0, "total_bets": 0, "wins": 0}}
    return {"results": [], "predictions": [], "stats": {"lose_streak": 0, "total_bets": 0, "wins": 0}}

def save_data(data):
    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def parse_numbers(text):
    clean = re.sub(r'\s+', '', text)
    return [n for n in re.findall(r'\d{5}', clean) if n]

# === SURVIVAL MODE ENGINE ===
class SurvivalAnalyzer:
    """Phân tích sinh tồn - Tập trung vào việc KHÔNG THUA"""
    
    def __init__(self, results, predictions_history):
        self.results = results
        self.history = predictions_history
        self.lose_streak = self._calculate_lose_streak()
        self.trap_pairs = self._identify_trap_pairs()
        
    def _calculate_lose_streak(self):
        """Tính số lần thua liên tiếp"""
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
        """Nhận diện cặp số BẪY (thua nhiều lần)"""
        pair_losses = defaultdict(int)
        
        for pred in self.history:
            if pred.get("ket_qua") == "LOSE" and "du_doan" in pred:
                pair_losses[pred["du_doan"]] += 1
        
        # Cặp nào thua >= 2 lần coi là bẫy
        return {pair for pair, count in pair_losses.items() if count >= 2}
    
    def should_skip_bet(self):
        """
        QUYẾT ĐỊNH SỐNG CÒN: Có nên đánh không?
        """
        # RULE 1: Thua 3+ lần liên tiếp -> NGHỈ
        if self.lose_streak >= 3:
            return True, f"ĐANG THUA {self.lose_streak} KỲ LIÊN TIẾP - BẮT BUỘC NGHỈ"
        
        # RULE 2: Thua 2 lần -> Chỉ đánh nếu confidence rất cao
        if self.lose_streak >= 2:
            return True, "ĐANG THUA 2 KỲ - CỰC KỲ THẬN TRỌNG"
        
        # RULE 3: Kiểm tra xem có đang trong chu kỳ thua không
        if len(self.history) >= 5:
            recent_losses = sum(1 for h in self.history[:5] if h.get("ket_qua") == "LOSE")
            if recent_losses >= 4:
                return True, "4/5 KỲ GẦN THUA - NHÀ CÁI ĐANG THU"
        
        return False, "OK"
    
    def analyze_pattern(self):
        """Phân tích pattern THUA để tránh"""
        if len(self.results) < 10:
            return {}
        
        # Tìm xem số nào ĐANG VỀ nhiều gần đây
        recent = self.results[-10:]
        all_digits = "".join(recent)
        hot_numbers = Counter(all_digits).most_common(5)
        
        # Tìm xem cặp nào ĐANG VỀ nhiều
        pair_freq = Counter()
        for num in recent:
            for pair in combinations(sorted(set(num)), 2):
                pair_freq["".join(pair)] += 1
        
        return {
            "hot_numbers": [n[0] for n in hot_numbers],
            "hot_pairs": [p for p, c in pair_freq.most_common(5)]
        }
    
    def get_survival_prediction(self):
        """
        Dự đoán sinh tồn: Ưu tiên KHÔNG THUA thay vì THẮNG
        """
        if len(self.results) < 15:
            return None, "Cần ít nhất 15 kỳ dữ liệu", 0
        
        # Check có nên nghỉ không
        should_skip, reason = self.should_skip_bet()
        if should_skip:
            return None, reason, 0
        
        # Phân tích pattern
        pattern = self.analyze_pattern()
        
        # Tìm cặp số:
        # 1. KHÔNG nằm trong trap_pairs
        # 2. Có gan vừa phải (3-8 kỳ)
        # 3. KHÔNG phải số đang hot (tránh bẫy)
        
        all_pairs = list(combinations("0123456789", 2))
        candidates = []
        
        for pair in all_pairs:
            pair_str = "".join(pair)
            
            # Loại trap pairs
            if pair_str in self.trap_pairs:
                continue
            
            # Loại hot pairs (đang về nhiều)
            if pair_str in pattern.get("hot_pairs", []):
                continue
            
            # Tính gan
            gan = 0
            for num in reversed(self.results):
                if set(pair).issubset(set(num)):
                    break
                gan += 1
            
            # Ưu tiên gan 4-10
            if 4 <= gan <= 10:
                score = 80 + (10 - abs(gan - 7)) * 2  # Gan 7 là đẹp nhất
                candidates.append((pair_str, score, gan))
            elif 2 <= gan <= 12:
                score = 50 + gan
                candidates.append((pair_str, score, gan))
        
        if not candidates:
            return None, "KHÔNG CÓ CẶP NÀO AN TOÀN - NÊN NGHỈ", 0
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        best = candidates[0]
        
        confidence = min(75, best[1])  # Max 75% để thận trọng
        
        return best[0], f"Gan {best[2]} kỳ - Score {best[1]}", confidence

# === UI ===
def main():
    data = load_data()
    
    st.markdown('<div class="main-header">🛡️ TITAN V52 - SURVIVOR</div>', unsafe_allow_html=True)
    st.markdown('<div class="warning-text" style="text-align:center; margin-bottom:20px;">CHẾ ĐỘ SINH TỒN - ƯU TIÊN KHÔNG THUA</div>', unsafe_allow_html=True)
    
    # Stats
    stats = data.get("stats", {})
    total_bets = stats.get("total_bets", 0)
    wins = stats.get("wins", 0)
    lose_streak = stats.get("lose_streak", 0)
    win_rate = (wins/total_bets*100) if total_bets > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tổng cược", total_bets)
    with col2:
        st.metric("Thắng", wins)
    with col3:
        st.metric("Thua liên tiếp", lose_streak)
    
    if lose_streak >= 3:
        st.markdown(f"""
        <div class="danger-box">
            <div class="warning-text">⚠️ CẢNH BÁO ĐỎ</div>
            <div style="font-size:20px; margin-top:10px;">
                ĐANG THUA {lose_streak} KỲ LIÊN TIẾP<br>
                <span style="color:#ff0000; font-weight:900;">BẮT BUỘC NGHỈ ÍT NHẤT 5 KỲ</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Input
    st.markdown("### 📥 NHẬP KẾT QUẢ MỚI")
    new_input = st.text_area("Dán kết quả kỳ vừa rồi:", height=80, 
                             placeholder="12345")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💾 LƯU", use_container_width=True):
            nums = parse_numbers(new_input)
            if nums:
                data["results"].extend(nums)
                data["results"] = data["results"][-100:]
                
                # Update prediction history with result
                if data.get("predictions"):
                    last_pred = data["predictions"][0] if data["predictions"] else None
                    if last_pred and last_pred.get("ket_qua") != "WIN":
                        last_pred["ket_qua"] = "LOSE"
                        stats["lose_streak"] = stats.get("lose_streak", 0) + 1
                        stats["total_bets"] = stats.get("total_bets", 0) + 1
                
                save_data(data)
                st.success("✅ Đã lưu")
                st.rerun()
    
    with col2:
        if st.button("🎯 DỰ ĐOÁN KỲ TIẾP", use_container_width=True, type="primary"):
            analyzer = SurvivalAnalyzer(data["results"], data.get("predictions", []))
            prediction, reason, confidence = analyzer.get_survival_prediction()
            
            if prediction:
                # Lưu prediction
                pred_record = {
                    "thoi_gian": datetime.now().isoformat(),
                    "du_doan": prediction,
                    "ly_do": reason,
                    "confidence": confidence,
                    "ket_qua": None  # Chưa có kết quả
                }
                data["predictions"].insert(0, pred_record)
                data["predictions"] = data["predictions"][:50]
                
                # Reset lose streak khi có prediction mới
                data["stats"]["lose_streak"] = 0
                
                save_data(data)
                
                st.markdown(f"""
                <div class="safe-box">
                    <div style="font-size:18px;">🎯 DỰ ĐOÁN:</div>
                    <div class="big-number">{prediction[0]} - {prediction[1]}</div>
                    <div style="margin-top:10px;">{reason}</div>
                    <div style="margin-top:10px; font-size:20px;">
                        Confidence: <span style="color:#00ff00; font-weight:900;">{confidence:.0f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="danger-box">
                    <div class="warning-text">⛔ KHÔNG NÊN ĐÁNH</div>
                    <div style="margin-top:10px;">{reason}</div>
                </div>
                """, unsafe_allow_html=True)
    
    with col3:
        if st.button("🗑️ XÓA DATA", use_container_width=True):
            data["results"] = []
            data["predictions"] = []
            data["stats"] = {"lose_streak": 0, "total_bets": 0, "wins": 0}
            save_data(data)
            st.warning("🗑️ Đã xóa")
            st.rerun()
    
    # History
    if data.get("predictions"):
        st.markdown("### 📊 LỊCH SỬ (5 kỳ gần nhất)")
        
        df_data = []
        for pred in data["predictions"][:5]:
            df_data.append({
                "Thời gian": pred.get("thoi_gian", "")[:16].replace("T", " "),
                "Dự đoán": pred.get("du_doan", ""),
                "Lý do": pred.get("ly_do", ""),
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
            <div style="font-size:16px;">TỶ LỆ THẮNG</div>
            <div style="font-size:36px; font-weight:900; color:{'#00ff00' if win_rate >= 40 else '#ff0000'};">
                {win_rate:.1f}% ({wins}/{total_bets})
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div style="text-align:center; color:#666; font-size:10px;">TITAN V52 SURVIVOR | Sinh tồn là ưu tiên</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()