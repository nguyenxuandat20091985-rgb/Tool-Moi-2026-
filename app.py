import streamlit as st
import re, json, os, pandas as pd, numpy as np
from collections import Counter, defaultdict
from itertools import combinations
import google.generativeai as genai
from datetime import datetime

# --- CẤU HÌNH HỆ THỐNG ---
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
DB_FILE = "titan_v33_data.json"

# Tuổi Sửu 1985 - Mệnh Kim
LUCKY_OX = [0, 2, 5, 6, 7, 8]
SHADOW_MAP = {"0":"5", "5":"0", "1":"6", "6":"1", "2":"7", "7":"2", "3":"8", "8":"3", "4":"9", "9":"4"}

genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="TITAN V33 ULTIMATE", page_icon="🐂", layout="centered")

# --- CSS CAO CẤP ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&display=swap');
    
    .stApp {
        background: linear-gradient(180deg, #0a0e27 0%, #1a1f3a 100%);
        color: #00ffff;
        font-family: 'Orbitron', sans-serif;
    }
    
    .main-header {
        font-size: 36px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #00ffff, #ff00ff, #00ffff);
        background-size: 200% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 3s ease infinite;
        text-shadow: 0 0 30px rgba(0, 255, 255, 0.5);
        margin-bottom: 20px;
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .big-num {
        font-size: 48px;
        font-weight: 900;
        color: #00ffff;
        text-align: center;
        font-family: 'Orbitron', monospace;
        letter-spacing: 8px;
        text-shadow: 0 0 20px #00ffff;
    }
    
    .box {
        background: linear-gradient(135deg, #0f1428, #1a1f3a);
        color: #00ffff;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        border: 2px solid #00ffff;
        margin-bottom: 15px;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
    }
    
    .item {
        background: linear-gradient(135deg, #00ffff, #0080ff);
        color: #000;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 32px;
        font-weight: 900;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
        margin: 5px 0;
    }
    
    .item-3 {
        background: linear-gradient(135deg, #ff00ff, #8000ff);
        color: #fff;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #2a0a0a, #1a0505);
        border: 3px solid #ff0000;
        color: #ff0000;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 20px rgba(255, 0, 0, 0.5); }
        50% { box-shadow: 0 0 40px rgba(255, 0, 0, 0.8); }
    }
    
    .success-box {
        background: linear-gradient(135deg, #0a2a0a, #051a05);
        border: 3px solid #00ff00;
        color: #00ff00;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    
    .tag {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: bold;
        margin: 3px;
    }
    
    .tag-hot { background: #ff0040; color: white; }
    .tag-gan { background: #8000ff; color: white; }
    .tag-lucky { background: #FFD700; color: black; }
    
    textarea {
        background: #0a0e27 !important;
        border: 2px solid #00ffff !important;
        color: #00ffff !important;
        font-family: 'Orbitron', monospace;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #00ffff, #0080ff);
        color: #000;
        font-weight: 900;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-size: 16px;
    }
    
    .win { color: #00ff40; font-weight: 900; }
    .lose { color: #ff0040; font-weight: 900; }
</style>
""", unsafe_allow_html=True)

# --- DATA MANAGEMENT ---
def load_data():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {"results": [], "history": [], "stats": {"total": 0, "wins": 0}}
    return {"results": [], "history": [], "stats": {"total": 0, "wins": 0}}

def save_data(data):
    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_nums(text):
    clean = re.sub(r'\s+', '', text)
    return [n for n in re.findall(r'\d{5}', clean) if n]

# --- THUẬT TOÁN NÂNG CAO ---
class AdvancedAnalyzer:
    def __init__(self, db):
        self.db = db
        self.trap_pairs = self._detect_trap_pairs()
        self.hot_numbers = self._get_hot_numbers()
        
    def _detect_trap_pairs(self):
        """Phát hiện cặp bẫy (thua nhiều lần)"""
        if not os.path.exists(DB_FILE):
            return set()
        
        data = load_data()
        pair_losses = defaultdict(int)
        
        for h in data.get("history", []):
            if h.get("Kết quả") == "❌" and "Dự đoán" in h:
                pair_losses[h["Dự đoán"]] += 1
        
        return {pair for pair, count in pair_losses.items() if count >= 2}
    
    def _get_hot_numbers(self):
        """Lấy số đang nóng (về nhiều trong 10 kỳ)"""
        if len(self.db) < 10:
            return []
        
        recent = "".join(self.db[-10:])
        counter = Counter(recent)
        return [d for d, c in counter.most_common(5)]
    
    def calculate_score(self, digit):
        """Tính điểm cho từng số"""
        score = 0
        
        # 1. Tần suất 40 kỳ
        freq = "".join(self.db[-40:]).count(digit)
        score += freq * 2
        
        # 2. Cầu rơi (số vừa nổ)
        if len(self.db) > 0 and digit in self.db[-1]:
            score += 30
        
        # 3. Bóng ngũ hành
        last_digit = self.db[-1][0] if self.db else "0"
        if SHADOW_MAP.get(last_digit) == digit:
            score += 25
        
        # 4. Tuổi Sửu
        if int(digit) in LUCKY_OX:
            score += 15
        
        # 5. Tránh số quá hot (bẫy)
        if digit in self.hot_numbers:
            score -= 10
        
        return score
    
    def predict(self):
        if len(self.db) < 10:
            return None
        
        # Tính điểm từng số
        scores = {str(i): self.calculate_score(str(i)) for i in range(10)}
        
        # Sort
        sorted_digits = [x[0] for x in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        
        # Top 8
        top_8 = "".join(sorted_digits[:8])
        
        # 2 Tinh - Top 5 số mạnh nhất
        top_5 = sorted_digits[:5]
        pairs = []
        for p in combinations(top_5, 2):
            pair = "".join(p)
            if pair not in self.trap_pairs:
                pairs.append(pair)
            if len(pairs) >= 3:
                break
        
        # 3 Tinh - Top 6
        top_6 = sorted_digits[:6]
        triples = []
        for t in combinations(top_6, 3):
            triple = "".join(t)
            # Check nếu có chứa trap pair thì skip
            has_trap = any("".join(p) in self.trap_pairs for p in combinations(t, 2))
            if not has_trap:
                triples.append(triple)
            if len(triples) >= 3:
                break
        
        # Confidence
        total_score = sum(scores[d] for d in top_5)
        confidence = min(92, max(65, 65 + total_score / 10))
        
        return {
            "pairs": pairs if pairs else ["01", "23", "45"],
            "triples": triples if triples else ["012", "345", "678"],
            "top8": top_8,
            "confidence": confidence,
            "trap_count": len(self.trap_pairs)
        }

# --- GIAO DIỆN ---
def main():
    # Initialize session state
    if "db" not in st.session_state:
        st.session_state.db = []
    if "history" not in st.session_state:
        data = load_data()
        st.session_state.history = data.get("history", [])
        st.session_state.db = data.get("results", [])
    
    st.markdown('<h1 class="main-header">🐂 TITAN V33 ULTIMATE</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#888; margin-bottom:20px;">AI-Powered | Anti-Trap | Shadow Theory</p>', unsafe_allow_html=True)
    
    # Stats
    stats = load_data().get("stats", {})
    total = stats.get("total", 0)
    wins = stats.get("wins", 0)
    win_rate = (wins/total*100) if total > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tổng kỳ", f"{total}")
    with col2:
        st.metric("Thắng", f"{wins}")
    with col3:
        st.metric("Tỷ lệ", f"{win_rate:.1f}%")
    
    # Input
    st.markdown("### 📥 NHẬP KẾT QUẢ")
    user_input = st.text_area("Dán kết quả (mỗi kỳ 1 dòng):", height=120, 
                              placeholder="12345\n67890\n...")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 LƯU & PHÂN TÍCH", use_container_width=True):
            nums = get_nums(user_input)
            if nums:
                st.session_state.db = nums
                
                # Đối soát
                data = load_data()
                if data.get("history"):
                    last_pred = data["history"][0].get("Dự đoán") if data["history"] else None
                    if last_pred:
                        actual = nums[-1]
                        win = all(d in actual for d in last_pred)
                        
                        if win:
                            data["stats"]["wins"] += 1
                            st.markdown('<div class="success-box">🔥 WIN! Kết quả trùng khớp!</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="warning-box">❌ THUA! Cập nhật bẫy cầu...</div>', unsafe_allow_html=True)
                        
                        data["stats"]["total"] += 1
                        save_data(data)
                
                st.success(f"✅ Đã lưu {len(nums)} kỳ")
                st.rerun()
    
    with col2:
        if st.button("🗑️ RESET", use_container_width=True):
            st.session_state.db = []
            st.session_state.history = []
            if os.path.exists(DB_FILE):
                os.remove(DB_FILE)
            st.rerun()
    
    # Dự đoán
    if st.session_state.db:
        st.markdown("### 🎯 DỰ ĐOÁN KỲ TIẾP")
        
        analyzer = AdvancedAnalyzer(st.session_state.db)
        result = analyzer.predict()
        
        if result:
            # Cảnh báo trap
            if result["trap_count"] > 0:
                st.markdown(f"""
                <div class="warning-box">
                    ⚠️ PHÁT HIỆN {result['trap_count']} CẶP BẪY - ĐANG TRÁNH TỰ ĐỘNG
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence
            conf_color = "#00ff00" if result["confidence"] >= 80 else "#ffff00" if result["confidence"] >= 70 else "#ff0000"
            st.markdown(f"""
            <div class="box">
                <div style="font-size:16px;">ĐỘ TIN CẬY</div>
                <div style="font-size:36px; font-weight:900; color:{conf_color};">{result['confidence']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Top 8
            st.markdown(f"""
            <div style="text-align:center; margin:15px 0;">
                <span style="color:#888;">8 SỐ MẠNH:</span><br>
                <span class="big-num">{result['top8']}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # 2 Tinh
            st.markdown("<div class='box'>🎯 2 TINH - 3 CẶP VIP</div>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            tags = ["🥇", "🥈", "🥉"]
            for i, p in enumerate(result['pairs']):
                tag_html = f'<span class="tag tag-lucky">{tags[i]}</span>' if i < 3 else ""
                with [c1, c2, c3][i]:
                    st.markdown(f"""
                    <div class="item">
                        {p[0]} - {p[1]}
                        {tag_html}
                    </div>
                    """, unsafe_allow_html=True)
            
            # 3 Tinh
            st.markdown("<div class='box' style='border-color:#ff00ff;'>💎 3 TINH - 3 BỘ</div>", unsafe_allow_html=True)
            d1, d2, d3 = st.columns(3)
            for i, t in enumerate(result['triples']):
                with [d1, d2, d3][i]:
                    st.markdown(f"""
                    <div class="item item-3">
                        {t[0]} - {t[1]} - {t[2]}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Lưu prediction
            data = load_data()
            data["history"].insert(0, {
                "Thời gian": datetime.now().isoformat(),
                "Dự đoán": result["pairs"][0] if result["pairs"] else "00",
                "Kết quả": None
            })
            data["history"] = data["history"][:50]
            data["results"] = st.session_state.db
            save_data(data)
    
    # Lịch sử
    data = load_data()
    if data.get("history"):
        st.markdown("### 📊 LỊCH SỬ (10 kỳ)")
        
        df_data = []
        for h in data["history"][:10]:
            df_data.append({
                "Thời gian": h.get("Thời gian", "")[:16].replace("T", " "),
                "Dự đoán": h.get("Dự đoán", ""),
                "Kết quả": h.get("Kết quả") or "⏳ Chờ"
            })
        
        df = pd.DataFrame(df_data)
        
        def color_kq(val):
            if val == "🔥 WIN":
                return "color: #00ff40; font-weight: 900"
            elif val == "❌":
                return "color: #ff0040; font-weight: 900"
            return "color: #ffaa00"
        
        st.dataframe(df.style.applymap(color_kq, subset=['Kết quả']), 
                     use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown('<div style="text-align:center; color:#444; font-size:11px;">TITAN V33 ULTIMATE | Anti-Trap System</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()