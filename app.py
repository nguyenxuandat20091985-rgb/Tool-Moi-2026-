import streamlit as st
import re, json, os, pandas as pd, numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from openai import OpenAI
import google.generativeai as genai
import math
import time

# --- CẤU HÌNH HỆ THỐNG ---
NVIDIA_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyD1-XMO6FsA9ZgAf2P6nIiXLPp8moTPMrc"
DB_FILE = "titan_v39_data.json"

# Tuổi Sửu 1985 - Mệnh Kim
LUCKY_OX = [0, 2, 5, 6, 7, 8]
SHADOW_MAP = {"0":"5", "5":"0", "1":"6", "6":"1", "2":"7", "7":"2", "3":"8", "8":"3", "4":"9", "9":"4"}

st.set_page_config(page_title="TITAN V39 AI PRO", page_icon="🐂🤖", layout="centered")

# --- GIAO DIỆN ---
st.markdown("""
<style>
    .stApp {background-color: #050505; color: #FFD700;}
    .big-num {font-size: 48px; font-weight: bold; color: #00FFCC; text-align: center; font-family: monospace; letter-spacing: 8px;}
    .box {background: linear-gradient(135deg, #1C3A3A, #050505); color: #FFD700; padding: 15px; border-radius: 12px; text-align: center; border: 2px solid #FFD700; margin-bottom: 10px;}
    .item {background: linear-gradient(135deg, #00FFCC, #008B8B); color: #000; padding: 15px; border-radius: 8px; text-align: center; font-size: 28px; font-weight: bold;}
    .item-3 {background: linear-gradient(135deg, #FFD700, #FFA500); color: #000;}
    .win {background: rgba(0,255,0,0.3); border: 2px solid #00FF00;}
    .lose {background: rgba(255,0,0,0.3); border: 2px solid #FF0000;}
</style>
""", unsafe_allow_html=True)

# --- LOGIC ---
def get_nums(text):
    return [n for n in re.findall(r"\d{5}", text) if n and len(n)==5]

def save_data(db):
    try:
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
    except:
        pass

def load_data():
    try:
        if os.path.exists(DB_FILE):
            with open(DB_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except:
        pass
    return []

def ai_predict_with_gemini(db, last_results=None):
    """SỬ DỤNG GEMINI AI ĐỂ DỰ ĐOÁN"""
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Chuẩn bị dữ liệu
        recent_50 = db[-50:] if len(db) >= 50 else db
        
        prompt = f"""
Phân tích xổ số 5D (5 chữ số). Dữ liệu {len(recent_50)} kỳ gần nhất:
{chr(10).join(recent_50)}

Nhiệm vụ:
1. Phân tích tần suất từng số 0-9
2. Tìm pattern/cầu đang chạy
3. Dự đoán 8 số có khả năng ra cao nhất
4. Chọn 3 cặp 2 số (2-tinh) mạnh nhất
5. Chọn 3 bộ 3 số (3-tinh) mạnh nhất

Luật 5-tinh: Thắng nếu TẤT CẢ số trong cặp/bộ XUẤT HIỆN trong kết quả (bất kỳ vị trí nào)

Trả về JSON format:
{{
  "top_8_digits": ["1","2","3","4","5","6","7","8"],
  "pairs_2tinh": ["12", "34", "56"],
  "triples_3tinh": ["123", "456", "789"],
  "reasoning": "Giải thích ngắn gọn tại sao chọn các số này"
}}

Lưu ý cho tuổi Sửu mệnh Kim 1985: Ưu tiên số 0,2,5,6,7,8 nếu có thể.
"""
        
        response = model.generate_content(prompt)
        
        # Extract JSON from response
        import re as regex
        json_match = regex.search(r'\{[\s\S]*\}', response.text)
        if json_match:
            result = json.loads(json_match.group())
            return result
        else:
            return None
            
    except Exception as e:
        st.error(f"AI Error: {str(e)[:100]}")
        return None

def statistical_predict(db):
    """THỐNG KÊ THUẦN TÚY (fallback)"""
    if len(db) < 10:
        return None
    
    # Tần suất
    all_digits = "".join(db[-40:])
    freq = Counter(all_digits)
    
    # Top 8
    top_8 = [d for d, c in freq.most_common(8)]
    
    # Pairs
    pair_counts = Counter()
    for num in db[-30:]:
        unique = sorted(set(num))
        for p in combinations(unique, 2):
            pair_counts["".join(p)] += 1
    
    top_pairs = [p for p, c in pair_counts.most_common(3)]
    
    # Triples
    triple_counts = Counter()
    for num in db[-30:]:
        unique = sorted(set(num))
        for t in combinations(unique, 3):
            triple_counts["".join(t)] += 1
    
    top_triples = [t for t, c in triple_counts.most_common(3)]
    
    return {
        "top_8_digits": top_8,
        "pairs_2tinh": top_pairs if top_pairs else ["01", "23", "45"],
        "triples_3tinh": top_triples if top_triples else ["012", "345", "678"],
        "reasoning": "Thống kê tần suất thuần túy"
    }

def check_win_2tinh(pair, result):
    """KIỂM TRA THẮNG 2-TINH: Cả 2 số phải có trong result"""
    if len(pair) != 2:
        return False
    return pair[0] in result and pair[1] in result

def check_win_3tinh(triple, result):
    """KIỂM TRA THẮNG 3-TINH: Cả 3 số phải có trong result"""
    if len(triple) != 3:
        return False
    return triple[0] in result and triple[1] in result and triple[2] in result

# --- GIAO DIỆN CHÍNH ---
st.markdown('<h1>🐂🤖 TITAN V39 AI PRO</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#888;font-size:12px;">Gemini AI + Statistical Analysis + Tuổi Sửu</p>', unsafe_allow_html=True)

if "db" not in st.session_state:
    st.session_state.db = load_data()
if "history" not in st.session_state:
    st.session_state.history = []
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None

user_input = st.text_area("📥 Dán kết quả (tối thiểu 30 kỳ, càng nhiều càng tốt):", 
                          height=150, 
                          placeholder="19626\n34479\n37882\n...")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🤖 AI DỰ ĐOÁN"):
        nums = get_nums(user_input)
        if len(nums) >= 30:
            st.session_state.db = nums[-100:]  # Giữ 100 kỳ
            save_data(st.session_state.db)
            
            with st.spinner("🤖 AI đang phân tích..."):
                # Thử AI trước
                ai_result = ai_predict_with_gemini(st.session_state.db)
                
                if ai_result:
                    st.session_state.last_pred = ai_result
                    st.success("✅ AI đã phân tích xong!")
                else:
                    # Fallback statistical
                    st.session_state.last_pred = statistical_predict(st.session_state.db)
                    st.warning("⚠️ AI lỗi, dùng thống kê thuần túy")
                
                # Đối soát nếu có kết quả mới
                if st.session_state.history and len(nums) > 1:
                    last_actual = nums[-1]
                    pred = st.session_state.last_pred
                    
                    # Check 2-tinh
                    win_2 = any(check_win_2tinh(p, last_actual) for p in pred.get("pairs_2tinh", []))
                    # Check 3-tinh
                    win_3 = any(check_win_3tinh(t, last_actual) for t in pred.get("triples_3tinh", []))
                    
                    st.session_state.history.insert(0, {
                        "Kỳ": last_actual,
                        "2-tinh": "✅" if win_2 else "❌",
                        "3-tinh": "✅" if win_3 else "❌"
                    })
                
                st.rerun()
        else:
            st.error(f"❌ Chỉ có {len(nums)} kỳ. Cần tối thiểu 30 kỳ!")

with col2:
    if st.button("📊 THỐNG KÊ"):
        st.session_state.show_stats = not st.session_state.get("show_stats", False)

with col3:
    if st.button("🗑️ XÓA"):
        st.session_state.clear()
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
        st.rerun()

# --- HIỂN THỊ KẾT QUẢ ---
if st.session_state.last_pred:
    pred = st.session_state.last_pred
    
    st.markdown("---")
    st.markdown('<div class="box">🎯 KẾT QUẢ DỰ ĐOÁN</div>', unsafe_allow_html=True)
    
    # Reasoning
    if pred.get("reasoning"):
        st.info(f"🧠 **AI phân tích:** {pred['reasoning']}")
    
    # Top 8
    top8 = pred.get("top_8_digits", [])
    st.markdown(f"<div class='box'>🔥 TOP 8 SỐ MẠNH: <span class='big-num'>{','.join(top8)}</span></div>", unsafe_allow_html=True)
    
    # 2-TINH
    st.markdown("<div class='box'>🎯 2 SỐ 5 TINH (Thắng nếu CẢ 2 số xuất hiện)</div>", unsafe_allow_html=True)
    pairs = pred.get("pairs_2tinh", [])
    c1, c2, c3 = st.columns(3)
    for i, pair in enumerate(pairs[:3]):
        with [c1, c2, c3][i]:
            st.markdown(f"<div class='item'>{pair[0]}-{pair[1]}</div>", unsafe_allow_html=True)
    
    # 3-TINH
    st.markdown("<div class='box' style='border-color:#FFD700;'>💎 3 SỐ 5 TINH (Thắng nếu CẢ 3 số xuất hiện)</div>", unsafe_allow_html=True)
    triples = pred.get("triples_3tinh", [])
    d1, d2, d3 = st.columns(3)
    for i, triple in enumerate(triples[:3]):
        with [d1, d2, d3][i]:
            st.markdown(f"<div class='item item-3'>{triple[0]}-{triple[1]}-{triple[2]}</div>", unsafe_allow_html=True)

# --- THỐNG KÊ ---
if st.session_state.get("show_stats", False):
    nums = get_nums(user_input)
    if nums:
        st.divider()
        st.subheader("📊 THỐNG KÊ CHI TIẾT")
        
        all_digits = "".join(nums)
        freq = Counter(all_digits)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**🔥 Số nóng (hay ra):**")
            for d, c in freq.most_common(5):
                st.write(f"• Số {d}: {c} lần")
        
        with col_b:
            st.markdown("**❄️ Số lạnh (hiếm ra):**")
            for d in range(10):
                if str(d) not in freq:
                    st.write(f"• Số {d}: 0 lần")

# --- LỊCH SỬ ---
if st.session_state.history:
    st.divider()
    st.subheader("📋 LỊCH SỬ ĐỐI SOÁT")
    
    total = len(st.session_state.history)
    win_2 = sum(1 for h in st.session_state.history if h.get("2-tinh") == "✅")
    win_3 = sum(1 for h in st.session_state.history if h.get("3-tinh") == "✅")
    
    rate_2 = (win_2 / total * 100) if total > 0 else 0
    rate_3 = (win_3 / total * 100) if total > 0 else 0
    
    st.markdown(f"""
    <div class='box'>
    <b>📊 TỶ LỆ THẮNG:</b><br>
    🎯 2-tinh: {win_2}/{total} = <span style="color:{'#00FF00' if rate_2>=40 else '#FFA500' if rate_2>=25 else '#FF0000'};font-size:24px;">{rate_2:.1f}%</span><br>
    💎 3-tinh: {win_3}/{total} = <span style="color:{'#00FF00' if rate_3>=25 else '#FFA500' if rate_3>=15 else '#FF0000'};font-size:24px;">{rate_3:.1f}%</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Bảng
    df = pd.DataFrame(st.session_state.history[:20])
    
    # Color coding
    def color_val(val):
        if val == "✅":
            return "color: #00FF00; font-weight: bold"
        else:
            return "color: #FF0000"
    
    styled_df = df.style.applymap(color_val, subset=["2-tinh", "3-tinh"])
    st.dataframe(styled_df, use_container_width=True)

st.markdown('<div style="text-align:center;color:#666;font-size:10px;margin-top:20px;">🐂🤖 TITAN V39 AI PRO - Gemini Powered</div>', unsafe_allow_html=True)