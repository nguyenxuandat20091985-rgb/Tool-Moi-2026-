import streamlit as st
import google.generativeai as genai
import re
import json
import os
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import time

# ==============================================================================
# 1. CONFIGURATION & STYLING
# ==============================================================================

st.set_page_config(
    page_title="TITAN v33.0 PRO",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for "Dark Theme v22"
st.markdown("""
<style>
    /* Global Variables */
    :root {
        --bg-color: #010409;
        --text-color: #e6edf3;
        --card-bg: #0d1117;
        --card-border: #30363d;
        --num-color: #ff5858;
        --lot-color: #58a6ff;
        --status-green: #238636;
        --status-red: #da3633;
        --status-yellow: #d29922;
    }

    /* Body & Background */
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Prediction Card */
    .prediction-card {
        background-color: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }

    /* Number Boxes */
    .num-box {
        font-size: 70px;
        font-weight: bold;
        color: var(--num-color);
        letter-spacing: 10px;
        text-align: center;
        display: inline-block;
        margin: 0 5px;
    }

    /* Lottery Boxes */
    .lot-box {
        font-size: 50px;
        font-weight: bold;
        color: var(--lot-color);
        letter-spacing: 5px;
        text-align: center;
        display: inline-block;
        margin: 0 5px;
    }

    /* Status Bar */
    .status-bar {
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 15px;
        text-transform: uppercase;
    }
    .status-green { background-color: var(--status-green); color: white; }
    .status-red { background-color: var(--status-red); color: white; }
    .status-yellow { background-color: var(--status-yellow); color: black; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #161b22;
        border-radius: 4px 4px 0px 0px;
        border: 1px solid var(--card-border);
        color: var(--text-color);
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d;
        border-bottom: 2px solid var(--num-color);
    }

    /* Buttons */
    .stButton > button {
        background-color: var(--status-green);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: bold;
        padding: 10px 24px;
    }
    .stButton > button:hover {
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SECRETS & API MANAGEMENT
# ==============================================================================

@st.cache_resource
def init_gemini():
    """Initialize Gemini API safely."""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-pro')
    except KeyError:
        st.error("⚠️ LỖI CẤU HÌNH: Không tìm thấy GEMINI_API_KEY trong secrets.toml")
        return None
    except Exception as e:
        st.error(f"⚠️ LỖI KHỞI TẠO AI: {str(e)}")
        return None

def check_quota():
    """Check and manage daily API quota."""
    if "quota_tracker" not in st.session_state:
        st.session_state.quota_tracker = {
            "count": 0,
            "last_reset": datetime.now().date().isoformat()
        }
    
    tracker = st.session_state.quota_tracker
    today = datetime.now().date().isoformat()
    
    # Reset if new day
    if tracker["last_reset"] != today:
        tracker["count"] = 0
        tracker["last_reset"] = today
    
    return tracker["count"] < 15  # Limit 15 requests/day

def increment_quota():
    if "quota_tracker" in st.session_state:
        st.session_state.quota_tracker["count"] += 1

# ==============================================================================
# 3. DATA CLEANING & MANAGEMENT
# ==============================================================================

def clean_input_data(raw_text, existing_db):
    """
    Clean raw text, extract 5-digit numbers, remove duplicates, 
    and filter out numbers already in DB.
    """
    if not raw_text:
        return [], 0, 0, 0, 0
    
    # Regex to find 5-digit sequences
    matches = re.findall(r'\b\d{5}\b', raw_text)
    
    # Process numbers
    new_entries = []
    seen = set()
    db_set = set(existing_db)
    
    stats = {"found": len(matches), "new": 0, "duplicate": 0, "existing": 0}
    
    for m in matches:
        if m in seen:
            stats["duplicate"] += 1
            continue
        seen.add(m)
        
        if m in db_set:
            stats["existing"] += 1
        else:
            new_entries.append(m)
            stats["new"] += 1
            
    return new_entries, stats["found"], stats["new"], stats["duplicate"], stats["existing"]

def update_database(new_numbers):
    """Add new numbers to session state DB."""
    if "lottery_db" not in st.session_state:
        st.session_state.lottery_db = []
    
    # Prepend new numbers (newest first)
    st.session_state.lottery_db = new_numbers + st.session_state.lottery_db
    
    # Limit to 3000 entries
    if len(st.session_state.lottery_db) > 3000:
        st.session_state.lottery_db = st.session_state.lottery_db[:3000]

# ==============================================================================
# 4. AI ENGINES
# ==============================================================================

def ai_statistical_analysis(data):
    """AI 1: Pure Statistical Analysis with weighted recency."""
    if len(data) < 10:
        return {"error": "Not enough data"}
    
    # Weighted frequency (Recent draws count more)
    weights = np.linspace(1.5, 0.5, len(data))[:len(data)] # Decay weight
    digit_counts = defaultdict(float)
    
    for i, number in enumerate(data):
        weight = weights[i] if i < len(weights) else 0.5
        for digit in number:
            digit_counts[int(digit)] += weight
            
    # Sort by weighted frequency
    sorted_digits = sorted(digit_counts.items(), key=lambda x: x[1], reverse=True)
    
    top_3 = [str(d[0]) for d in sorted_digits[:3]]
    support_4 = [str(d[0]) for d in sorted_digits[3:7]]
    
    return {
        "main_3": top_3,
        "support_4": support_4,
        "confidence": min(95, 40 + len(data)/10), # Base confidence
        "method": "Thống kê trọng số"
    }

def ai_pattern_recognition(data):
    """AI 2: Pattern Recognition (Streaks, Gaps)."""
    if len(data) < 20:
        return {"error": "Not enough data for patterns"}
    
    # Analyze last 20 draws
    recent = data[:20]
    digit_history = defaultdict(list) # digit -> [indices where it appeared]
    
    for idx, num in enumerate(recent):
        for d in num:
            digit_history[int(d)].append(idx)
            
    patterns_found = []
    scores = defaultdict(int)
    
    for digit, indices in digit_history.items():
        score = 0
        # Check for streaks (consecutive)
        for i in range(len(indices)-1):
            if indices[i] - indices[i+1] == 1:
                score += 5 # Streak bonus
                patterns_found.append(f"Số {digit} ra cầu bệt")
        
        # Check for rhythm (gap of 1)
        for i in range(len(indices)-1):
            if indices[i] - indices[i+1] == 2:
                score += 3
                patterns_found.append(f"Số {digit} ra cầu nhịp 2")
        
        scores[digit] = score
        
    sorted_digits = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_3 = [str(d[0]) for d in sorted_digits[:3] if d[1] > 0]
    
    # Fallback if no strong patterns
    if len(top_3) < 3:
        # Fallback to simple frequency for missing slots
        all_digits = [int(d) for num in recent for d in num]
        freq = Counter(all_digits)
        missing = [str(d) for d, c in freq.most_common(10) if str(d) not in top_3]
        top_3.extend(missing[:3-len(top_3)])
        
    return {
        "main_3": top_3[:3],
        "support_4": [str(d[0]) for d in sorted_digits[3:7]],
        "confidence": 60 if patterns_found else 40,
        "patterns": patterns_found[:3],
        "method": "Nhận diện mẫu hình"
    }

def ai_gemini_analysis(data, model):
    """AI 3: Gemini Deep Analysis."""
    if not model:
        return {"error": "AI Model not initialized"}
    
    if not check_quota():
        return {"error": "Quota exceeded", "fallback": True}
    
    try:
        # Prepare context (last 50 draws)
        context_data = data[:50]
        prompt = f"""
        Phân tích dãy số xổ số 5 chữ số sau (mới nhất ở đầu):
        {', '.join(context_data)}
        
        Nhiệm vụ: Dự đoán 3 con số (0-9) có khả năng xuất highest nhất trong kết quả tiếp theo.
        Yêu cầu trả về JSON STRICT format:
        {{
            "main_3": ["1", "2", "3"],
            "support_4": ["4", "5", "6", "7"],
            "decision": "MUA" or "DỪNG",
            "confidence": 0-100,
            "logic": "Giải thích ngắn gọn bằng tiếng Việt",
            "method": "Gemini Deep Learning"
        }}
        Chỉ trả về JSON, không thêm markdown.
        """
        
        response = model.generate_content(prompt)
        text = response.text
        
        # Robust JSON parsing
        try:
            # Try direct load
            result = json.loads(text)
        except:
            # Try regex extraction for code block
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                result = json.loads(match.group())
            else:
                raise ValueError("Invalid JSON format from AI")
                
        increment_quota()
        return result
        
    except Exception as e:
        return {"error": str(e), "fallback": True}

def consensus_engine(results):
    """Combine results from all AIs."""
    valid_results = [r for r in results if "error" not in r or r.get("fallback")]
    
    if not valid_results:
        return None
        
    all_suggestions = []
    total_confidence = 0
    
    for res in valid_results:
        if "main_3" in res:
            all_suggestions.extend(res["main_3"])
            total_confidence += res.get("confidence", 50)
            
    if not all_suggestions:
        return None
        
    # Vote counting
    votes = Counter(all_suggestions)
    # Get top 3 most voted
    top_3 = [item for item, count in votes.most_common(3)]
    
    # Support 4: Next most common excluding top 3
    remaining = [item for item, count in votes.most_common() if item not in top_3]
    support_4 = remaining[:4]
    
    avg_conf = int(total_confidence / len(valid_results))
    
    return {
        "main_3": top_3,
        "support_4": support_4,
        "confidence": avg_conf,
        "method": "Consensus (3 AI)"
    }

# ==============================================================================
# 5. RISK & LOGIC
# ==============================================================================

def calculate_risk(data):
    """Calculate Risk Score 0-100."""
    if len(data) < 20:
        return 50, "Thiếu dữ liệu"
        
    recent = data[:20]
    digit_freq = Counter([d for num in recent for d in num])
    
    risk_score = 0
    reasons = []
    
    # 1. Over-representation (>15 times in 20 draws = 75% appearance rate per digit slot roughly)
    # Max possible appearances for a digit in 20 draws of 5 digits = 100 slots.
    # If a digit appears > 25 times (25% of all slots), it's hot.
    for digit, count in digit_freq.items():
        if count > 25: # Very hot
            risk_score += 20
            reasons.append(f"Số {digit} quá nóng ({count} lần)")
            
    # 2. Entropy Check (Too uniform = suspicious/random noise)
    counts = list(digit_freq.values())
    if counts:
        std_dev = np.std(counts)
        if std_dev < 2: # Too uniform
            risk_score += 30
            reasons.append("Phân phối quá đều (Ngẫu nhiên cao)")
            
    # 3. Streak detection
    # (Simplified for this demo)
    
    risk_score = min(100, risk_score)
    
    if risk_score >= 60:
        status = "DỪNG"
        color_class = "status-red"
    elif risk_score >= 40:
        status = "THEO DÕI"
        color_class = "status-yellow"
    else:
        status = "ĐÁNH"
        color_class = "status-green"
        
    return risk_score, status, color_class, reasons

# ==============================================================================
# 6. UI COMPONENTS
# ==============================================================================

def render_prediction_card(result, risk_info):
    risk_score, status, color_class, reasons = risk_info
    
    if not result or "main_3" not in result:
        st.warning("Chưa có đủ dữ liệu để dự đoán.")
        return

    main_3 = result["main_3"]
    support_4 = result.get("support_4", ["?", "?", "?", "?"])
    
    # Ensure lists have correct length for display
    while len(main_3) < 3: main_3.append("?")
    while len(support_4) < 4: support_4.append("?")

    st.markdown(f"""
    <div class="prediction-card">
        <div class="status-bar {color_class}">
            RISK SCORE: {risk_score}/100 | KHUYẾN NGHỊ: {status}
        </div>
        <div style="text-align:center; margin-bottom:10px; color:#8b949e; font-size:14px;">
            {result.get('method', 'Phương pháp không xác định')} - Độ tin cậy: {result.get('confidence', 0)}%
        </div>
        <div style="text-align:center;">
            <span class="num-box">{main_3[0]}</span>
            <span class="num-box">{main_3[1]}</span>
            <span class="num-box">{main_3[2]}</span>
        </div>
        <div style="text-align:center; margin-top:10px; border-top:1px solid #30363d; padding-top:10px;">
            <span style="font-size:14px; color:#8b949e;">SỐ LÓT:</span><br>
            <span class="lot-box">{support_4[0]}</span>
            <span class="lot-box">{support_4[1]}</span>
            <span class="lot-box">{support_4[2]}</span>
            <span class="lot-box">{support_4[3]}</span>
        </div>
        <div style="margin-top:15px; font-size:13px; color:#8b949e; background:#010409; padding:10px; border-radius:6px;">
            <strong>Logic:</strong> {result.get('logic', 'Phân tích tổng hợp từ 3 nguồn AI.')}
            {f"<br><em>Cảnh báo: {', '.join(reasons)}</em>" if reasons else ""}
        </div>
        <div style="text-align:right; margin-top:10px;">
            <button style="background:none; border:1px solid #30363d; color:#e6edf3; padding:5px 10px; border-radius:4px; cursor:pointer;" 
            onclick="navigator.clipboard.writeText('{','.join(main_3 + support_4)}')">
                📋 Copy Dàn 7 Số
            </button>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_stats_tab():
    st.header("📊 THỐNG KÊ & DATABASE")
    
    if "lottery_db" not in st.session_state or not st.session_state.lottery_db:
        st.info("Chưa có dữ liệu trong database.")
        return

    db = st.session_state.lottery_db
    
    # 1. Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Tần suất 50 kỳ gần")
        recent_50 = db[:50]
        all_digits = [d for num in recent_50 for d in num]
        freq = Counter(all_digits)
        df_freq = pd.DataFrame(list(freq.items()), columns=['Số', 'Tần suất']).sort_values('Tần suất')
        st.bar_chart(df_freq.set_index('Số'))
        
    with col2:
        st.subheader("Top Số Nóng (Toàn bộ)")
        all_digits_full = [d for num in db for d in num]
        freq_full = Counter(all_digits_full)
        top_hot = freq_full.most_common(5)
        hot_html = "<div style='display:flex; gap:10px;'>"
        for num, count in top_hot:
            hot_html += f"<div style='background:#21262d; padding:10px; border-radius:8px; text-align:center;'><div style='font-size:24px; color:#ff5858;'>{num}</div><div style='font-size:12px; color:#8b949e;'>{count} lần</div></div>"
        hot_html += "</div>"
        st.markdown(hot_html, unsafe_allow_html=True)

    # 2. Data Management
    st.divider()
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.download_button(
            label="📥 Download Database (JSON)",
            data=json.dumps(db, indent=2),
            file_name="titan_v33_db.json",
            mime="application/json"
        )
        
    with c2:
        uploaded_file = st.file_uploader("📤 Upload Database", type="json")
        if uploaded_file:
            try:
                new_db = json.load(uploaded_file)
                if isinstance(new_db, list):
                    st.session_state.lottery_db = new_db
                    st.success("Đã tải database thành công!")
                    st.rerun()
                else:
                    st.error("File JSON không hợp lệ (phải là danh sách).")
            except Exception as e:
                st.error(f"Lỗi file: {e}")
                
    with c3:
        if st.button("🗑️ XÓA TOÀN BỘ DỮ LIỆU", type="primary"):
            st.session_state.lottery_db = []
            st.success("Đã xóa dữ liệu!")
            st.rerun()

    # 3. Recent History Table
    st.subheader("Lịch sử 20 kỳ gần nhất")
    df_hist = pd.DataFrame(st.session_state.lottery_db[:20], columns=["Kết Quả"])
    df_hist.index = range(1, len(df_hist)+1)
    st.dataframe(df_hist, use_container_width=True, hide_index=True)

# ==============================================================================
# 7. MAIN APP LOGIC
# ==============================================================================

def main():
    # Initialize Session State
    if "lottery_db" not in st.session_state:
        st.session_state.lottery_db = []
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "risk_info" not in st.session_state:
        st.session_state.risk_info = None

    # Header
    st.title("🔮 TITAN v33.0 PRO")
    st.caption("Multi-AI Lottery Prediction System | 3-Number Strategy")
    
    # Quota Display
    if "quota_tracker" in st.session_state:
        q = st.session_state.quota_tracker
        st.sidebar.info(f"🤖 Gemini Quota: {15 - q['count']}/15 còn lại hôm nay")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📝 NHẬP DỮ LIỆU", "🎯 KẾT QUẢ", "📊 THỐNG KÊ"])
    
    # --- TAB 1: INPUT ---
    with tab1:
        st.header("Nhập Kết Quả Xổ Số")
        st.markdown("Dán kết quả (5 số/dòng). Hệ thống tự động làm sạch và phân tích.")
        
        input_data = st.text_area(
            "Dữ liệu thô", 
            height=200, 
            placeholder="Ví dụ:\n12345\n67890\n54321\n..."
        )
        
        col_btn, col_info = st.columns([1, 3])
        
        with col_btn:
            process_btn = st.button("THÊM & PHÂN TÍCH TỰ ĐỘNG", type="primary", use_container_width=True)
            
        if process_btn and input_data:
            with st.spinner("Đang xử lý dữ liệu và gọi AI..."):
                # 1. Clean Data
                new_nums, f, n, d, e = clean_input_data(input_data, st.session_state.lottery_db)
                
                if n > 0:
                    update_database(new_nums)
                    st.success(f"✅ Đã thêm {n} số mới. (Tìm thấy: {f}, Trùng trong input: {d}, Có trong DB: {e})")
                    
                    # 2. Run AI Engines
                    model = init_gemini()
                    
                    res_stat = ai_statistical_analysis(st.session_state.lottery_db)
                    res_pat = ai_pattern_recognition(st.session_state.lottery_db)
                    res_gem = ai_gemini_analysis(st.session_state.lottery_db, model) if model else {"error": "No Model"}
                    
                    # 3. Consensus
                    final_result = consensus_engine([res_stat, res_pat, res_gem])
                    
                    # 4. Risk Calculation
                    risk_info = calculate_risk(st.session_state.lottery_db)
                    
                    # Save to state
                    st.session_state.analysis_result = final_result
                    st.session_state.risk_info = risk_info
                    
                    # Switch to Tab 2 automatically (via session state logic or just inform user)
                    st.info("Phân tích hoàn tất! Chuyển sang tab 'KẾT QUẢ' để xem.")
                else:
                    st.warning("Không có số mới nào được thêm vào. Kiểm tra lại dữ liệu.")
                    
        elif process_btn and not input_data:
            st.error("Vui lòng nhập dữ liệu!")

    # --- TAB 2: RESULTS ---
    with tab2:
        st.header("Dự Đoán Thông Minh")
        
        if st.session_state.analysis_result:
            render_prediction_card(st.session_state.analysis_result, st.session_state.risk_info)
            
            # Detailed AI Breakdown (Accordion)
            with st.expander("🔍 Chi tiết phân tích từ các AI"):
                st.json(st.session_state.analysis_result)
                if "quota_tracker" in st.session_state:
                    st.write(f"**Trạng thái Gemini:** {'Hoạt động' if st.session_state.quota_tracker['count'] < 15 else 'Hết Quota'}")
        else:
            st.info("Chưa có kết quả phân tích. Vui lòng nhập dữ liệu ở Tab 1.")

    # --- TAB 3: STATS ---
    with tab3:
        render_stats_tab()

if __name__ == "__main__":
    main()