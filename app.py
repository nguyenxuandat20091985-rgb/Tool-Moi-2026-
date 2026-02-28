import streamlit as st
from datetime import datetime
import re

st.set_page_config(page_title="TITAN v35.0 - PRO", layout="centered", page_icon="🎯")

# --- CSS PROFESSIONAL ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
    
    .recommendation-box {
        background: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin: 15px 0;
    }
    
    .title {
        font-size: 1.8em;
        font-weight: 700;
        color: #1a202c;
        text-align: center;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .label {
        font-size: 0.95em;
        font-weight: 600;
        color: #4a5568;
        margin: 12px 0 5px 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .value {
        font-size: 1.4em;
        font-weight: 700;
        margin: 5px 0;
    }
    
    .position {
        color: #ed8936;
    }
    
    .tai {
        color: #e53e3e;
    }
    
    .xiu {
        color: #3182ce;
    }
    
    .bet-amount {
        color: #38a169;
    }
    
    .odds {
        color: #2d3748;
        font-size: 1.1em;
    }
    
    .reason-box {
        background: #fffaf0;
        border: 1px solid #fbd38d;
        border-radius: 8px;
        padding: 12px;
        margin: 15px 0;
        color: #c05621;
        font-weight: 600;
    }
    
    .confidence-bar {
        background: #e2e8f0;
        border-radius: 10px;
        height: 25px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #48bb78 0%, #38a169 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 0.9em;
    }
    
    .countdown {
        background: #fed7d7;
        color: #c53030;
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        font-weight: 700;
        font-size: 1.2em;
        margin: 10px 0;
    }
    
    .btn-win {
        background: #48bb78;
        color: white;
        font-weight: 700;
        padding: 12px;
        border-radius: 8px;
        border: none;
        width: 100%;
        margin: 5px 0;
    }
    
    .btn-lose {
        background: #f56565;
        color: white;
        font-weight: 700;
        padding: 12px;
        border-radius: 8px;
        border: none;
        width: 100%;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 500000
if 'period_count' not in st.session_state:
    st.session_state.period_count = 690
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

# --- HÀM PHÂN TÍCH ---
def analyze_data(raw_text):
    lines = [re.sub(r'[^\d]', '', l.strip()) for l in raw_text.strip().split('\n')]
    valid = [l for l in lines if len(l) == 5]
    
    if len(valid) < 5:
        return None, "Cần ít nhất 5 kỳ hợp lệ"
    
    positions = ["Chục Ngàn", "Ngàn", "Trăm", "Chục", "Đơn Vị"]
    best_pick = None
    best_score = -1
    
    for pos_idx, pos_name in enumerate(positions):
        digits = [int(line[pos_idx]) for line in valid[:10]]
        tai_count = sum(1 for d in digits if d >= 5)
        tai_rate = tai_count / len(digits)
        
        if tai_rate >= 0.7:
            score = tai_rate
            pred = "XỈU"
            reason = f"Bệt Tài {tai_count}/10 → Bẻ cầu"
        elif tai_rate <= 0.3:
            score = 1 - tai_rate
            pred = "TÀI"
            reason = f"Bệt Xỉu {10-tai_count}/10 → Bẻ cầu"
        else:
            continue
        
        if score > best_score:
            best_score = score
            best_pick = {
                "position": pos_name,
                "bet": pred,
                "reason": reason,
                "confidence": int(score * 100),
                "bet_amount": min(20000, int(st.session_state.bankroll * 0.02))
            }
    
    if not best_pick:
        last_digit = int(valid[0][4])
        best_pick = {
            "position": "Đơn Vị",
            "bet": "TÀI" if last_digit < 5 else "XỈU",
            "reason": "Cầu nhảy → Theo kỳ trước ngược",
            "confidence": 55,
            "bet_amount": min(10000, int(st.session_state.bankroll * 0.01))
        }
    
    return best_pick, None

# --- GIAO DIỆN ---
st.title("🎯 TITAN v35.0 - PROFESSIONAL")

# Countdown
now = datetime.now()
seconds = now.second
remaining = 60 - seconds if seconds < 30 else 30 - (seconds - 30)
st.markdown(f'<div class="countdown">🕒 Kỳ tiếp theo: {remaining:02d} giây</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("💰 Quản lý vốn")
    st.session_state.bankroll = st.number_input(
        "Vốn hiện tại (đ)", 
        value=st.session_state.bankroll, 
        step=10000
    )
    recommended_bet = min(20000, int(st.session_state.bankroll * 0.02))
    st.info(f"✅ Cược đề xuất: {recommended_bet:,}đ (2% vốn)")
    
    st.divider()
    st.metric("💵 Vốn hiện tại", f"{st.session_state.bankroll:,.0f}đ")
    
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.period_count = 690
        st.session_state.last_result = None
        st.rerun()

# Form nhập liệu
with st.form("input_form", clear_on_submit=False):
    st.subheader("📥 Nhập kết quả")
    raw = st.text_area(
        "Dán 10 kỳ gần nhất (mới nhất trên cùng):",
        placeholder="87746\n56421\n69137\n...",
        height=150
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        submitted = st.form_submit_button("⚡ Phân tích ngay", type="primary", use_container_width=True)
    with col2:
        st.form_submit_button("🗑️ Xoá", use_container_width=True)

# Xử lý kết quả
if submitted and raw:
    result, error = analyze_data(raw)
    
    if error:
        st.warning(f"⚠️ {error}")
    elif result:
        st.session_state.period_count += 1
        st.session_state.last_result = result
        
        profit = int(result['bet_amount'] * 0.985)
        bet_class = "tai" if result['bet'] == "TÀI" else "xiu"
        
        # Hiển thị kết quả chuyên nghiệp
        st.markdown(f"""
        <div class="main-container">
            <div class="recommendation-box">
                <div class="title">🎯 KHUYẾN NGHỊ KỲ {st.session_state.period_count}</div>
                
                <div class="label">📍 Vị trí:</div>
                <div class="value position">{result['position']}</div>
                
                <div class="label">🔴 Đánh:</div>
                <div class="value {bet_class}">{result['bet']}</div>
                
                <div class="label">💰 Mức cược:</div>
                <div class="value bet-amount">{result['bet_amount']:,}đ</div>
                
                <div class="odds">🎯 Odds: 1.985 → Thắng +{profit:,}đ</div>
                
                <div class="reason-box">
                    📊 {result['reason']}
                </div>
                
                <div class="label">⚡ Độ tin cậy:</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {result['confidence']}%">
                        {result['confidence']}%
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Nút kết quả
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Thắng", type="primary", use_container_width=True, key="win_btn"):
                st.session_state.bankroll += profit
                st.balloons()
                st.success(f"🎉 +{profit:,}đ")
                st.rerun()
        with c2:
            if st.button("❌ Thua", type="secondary", use_container_width=True, key="lose_btn"):
                st.session_state.bankroll -= result['bet_amount']
                st.error(f"💸 -{result['bet_amount']:,}đ")
                st.rerun()
        
        # Cảnh báo
        if st.session_state.bankroll < 400000:
            st.error("🛑 Dừng ngay! Đã mất >20% vốn.")

# Footer
st.markdown("---")
st.caption("🎯 TITAN v35.0 | Phân tích chuyên nghiệp | Chơi có trách nhiệm 🙏")