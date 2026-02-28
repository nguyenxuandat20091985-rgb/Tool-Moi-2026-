import streamlit as st
from datetime import datetime
import time
import re

st.set_page_config(page_title="TITAN v35.0 - PRO", layout="centered", page_icon="🎯")

# --- CSS TỐI ƯU ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin: 20px 0;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .result-box {
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 20px 0;
        border: 3px solid #1E88E5;
    }
    .position-text {
        font-size: 1.5em;
        font-weight: bold;
        color: #FF6F00;
        text-align: center;
        margin: 15px 0;
    }
    .bet-text {
        font-size: 4em;
        font-weight: bold;
        text-align: center;
        margin: 20px 0;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.2);
    }
    .tai-text {
        color: #E53935;
    }
    .xiu-text {
        color: #1E88E5;
    }
    .bet-amount {
        font-size: 2.5em;
        font-weight: bold;
        color: #43A047;
        text-align: center;
        margin: 15px 0;
    }
    .odds-text {
        font-size: 1.5em;
        color: #333;
        text-align: center;
        margin: 15px 0;
        padding: 15px;
        background: #FFF9C4;
        border-radius: 10px;
    }
    .reason-box {
        font-size: 1.3em;
        color: #D84315;
        font-weight: bold;
        text-align: center;
        margin: 20px 0;
        padding: 15px;
        background: #FFE0B2;
        border-radius: 10px;
        border-left: 5px solid #D84315;
    }
    .confidence-bar {
        font-size: 1.3em;
        color: #333;
        text-align: center;
        margin: 15px 0;
    }
    .countdown {
        font-size: 2em;
        font-weight: bold;
        color: #E53935;
        text-align: center;
        padding: 15px;
        background: #FFEBEE;
        border-radius: 10px;
        margin: 10px 0;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .stButton>button {
        font-size: 1.5em;
        font-weight: bold;
        padding: 15px 30px;
        width: 100%;
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
def quick_analyze(raw_text):
    lines = [re.sub(r'[^\d]', '', l.strip()) for l in raw_text.strip().split('\n')]
    valid = [l for l in lines if len(l) == 5]
    
    if len(valid) < 5:
        return None
    
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
    
    return best_pick

# --- GIAO DIỆN ---
st.markdown('<div class="main-header">🎯 TITAN v35.0 - PRO</div>', unsafe_allow_html=True)

# Countdown
now = datetime.now()
seconds = now.second
remaining = 60 - seconds if seconds < 30 else 30 - (seconds - 30)
st.markdown(f'<div class="countdown">🕒 KỲ TIẾP THEO: {remaining:02d} GIÂY</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("💰 QUẢN LÝ VỐN")
    st.session_state.bankroll = st.number_input(
        "Vốn hiện tại (đ)", 
        value=st.session_state.bankroll, 
        step=10000,
        min_value=0
    )
    
    recommended = min(20000, int(st.session_state.bankroll * 0.02))
    st.success(f"✅ Cược đề xuất:\n\n**{recommended:,}đ**\n\n(2% vốn)")
    
    st.divider()
    st.info(f"💵 Vốn: {st.session_state.bankroll:,}đ")
    
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.period_count = 690
        st.session_state.last_result = None
        st.rerun()

# Form nhập liệu
st.markdown("### 📥 NHẬP KẾT QUẢ 10 KỲ GẦN NHẤT")
raw = st.text_area(
    "Dán kết quả tại đây (mỗi dòng 5 số, kỳ mới nhất trên cùng):",
    placeholder="87746\n56421\n69137\n00443\n04475\n...",
    height=200,
    label_visibility="collapsed"
)

col1, col2 = st.columns([3, 1])
with col1:
    analyze_btn = st.button("⚡ PHÂN TÍCH NGAY", type="primary", use_container_width=True)
with col2:
    if st.button("🗑️ XOÁ", use_container_width=True):
        st.session_state.last_result = None
        st.rerun()

# Kết quả
if analyze_btn and raw:
    result = quick_analyze(raw)
    
    if result:
        st.session_state.period_count += 1
        st.session_state.last_result = result
        
        profit = int(result['bet_amount'] * 0.985)
        
        # HIỂN THỊ KẾT QUẢ - SỬ DỤNG STREAMLIT NATIVE
        st.markdown("### 🎯 KHUYẾN NGHỊ KỲ " + str(st.session_state.period_count))
        st.divider()
        
        # Vị trí
        st.markdown(f"""
        <div class="position-text">
            📍 VỊ TRÍ:<br>{result['position']}
        </div>
        """, unsafe_allow_html=True)
        
        # ĐÁNH GÌ - CHỮ TO SIÊU RÕ
        bet_class = "tai-text" if result['bet'] == "TÀI" else "xiu-text"
        st.markdown(f"""
        <div class="bet-text {bet_class}">
            🔴 {result['bet']}
        </div>
        """, unsafe_allow_html=True)
        
        # Mức cược
        st.markdown(f"""
        <div class="bet-amount">
            💰 CƯỢC: {result['bet_amount']:,}đ
        </div>
        """, unsafe_allow_html=True)
        
        # Odds
        st.markdown(f"""
        <div class="odds-text">
            🎯 Odds: 1.985<br>
            👉 Thắng: +{profit:,}đ
        </div>
        """, unsafe_allow_html=True)
        
        # Lý do
        st.markdown(f"""
        <div class="reason-box">
            📊 {result['reason']}
        </div>
        """, unsafe_allow_html=True)
        
        # Độ tin cậy
        bars = "🟩" * (result['confidence'] // 10) + "⬜" * (10 - result['confidence'] // 10)
        st.markdown(f"""
        <div class="confidence-bar">
            ⚡ Độ tin cậy:<br>{bars}<br>{result['confidence']}%
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Nút kết quả
        st.markdown("### ✅ KẾT QUẢ THỰC TẾ:")
        c1, c2 = st.columns(2)
        
        with c1:
            if st.button(" THẮNG", type="primary", use_container_width=True, key="win_btn"):
                st.session_state.bankroll += profit
                st.balloons()
                st.success(f"🎉 Chúc mừng! +{profit:,}đ")
                st.info(f"💵 Vốn mới: {st.session_state.bankroll:,}đ")
                st.rerun()
        
        with c2:
            if st.button("🔴 THUA", type="secondary", use_container_width=True, key="lose_btn"):
                st.session_state.bankroll -= result['bet_amount']
                st.error(f"💸 Thua: -{result['bet_amount']:,}đ")
                st.info(f"💵 Vốn mới: {st.session_state.bankroll:,}đ")
                st.rerun()
        
        # Cảnh báo stop-loss
        if st.session_state.bankroll < 400000:
            st.error("🛑 **DỪNG NGAY!** Đã mất >20% vốn. Nghỉ ngơi và quay lại sau!")
        
        if st.session_state.bankroll > 575000:
            st.success("🎉 **TUYỆT VỜI!** Đã thắng >15%. Nên chốt lời!")

    else:
        st.warning("⚠️ Dữ liệu chưa đủ 5 kỳ hợp lệ! Vui lòng nhập ít nhất 5 dòng 5 chữ số.")

# Footer
st.markdown("---")
st.caption("""
**🎯 TITAN v35.0 - PROFESSIONAL**  
⚡ Phân tích nhanh - Chữ to rõ ràng - Dễ sử dụng  
⚠️ Chơi có trách nhiệm - Biết dừng đúng lúc 🙏
""")

# Hướng dẫn
with st.expander("📖 HƯỚNG DẪN SỬ DỤNG"):
    st.markdown("""
    **Bước 1:** Copy 10 kỳ kết quả gần nhất từ 5D KU
    
    **Bước 2:** Dán vào ô text above (kỳ mới nhất ở TRÊN cùng)
    
    **Bước 3:** Bấm "⚡ PHÂN TÍCH NGAY"
    
    **Bước 4:** Nhìn dòng chữ TO nhất → Đó là khuyến nghị:
    - 📍 VỊ TRÍ: Đánh ở cột nào (Chục Ngàn/Ngàn/Trăm/Chục/Đơn Vị)
    - 🔴 ĐÁNH: TÀI (màu đỏ) hoặc XỈU (màu xanh)
    - 💰 CƯỢC: Số tiền nên đánh
    
    **Bước 5:** Vào game 5D KU → Kèo Đôi → Đơn Thức → Chọn vị trí và Tài/Xỉu → Nhập tiền → Xác nhận
    
    **Bước 6:** Sau khi có kết quả, bấm "🟢 THẮNG" hoặc "🔴 THUA" để cập nhật vốn
    """)