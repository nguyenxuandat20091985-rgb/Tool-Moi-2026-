import streamlit as st
from datetime import datetime
import re

st.set_page_config(page_title="TITAN v36.0 - STABLE", layout="centered", page_icon="🎯")

# --- SESSION STATE ---
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 500000
if 'period_count' not in st.session_state:
    st.session_state.period_count = 690

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
st.title("🎯 TITAN v36.0 - ỔN ĐỊNH")

# Countdown
now = datetime.now()
seconds = now.second
remaining = 60 - seconds if seconds < 30 else 30 - (seconds - 30)
st.info(f"🕒 **Kỳ tiếp theo sau: {remaining:02d} giây**")

# Sidebar
with st.sidebar:
    st.header("💰 Quản lý vốn")
    st.session_state.bankroll = st.number_input(
        "Vốn hiện tại (đ)", 
        value=st.session_state.bankroll, 
        step=10000
    )
    recommended_bet = min(20000, int(st.session_state.bankroll * 0.02))
    st.info(f"✅ Cược đề xuất: {recommended_bet:,}đ")
    st.metric("💵 Vốn", f"{st.session_state.bankroll:,.0f}đ")
    
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.period_count = 690
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
        profit = int(result['bet_amount'] * 0.985)
        
        # HIỂN THỊ KẾT QUẢ - DÙNG STREAMLIT COMPONENTS
        st.success(f"## 🎯 KHUYẾN NGHỊ KỲ {st.session_state.period_count}")
        st.divider()
        
        # Cột 1: Thông tin chính
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("📍 Vị trí", result['position'])
            st.metric("🔴 Đánh", result['bet'])
        
        with col2:
            st.metric("💰 Mức cược", f"{result['bet_amount']:,}đ")
            st.metric("🎯 Thắng nhận", f"+{profit:,}đ")
        
        st.divider()
        
        # Hiển thị lý do
        st.info(f"📊 **Phân tích:** {result['reason']}")
        
        # Thanh độ tin cậy
        st.write(f"⚡ **Độ tin cậy:** {result['confidence']}%")
        st.progress(result['confidence'] / 100)
        
        # Gợi ý màu sắc
        if result['bet'] == "TÀI":
            st.error("### 🔴 ĐÁNH: TÀI")
        else:
            st.success("### 🔵 ĐÁNH: XỈU")
        
        st.divider()
        
        # Nút kết quả
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ THẮNG", type="primary", use_container_width=True, key="win_btn"):
                st.session_state.bankroll += profit
                st.balloons()
                st.success(f"🎉 +{profit:,}đ")
                st.rerun()
        with c2:
            if st.button("❌ THUA", type="secondary", use_container_width=True, key="lose_btn"):
                st.session_state.bankroll -= result['bet_amount']
                st.error(f"💸 -{result['bet_amount']:,}đ")
                st.rerun()
        
        # Cảnh báo
        if st.session_state.bankroll < 400000:
            st.error("🛑 **Dừng ngay!** Đã mất >20% vốn.")

# Footer
st.markdown("---")
st.caption("🎯 TITAN v36.0 | Ổn định - Không lỗi HTML | Chơi có trách nhiệm 🙏")