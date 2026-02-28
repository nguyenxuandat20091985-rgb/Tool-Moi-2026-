import streamlit as st
from datetime import datetime, timedelta
import time
import re

st.set_page_config(page_title="TITAN v33.0 - 1 PHÚT", layout="centered", page_icon="⚡")

# --- CSS TỐI GIẢN + MÀU SẮC RÕ RÀNG ---
st.markdown("""
<style>
    .recommendation { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px; border-radius: 15px; color: white; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3); margin: 10px 0;
    }
    .tai { color: #FF4B4B; font-weight: bold; font-size: 2em; }
    .xiu { color: #1F77B4; font-weight: bold; font-size: 2em; }
    .position { font-size: 1.3em; font-weight: bold; color: #FFD700; }
    .bet-amount { font-size: 1.5em; color: #00FF00; font-weight: bold; }
    .countdown { font-size: 1.2em; color: #FF6B6B; font-weight: bold; }
    .quick-btn { width: 100%; padding: 12px; font-size: 1.1em; margin: 5px 0; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 500000
if 'last_recommendation' not in st.session_state:
    st.session_state.last_recommendation = None
if 'period_count' not in st.session_state:
    st.session_state.period_count = 0

# --- HÀM PHÂN TÍCH SIÊU NHANH ---
def quick_analyze(raw_text):
    """Phân tích nhanh, trả về 1 khuyến nghị duy nhất"""
    lines = [re.sub(r'[^\d]', '', l.strip()) for l in raw_text.strip().split('\n')]
    valid = [l for l in lines if len(l) == 5]
    
    if len(valid) < 5:
        return None
    
    # Tính cho từng vị trí
    positions = ["Chục Ngàn", "Ngàn", "Trăm", "Chục", "Đơn Vị"]
    best_pick = None
    best_score = -1
    
    for pos_idx, pos_name in enumerate(positions):
        digits = [int(line[pos_idx]) for line in valid[:10]]
        tai_count = sum(1 for d in digits if d >= 5)
        tai_rate = tai_count / len(digits)
        
        # Score: càng lệch càng dễ bẻ
        if tai_rate >= 0.7:  # 7/10 kỳ là Tài → Đánh Xỉu
            score = tai_rate
            pred = "XỈU"
            reason = f"Bệt Tài {tai_count}/10 → Bẻ cầu"
        elif tai_rate <= 0.3:  # 3/10 kỳ là Tài → Đánh Tài
            score = 1 - tai_rate
            pred = "TÀI"
            reason = f"Bệt Xỉu {10-tai_count}/10 → Bẻ cầu"
        else:
            continue  # Bỏ qua nếu không rõ xu hướng
        
        if score > best_score:
            best_score = score
            best_pick = {
                "position": pos_name,
                "bet": pred,
                "reason": reason,
                "confidence": int(score * 100),
                "bet_amount": min(20000, int(st.session_state.bankroll * 0.02))
            }
    
    # Fallback: nếu không có xu hướng rõ, chọn vị trí Đơn Vị (ổn định nhất)
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

# --- GIAO DIỆN CHÍNH ---
st.title("⚡ TITAN v33.0 - 5D 1 PHÚT")

# Countdown giả lập (đồng bộ với game thật)
now = datetime.now()
seconds = now.second
remaining = 60 - seconds if seconds < 30 else 30 - (seconds - 30)
st.markdown(f'<p class="countdown">🕒 Kỳ tiếp theo sau: {remaining:02d} giây</p>', unsafe_allow_html=True)

# Sidebar: Vốn
with st.sidebar:
    st.header("💰 Vốn")
    st.session_state.bankroll = st.number_input("Vốn hiện tại", value=st.session_state.bankroll, step=10000)
    st.info(f"✅ Cược đề xuất: 1-2% vốn = {min(20000, int(st.session_state.bankroll*0.02)):,}đ")
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.period_count = 0
        st.rerun()

# Form nhập liệu tối giản
with st.form("quick_form", clear_on_submit=False):
    raw = st.text_area(
        "📥 Dán 10 kỳ gần nhất (mới nhất trên cùng):",
        placeholder="95231\n18472\n03659\n...",
        height=150
    )
    col1, col2 = st.columns([3, 1])
    with col1:
        go = st.form_submit_button("⚡ PHÂN TÍCH NGAY", type="primary", use_container_width=True)
    with col2:
        st.form_submit_button("🗑️ Xoá", use_container_width=True)

# Kết quả - HIỂN THỊ TO RÕ
if go and raw:
    rec = quick_analyze(raw)
    
    if rec:
        st.session_state.last_recommendation = rec
        st.session_state.period_count += 1
        
        # 🎯 KHUNG KHUYẾN NGHỊ CHÍNH
        bet_class = "tai" if rec['bet'] == "TÀI" else "xiu"
        st.markdown(f"""
        <div class="recommendation">
            <h3>🎯 KHUYẾN NGHỊ KỲ {st.session_state.period_count + 689}</h3>
            <hr style="border-color: rgba(255,255,255,0.3)">
            <p class="position">📍 VỊ TRÍ: {rec['position']}</p>
            <p class="{bet_class}">🔴 ĐÁNH: {rec['bet']}</p>
            <p class="bet-amount">💰 CƯỢC: {rec['bet_amount']:,}đ</p>
            <p>🎯 Odds: 1.985 → Thắng +{int(rec['bet_amount']*0.985):,}đ</p>
            <hr style="border-color: rgba(255,255,255,0.3)">
            <p>📊 {rec['reason']}</p>
            <p>⚡ Độ tin cậy: {"█" * (rec['confidence']//10)}{"░" * (10 - rec['confidence']//10)} {rec['confidence']}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Nút hành động nhanh
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ ĐÃ ĐÁNH", type="primary", use_container_width=True, key="win"):
                st.session_state.bankroll += int(rec['bet_amount'] * 0.985)
                st.success(f"🎉 +{int(rec['bet_amount']*0.985):,}đ")
                st.rerun()
        with c2:
            if st.button("❌ THUA", type="secondary", use_container_width=True, key="lose"):
                st.session_state.bankroll -= rec['bet_amount']
                st.error(f"💸 -{rec['bet_amount']:,}đ")
                st.rerun()
        
        # Cảnh báo stop-loss
        if st.session_state.bankroll < 400000:  # Mất >20%
            st.error("🛑 DỪNG NGAY! Đã mất >20% vốn. Nghỉ ngơi nhé anh!")

    else:
        st.warning("⚠️ Dữ liệu chưa đủ 5 kỳ hợp lệ!")

# Footer cố định
st.markdown("---")
st.caption("⚡ TITAN v33.0 | 1 dòng khuyến nghị - 3 giây quyết định | Chơi có trách nhiệm 🙏")

# Auto-refresh hint
st.info("💡 Mẹo: Giữ tab mở, dán kết quả mới mỗi kỳ → Tool chạy trong 1 giây!")