import streamlit as st
from datetime import datetime
import re
import time

st.set_page_config(page_title="TITAN v39.0 - 5 HÀNG", layout="wide", initial_sidebar_state="collapsed")

# --- CSS TỐI ƯU MOBILE ---
st.markdown("""
<style>
    .main > div {padding-top: 1rem;}
    .stAlert {padding: 0.5rem;}
    div[data-testid="stMetricValue"] {font-size: 1.2rem;}
    .tai {color: #FF4444; font-weight: bold;}
    .xiu {color: #4444FF; font-weight: bold;}
    .bet-box {
        background: white;
        border: 2px solid #ddd;
        border-radius: 8px;
        padding: 8px;
        margin: 2px;
        text-align: center;
    }
    .position-name {
        font-size: 0.85em;
        color: #666;
        font-weight: 600;
    }
    .prediction {
        font-size: 1.5em;
        font-weight: bold;
        margin: 5px 0;
    }
    .confidence {
        font-size: 0.8em;
        color: #28a745;
    }
    .reason {
        font-size: 0.75em;
        color: #dc3545;
        font-weight: 600;
    }
    .quick-input {
        font-size: 1.1em;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 500000
if 'last_results' not in st.session_state:
    st.session_state.last_results = []
if 'auto_analyze' not in st.session_state:
    st.session_state.auto_analyze = True

# --- HÀM PHÂN TÍCH THÔNG MINH ---
def smart_analyze(raw_text):
    lines = [re.sub(r'[^\d]', '', l.strip()) for l in raw_text.strip().split('\n')]
    valid = [l for l in lines if len(l) == 5 and l.isdigit()]
    
    if len(valid) < 5:
        return None, f"Cần ít nhất 5 kỳ (hiện có: {len(valid)})"
    
    positions = ["C.Ngàn", "Ngàn", "Trăm", "Chục", "Đơn Vị"]
    predictions = []
    
    for pos_idx, pos_name in enumerate(positions):
        # Lấy 15 kỳ gần nhất
        digits = [int(line[pos_idx]) for line in valid[:15]]
        total = len(digits)
        
        # Tính thống kê
        tai_count = sum(1 for d in digits if d >= 5)
        xiu_count = total - tai_count
        tai_rate = tai_count / total
        
        # Phân tích xu hướng 5 kỳ gần nhất
        last_5 = digits[:5]
        last_5_tai = sum(1 for d in last_5 if d >= 5)
        
        # AI LOGIC - Phát hiện mẫu
        prediction = ""
        confidence = 50
        reason = ""
        bet_type = "TÀI/XỈU"
        
        # 1. Cầu bệt (4-5 kỳ cùng 1 bên) → Đánh bẻ
        if last_5_tai >= 4:
            prediction = "XỈU"
            confidence = 70 + (last_5_tai - 4) * 10
            reason = f"🔥 Bệt TÀI {last_5_tai}/5 → BẺ"
        elif last_5_tai <= 1:
            prediction = "TÀI"
            confidence = 70 + (1 - last_5_tai) * 10
            reason = f"🔥 Bệt XỈU {5-last_5_tai}/5 → BẺ"
        
        # 2. Độ lệch thống kê (>70% hoặc <30%)
        elif tai_rate >= 0.7:
            prediction = "XỈU"
            confidence = int(tai_rate * 100)
            reason = f"📊 Lệch TÀI {tai_count}/{total} → BÙ"
        elif tai_rate <= 0.3:
            prediction = "TÀI"
            confidence = int((1-tai_rate) * 100)
            reason = f"📊 Lệch XỈU {xiu_count}/{total} → BÙ"
        
        # 3. Cầu nhảy (3-2) → Theo kỳ gần nhất
        elif last_5_tai == 3:
            # Kiểm tra xu hướng giảm
            if sum(last_5[:3]) >= 15:  # 3 kỳ đầu Tài mạnh
                prediction = "XỈU"
                confidence = 60
                reason = "📉 Cầu nhảy → Giảm"
            else:
                prediction = "TÀI"
                confidence = 55
                reason = "📈 Cầu nhảy → Tăng"
        elif last_5_tai == 2:
            # Kiểm tra kỳ gần nhất
            if digits[0] >= 5:
                prediction = "TÀI"
                confidence = 55
                reason = "⚡ Theo kỳ mới"
            else:
                prediction = "XỈU"
                confidence = 55
                reason = "⚡ Theo kỳ mới"
        
        # 4. Mặc định - Theo thống kê
        else:
            if tai_rate > 0.5:
                prediction = "XỈU"
                confidence = 52
                reason = "📊 Thống kê nghiêng TÀI"
            else:
                prediction = "TÀI"
                confidence = 52
                reason = "📊 Thống kê nghiêng XỈU"
        
        # Tính số kỳ liên tiếp hiện tại
        current = digits[0] >= 5
        streak = 1
        for i in range(1, len(digits)):
            if (digits[i] >= 5) == current:
                streak += 1
            else:
                break
        
        predictions.append({
            'position': pos_name,
            'prediction': prediction,
            'confidence': min(confidence, 90),  # Max 90%
            'reason': reason,
            'tai_rate': tai_rate,
            'streak': streak,
            'current': 'TÀI' if current else 'XỈU'
        })
    
    return predictions, None

# --- GIAO DIỆN ---
st.title("🎯 TITAN v39.0 - 5 HÀNG SIÊU TỐC")

# Countdown
now = datetime.now()
seconds = now.second
remaining = 60 - seconds if seconds < 30 else 30 - (seconds - 30)
st.info(f"🕒 **Kỳ tiếp sau: {remaining:02d}s** | 💰 Vốn: {st.session_state.bankroll:,}đ")

# Input - AUTO ANALYZE
st.subheader("📥 DÁN KẾT QUẢ (Tự động phân tích)")
raw = st.text_area(
    "",
    placeholder="Dán 10-15 kỳ mới nhất vào đây...\nMỗi dòng 5 số\nKỳ mới nhất ở TRÊN CÙNG",
    height=120,
    key="auto_input",
    help="Tool sẽ tự động phân tích ngay khi bạn dán số!"
)

# Nút điều khiển
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])
with col_ctrl1:
    if st.button("🔄 LÀM MỚI", use_container_width=True):
        st.rerun()
with col_ctrl2:
    if st.button("🗑️ XÓA", use_container_width=True):
        st.session_state.last_results = []
        st.rerun()
with col_ctrl3:
    bet_amount = st.number_input("💵 Mức cược", min_value=1000, value=10000, step=1000)

# AUTO ANALYZE
if raw and len([l for l in raw.split('\n') if l.strip() and len(re.sub(r'[^\d]', '', l.strip())) == 5]) >= 5:
    predictions, error = smart_analyze(raw)
    
    if error:
        st.warning(f"⚠️ {error}")
    else:
        st.success(f"✅ ĐÃ PHÂN TÍCH {len([l for l in raw.split('\n') if l.strip()])} KỲ")
        
        # HIỂN THỊ 5 VỊ TRÍ - 1 HÀNG NGANG
        st.subheader("🎯 DỰ ĐOÁN 5 VỊ TRÍ")
        
        cols = st.columns(5)
        for idx, pred in enumerate(predictions):
            with cols[idx]:
                is_tai = pred['prediction'] == "TÀI"
                color_class = "tai" if is_tai else "xiu"
                
                st.markdown(f"""
                <div class="bet-box" style="border-color: {'#FF4444' if is_tai else '#4444FF'}">
                    <div class="position-name">{pred['position']}</div>
                    <div class="prediction {color_class}">{pred['prediction']}</div>
                    <div class="confidence">⚡ {pred['confidence']}%</div>
                    <div class="reason">{pred['reason']}</div>
                    <div style="font-size:0.7em; margin-top:5px; color:#666">
                        streak: {pred['streak']} | {pred['current']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Nút đánh nhanh
                if st.button(f"✅ ĐÁNH {pred['position']}", key=f"bet_{idx}", use_container_width=True):
                    profit = int(bet_amount * 0.985)
                    st.session_state.bankroll += profit
                    st.success(f"🎉 +{profit:,}đ")
                    st.rerun()
        
        # Gợi ý vị trí tốt nhất
        best = max(predictions, key=lambda x: x['confidence'])
        st.info(f"💡 **Vị trí tốt nhất:** {best['position']} → {best['prediction']} ({best['confidence']}%) - {best['reason']}")
        
        # Thống kê tổng
        st.divider()
        st.caption("📊 **Thống kê nhanh:** " + " | ".join([f"{p['position']}: T{int(p['tai_rate']*100)}%" for p in predictions]))

# Footer cố định
st.markdown("---")
st.caption("⚡ TITAN v39.0 | Auto-analyze | 5D KU 1 phút | Chơi có trách nhiệm 🙏")

# Auto-refresh hint
if not raw:
    st.info("👉 **Mẹo:** Copy kết quả từ nhà cái → Dán vào ô trên → Tool tự động phân tích trong 1 giây!")