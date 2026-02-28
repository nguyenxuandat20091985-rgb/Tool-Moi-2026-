import streamlit as st
from datetime import datetime
import re
import time

st.set_page_config(page_title="TITAN v39.0 - 5 VỊ TRÍ", layout="wide", page_icon="🎯")

# --- CSS ĐƠN GIẢN ---
st.markdown("""
<style>
    .position-box {
        background: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #e2e8f0;
        margin: 5px;
    }
    .tai { color: #E53E3E; font-size: 1.8em; font-weight: bold; }
    .xiu { color: #3182CE; font-size: 1.8em; font-weight: bold; }
    .position-name { 
        font-size: 1.1em; 
        font-weight: bold; 
        color: #2D3748;
        margin-bottom: 10px;
    }
    .confidence {
        font-size: 0.9em;
        color: #718096;
        margin-top: 5px;
    }
    .main-title {
        background: #48BB78;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 500000
if 'period_count' not in st.session_state:
    st.session_state.period_count = 754
if 'last_input' not in st.session_state:
    st.session_state.last_input = ""

# --- HÀM PHÂN TÍCH TÀI/XỈU ---
def analyze_tai_xiu(raw_text):
    lines = [re.sub(r'[^\d]', '', l.strip()) for l in raw_text.strip().split('\n')]
    valid = [l for l in lines if len(l) == 5 and l.isdigit()]
    
    if len(valid) < 5:
        return None, f"Cần ít nhất 5 kỳ (hiện có: {len(valid)})"
    
    positions = ["Chục Ngàn", "Ngàn", "Trăm", "Chục", "Đơn Vị"]
    predictions = []
    
    for pos_idx, pos_name in enumerate(positions):
        # Lấy 10 kỳ gần nhất
        digits = [int(line[pos_idx]) for line in valid[:10]]
        total = len(digits)
        
        # Đếm Tài (5-9) và Xỉu (0-4)
        tai_count = sum(1 for d in digits if d >= 5)
        xiu_count = total - tai_count
        tai_rate = tai_count / total
        
        # Dự đoán: Nếu đang ra nhiều Tài → đánh Xỉu (bẻ cầu)
        # Nếu đang ra nhiều Xỉu → đánh Tài (bẻ cầu)
        if tai_rate >= 0.6:
            prediction = "XỈU"
            confidence = int(tai_rate * 100)
            reason = f"Bệt Tài {tai_count}/{total} → Bẻ"
        elif tai_rate <= 0.4:
            prediction = "TÀI"
            confidence = int((1 - tai_rate) * 100)
            reason = f"Bệt Xỉu {xiu_count}/{total} → Bẻ"
        else:
            # Cầu nhảy → theo kỳ gần nhất
            last_digit = digits[0]
            prediction = "TÀI" if last_digit < 5 else "XỈU"
            confidence = 55
            reason = "Cầu nhảy → Theo ngược"
        
        predictions.append({
            'position': pos_name,
            'prediction': prediction,
            'confidence': confidence,
            'reason': reason,
            'stats': f"Tài: {tai_count}, Xỉu: {xiu_count}"
        })
    
    return predictions, None

# --- GIAO DIỆN ---
st.title("🎯 TITAN v39.0 - DỰ ĐOÁN 5 VỊ TRÍ TÀI/XỈU")

# Countdown
now = datetime.now()
seconds = now.second
remaining = 60 - seconds if seconds < 30 else 30 - (seconds - 30)
st.info(f"🕒 **Kỳ {st.session_state.period_count} - Còn lại: {remaining:02d} giây**")

# Sidebar
with st.sidebar:
    st.header("💰 Vốn")
    st.session_state.bankroll = st.number_input("Vốn (đ)", value=st.session_state.bankroll, step=10000)
    st.metric("💵 Vốn hiện tại", f"{st.session_state.bankroll:,.0f}đ")
    
    st.divider()
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.period_count = 754
        st.rerun()

# Form nhập liệu
st.subheader("📥 Nhập kết quả 10 kỳ gần nhất (mới nhất trên cùng)")
raw = st.text_area(
    "Dán số tại đây:",
    placeholder="95573\n87746\n56421\n69137\n...",
    height=150,
    key="input_data"
)

# Auto-analyze khi input thay đổi
if raw and raw != st.session_state.last_input:
    st.session_state.last_input = raw
    
    with st.spinner("🔄 Đang phân tích..."):
        time.sleep(0.3)
        predictions, error = analyze_tai_xiu(raw)
        
        if error:
            st.warning(f"⚠️ {error}")
        else:
            # Hiển thị kết quả
            st.markdown(f'<div class="main-title">✅ KỲ {st.session_state.period_count} - DỰ ĐOÁN TÀI/XỈU</div>', 
                       unsafe_allow_html=True)
            
            # Hiển thị 5 vị trí ngang
            cols = st.columns(5)
            
            for idx, pred in enumerate(predictions):
                with cols[idx]:
                    bet_class = "tai" if pred['prediction'] == "TÀI" else "xiu"
                    
                    st.markdown(f"""
                    <div class="position-box">
                        <div class="position-name">{pred['position']}</div>
                        <div class="{bet_class}">{pred['prediction']}</div>
                        <div class="confidence">⚡ {pred['confidence']}%</div>
                        <div style="font-size: 0.85em; color: #718096; margin-top: 5px;">
                            {pred['stats']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.divider()
            
            # Gợi ý cược
            st.subheader("💡 GỢI Ý CƯỢC")
            
            # Tìm vị trí có độ tin cậy cao nhất
            best_pick = max(predictions, key=lambda x: x['confidence'])
            st.success(f"🎯 **Vị trí tốt nhất:** {best_pick['position']} → {best_pick['prediction']} ({best_pick['confidence']}%)")
            
            # Xiên 2 gợi ý
            top_2 = sorted(predictions, key=lambda x: x['confidence'], reverse=True)[:2]
            if len(top_2) >= 2:
                st.info(f"🔗 **Xiên 2 gợi ý:** {top_0['position']} ({top_0['prediction']}) + {top_2[1]['position']} ({top_2[1]['prediction']})")
            
            # Nút cập nhật kết quả
            st.divider()
            st.subheader("📊 Cập nhật kết quả")
            c1, c2, c3 = st.columns(3)
            
            with c1:
                if st.button("✅ THẮNG TẤT CẢ", type="primary", use_container_width=True):
                    profit = int(st.session_state.bankroll * 0.1)  # Giả sử thắng 10%
                    st.session_state.bankroll += profit
                    st.session_state.period_count += 1
                    st.balloons()
                    st.success(f"🎉 +{profit:,}đ → Kỳ {st.session_state.period_count}")
                    st.rerun()
            
            with c2:
                if st.button("❌ THUA TẤT CẢ", type="secondary", use_container_width=True):
                    loss = int(st.session_state.bankroll * 0.05)  # Giả sử thua 5%
                    st.session_state.bankroll -= loss
                    st.session_state.period_count += 1
                    st.error(f"💸 -{loss:,}đ → Kỳ {st.session_state.period_count}")
                    st.rerun()
            
            with c3:
                if st.button("⏭️ BỎ QUA", use_container_width=True):
                    st.session_state.period_count += 1
                    st.rerun()

elif not raw:
    st.info("👆 Anh dán kết quả 10 kỳ gần nhất vào ô trên để xem dự đoán!")

# Footer
st.markdown("---")
st.caption("🎯 TITAN v39.0 | Tự động hiển thị 5 vị trí Tài/Xỉu | Chơi có trách nhiệm 🙏")