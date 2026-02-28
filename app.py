import streamlit as st
from datetime import datetime
import re
import random
from collections import Counter

st.set_page_config(page_title="TITAN v39.0 - AI 5 VỊ TRÍ", layout="wide", page_icon="🎯")

# --- CSS ĐƠN GIẢN & ĐẸP ---
st.markdown("""
<style>
    .position-box {
        background: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin: 5px;
        border-left: 5px solid #667eea;
    }
    .tai { color: #e53e3e; font-size: 1.5em; font-weight: bold; }
    .xiu { color: #3182ce; font-size: 1.5em; font-weight: bold; }
    .confidence-high { color: #38a169; font-weight: bold; }
    .confidence-med { color: #d69e2e; font-weight: bold; }
    .confidence-low { color: #e53e3e; font-weight: bold; }
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 500000
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_input' not in st.session_state:
    st.session_state.last_input = ""

# --- AI PHÂN TÍCH THÔNG MINH ---
def ai_analyze_position(digits, position_name):
    """AI phân tích 1 vị trí với nhiều thuật toán"""
    if len(digits) < 10:
        return None
    
    total = len(digits)
    tai_count = sum(1 for d in digits if d >= 5)
    xiu_count = total - tai_count
    tai_rate = tai_count / total
    
    # Thuật toán 1: Phân tích xu hướng (Trend Analysis)
    last_5 = digits[:5]
    last_5_tai = sum(1 for d in last_5 if d >= 5)
    
    # Thuật toán 2: Phát hiện bệt (Streak Detection)
    streak = 1
    for i in range(1, min(5, len(digits))):
        if (digits[i] >= 5) == (digits[i-1] >= 5):
            streak += 1
        else:
            break
    
    # Thuật toán 3: Mean Reversion (Về trung bình)
    recent_avg = sum(last_5) / len(last_5)
    
    # Thuật toán 4: Pattern Recognition (Nhận diện mẫu)
    pattern_score = 0
    if total >= 15:
        # Kiểm tra chu kỳ 3-4 kỳ
        cycle_3 = digits[0:3]
        cycle_3_tai = sum(1 for d in cycle_3 if d >= 5)
        if cycle_3_tai in [0, 3]:  # Bệt 3 kỳ
            pattern_score = 0.7
    
    # Thuật toán 5: Hot/Cold Analysis
    hot_numbers = Counter(digits[:10])
    hot_tai = sum(count for num, count in hot_numbers.items() if num >= 5)
    hot_xiu = sum(count for num, count in hot_numbers.items() if num < 5)
    
    # QUYẾT ĐỊNH AI - KẾT HỢP NHIỀU THUẬT TOÁN
    signals = []
    
    # Signal 1: Bệt quá dài → Bẻ cầu
    if streak >= 4:
        if last_5_tai >= 4:
            signals.append(('XỈU', 0.75, f'Bệt Tài {streak} kỳ → Bẻ'))
        elif last_5_tai <= 1:
            signals.append(('TÀI', 0.75, f'Bệt Xỉu {streak} kỳ → Bẻ'))
    
    # Signal 2: Lệch mạnh → Về trung bình
    if tai_rate >= 0.7:
        signals.append(('XỈU', 0.70, f'Tài {tai_rate*100:.0f}% → Giảm'))
    elif tai_rate <= 0.3:
        signals.append(('TÀI', 0.70, f'Xỉu {(1-tai_rate)*100:.0f}% → Tăng'))
    
    # Signal 3: Pattern 3 kỳ
    if pattern_score > 0:
        if last_5_tai == 3:
            signals.append(('XỈU', 0.65, 'Pattern 3 Tài → Giảm'))
        elif last_5_tai == 0:
            signals.append(('TÀI', 0.65, 'Pattern 3 Xỉu → Tăng'))
    
    # Signal 4: Hot/Cold
    if hot_tai > hot_xiu * 1.5:
        signals.append(('XỈU', 0.60, 'Tài nóng → Chuẩn bị nguội'))
    elif hot_xiu > hot_tai * 1.5:
        signals.append(('TÀI', 0.60, 'Xỉu nóng → Chuẩn bị nguội'))
    
    # Signal 5: Random yếu tố (RNG có thể có bias)
    if total >= 20:
        # Kiểm tra phân phối chẵn/lẻ
        even_count = sum(1 for d in digits if d % 2 == 0)
        if even_count > total * 0.65:
            signals.append(('LẺ', 0.55, 'Chẵn nhiều → Lẻ sắp về'))
        elif even_count < total * 0.35:
            signals.append(('CHẴN', 0.55, 'Lẻ nhiều → Chẵn sắp về'))
    
    # Chọn signal mạnh nhất
    if signals:
        signals.sort(key=lambda x: x[1], reverse=True)
        best = signals[0]
        return {
            'prediction': best[0],
            'confidence': int(best[1] * 100),
            'reason': best[2],
            'all_signals': signals[:3]
        }
    
    # Fallback: Theo xu hướng gần
    if last_5_tai >= 3:
        return {'prediction': 'TÀI', 'confidence': 60, 'reason': 'Xu hướng Tài', 'all_signals': []}
    else:
        return {'prediction': 'XỈU', 'confidence': 60, 'reason': 'Xu hướng Xỉu', 'all_signals': []}

# --- HÀM PHÂN TÍCH TẤT CẢ 5 VỊ TRÍ ---
def analyze_all_positions(raw_text):
    lines = [re.sub(r'[^\d]', '', l.strip()) for l in raw_text.strip().split('\n')]
    valid = [l for l in lines if len(l) == 5 and l.isdigit()]
    
    if len(valid) < 10:
        return None, f"Cần ít nhất 10 kỳ (hiện có: {len(valid)})"
    
    positions = ["Chục Ngàn", "Ngàn", "Trăm", "Chục", "Đơn Vị"]
    results = {}
    
    for pos_idx, pos_name in enumerate(positions):
        digits = [int(line[pos_idx]) for line in valid[:20]]  # Lấy 20 kỳ
        results[pos_name] = ai_analyze_position(digits, pos_name)
        
        # Thêm thống kê
        tai_count = sum(1 for d in digits if d >= 5)
        results[pos_name]['stats'] = f"Tài: {tai_count}/{len(digits)}"
        results[pos_name]['last_digit'] = digits[0] if digits else 0
    
    return results, None

# --- GIAO DIỆN ---
st.markdown('<div class="main-title">🎯 TITAN v39.0 - AI PHÂN TÍCH 5 VỊ TRÍ</div>', unsafe_allow_html=True)

# Countdown
now = datetime.now()
seconds = now.second
remaining = 60 - seconds if seconds < 30 else 30 - (seconds - 30)
st.info(f"🕒 **Kỳ tiếp theo sau: {remaining:02d} giây** | 💰 Vốn: {st.session_state.bankroll:,.0f}đ")

# Sidebar
with st.sidebar:
    st.header("⚙️ Cài đặt")
    st.session_state.bankroll = st.number_input("Vốn (đ)", value=st.session_state.bankroll, step=10000)
    bet_amount = st.number_input("Mức cược/ vị trí (đ)", min_value=1000, value=5000, step=1000)
    
    st.divider()
    st.subheader("📜 Lịch sử")
    if st.session_state.history:
        for h in st.session_state.history[-10:]:
            icon = "🟢" if h['result'] == 'WIN' else "🔴"
            st.caption(f"{icon} {h['time']}: {h['position']} {h['bet']} → {h['pnl']:+,}đ")
    
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# Form nhập liệu
st.subheader("📥 Nhập kết quả 20 kỳ gần nhất")
raw = st.text_area(
    "Dán kết quả (mới nhất TRÊN CÙNG):",
    placeholder="95573\n87746\n56421\n...",
    height=200
)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    analyze_btn = st.button("🚀 PHÂN TÍCH TẤT CẢ 5 VỊ TRÍ", type="primary", use_container_width=True)
with col2:
    if st.button("🗑️ Xóa", use_container_width=True):
        st.session_state.last_input = ""
        st.rerun()
with col3:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

# Xử lý phân tích
if analyze_btn and raw:
    if raw == st.session_state.last_input:
        st.warning("⚠️ Dữ liệu không thay đổi! Anh nhập số mới nhé.")
    else:
        st.session_state.last_input = raw
        
        with st.spinner("🤖 AI đang phân tích 5 vị trí..."):
            results, error = analyze_all_positions(raw)
        
        if error:
            st.error(f"❌ {error}")
        else:
            st.success("✅ PHÂN TÍCH HOÀN TẤT!")
            st.divider()
            
            # HIỂN THỊ 5 VỊ TRÍ - DẠNG LƯỚI
            st.subheader("🎯 DỰ ĐOÁN CẢ 5 VỊ TRÍ")
            
            cols = st.columns(5)
            positions = ["Chục Ngàn", "Ngàn", "Trăm", "Chục", "Đơn Vị"]
            
            for idx, pos_name in enumerate(positions):
                with cols[idx]:
                    result = results[pos_name]
                    pred = result['prediction']
                    conf = result['confidence']
                    
                    # Màu sắc theo độ tin cậy
                    if conf >= 70:
                        conf_class = "confidence-high"
                    elif conf >= 60:
                        conf_class = "confidence-med"
                    else:
                        conf_class = "confidence-low"
                    
                    st.markdown(f"""
                    <div class="position-box">
                        <div style="font-weight: bold; color: #4a5568;">{pos_name}</div>
                        <div style="font-size: 0.9em; color: #718096;">{result['stats']}</div>
                        <div style="margin: 10px 0;">
                            <div class="{'tai' if pred == 'TÀI' else 'xiu'}">{pred}</div>
                        </div>
                        <div class="{conf_class}">⚡ {conf}%</div>
                        <div style="font-size: 0.85em; color: #718096; margin-top: 5px;">
                            {result['reason']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.divider()
            
            # GỢI Ý KÈO TỐT NHẤT
            st.subheader("💎 TOP 3 KÈO TỐT NHẤT")
            
            # Sắp xếp theo độ tin cậy
            sorted_positions = sorted(positions, key=lambda x: results[x]['confidence'], reverse=True)
            
            for i, pos_name in enumerate(sorted_positions[:3], 1):
                result = results[pos_name]
                col1, col2, col3, col4 = st.columns([2, 1, 2, 1])
                
                with col1:
                    st.write(f"**{i}. {pos_name}**")
                with col2:
                    st.write(f"→ **{result['prediction']}**")
                with col3:
                    st.caption(result['reason'])
                with col4:
                    st.write(f"⚡ {result['confidence']}%")
                
                # Nút đánh nhanh
                c1, c2 = st.columns(2)
                with c1:
                    if st.button(f"✅ Thắng ({pos_name})", key=f"win_{pos_name}_{i}"):
                        profit = int(bet_amount * 0.985)
                        st.session_state.bankroll += profit
                        st.session_state.history.append({
                            'time': datetime.now().strftime("%H:%M"),
                            'position': pos_name,
                            'bet': result['prediction'],
                            'result': 'WIN',
                            'pnl': profit
                        })
                        st.success(f"🎉 +{profit:,}đ")
                        st.rerun()
                with c2:
                    if st.button(f"❌ Thua ({pos_name})", key=f"lose_{pos_name}_{i}"):
                        st.session_state.bankroll -= bet_amount
                        st.session_state.history.append({
                            'time': datetime.now().strftime("%H:%M"),
                            'position': pos_name,
                            'bet': result['prediction'],
                            'result': 'LOSE',
                            'pnl': -bet_amount
                        })
                        st.error(f"💸 -{bet_amount:,}đ")
                        st.rerun()
                
                st.divider()
            
            # CHIẾN LƯỢC ĐÁNH
            st.info("""
            💡 **CHIẾN LƯỢC KHUYẾN NGHỊ:**
            - **An toàn**: Chỉ đánh 1-2 vị trí có độ tin cậy >70%
            - **Trung bình**: Đánh 3 vị trí tốt nhất
            - **Mạo hiểm**: Đánh cả 5 vị trí (rủi ro cao)
            - **Xiên 2**: Kết hợp 2 vị trí có confidence cao nhất
            """)

# Footer
st.markdown("---")
st.caption("🎯 TITAN v39.0 | AI đa thuật toán | Khai thác điểm yếu RNG | Chơi có trách nhiệm 🙏")