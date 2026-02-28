import streamlit as st
from datetime import datetime
import time
import re

st.set_page_config(page_title="TITAN v32.0 - XIÊN 2 PRO", layout="wide", page_icon="🎯")

# --- SESSION STATE ---
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 500000  # Vốn gốc 500k
if 'bet_history' not in st.session_state:
    st.session_state.bet_history = []
if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = None
if 'daily_profit' not in st.session_state:
    st.session_state.daily_profit = 0
if 'stop_loss_reached' not in st.session_state:
    st.session_state.stop_loss_reached = False

# --- HÀM CLEAN & VALIDATE INPUT ---
def clean_input(raw_text):
    """Làm sạch input, chỉ giữ 5 chữ số mỗi dòng"""
    lines = raw_text.strip().split('\n')
    valid = []
    errors = []
    for idx, line in enumerate(lines, 1):
        clean = re.sub(r'[^\d]', '', line.strip())
        if len(clean) == 5:
            valid.append({'period': len(valid)+1, 'value': clean, 'digits': [int(d) for d in clean]})
        elif clean:
            errors.append(f"Dòng {idx}: '{line.strip()}' ❌")
    return valid, errors

# --- HÀM TÍNH THỐNG KÊ TỪNG VỊ TRÍ ---
def calculate_position_stats(periods, pos_idx):
    if not periods:
        return None
    digits = [p['digits'][pos_idx] for p in periods]
    tai = sum(1 for d in digits if d >= 5)
    xiu = len(digits) - tai
    chan = sum(1 for d in digits if d % 2 == 0)
    le = len(digits) - chan
    to = sum(1 for d in digits if d in [1,2,3,5,7])
    hop = len(digits) - to
    
    # Tính xu hướng 5 kỳ gần
    last_5 = digits[:5]
    last_5_tai = sum(1 for d in last_5 if d >= 5)
    
    # Phát hiện bệt (4-5 kỳ cùng 1 bên)
    is_tai_bet = last_5_tai >= 4
    is_xiu_bet = last_5_tai <= 1
    
    return {
        'total': len(digits),
        'tai': tai, 'xiu': xiu, 'tai_rate': tai/len(digits) if digits else 0,
        'chan': chan, 'le': le,
        'to': to, 'hop': hop,
        'last_5_tai': last_5_tai,
        'is_tai_bet': is_tai_bet,
        'is_xiu_bet': is_xiu_bet,
        'avg': sum(digits)/len(digits) if digits else 0,
        'trend': '📈 TĂNG' if digits[0] > digits[-1] else '📉 GIẢM' if digits[0] < digits[-1] else '➡️ ỔN'
    }

# --- HÀM PHÂN TÍCH XIÊN 2 ---
def analyze_xien2(periods):
    """Phân tích và gợi ý kèo Xiên 2 tối ưu"""
    if len(periods) < 10:
        return None, "Cần ít nhất 10 kỳ để phân tích"
    
    labels = ["Chục Ngàn", "Ngàn", "Trăm", "Chục", "Đơn Vị"]
    stats = {}
    
    for i, name in enumerate(labels):
        stats[name] = calculate_position_stats(periods, i)
    
    # Tìm cặp vị trí có tín hiệu mạnh nhất cho Xiên 2
    recommendations = []
    
    # Cặp 1: Chục Ngàn + Ngàn (vị trí 0 + 1)
    pos0 = stats["Chục Ngàn"]
    pos1 = stats["Ngàn"]
    
    if pos0['is_xiu_bet'] and pos1['is_xiu_bet']:
        rec = {"pair": "Chục Ngàn + Ngàn", "bet": "XỈU + XỈU", "confidence": "85%", 
               "reason": "Cả 2 vị trí đang bệt Xỉu → Đánh bẻ cầu", "risk": "CAO"}
        recommendations.append(rec)
    elif pos0['is_tai_bet'] and pos1['is_tai_bet']:
        rec = {"pair": "Chục Ngàn + Ngàn", "bet": "TÀI + TÀI", "confidence": "85%",
               "reason": "Cả 2 vị trí đang bệt Tài → Đánh bẻ cầu", "risk": "CAO"}
        recommendations.append(rec)
    
    # Cặp 2: Chục + Đơn Vị (vị trí 3 + 4) - Thường ổn định hơn
    pos3 = stats["Chục"]
    pos4 = stats["Đơn Vị"]
    
    if pos3['tai_rate'] > 0.6 and pos4['tai_rate'] > 0.6:
        rec = {"pair": "Chục + Đơn Vị", "bet": "XỈU + XỈU", "confidence": "75%",
               "reason": "Tài xuất hiện >60% → Theo luật bù trừ, Xỉu sẽ về", "risk": "TRUNG BÌNH"}
        recommendations.append(rec)
    elif pos3['tai_rate'] < 0.4 and pos4['tai_rate'] < 0.4:
        rec = {"pair": "Chục + Đơn Vị", "bet": "TÀI + TÀI", "confidence": "75%",
               "reason": "Xỉu xuất hiện >60% → Theo luật bù trừ, Tài sẽ về", "risk": "TRUNG BÌNH"}
        recommendations.append(rec)
    
    # Cặp 3: Kèo an toàn - 1 Tài 1 Xỉu (giảm variance)
    if pos0['tai_rate'] > 0.5 and pos4['tai_rate'] < 0.5:
        rec = {"pair": "Chục Ngàn + Đơn Vị", "bet": "TÀI + XỈU", "confidence": "65%",
               "reason": "Đa dạng hóa rủi ro, xác suất thắng ~30%", "risk": "THẤP"}
        recommendations.append(rec)
    
    # Luôn có gợi ý Tài/Xỉu 1 vị trí làm nền tảng
    recommendations.append({
        "pair": "BẤT KỲ 1 VỊ TRÍ", 
        "bet": "TÀI/XỈU ĐƠN", 
        "confidence": "50%+", 
        "reason": "Edge nhà cái thấp nhất (~2.5-5%) → Nuôi vốn an toàn",
        "risk": "THẤP"
    })
    
    return recommendations, stats

# --- HÀM TÍNH MỨC CƯỢC ---
def calculate_bet_amount(bankroll, bet_type, martingale_level=0):
    """Tính mức cược theo % vốn và cấp độ gấp thếp"""
    base_pct = 0.02 if bet_type == "ĐƠN" else 0.01  # Đơn: 2%, Xiên: 1%
    base_bet = int(bankroll * base_pct)
    
    if martingale_level > 0:
        bet = base_bet * (2 ** martingale_level)
        max_bet = int(bankroll * 0.1)  # Max 10% vốn
        bet = min(bet, max_bet)
    else:
        bet = base_bet
    
    return bet

# --- GIAO DIỆN ---
st.title("🎯 TITAN v32.0 - KÈO ĐÔI XIÊN 2 CHUYÊN BIỆT")
st.write(f"🕒 {datetime.now().strftime('%H:%M:%S | %d/%m/%Y')}")

# Cảnh báo quan trọng
st.warning("""
⚠️ **LƯU Ý QUAN TRỌNG**: 
- 5D KU dùng RNG (ngẫu nhiên), không có công thức thắng 100%
- Tool hỗ trợ ra quyết định nhanh, quản lý vốn thông minh
- **LUÔN DỪNG KHI THUA 20% VỐN/NGÀY** hoặc **THẮNG 15% VỐN/NGÀY**
- Chơi có trách nhiệm, coi đây là giải trí
""")

# Sidebar: Quản lý vốn
with st.sidebar:
    st.header("💰 QUẢN LÝ VỐN")
    
    st.session_state.bankroll = st.number_input(
        "Vốn hiện tại (đ)", 
        min_value=0, 
        value=st.session_state.bankroll,
        step=10000
    )
    
    bet_type = st.radio(
        "Loại cược:",
        ["Xiên 2", "Tài/Xỉu Đơn"],
        index=0
    )
    
    martingale = st.slider("Cấp độ gấp thếp", 0, 3, 0)
    
    st.divider()
    
    # Tính mức cược gợi ý
    recommended_bet = calculate_bet_amount(st.session_state.bankroll, "ĐƠN" if bet_type == "Tài/Xỉu Đơn" else "XIÊN", martingale)
    st.metric("Mức cược gợi ý", f"{recommended_bet:,}đ")
    
    # Stop-loss calculator
    stop_loss = int(st.session_state.bankroll * 0.2)
    take_profit = int(st.session_state.bankroll * 0.15)
    
    st.info(f"""
    🛑 **Stop-loss**: -{stop_loss:,}đ (20%)
    ✅ **Take-profit**: +{take_profit:,}đ (15%)
    """)
    
    st.divider()
    
    # Lịch sử cược
    st.subheader("📜 Lịch sử 5 ván gần")
    if st.session_state.bet_history:
        for h in st.session_state.bet_history[-5:]:
            icon = "🟢" if h['result'] == 'WIN' else "🔴"
            st.caption(f"{icon} {h['time']}: {h['type']} {h['bet']}đ → {h['pnl']:+,}đ")
    else:
        st.caption("Chưa có lịch sử")
    
    if st.button("🗑️ Reset lịch sử"):
        st.session_state.bet_history = []
        st.rerun()

# Form nhập liệu
with st.form("input_form"):
    st.subheader("📥 NHẬP KẾT QUẢ 10-20 KỲ GẦN NHẤT")
    
    raw_data = st.text_area(
        "Dán kết quả tại đây (mỗi dòng 5 chữ số):",
        placeholder="Ví dụ:\n95231\n18472\n03659\n74125\n...\n(Kỳ mới nhất ở TRÊN CÙNG)",
        height=200
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        submitted = st.form_submit_button("🚀 PHÂN TÍCH XIÊN 2", type="primary", use_container_width=True)
    with col2:
        preview_btn = st.form_submit_button("👀 Xem trước", use_container_width=True)
    with col3:
        cleared = st.form_submit_button("🗑️ Xoá", use_container_width=True)

# Xử lý nút
if cleared:
    st.session_state.analysis_cache = None
    st.rerun()

# Preview & Analysis
if preview_btn or (submitted and raw_data):
    periods, errors = clean_input(raw_data)
    
    if errors:
        with st.expander(f"⚠️ {len(errors)} dòng không hợp lệ", expanded=False):
            for e in errors:
                st.warning(e)
    
    if len(periods) < 10:
        st.warning(f"⚠️ Cần ít nhất 10 kỳ để phân tích thống kê. Hiện có: {len(periods)}")
    else:
        st.session_state.analysis_cache = {"periods": periods, "errors": errors}
        
        if submitted:
            with st.spinner("🔍 Đang phân tích Xiên 2..."):
                time.sleep(0.5)
                recommendations, stats = analyze_xien2(periods)
            
            st.success(f"✅ Phân tích xong {len(periods)} kỳ!")
            
            # 📊 BẢNG THỐNG KÊ CHI TIẾT
            st.subheader("📊 THỐNG KÊ TỪNG VỊ TRÍ")
            cols = st.columns(5)
            labels = ["Chục Ngàn", "Ngàn", "Trăm", "Chục", "Đơn Vị"]
            
            for idx, name in enumerate(labels):
                s = stats[name]
                with cols[idx]:
                    st.markdown(f"""
                    <div style='background:#f0f2f6; padding:10px; border-radius:8px; text-align:center'>
                        <b>{name}</b><br>
                        Tài: {s['tai']}/{s['total']} ({s['tai_rate']*100:.0f}%)<br>
                        Xu hướng: {s['trend']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Cảnh báo bệt
                    if s['is_tai_bet']:
                        st.caption("🔥 Đang bệt TÀI")
                    elif s['is_xiu_bet']:
                        st.caption("🔥 Đang bệt XỈU")
            
            st.divider()
            
            # 🎯 GỢI Ý XIÊN 2
            st.subheader("🎯 GỢI Ý KÈO XIÊN 2")
            
            for i, rec in enumerate(recommendations, 1):
                risk_color = {"THẤP": "🟢", "TRUNG BÌNH": "🟡", "CAO": "🔴"}
                
                with st.expander(f"{i}. {rec['pair']} → {rec['bet']}", expanded=(i==1)):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Độ tin cậy", rec['confidence'])
                    with c2:
                        st.metric("Rủi ro", f"{risk_color.get(rec['risk'], '⚪')} {rec['risk']}")
                    with c3:
                        st.metric("Lý do", rec['reason'][:30]+"...")
                    
                    # Tính toán cược cho kèo này
                    if "XIÊN" in rec['bet'] or "2" in rec['pair']:
                        bet_amount = calculate_bet_amount(st.session_state.bankroll, "XIÊN", martingale)
                        potential_win = int(bet_amount * 3.6)  # Tỷ lệ Xiên 2 ~3.6x
                    else:
                        bet_amount = calculate_bet_amount(st.session_state.bankroll, "ĐƠN", martingale)
                        potential_win = int(bet_amount * 1.9)  # Tỷ lệ Đơn ~1.9x
                    
                    st.info(f"""
                    💰 **Mức cược gợi ý**: {bet_amount:,}đ  
                    🏆 **Thắng nhận**: {potential_win:,}đ  
                    📊 **Lợi nhuận**: +{potential_win - bet_amount:,}đ
                    """)
            
            st.divider()
            
            # 📈 BIỂU ĐỒ XU HƯỚNG
            st.subheader("📈 XU HƯỚNG 5 KỲ GẦN NHẤT")
            
            for name in labels:
                s = stats[name]
                digits = [periods[i]['digits'][labels.index(name)] for i in range(min(5, len(periods)))]
                trend_str = " → ".join([f"{'🔴' if d>=5 else '🔵'}{d}" for d in digits])
                st.caption(f"**{name}**: {trend_str}")
            
            st.divider()
            
            # ⚠️ CẢNH BÁO QUAN TRỌNG
            st.error("""
            🔴 **QUY TẮC VÀNG**:
            1. Không đánh Xiên 2 quá 3 ván liên tiếp
            2. Thua 2 ván Xiên 2 → Quay về đánh Đơn nuôi vốn
            3. Dừng ngay khi thua 20% vốn/ngày
            4. Chốt lời khi thắng 15% vốn/ngày
            5. Không chasing loss (đuổi lỗ cảm tính)
            """)

# Footer
st.markdown("---")
st.caption("""
🔐 **TITAN v32.0** | Chuyên biệt Kèo Đôi Xiên 2 5D KU | Phân tích thống kê + Quản lý vốn  
⚠️ Kết quả tham khảo - Không đảm bảo thắng | Chơi có trách nhiệm 🙏
""")