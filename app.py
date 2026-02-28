import streamlit as st
from datetime import datetime
import re
import time

st.set_page_config(page_title="TITAN v38.0 - AUTO UPDATE", layout="centered", page_icon="🎯")

# --- SESSION STATE ---
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 500000
if 'period_count' not in st.session_state:
    st.session_state.period_count = 690
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_analysis' not in st.session_state:
    st.session_state.last_analysis = None
if 'last_input_hash' not in st.session_state:
    st.session_state.last_input_hash = ""

# --- HÀM PHÂN LOẠI SỐ ---
def get_classifications(digit):
    digit = int(digit)
    res = {}
    res['tai_xiu'] = 'TÀI' if digit >= 5 else 'XỈU'
    res['le_chan'] = 'LẺ' if digit % 2 != 0 else 'CHẴN'
    res['to_hop'] = 'TỐ' if digit in [1,2,3,5,7] else 'HỢP'
    return res

# --- HÀM PHÂN TÍCH ---
def analyze_kèo_đôi(raw_text):
    lines = [re.sub(r'[^\d]', '', l.strip()) for l in raw_text.strip().split('\n')]
    valid = [l for l in lines if len(l) == 5 and l.isdigit()]
    
    if len(valid) < 5:
        return None, None, f"Cần ít nhất 5 kỳ hợp lệ (hiện có: {len(valid)})"
    
    positions = ["Chục Ngàn", "Ngàn", "Trăm", "Chục", "Đơn Vị"]
    signals = []
    
    # Phân tích từng vị trí
    for pos_idx, pos_name in enumerate(positions):
        digits = [int(line[pos_idx]) for line in valid[:15]]
        total = len(digits)
        
        tai_count = sum(1 for d in digits if d >= 5)
        le_count = sum(1 for d in digits if d % 2 != 0)
        to_count = sum(1 for d in digits if d in [1,2,3,5,7])
        
        tai_rate = tai_count / total
        le_rate = le_count / total
        to_rate = to_count / total
        
        # Tín hiệu Tài/Xỉu
        if tai_rate >= 0.7:
            signals.append({'position': pos_name, 'type': 'TÀI/XỈU', 'bet': 'XỈU', 
                          'confidence': int(tai_rate * 100), 'reason': f'Bệt Tài {tai_count}/{total}',
                          'stats': f'Tài: {tai_count}, Xỉu: {total-tai_count}'})
        elif tai_rate <= 0.3:
            signals.append({'position': pos_name, 'type': 'TÀI/XỈU', 'bet': 'TÀI', 
                          'confidence': int((1-tai_rate) * 100), 'reason': f'Bệt Xỉu {total-tai_count}/{total}',
                          'stats': f'Tài: {tai_count}, Xỉu: {total-tai_count}'})
            
        # Tín hiệu Lẻ/Chẵn
        if le_rate >= 0.7:
            signals.append({'position': pos_name, 'type': 'LẺ/CHẴN', 'bet': 'CHẴN', 
                          'confidence': int(le_rate * 100), 'reason': f'Bệt Lẻ {le_count}/{total}',
                          'stats': f'Lẻ: {le_count}, Chẵn: {total-le_count}'})
        elif le_rate <= 0.3:
            signals.append({'position': pos_name, 'type': 'LẺ/CHẴN', 'bet': 'LẺ', 
                          'confidence': int((1-le_rate) * 100), 'reason': f'Bệt Chẵn {total-le_count}/{total}',
                          'stats': f'Lẻ: {le_count}, Chẵn: {total-le_count}'})
            
        # Tín hiệu Tố/Hợp
        if to_rate >= 0.7:
            signals.append({'position': pos_name, 'type': 'TỐ/HỢP', 'bet': 'HỢP', 
                          'confidence': int(to_rate * 100), 'reason': f'Bệt Tố {to_count}/{total}',
                          'stats': f'Tố: {to_count}, Hợp: {total-to_count}'})
        elif to_rate <= 0.3:
            signals.append({'position': pos_name, 'type': 'TỐ/HỢP', 'bet': 'TỐ', 
                          'confidence': int((1-to_rate) * 100), 'reason': f'Bệt Hợp {total-to_count}/{total}',
                          'stats': f'Tố: {to_count}, Hợp: {total-to_count}'})
    
    # Sắp xếp theo độ tin cậy
    signals.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Chọn kèo Đơn Thức tốt nhất
    single_bet = signals[0] if signals else None
    
    # Chọn kèo Xiên 2 (2 tín hiệu mạnh nhất khác vị trí)
    xien2_bet = None
    if len(signals) >= 2:
        pos_used = [single_bet['position']]
        for s in signals[1:]:
            if s['position'] not in pos_used:
                xien2_bet = [single_bet, s]
                break
    
    return single_bet, xien2_bet, None

# --- GIAO DIỆN ---
st.title("🎯 TITAN v38.0 - TỰ ĐỘNG CẬP NHẬT")

# Countdown
now = datetime.now()
seconds = now.second
remaining = 60 - seconds if seconds < 30 else 30 - (seconds - 30)
st.info(f"🕒 **Kỳ tiếp theo sau: {remaining:02d} giây**")

# Sidebar
with st.sidebar:
    st.header("💰 Quản lý vốn")
    st.session_state.bankroll = st.number_input("Vốn hiện tại (đ)", value=st.session_state.bankroll, step=10000)
    
    recommended_bet = min(20000, int(st.session_state.bankroll * 0.02))
    st.metric("✅ Cược đề xuất", f"{recommended_bet:,}đ")
    st.metric("💵 Vốn hiện tại", f"{st.session_state.bankroll:,.0f}đ")
    
    st.divider()
    st.subheader("📜 Lịch sử")
    if st.session_state.history:
        for h in st.session_state.history[-5:]:
            icon = "🟢" if h['result'] == 'WIN' else "🔴"
            st.caption(f"{icon} Kỳ {h['period']}: {h['type']} → {h['pnl']:+,}đ")
    else:
        st.caption("Chưa có lịch sử")
    
    if st.button("🔄 Reset toàn bộ", use_container_width=True):
        st.session_state.history = []
        st.session_state.period_count = 690
        st.session_state.last_analysis = None
        st.session_state.last_input_hash = ""
        st.rerun()

# Form nhập liệu
st.subheader("📥 Nhập kết quả 15 kỳ gần nhất")
raw = st.text_area(
    "Dán kết quả (mới nhất trên cùng):",
    placeholder="87746\n56421\n69137\n...",
    height=150,
    key="input_text"  # Key cố định để track changes
)

# Kiểm tra nếu input thay đổi
current_hash = hash(raw) if raw else ""
input_changed = (current_hash != st.session_state.last_input_hash) and len(raw.strip()) > 0

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    submitted = st.button("⚡ PHÂN TÍCH NGAY", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ XÓA INPUT", use_container_width=True)
with col3:
    if st.button("🔄 LÀM MỚI", use_container_width=True):
        st.rerun()

# Xử lý khi xóa
if clear_btn:
    st.session_state.last_analysis = None
    st.session_state.last_input_hash = ""
    st.rerun()

# Xử lý khi submit HOẶC input thay đổi
if (submitted or input_changed) and raw:
    with st.spinner(f"🔄 Đang phân tích {len([l for l in raw.split('\\n') if l.strip()])} kỳ..."):
        time.sleep(0.3)  # Hiệu ứng loading
        
        # Cập nhật hash
        st.session_state.last_input_hash = current_hash
        
        single, xien2, error = analyze_kèo_đôi(raw)
        
        if error:
            st.warning(f"⚠️ {error}")
        else:
            # Lưu kết quả phân tích
            st.session_state.last_analysis = {
                'single': single,
                'xien2': xien2,
                'time': datetime.now().strftime("%H:%M:%S"),
                'period': st.session_state.period_count + 1
            }
            
            # Hiển thị kết quả
            st.success(f"## ✅ PHÂN TÍCH KỲ {st.session_state.last_analysis['period']}")
            st.caption(f"⏰ Cập nhật lúc: {st.session_state.last_analysis['time']}")
            st.divider()
            
            # --- KHUYẾN NGHỊ ĐƠN THỨC ---
            st.subheader("1️⃣ KÈO ĐÔI - ĐƠN THỨC (An toàn)")
            if single:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📍 Vị trí", single['position'])
                with col2:
                    st.metric("🔴 Đánh", f"{single['type']} → {single['bet']}")
                with col3:
                    st.metric("⚡ Độ tin cậy", f"{single['confidence']}%")
                
                st.info(f"📊 **Lý do:** {single['reason']}")
                st.caption(f"📈 Thống kê: {single.get('stats', 'N/A')}")
                
                # Nút hành động
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ THẮNG (Đơn)", type="primary", use_container_width=True, key="win_single"):
                        profit = int(recommended_bet * 0.985)
                        st.session_state.bankroll += profit
                        st.session_state.history.append({
                            'period': st.session_state.last_analysis['period'], 
                            'type': 'Đơn', 
                            'result': 'WIN', 
                            'pnl': profit
                        })
                        st.balloons()
                        st.success(f"🎉 +{profit:,}đ")
                        st.rerun()
                with c2:
                    if st.button("❌ THUA (Đơn)", type="secondary", use_container_width=True, key="lose_single"):
                        st.session_state.bankroll -= recommended_bet
                        st.session_state.history.append({
                            'period': st.session_state.last_analysis['period'], 
                            'type': 'Đơn', 
                            'result': 'LOSE', 
                            'pnl': -recommended_bet
                        })
                        st.error(f"💸 -{recommended_bet:,}đ")
                        st.rerun()
            else:
                st.warning("Không có tín hiệu đơn thức đủ mạnh.")
                
            st.divider()
            
            # --- KHUYẾN NGHỊ XIÊN 2 ---
            st.subheader("2️⃣ KÈO ĐÔI - CƯỢC XIÊN 2 (Hiệu quả cao)")
            if xien2:
                s1, s2 = xien2
                st.warning(f"🔗 **Kết hợp:** {s1['position']} ({s1['bet']}) + {s2['position']} ({s2['bet']})")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📍 Vị trí 1", f"{s1['position']}\n{s1['bet']}")
                with col2:
                    st.metric("📍 Vị trí 2", f"{s2['position']}\n{s2['bet']}")
                
                avg_conf = int((s1['confidence'] + s2['confidence']) / 2)
                st.progress(avg_conf / 100)
                st.caption(f"⚡ Độ tin cậy trung bình: {avg_conf}%")
                
                xien_bet_amount = min(10000, int(st.session_state.bankroll * 0.01))
                st.info(f"💰 **Mức cược Xiên gợi ý:** {xien_bet_amount:,}đ (1% vốn)")
                
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✅ THẮNG (Xiên)", type="primary", use_container_width=True, key="win_xien"):
                        profit = int(xien_bet_amount * 3.6)
                        st.session_state.bankroll += profit
                        st.session_state.history.append({
                            'period': st.session_state.last_analysis['period'], 
                            'type': 'Xiên 2', 
                            'result': 'WIN', 
                            'pnl': profit
                        })
                        st.balloons()
                        st.success(f"🎉 +{profit:,}đ")
                        st.rerun()
                with c2:
                    if st.button("❌ THUA (Xiên)", type="secondary", use_container_width=True, key="lose_xien"):
                        st.session_state.bankroll -= xien_bet_amount
                        st.session_state.history.append({
                            'period': st.session_state.last_analysis['period'], 
                            'type': 'Xiên 2', 
                            'result': 'LOSE', 
                            'pnl': -xien_bet_amount
                        })
                        st.error(f"💸 -{xien_bet_amount:,}đ")
                        st.rerun()
            else:
                st.warning("Không có tín hiệu Xiên 2 đủ mạnh.")
            
            if st.session_state.bankroll < 400000:
                st.error("🛑 **CẢNH BÁO:** Vốn giảm >20%. Nên dừng lại!")

elif not raw:
    st.info("👆 Anh nhập số vào ô trên rồi bấm 'PHÂN TÍCH NGAY' nhé!")

# Footer
st.markdown("---")
st.caption("🎯 TITAN v38.0 | Tự động cập nhật khi nhập số mới | Chơi có trách nhiệm 🙏")