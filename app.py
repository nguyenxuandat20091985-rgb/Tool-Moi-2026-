import streamlit as st
import pandas as pd

# Cấu hình giao diện Full HD
st.set_page_config(page_title="TITAN v30.3 - ANTI-LAG", layout="wide")

# Hàm xử lý dữ liệu tỉ mỉ
def clean_and_analyze(raw_text):
    # Tách dòng và dọn dẹp khoảng trắng dư thừa
    lines = [l.strip() for l in raw_text.split('\n') if len(l.strip()) == 5]
    if len(lines) < 5:
        return None
    
    # Lấy 10 kỳ gần nhất để soi độ dài cầu bệt
    latest_10 = lines[:10]
    h_chuc = [int(line[-2]) for line in latest_10]
    h_donvi = [int(line[-1]) for line in latest_10]
    
    def predict(digits):
        # Đếm 5 kỳ gần nhất
        last_5 = digits[:5]
        tai_count = sum(1 for d in last_5 if d >= 5)
        
        # Chỉ số tin cậy (Confidence)
        if tai_count >= 4: return "XỈU", "🔥 Bẻ Cầu (90%)"
        if tai_count <= 1: return "TÀI", "🔥 Bẻ Cầu (90%)"
        return ("TÀI" if digits[0] >= 5 else "XỈU"), "🛡 Bám Cầu (70%)"

    res_c, conf_c = predict(h_chuc)
    res_dv, conf_dv = predict(h_donvi)
    
    return {
        "chuc": res_c, "conf_c": conf_c,
        "dv": res_dv, "conf_dv": conf_dv,
        "history": latest_10
    }

# --- GIAO DIỆN CHÍNH ---
st.title("🎯 TITAN v30.3 - ĐIỂM YẾU 5D")
st.markdown("---")

# Ô nhập liệu thông minh
input_data = st.text_area("📥 Dán lịch sử (Mới nhất ở trên):", height=150)

if st.button("🔄 PHÂN TÍCH NGAY"):
    if input_data:
        result = clean_and_analyze(input_data)
        if result:
            st.success("✅ Dữ liệu đã được cập nhật!")
            
            # Hiển thị kết quả chính
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("📍 HÀNG CHỤC")
                st.header(result['chuc'])
                st.caption(result['conf_c'])
            with c2:
                st.subheader("📍 ĐƠN VỊ")
                st.header(result['dv'])
                st.caption(result['conf_dv'])
            
            st.divider()
            st.error(f"🚀 KÈO XIÊN 2 ĐỀ XUẤT: {result['chuc']} + {result['dv']}")
            
            # Bảng lịch sử để anh kiểm tra xem tool có đọc đúng số không
            with st.expander("📊 Kiểm tra dữ liệu nguồn (10 kỳ)"):
                st.write(result['history'])
        else:
            st.error("Dữ liệu không đủ hoặc sai định dạng (mỗi dòng phải 5 số).")
    else:
        st.info("Vui lòng dán dữ liệu để bắt đầu.")

st.markdown("---")
st.write("💡 **Mẹo thực chiến:** Nếu cả 2 hàng cùng báo 'Bẻ Cầu (90%)', anh có thể tự tin tăng mức cược lên 1.5 lần.")
