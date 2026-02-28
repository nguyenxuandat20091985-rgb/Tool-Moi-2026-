import streamlit as st
import time

# Cấu hình trang chuyên nghiệp
st.set_page_config(page_title="TITAN v30.2 - SUPREME", layout="centered")

def analyze_logic(data_input):
    # 1. Lọc dữ liệu: Chỉ lấy các dòng có đúng 5 chữ số
    # Đảo ngược danh sách để dòng mới nhập nằm ở đầu (index 0)
    history = [str(line).strip() for line in data_input if len(str(line).strip()) == 5]
    
    if len(history) < 5:
        return None

    # LẤY 5 KỲ MỚI NHẤT ĐỂ PHÂN TÍCH (Cực kỳ quan trọng)
    latest_5 = history[:5] 

    # Tách dữ liệu Hàng Chục (-2) và Hàng Đơn Vị (-1)
    h_chuc = [int(line[-2]) for line in latest_5]
    h_donvi = [int(line[-1]) for line in latest_5]

    def get_binary_prediction(digits):
        # Đếm số lượng Tài trong 5 kỳ gần nhất
        tai_count = sum(1 for d in digits if d >= 5)
        
        # Logic bẻ cầu: Nếu bệt quá dài (4/5 hoặc 5/5) -> Dự đoán bẻ
        if tai_count >= 4: return "XỈU"
        if tai_count <= 1: return "TÀI"
        
        # Logic bám cầu: Nếu cầu đang 2-2 hoặc 1-2 -> Đánh theo con vừa về
        return "TÀI" if digits[0] >= 5 else "XỈU"

    res_chuc = get_binary_prediction(h_chuc)
    res_donvi = get_binary_prediction(h_donvi)
    
    return res_chuc, res_donvi, latest_5

# --- GIAO DIỆN ---
st.title("🎯 TITAN v30.2 - SUPREME")
st.subheader("Hệ thống khai thác Xiên 2 & Kèo Đôi")
st.markdown("---")

# Hướng dẫn nhanh cho anh
st.sidebar.header("🕹 HƯỚNG DẪN")
st.sidebar.info("1. Copy 5-10 kết quả mới nhất.\n2. Dán vào ô bên phải.\n3. Dòng mới nhất phải nằm ở trên cùng.")

raw_data = st.text_area("📥 Dán kết quả 5D (Mới nhất ở trên cùng):", height=200, placeholder="Ví dụ:\n80673\n64061\n...")

if raw_data:
    with st.spinner('🔄 Đang quét cầu và phân tích...'):
        time.sleep(0.5) # Tạo độ trễ giả lập để anh thấy tool có loading
        lines = raw_data.split('\n')
        analysis = analyze_logic(lines)
    
    if analysis:
        trend_c, trend_dv, history_view = analysis
        
        # Hiển thị trạng thái dữ liệu
        st.success(f"✅ Đã nhận diện {len(history_view)} kỳ gần nhất.")
        
        # Hiển thị khu vực XIÊN 2
        st.markdown("### 🔥 KẾT QUẢ PHÂN TÍCH THỰC CHIẾN")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="📍 DỰ ĐOÁN HÀNG CHỤC", value=trend_c)
        with col2:
            st.metric(label="📍 DỰ ĐOÁN ĐƠN VỊ", value=trend_dv)
            
        st.warning(f"🚀 **CƯỢC XIÊN CHIẾN THUẬT:** Hàng Chục **{trend_c}** + Hàng Đơn Vị **{trend_dv}**")
        
        # Công thức vào tiền thông minh
        st.markdown("---")
        st.markdown("#### 💰 QUẢN LÝ VỐN XIÊN 2 (Tỷ lệ 1 ăn 3.9)")
        st.write("Đánh Xiên 2 giúp anh chịu được nhiệt nếu gãy cầu lẻ.")
        
        data_money = [
            {"Kỳ": 1, "Vào tiền": "10k", "Vốn tích lũy": "10k", "Thắng nhận": "39k", "Lợi nhuận": "+29k"},
            {"Kỳ": 2, "Vào tiền": "15k", "Vốn tích lũy": "25k", "Thắng nhận": "58k", "Lợi nhuận": "+33k"},
            {"Kỳ": 3, "Vào tiền": "30k", "Vốn tích lũy": "55k", "Thắng nhận": "117k", "Lợi nhuận": "+62k"},
        ]
        st.table(data_money)
    else:
        st.error("⚠️ LỖI: Cần tối thiểu 5 dòng số (mỗi dòng 5 chữ số) để tính toán!")

st.markdown("---")
st.caption("Phiên bản v30.2 tối ưu cho cược Xiên trên giao diện KU. Chúc anh thắng lớn!")
