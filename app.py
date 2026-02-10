import streamlit as st
import collections
import time

# Cấu hình trang trang nhã, chuyên nghiệp hơn
st.set_page_config(page_title="AI 3-TINH PRO v33", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #0b0f13; color: #e0e0e0; }
    /* Card kết quả tinh tế hơn */
    .result-card { 
        border: 1px solid #00e5ff; 
        border-radius: 15px; 
        padding: 25px; 
        background: #161b22; 
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        margin-top: 20px;
    }
    .label-text { font-size: 18px; color: #8b949e; margin-bottom: 10px; }
    /* Số kết quả vừa phải, dễ nhìn không bị lóa */
    .numbers-display { 
        font-size: 70px !important; 
        color: #00ffcc; 
        font-weight: bold; 
        letter-spacing: 15px;
        text-shadow: 0 0 10px rgba(0, 255, 204, 0.3);
        margin: 10px 0;
    }
    .status-bar { 
        padding: 10px 20px; 
        border-radius: 8px; 
        font-weight: bold; 
        margin-top: 15px;
        font-size: 14px;
    }
    /* Tùy chỉnh ô nhập liệu */
    .stTextArea textarea { background-color: #0d1117 !important; color: #00ffcc !important; border: 1px solid #30363d !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎯 HỆ THỐNG SOI 3 TINH v33.0")
st.write("---")

# Nhập chuỗi số
data_input = st.text_area("📡 Dán chuỗi số thực tế (Nhập từ 8 số trở lên):", height=100, placeholder="Ví dụ: 01458923...")

if st.button("🚀 TRUY QUÉT NHỊP MÁY", use_container_width=True):
    if len(data_input.strip()) < 8:
        st.error("⚠️ Dữ liệu quá ngắn! AI nhà cái rất lọc lõi, anh cần dán thêm số để em tính toán chính xác.")
    else:
        with st.spinner('Đang dò sóng thuật toán...'):
            time.sleep(0.6)
            raw = "".join(filter(str.isdigit, data_input))
            
            # --- THUẬT TOÁN "BÓNG NHẢY" CẬP NHẬT ---
            counts = collections.Counter(raw)
            last_num = int(raw[-1])
            
            # Phân tích chu kỳ dựa trên 10 con số
            all_nums = [str(i) for i in range(10)]
            # Ưu tiên những con số đang "vào nhịp" (không quá khan nhưng cũng không quá dày)
            potential = sorted(all_nums, key=lambda x: counts[x])
            
            # Logic: Lấy 1 con bóng, 1 con kề, 1 con lặp (tạo thành dàn 3 tinh vững)
            t1 = str((last_num + 5) % 10) # Số bóng
            t2 = potential[0] # Số đang bị giam (khả năng nổ bù)
            t3 = potential[1] # Số nhịp trung bình
            
            tinh3_list = list(set([t1, t2, t3]))
            # Đảm bảo luôn đủ 3 số
            while len(tinh3_list) < 3:
                tinh3_list.append(str((int(tinh3_list[-1]) + 1) % 10))
            
            tinh3_display = " ".join(tinh3_list[:3])

        # HIỂN THỊ KẾT QUẢ
        st.markdown(f"""
            <div class='result-card'>
                <p class='label-text'>🥈 DÀN 3 TINH ĐỀ XUẤT</p>
                <p class='numbers-display'>{tinh3_display}</p>
                <p style='color: #58a6ff; font-size: 14px;'>Nhịp cuối ghi nhận: {raw[-1]}</p>
            </div>
        """, unsafe_allow_html=True)

        # Cảnh báo nhịp độ
        if len(set(raw[-4:])) <= 2:
            st.markdown("<div class='status-bar' style='background: #3e1b1b; color: #ff7b72;'>🚨 CẢNH BÁO: Cầu đang bệt/lặp. Đánh nhẹ tay chờ nhịp gãy!</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='status-bar' style='background: #1b2e1b; color: #7ee787;'>✅ TÍN HIỆU: Nhịp nhảy đều. Có thể vào tiền dàn 3.</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("💡 Mẹo: Khi vốn cạn, anh hãy đánh theo kiểu 'du kích'. Thắng 1 tay dàn 3 là nghỉ, chờ 5-10 ván sau mới dán số quét lại một lần.")
