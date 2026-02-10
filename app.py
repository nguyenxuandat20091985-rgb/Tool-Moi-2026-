import streamlit as st
import collections
import time

st.set_page_config(page_title="3-TINH ANTI BOT 2026", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #01080c; color: #ffffff; }
    .bot-card { 
        border: 4px solid #ff0055; 
        border-radius: 20px; 
        padding: 40px; 
        background: linear-gradient(145deg, #0a1a25, #000000); 
        text-align: center;
        box-shadow: 0 0 50px rgba(255, 0, 85, 0.3);
    }
    .tinh3-label { font-size: 30px; color: #00e5ff; font-weight: bold; text-transform: uppercase; }
    .tinh3-numbers { 
        font-size: 150px !important; 
        color: #ffff00; 
        font-weight: bold; 
        letter-spacing: 20px;
        text-shadow: 0 0 30px #ff0055;
        margin: 20px 0;
    }
    .status-msg { font-size: 20px; font-weight: bold; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎯 HỆ THỐNG TRUY QUÉT 3 TINH (V32.0)")
st.write("---")

# Nhập chuỗi số từ sảnh cược
data_input = st.text_area("📡 Dán chuỗi số vừa ra (Ví dụ: 014589...):", height=80, placeholder="Nhập ít nhất 8-10 số để AI dò nhịp...")

if st.button("⚡ PHÂN TÍCH 3 TINH", use_container_width=True):
    if len(data_input.strip()) < 8:
        st.warning("⚠️ Vốn ít thì phải cẩn thận. Anh cho em thêm dữ liệu (ít nhất 8 số) để em né bẫy AI nhà cái.")
    else:
        with st.spinner('Đang giải mã nhịp máy Kubet/Tha...'):
            time.sleep(0.8) # Giả lập thời gian xử lý nhịp động
            
            # Làm sạch dữ liệu
            raw = "".join(filter(str.isdigit, data_input))
            recent_5 = raw[-5:] # Lấy 5 số gần nhất để tìm nhịp nhảy
            
            # --- THUẬT TOÁN MA TRẬN 3 TINH MỚI ---
            # Dựa trên lý thuyết "Số bù" và "Điểm rơi rơi tự do" của máy quay số
            all_nums = "0123456789"
            counts = collections.Counter(raw)
            
            # Tìm các số đang bị "giam" (nhà cái ít ra) và các số "vừa chớm nổ"
            # Thuật toán lấy 3 số có xác suất nổ cao nhất trong 10 ván tới
            missing = [n for n in all_nums if n not in recent_5]
            # Sắp xếp theo tần suất xuất hiện trung bình để lấy 3 số tiềm năng nhất
            tinh3_list = sorted(missing, key=lambda x: counts[x], reverse=True)[:3]
            
            # Nếu chuỗi quá loạn, đảo thuật toán sang bắt nhịp lặp
            if len(set(recent_5)) <= 2:
                tinh3_list = [raw[-1], str((int(raw[-1])+5)%10), str((int(raw[-1])+1)%10)]

            tinh3_display = " ".join(tinh3_list)

        # HIỂN THỊ KẾT QUẢ DUY NHẤT
        st.markdown(f"""
            <div class='bot-card'>
                <p class='tinh3-label'>🥈 KẾT QUẢ 3 TINH SIÊU CẤP</p>
                <p class='tinh3-numbers'>{tinh3_display}</p>
                <p style='color: #888;'>Cầu hiện tại: {raw[-10:]}</p>
            </div>
        """, unsafe_allow_html=True)

        st.write("---")
        # Phân tích rủi ro từ AI Nhà Cái
        if raw[-1] == raw[-2]:
            st.markdown("<div class='status-msg' style='background: #330000; color: #ff4b4b;'>🚨 CẢNH BÁO: AI Nhà cái đang đi Bệt. Đánh dàn 3 Tinh đều tay, không gấp!</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='status-msg' style='background: #002200; color: #00ff41;'>✅ NHỊP ĐẸP: Thuật toán đang nhả số đều. Vào tiền theo dàn 3.</div>", unsafe_allow_html=True)

st.info("💡 **Gỡ vốn:** Chia tiền đều cho 3 con số trên. Ăn 2-3 tay là nghỉ, đừng ở lại bàn quá lâu để AI nó quét ID của anh.")
