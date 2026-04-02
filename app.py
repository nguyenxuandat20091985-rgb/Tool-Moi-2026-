import streamlit as st
import pandas as pd

st.title("📊 Tool Soi Cầu 5D Bet - Hỗ Trợ Về Bờ")

# Nhập dữ liệu lịch sử (Nhập dãy số của 5-10 kỳ gần nhất)
data_input = st.text_area("Nhập kết quả các kỳ gần nhất (mỗi kỳ 5 số, cách nhau bằng dấu phẩy):", 
                         placeholder="12345, 67890, 11223...")

def analyze_5d(data_str):
    try:
        rows = [list(map(int, list(s.strip()))) for s in data_str.split(",") if len(s.strip()) == 5]
        df = pd.DataFrame(rows, columns=['Hàng 1', 'Hàng 2', 'Hàng 3', 'Hàng 4', 'Hàng 5'])
        
        st.subheader("📈 Kết quả phân tích:")
        for col in df.columns:
            most_common = df[col].mode()[0]
            st.write(f"**{col}:** Con số hay về nhất là **{most_common}**. Nên né hoặc đánh theo tùy cầu.")
            
        # Thuật toán dự đoán Tài/Xỉu hàng đơn vị (Hàng 5)
        last_val = df['Hàng 5'].iloc[-1]
        prediction = "TÀI" if last_val <= 4 else "XỈU" # Đánh nghịch đảo cầu
        st.success(f"🔮 Dự đoán kỳ tới (Hàng 5): {prediction}")
        
    except:
        st.warning("Vui lòng nhập đúng định dạng 5 chữ số mỗi kỳ.")

if st.button("Bắt đầu soi cầu"):
    if data_input:
        analyze_5d(data_input)
