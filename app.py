# ================= CAPITAL MANAGEMENT (FUN CODE) =================
def render_money_management(win_rate):
    st.divider()
    st.subheader("💰 CHIẾN THUẬT QUẢN LÝ VỐN")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        base_bet = st.number_input("Tiền cược cơ sở (VNĐ)", min_value=10000, value=50000, step=10000)
        strategy = st.selectbox("Chiến thuật", ["An toàn (Cố định)", "Gấp thếp (Martingale)", "Thông minh (Kelly Criterion)"])
    
    with col2:
        if strategy == "An toàn (Cố định)":
            st.info(f"Mỗi kỳ đánh đúng: **{base_bet:,} VNĐ**. Mục tiêu bền bỉ.")
        
        elif strategy == "Gấp thếp (Martingale)":
            st.warning("⚠️ Cẩn thận: Chỉ dành cho vốn dày!")
            steps = [base_bet * (2**i) for i in range(5)]
            st.write("Lộ trình vào tiền (nếu chưa về):")
            st.code(" -> ".join([f"{x:,}" for x in steps]))
            
        elif strategy == "Thông minh (Kelly Criterion)":
            # Công thức Kelly: f* = (bp - q) / b 
            # p: tỷ lệ thắng, q: tỷ lệ thua, b: tỷ lệ ăn (ở đây Lotobet 2 tinh thường là 1 ăn 95-99)
            b = 95 
            p = win_rate / 100
            q = 1 - p
            kelly_f = max(0, (b * p - q) / b) * 0.1 # Chỉ dùng 10% của Kelly để an toàn
            
            suggested = base_bet * (1 + kelly_f)
            st.success(f"Dựa trên tỷ lệ thắng {win_rate:.1f}%, AI khuyên vào: **{suggested:,.0f} VNĐ**")

    # Vẽ biểu đồ mô phỏng tăng trưởng vốn
    st.caption("Biểu đồ mô phỏng tăng trưởng vốn dự kiến")
    simulation = pd.DataFrame({
        "Kỳ": np.arange(1, 11),
        "Vốn dự kiến": np.cumsum(np.random.normal(win_rate - 50, 20, 10)) + 1000 # Demo vui vẻ
    })
    st.line_chart(simulation, x="Kỳ", y="Vốn dự kiến")

# Thêm dòng này vào cuối hàm main() trong code của anh:
# render_money_management(win_rate)
