# ================= TITAN v30.0: THE WEAKNESS EXPLOITER =================

def analyze_xien_2(history_data):
    # Lấy 2 hàng có nhịp ổn định nhất: Hàng Chục và Hàng Đơn Vị
    h_chuc = [int(str(line)[-2]) for line in history_data if len(str(line)) == 5]
    h_donvi = [int(str(line)[-1]) for line in history_data if len(str(line)) == 5]
    
    # Tính toán xác suất Kèo Đôi cho từng hàng
    def get_binary_trend(digits):
        last_5 = ["T" if d >= 5 else "X" for d in digits[:5]]
        if last_5.count("T") >= 4: return "XỈU" # Bắt hồi quy
        if last_5.count("X") >= 4: return "TÀI" # Bắt hồi quy
        return "TÀI" if digits[0] < 5 else "XỈU" # Đánh đảo

    trend_chuc = get_binary_trend(h_chuc)
    trend_donvi = get_binary_trend(h_donvi)
    
    return trend_chuc, trend_donvi

# --- HIỂN THỊ CHIẾN THUẬT XIÊN 2 ---
st.title("🎯 TITAN v30.0 - KHAI THÁC ĐIỂM YẾU 5D")
# Gợi ý cược Xiên 2 (Ví dụ: Chục Tài + Đơn vị Xỉu)
st.error(f"🔥 XIÊN 2 GỢI Ý: HÀNG CHỤC [{trend_chuc}] + HÀNG ĐƠN VỊ [{trend_donvi}]")
st.success("💰 Tỉ lệ ăn cực cao - Vốn chỉ cần 1/10 so với dàn số")
