# ============= 🎯 CÁC THUẬT TOÁN BỔ SUNG ============

def detect_special_patterns(db):
    """
    Phát hiện các cầu đặc biệt nhà cái hay dùng
    """
    if len(db) < 20:
        return {}
    
    patterns = {
        'cau_rong': [],      # Cầu rồng: số tăng dần
        'cau_gan': [],       # Cầu gan: số lâu chưa về
        'cau_bac_nho': [],   # Cầu bạc nhớ: số hay về sau số khác
        'cau_total': [],     # Cầu tổng: tổng 2 số cuối
        'cau_dau_duoi': []   # Cầu đầu đuôi: ghép đầu kỳ trước + đuôi kỳ này
    }
    
    # 1. Cầu Rồng (Sequence tăng/giảm)
    for i in range(len(db)-3):
        curr = db[i]
        next_num = db[i+1]
        # Kiểm tra sequence tăng dần từng vị trí
        is_sequence = True
        for pos in range(5):
            if int(next_num[pos]) != (int(curr[pos]) + 1) % 10:
                is_sequence = False
                break
        if is_sequence:
            patterns['cau_rong'].append(next_num)
    
    # 2. Cầu Gan (Số lâu chưa về)
    for digit in range(10):
        ds = str(digit)
        last_seen = -1
        for i in range(len(db)-1, -1, -1):
            if ds in db[i]:
                last_seen = len(db) - 1 - i
                break
        if last_seen > 5:  # Gan trên 5 kỳ
            patterns['cau_gan'].append((ds, last_seen))
    
    # 3. Cầu Bạc Nhớ (Số A về thì số B hay về)
    memory = defaultdict(lambda: Counter())
    for i in range(len(db)-1):
        curr_digits = set(db[i])
        next_digits = set(db[i+1])
        for d1 in curr_digits:
            for d2 in next_digits:
                memory[d1][d2] += 1
    
    # Tìm các cặp có tần suất > 3
    for d1, counter in memory.items():
        for d2, count in counter.items():
            if count >= 3 and d1 != d2:
                patterns['cau_bac_nho'].append((d1, d2, count))
    
    # 4. Cầu Tổng (Tổng 2 số cuối)
    totals = Counter()
    for num in db[-30:]:
        total = (int(num[-2]) + int(num[-1])) % 10
        totals[total] += 1
    
    for total, count in totals.most_common(3):
        patterns['cau_total'].append((total, count))
    
    # 5. Cầu Đầu Đuôi
    dau_duoi = Counter()
    for i in range(len(db)-1):
        dau = db[i][0]  # Đầu kỳ trước
        duoi = db[i+1][-1]  # Đuôi kỳ này
        dau_duoi[dau + duoi] += 1
    
    for combo, count in dau_duoi.most_common(5):
        patterns['cau_dau_duoi'].append((combo, count))
    
    return patterns

def predict_with_folk_wisdom(db, patterns):
    """
    Dự đoán theo kinh nghiệm dân gian Việt Nam
    """
    predictions = {
        'lo_kep': [],      # Lô kép
        'lo_ra_cung': [],  # Lô ra cùng
        'lo_cam': [],      # Lô câm (đầu/đuôi câm)
        'lo_bach_thu': []  # Bạch thủ
    }
    
    if len(db) < 15:
        return predictions
    
    last_num = db[-1]
    
    # 1. Lô Kép (00, 11, 22, ..., 99)
    # Kiểm tra xem kỳ trước có về kép không
    for i in range(5):
        if last_num[i] == last_num[(i+1)%5]:
            # Về kép → kỳ sau hay về kép khác
            kep_so = last_num[i]
            predictions['lo_kep'].append(f"{kep_so}{kep_so}")
    
    # 2. Lô Ra Cùng (Số hay về cùng nhau)
    pair_freq = Counter()
    for num in db[-40:]:
        digits = list(set(num))
        for i in range(len(digits)):
            for j in range(i+1, len(digits)):
                pair = "".join(sorted([digits[i], digits[j]]))
                pair_freq[pair] += 1
    
    for pair, count in pair_freq.most_common(5):
        predictions['lo_ra_cung'].append((pair, count))
    
    # 3. Lô Câm (Đầu/đuôi không về)
    dau_ve = Counter(num[0] for num in db[-20:])
    duoi_ve = Counter(num[-1] for num in db[-20:])
    
    for d in range(10):
        ds = str(d)
        if dau_ve[ds] == 0:
            predictions['lo_cam'].append(f"Đầu {ds} câm")
        if duoi_ve[ds] == 0:
            predictions['lo_cam'].append(f"Đuôi {ds} câm")
    
    # 4. Bạch Thủ (Số chắc nhất)
    all_digits = "".join(db[-30:])
    freq = Counter(all_digits)
    top_digit = freq.most_common(1)[0][0]
    predictions['lo_bach_thu'].append(top_digit)
    
    return predictions

def detect_house_algorithm(db):
    """
    Phát hiện thuật toán nhà cái đang dùng
    """
    if len(db) < 30:
        return "Không đủ dữ liệu"
    
    checks = {
        'random': 0,
        'pattern': 0,
        'cycle': 0,
        'balanced': 0
    }
    
    # Kiểm tra tính ngẫu nhiên
    all_digits = "".join(db[-50:])
    freq = Counter(all_digits)
    expected = len(all_digits) / 10
    variance = sum((c - expected)**2 for c in freq.values()) / 10
    
    if variance < 5:
        checks['random'] += 1
    else:
        checks['pattern'] += 1
    
    # Kiểm tra chu kỳ
    for period in [5, 7, 10]:
        matches = 0
        for i in range(len(db) - period):
            if db[i] == db[i + period]:
                matches += 1
        if matches > 2:
            checks['cycle'] += 1
    
    # Kiểm tra cân bằng
    total_sum = sum(int(d) for d in all_digits)
    avg = total_sum / len(all_digits)
    if 4.0 <= avg <= 5.0:
        checks['balanced'] += 1
    
    # Kết luận
    max_check = max(checks, key=checks.get)
    algorithms = {
        'random': "🎲 RNG thuần ngẫu nhiên",
        'pattern': "📊 Pattern-based (có quy luật)",
        'cycle': "🔄 Cycle-based (chu kỳ)",
        'balanced': "⚖️ Balanced distribution"
    }
    
    return algorithms[max_check]

def enhance_prediction_with_deep_analysis(db):
    """
    Dự đoán nâng cao với phân tích sâu
    """
    if len(db) < 20:
        return None
    
    # 1. Phân tích pattern nhà cái
    house_algo = detect_house_algorithm(db)
    
    # 2. Phát hiện special patterns
    patterns = detect_special_patterns(db)
    
    # 3. Folk wisdom predictions
    folk_preds = predict_with_folk_wisdom(db, patterns)
    
    # 4. Tính toán nâng cao cho 2-tinh và 3-tinh
    enhanced_pairs = []
    enhanced_triples = []
    
    # Kết hợp nhiều yếu tố
    all_digits = "".join(db[-50:])
    freq = Counter(all_digits)
    
    # Score cho từng cặp số
    for i in range(10):
        for j in range(i+1, 10):
            pair = f"{i}{j}"
            score = 0
            
            # Tần suất
            score += freq[str(i)] + freq[str(j)]
            
            # Cầu bạc nhớ
            for d1, d2, count in patterns.get('cau_bac_nho', []):
                if (str(i) == d1 and str(j) == d2) or (str(i) == d2 and str(j) == d1):
                    score += count * 3
            
            # Lô ra cùng
            for p, count in folk_preds.get('lo_ra_cung', []):
                if p == pair:
                    score += count * 2
            
            # Cầu gan
            for digit, gan_count in patterns.get('cau_gan', []):
                if digit in pair:
                    score += gan_count
            
            enhanced_pairs.append((pair, score))
    
    # Sort và lấy top 3
    enhanced_pairs.sort(key=lambda x: x[1], reverse=True)
    top_pairs = [p[0] for p in enhanced_pairs[:3]]
    
    # Tương tự cho 3-tinh
    for i in range(10):
        for j in range(i+1, 10):
            for k in range(j+1, 10):
                triple = f"{i}{j}{k}"
                score = freq[str(i)] + freq[str(j)] + freq[str(k)]
                
                # Bonus nếu có trong bạc nhớ
                for d1, d2, count in patterns.get('cau_bac_nho', []):
                    if str(i) in (d1,d2) or str(j) in (d1,d2) or str(k) in (d1,d2):
                        score += count
                
                enhanced_triples.append((triple, score))
    
    enhanced_triples.sort(key=lambda x: x[1], reverse=True)
    top_triples = [t[0] for t in enhanced_triples[:3]]
    
    return {
        'house_algorithm': house_algo,
        'patterns': patterns,
        'folk_predictions': folk_preds,
        'enhanced_pairs': top_pairs,
        'enhanced_triples': top_triples,
        'analysis': {
            'cau_rong_count': len(patterns.get('cau_rong', [])),
            'cau_gan_count': len(patterns.get('cau_gan', [])),
            'bac_nho_count': len(patterns.get('cau_bac_nho', []))
        }
    }

# ============= 🖥️ THÊM VÀO GIAO DIỆN =============

# Thêm vào sau phần user_input:

if st.button("🔍 PHÂN TÍCH SÂU CẦU"):
    nums = get_nums(user_input)
    if len(nums) >= 30:
        with st.spinner("🔄 Đang phân tích chuyên sâu..."):
            deep_analysis = enhance_prediction_with_deep_analysis(nums)
            st.session_state.deep_analysis = deep_analysis
            st.rerun()
    else:
        st.warning("Cần ít nhất 30 kỳ để phân tích sâu!")

# Hiển thị kết quả phân tích sâu
if "deep_analysis" in st.session_state:
    da = st.session_state.deep_analysis
    
    st.markdown("---")
    st.markdown("### 🔬 PHÂN TÍCH CHUYÊN SÂU")
    
    # Thuật toán nhà cái
    st.info(f"**🎯 Thuật toán nhà cái:** {da['house_algorithm']}")
    
    # Special patterns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🐉 Cầu Rồng", da['analysis']['cau_rong_count'])
    with col2:
        st.metric("💀 Cầu Gan", da['analysis']['cau_gan_count'])
    with col3:
        st.metric("💰 Cầu Bạc Nhớ", da['analysis']['bac_nho_count'])
    
    # Dự đoán nâng cao
    st.markdown("### 🎯 DỰ ĐOÁN NÂNG CAO")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**2 TINH TỐT NHẤT:**")
        for pair in da['enhanced_pairs']:
            st.markdown(f"🔸 {pair}")
    
    with col_b:
        st.markdown("**3 TINH TỐT NHẤT:**")
        for triple in da['enhanced_triples']:
            st.markdown(f"🔸 {triple}")
    
    # Folk wisdom
    if da['folk_predictions']['lo_kep']:
        st.warning(f"**Lô kép:** {', '.join(da['folk_predictions']['lo_kep'])}")
    
    if da['folk_predictions']['lo_cam']:
        st.error(f"**Lô câm:** {', '.join(da['folk_predictions']['lo_cam'])}")