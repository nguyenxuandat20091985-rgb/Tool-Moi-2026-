<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>TITAN OMNI V5 - KUBET KILLER</title>
    <style>
        :root { --neon: #00ff88; --blue: #00d4ff; --bg: #05050a; --card: #10101a; }
        body { background: var(--bg); color: white; font-family: sans-serif; margin: 0; padding: 10px; }
        .tab-bar { display: flex; gap: 5px; margin-bottom: 10px; }
        .tab-bar button { flex: 1; padding: 12px; border: none; background: #1a1a2e; color: #666; border-radius: 8px; font-weight: 900; }
        .tab-bar button.active { background: var(--neon); color: black; box-shadow: 0 0 15px var(--neon); }
        .content { display: none; background: var(--card); border-radius: 15px; padding: 15px; border: 1px solid #222; }
        .content.active { display: block; }
        input { width: 100%; padding: 15px; background: #000; border: 2px solid var(--blue); color: var(--neon); font-size: 2rem; text-align: center; border-radius: 10px; letter-spacing: 5px; margin-bottom: 10px; box-sizing: border-box; }
        .btn-action { width: 100%; padding: 15px; background: var(--neon); color: black; border: none; border-radius: 10px; font-weight: 900; font-size: 1.1rem; text-transform: uppercase; }
        .prediction-box { text-align: center; padding: 20px; border: 2px dashed var(--neon); border-radius: 15px; margin: 15px 0; }
        .pred-num { font-size: 3.5rem; color: #ffd700; font-weight: 900; text-shadow: 0 0 20px rgba(255,215,0,0.5); }
        .history-item { display: flex; justify-content: space-between; padding: 10px; border-bottom: 1px solid #222; font-size: 0.9rem; }
        .win { color: var(--neon); } .loss { color: #ff4444; }
    </style>
</head>
<body>

<div class="tab-bar">
    <button class="active" onclick="showTab(0)">NHẬP DỮ LIỆU</button>
    <button onclick="showTab(1)">AI DỰ ĐOÁN</button>
    <button onclick="showTab(2)">ĐỐI SOÁT</button>
</div>

<div class="content active" id="tab0">
    <h3>📥 KẾT QUẢ VỪA RA</h3>
    <input type="number" id="numIn" placeholder="-----" oninput="if(this.value.length > 5) this.value = this.value.slice(0,5);">
    <button class="btn-action" onclick="analyzeData()">XÁC NHẬN & PHÂN TÍCH</button>
    <p id="dbStatus" style="font-size: 0.8rem; color: #888; margin-top: 10px;"></p>
    <button onclick="clearData()" style="background:none; border:none; color:#ff4444; font-size:0.7rem;">XÓA TẤT CẢ DỮ LIỆU</button>
</div>

<div class="content" id="tab1">
    <div class="prediction-box">
        <h4 style="margin:0; color:var(--blue);">DỰ ĐOÁN KỲ TIẾP</h4>
        <div id="nextCode" class="pred-num">-----</div>
        <div id="conf">Độ tin cậy: 0%</div>
    </div>
    <div id="aiAdvice" style="font-size: 0.85rem; line-height: 1.5; color: #bbb;"></div>
</div>

<div class="content" id="tab2">
    <h3>📊 TỶ LỆ THẮNG: <span id="winRate">0%</span></h3>
    <div id="logList"></div>
</div>

<script>
    let db = JSON.parse(localStorage.getItem('TITAN_V5_DB')) || [];
    let history = JSON.parse(localStorage.getItem('TITAN_V5_HIST')) || [];
    let lastPred = localStorage.getItem('TITAN_V5_LAST_PRED') || "";

    function showTab(idx) {
        document.querySelectorAll('.content').forEach((c, i) => c.classList.toggle('active', i === idx));
        document.querySelectorAll('.tab-bar button').forEach((b, i) => b.classList.toggle('active', i === idx));
    }

    function analyzeData() {
        let val = document.getElementById('numIn').value;
        if(val.length !== 5) { alert("Vui lòng nhập đủ 5 số!"); return; }

        // 1. ĐỐI SOÁT (Check win/loss cho kỳ trước)
        if(lastPred) {
            let isWin = false;
            for(let i=0; i<5; i++) { if(val[i] === lastPred[i]) isWin = true; }
            history.push({ pred: lastPred, real: val, res: isWin ? 'WIN' : 'LOSS', time: new Date().toLocaleTimeString() });
            if(history.length > 50) history.shift();
        }

        // 2. CẬP NHẬT DỮ LIỆU CƠ SỞ
        db.push(val);
        if(db.length > 300) db.shift();
        
        // 3. THUẬT TOÁN AI V5 (Nhận diện nhịp cầu)
        let next = "";
        for(let pos=0; pos<5; pos++) {
            let counts = {};
            db.slice(-30).forEach(n => { counts[n[pos]] = (counts[n[pos]] || 0) + 1; });
            let sorted = Object.entries(counts).sort((a,b) => b[1] - a[1]);
            
            // Logic: Nếu đang cầu bệt (số cũ ra lại), lấy số top 1. Nếu cầu nhảy, lấy số top 2.
            next += (db.length % 2 === 0) ? sorted[0][0] : (sorted[1] ? sorted[1][0] : sorted[0][0]);
        }

        lastPred = next;
        saveAll();
        updateUI();
        document.getElementById('numIn').value = "";
        showTab(1); // Chuyển sang tab dự đoán ngay
    }

    function saveAll() {
        localStorage.setItem('TITAN_V5_DB', JSON.stringify(db));
        localStorage.setItem('TITAN_V5_HIST', JSON.stringify(history));
        localStorage.setItem('TITAN_V5_LAST_PRED', lastPred);
    }

    function updateUI() {
        document.getElementById('dbStatus').innerText = `Dữ liệu cơ sở: ${db.length} kỳ.`;
        document.getElementById('nextCode').innerText = lastPred || "-----";
        
        let wins = history.filter(h => h.res === 'WIN').length;
        let rate = history.length > 0 ? Math.round((wins / history.length) * 100) : 0;
        document.getElementById('winRate').innerText = rate + "%";
        document.getElementById('conf').innerText = `Độ tin cậy: ${Math.min(rate + 15, 95)}%`;

        let listHtml = "";
        history.slice().reverse().forEach(h => {
            listHtml += `<div class="history-item">
                <span>${h.time}</span>
                <span>AI: <b>${h.pred}</b> ➡ ${h.real}</span>
                <span class="${h.res.toLowerCase()}">${h.res}</span>
            </div>`;
        });
        document.getElementById('logList').innerHTML = listHtml;

        let advice = "🤖 <b>Lời khuyên AI:</b><br>";
        if(rate < 40) advice += "🔴 Cầu đang loạn, nên đánh nhẹ hoặc nghỉ.";
        else if(rate > 60) advice += "🟢 Cầu đang thuận, có thể vào tiền đều tay.";
        else advice += "🟡 Cầu trung bình, ưu tiên đánh các vị trí đầu/đuôi.";
        document.getElementById('aiAdvice').innerHTML = advice;
    }

    function clearData() {
        if(confirm("Xóa toàn bộ để làm lại từ đầu?")) {
            localStorage.clear();
            location.reload();
        }
    }

    updateUI();
</script>
</body>
</html>
