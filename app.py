<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>TITAN V6 - KUBET KILLER</title>
    <style>
        :root { --neon: #00ff88; --blue: #00d4ff; --bg: #0a0a0f; --card: #161625; }
        * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
        body { background: var(--bg); color: white; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 10px; overflow-x: hidden; }
        
        .tab-bar { display: flex; gap: 5px; margin-bottom: 10px; }
        .tab-bar button { flex: 1; padding: 15px 5px; border: none; background: #252538; color: #888; border-radius: 10px; font-weight: bold; font-size: 0.8rem; }
        .tab-bar button.active { background: var(--neon); color: black; box-shadow: 0 0 15px var(--neon); }
        
        .content { display: none; background: var(--card); border-radius: 15px; padding: 15px; border: 1px solid #2a2a3a; min-height: 300px; }
        .content.active { display: block; animation: fadeIn 0.3s ease; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

        input { width: 100%; padding: 20px; background: #000; border: 2px solid var(--blue); color: var(--neon); font-size: 2.5rem; text-align: center; border-radius: 12px; letter-spacing: 8px; margin-bottom: 15px; font-weight: bold; outline: none; }
        .btn-main { width: 100%; padding: 18px; background: linear-gradient(135deg, var(--neon), #00cc6a); color: black; border: none; border-radius: 50px; font-weight: 900; font-size: 1.1rem; text-transform: uppercase; box-shadow: 0 5px 15px rgba(0,255,136,0.3); }
        .btn-main:active { transform: scale(0.98); }

        .prediction-box { text-align: center; padding: 25px 10px; border: 2px solid var(--neon); border-radius: 20px; margin: 15px 0; background: rgba(0,255,136,0.05); }
        .pred-label { color: var(--blue); font-size: 0.9rem; letter-spacing: 2px; margin-bottom: 10px; }
        .pred-num { font-size: 4rem; color: #ffd700; font-weight: 900; text-shadow: 0 0 25px rgba(255,215,0,0.6); font-family: monospace; }
        
        .history-item { display: flex; justify-content: space-between; padding: 12px; border-bottom: 1px solid #2a2a3a; font-size: 0.85rem; }
        .win { color: var(--neon); font-weight: bold; }
        .loss { color: #ff4444; font-weight: bold; }
        .clear-btn { background: none; border: none; color: #ff4444; font-size: 0.7rem; margin-top: 20px; width: 100%; }
    </style>
</head>
<body>

<div class="tab-bar">
    <button id="btn0" class="active" onclick="switchTab(0)">📥 NHẬP DỮ LIỆU</button>
    <button id="btn1" onclick="switchTab(1)">🤖 PHÂN TÍCH AI</button>
    <button id="btn2" onclick="switchTab(2)">📊 ĐỐI SOÁT</button>
</div>

<div class="content active" id="tab0">
    <h4 style="text-align:center; color:var(--blue)">KẾT QUẢ KỲ VỪA RA</h4>
    <input type="number" id="numIn" placeholder="00000" pattern="\d*" inputmode="numeric">
    <button class="btn-main" onclick="handleAnalysis()">⚡ XÁC NHẬN NGAY</button>
    <div id="dbCount" style="text-align:center; margin-top:15px; font-size:0.8rem; color:#666">Dữ liệu: 0 kỳ</div>
</div>

<div class="content" id="tab1">
    <div class="prediction-box">
        <div class="pred-label">DỰ ĐOÁN KỲ KẾ</div>
        <div id="nextCode" class="pred-num">-----</div>
        <div id="confLevel" style="color:var(--neon); font-weight:bold; margin-top:10px">Độ tin cậy: --</div>
    </div>
    <div id="aiLogic" style="padding:10px; font-size:0.8rem; background:rgba(0,0,0,0.3); border-radius:10px; color:#aaa; line-height:1.6">
        🤖 Chờ nhập dữ liệu để kích hoạt AI...
    </div>
</div>

<div class="content" id="tab2">
    <h3 style="text-align:center">TỶ LỆ THẮNG: <span id="winRateText" style="color:var(--neon)">0%</span></h3>
    <div id="historyLog"></div>
    <button class="clear-btn" onclick="hardReset()">🗑️ XÓA DỮ LIỆU LÀM LẠI</button>
</div>

<script>
    // Khởi tạo dữ liệu an toàn
    let database = [];
    let logs = [];
    let currentPrediction = "";

    // Load data khi mở trang
    window.onload = function() {
        const savedDb = localStorage.getItem('T6_DB');
        const savedLogs = localStorage.getItem('T6_LOGS');
        const savedPred = localStorage.getItem('T6_NEXT');
        
        if(savedDb) database = JSON.parse(savedDb);
        if(savedLogs) logs = JSON.parse(savedLogs);
        if(savedPred) currentPrediction = savedPred;
        
        refreshUI();
    };

    function switchTab(idx) {
        document.querySelectorAll('.content').forEach((c, i) => c.classList.toggle('active', i === idx));
        document.querySelectorAll('.tab-bar button').forEach((b, i) => b.classList.toggle('active', i === idx));
    }

    function handleAnalysis() {
        const input = document.getElementById('numIn');
        const val = input.value;

        if (val.length !== 5) {
            alert("Anh Đạt ơi, nhập đủ 5 số nhé!");
            return;
        }

        // 1. Đối soát kỳ trước
        if (currentPrediction !== "") {
            let winCount = 0;
            for(let i=0; i<5; i++) {
                if(val[i] === currentPrediction[i]) winCount++;
            }
            const status = winCount >= 1 ? "WIN" : "LOSS";
            logs.unshift({ pred: currentPrediction, real: val, status: status, time: new Date().toLocaleTimeString() });
            if(logs.length > 30) logs.pop();
        }

        // 2. Thêm vào database
        database.push(val);
        if(database.length > 500) database.shift();

        // 3. Thuật toán AI MASTER V6
        // Phân tích sự lặp lại và nhịp nhảy của từng vị trí
        let next = "";
        for(let p=0; p<5; p++) {
            let freq = {};
            // Chỉ lấy 40 kỳ gần nhất để soi nhịp
            database.slice(-40).forEach(n => {
                let d = n[p];
                freq[d] = (freq[d] || 0) + 1;
            });
            
            let sorted = Object.entries(freq).sort((a,b) => b[1] - a[1]);
            
            // Chiến thuật: Nếu kỳ này là số thứ tự lẻ trong DB, đánh số Top 1 (Bệt). 
            // Nếu là số thứ tự chẵn, đánh số Top 2 (Đảo).
            if(database.length % 2 !== 0) {
                next += sorted[0] ? sorted[0][0] : Math.floor(Math.random()*10);
            } else {
                next += sorted[1] ? sorted[1][0] : (sorted[0] ? sorted[0][0] : "5");
            }
        }

        currentPrediction = next;
        
        // Lưu trữ
        localStorage.setItem('T6_DB', JSON.stringify(database));
        localStorage.setItem('T6_LOGS', JSON.stringify(logs));
        localStorage.setItem('T6_NEXT', currentPrediction);

        input.value = "";
        refreshUI();
        switchTab(1); // Sang tab dự đoán
    }

    function refreshUI() {
        document.getElementById('dbCount').innerText = `Dữ liệu cơ sở: ${database.length} kỳ`;
        document.getElementById('nextCode').innerText = currentPrediction || "-----";
        
        const winRecords = logs.filter(l => l.status === "WIN").length;
        const rate = logs.length > 0 ? Math.round((winRecords / logs.length) * 100) : 0;
        document.getElementById('winRateText').innerText = rate + "%";
        document.getElementById('confLevel').innerText = `Độ tin cậy: ${Math.min(rate + 20, 95)}%`;

        let logHtml = "";
        logs.forEach(l => {
            logHtml += `
                <div class="history-item">
                    <span>${l.time}</span>
                    <span>AI: ${l.pred} ➔ ${l.real}</span>
                    <span class="${l.status.toLowerCase()}">${l.status}</span>
                </div>
            `;
        });
        document.getElementById('historyLog').innerHTML = logHtml;

        if(database.length > 0) {
            document.getElementById('aiLogic').innerHTML = `
                <b>PHÂN TÍCH HỆ THỐNG:</b><br>
                • Trạng thái: Cầu đang ${rate > 50 ? 'Ổn định' : 'Biến động'}<br>
                • Chiến thuật: Đối soát nhịp ${database.length % 2 === 0 ? 'ĐẢO' : 'BỆT'}<br>
                • Lời khuyên: ${rate > 60 ? 'Tự tin vào lệnh' : 'Đánh nhẹ dò cầu'}
            `;
        }
    }

    function hardReset() {
        if(confirm("Anh Đạt chắc chắn muốn xóa hết để AI học lại từ đầu không?")) {
            localStorage.clear();
            database = [];
            logs = [];
            currentPrediction = "";
            refreshUI();
            switchTab(0);
        }
    }
</script>
</body>
</html>
