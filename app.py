<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>TITAN 5D OMNI V4</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --neon-green: #00ff88;
            --neon-blue: #00ccff;
            --bg-dark: #0a0a0f;
            --card-bg: #161625;
        }
        body {
            background-color: var(--bg-dark);
            color: white;
            font-family: 'Segoe UI', sans-serif;
            margin: 0; padding: 10px;
        }
        /* Tab System */
        .tab-container { display: flex; gap: 5px; margin-bottom: 15px; }
        .tab-btn {
            flex: 1; padding: 12px; border: none; background: #252538;
            color: #888; border-radius: 8px; font-weight: bold;
        }
        .tab-btn.active {
            background: linear-gradient(135deg, var(--neon-green), var(--neon-blue));
            color: #000; box-shadow: 0 0 15px var(--neon-green);
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }

        /* Cyber UI */
        .glass-card {
            background: var(--card-bg); border: 1px solid #2a2a3a;
            border-radius: 15px; padding: 15px; margin-bottom: 15px;
        }
        #numberInput {
            width: 100%; padding: 20px; font-size: 2rem; text-align: center;
            background: #000; border: 2px solid var(--neon-green);
            color: var(--neon-green); border-radius: 10px; margin-bottom: 10px;
            letter-spacing: 10px;
        }
        .btn-main {
            width: 100%; padding: 15px; border-radius: 30px; border: none;
            background: var(--neon-green); color: black; font-weight: 900;
            text-transform: uppercase; cursor: pointer;
        }
        .prediction-val {
            font-size: 3rem; text-align: center; color: #ffd700;
            text-shadow: 0 0 20px rgba(255,215,0,0.5); font-weight: 900;
        }
        .status-win { color: var(--neon-green); font-weight: bold; }
        .status-loss { color: #ff4444; font-weight: bold; }
    </style>
</head>
<body>

<div class="tab-container">
    <button class="tab-btn active" onclick="openTab('input-tab')">📥 NHẬP LIỆU</button>
    <button class="tab-btn" onclick="openTab('ai-tab')">🤖 AI MASTER</button>
    <button class="tab-btn" onclick="openTab('history-tab')">📊 ĐỐI SOÁT</button>
</div>

<div id="input-tab" class="tab-content active">
    <div class="glass-card">
        <h3>🔢 DỮ LIỆU ĐẦU VÀO</h3>
        <input type="text" id="numberInput" placeholder="00000" maxlength="5">
        <button class="btn-main" onclick="processData()">⚡ LƯU & PHÂN TÍCH</button>
    </div>
    <div class="glass-card">
        <h4>📊 THỐNG KÊ CƠ SỞ</h4>
        <div id="baseStats">Chưa có dữ liệu...</div>
    </div>
</div>

<div id="ai-tab" class="tab-content">
    <div class="glass-card" style="border-color: var(--neon-blue);">
        <h2 style="text-align: center; color: var(--neon-blue);">DỰ ĐOÁN KỲ TIẾP</h2>
        <div id="predictionDisplay" class="prediction-val">-----</div>
        <div id="aiReasoning" style="font-size: 0.8rem; color: #888; margin-top: 10px;"></div>
    </div>
    <div class="glass-card">
        <h4>🔍 NHẬN DIỆN CẦU</h4>
        <div id="trendAnalysis">Đang chờ dữ liệu...</div>
    </div>
</div>

<div id="history-tab" class="tab-content">
    <div class="glass-card">
        <h4>📈 TỶ LỆ THẮNG THỰC TẾ</h4>
        <h1 id="winRateDisplay" style="text-align: center; color: var(--neon-green);">0%</h1>
    </div>
    <div id="historyList"></div>
</div>

<script>
    let db = JSON.parse(localStorage.getItem('titan_v4_db')) || [];
    let predictions = JSON.parse(localStorage.getItem('titan_v4_preds')) || [];

    function openTab(tabId) {
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.getElementById(tabId).classList.add('active');
        event.currentTarget.classList.add('active');
    }

    // --- THUẬT TOÁN LÕI (KHÔNG DỰ ĐOÁN MÒ) ---
    function analyzeLogic(data) {
        if(data.length < 10) return "Cần thêm dữ liệu";
        
        let lastNum = data[data.length - 1];
        let posScores = [{},{},{},{},{}];
        
        // 1. Phân tích tần suất 20 kỳ gần nhất
        let recent = data.slice(-20);
        recent.forEach(num => {
            for(let i=0; i<5; i++) {
                let d = num[i];
                posScores[i][d] = (posScores[i][d] || 0) + 1;
            }
        });

        // 2. Thuật toán Săn Cầu Bệt (Nếu số vừa ra lặp lại nhiều lần)
        let result = "";
        for(let i=0; i<5; i++) {
            let sorted = Object.entries(posScores[i]).sort((a,b) => b[1] - a[1]);
            // Nếu số cũ đang "nóng", AI sẽ ưu tiên bám theo thay vì bỏ
            result += sorted[0][0];
        }
        return result;
    }

    function processData() {
        let val = document.getElementById('numberInput').value;
        if(val.length !== 5) { alert("Nhập đủ 5 số!"); return; }

        // 1. Đối soát trước khi lưu mới
        if(predictions.length > 0) {
            let lastPred = predictions[predictions.length - 1];
            if(lastPred.status === 'pending') {
                let matchCount = 0;
                for(let i=0; i<5; i++) if(val[i] === lastPred.code[i]) matchCount++;
                lastPred.status = matchCount >= 1 ? 'WIN' : 'LOSS';
                lastPred.real = val;
            }
        }

        // 2. Lưu dữ liệu cơ sở
        db.push(val);
        if(db.length > 500) db.shift();
        
        // 3. Chạy thuật toán AI
        let nextPred = analyzeLogic(db);
        predictions.push({ code: nextPred, status: 'pending', time: new Date().toLocaleTimeString() });

        localStorage.setItem('titan_v4_db', JSON.stringify(db));
        localStorage.setItem('titan_v4_preds', JSON.stringify(predictions));

        updateUI();
        document.getElementById('numberInput').value = "";
        alert("Đã ghi nhận & Phân tích xong!");
    }

    function updateUI() {
        // Cập nhật màn hình AI
        if(predictions.length > 0) {
            document.getElementById('predictionDisplay').innerText = predictions[predictions.length-1].code;
            document.getElementById('aiReasoning').innerText = "Dựa trên chuỗi " + db.length + " kỳ gần nhất. Ưu tiên số nóng.";
        }

        // Cập nhật đối soát
        let historyHtml = "";
        let wins = 0;
        predictions.slice(-10).reverse().forEach(p => {
            if(p.status === 'WIN') wins++;
            historyHtml += `
                <div class="glass-card" style="display:flex; justify-content:space-between;">
                    <span>Kỳ ${p.time}</span>
                    <span class="prediction-val" style="font-size:1rem">${p.code}</span>
                    <span class="${p.status === 'WIN' ? 'status-win' : 'status-loss'}">${p.status}</span>
                </div>
            `;
        });
        document.getElementById('historyList').innerHTML = historyHtml;
        document.getElementById('winRateDisplay').innerText = (wins * 10) + "%";
        document.getElementById('baseStats').innerText = "Tổng dữ liệu cơ sở: " + db.length + " kỳ.";
    }

    updateUI();
</script>
</body>
</html>
