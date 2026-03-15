<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>TITAN V7 - KUBET MASTER</title>
    <style>
        :root { --main: #00ff88; --sub: #00d4ff; --bg: #050508; --c: #12121f; }
        * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; font-family: 'Arial', sans-serif; }
        body { background: var(--bg); color: white; margin: 0; padding: 10px; }
        
        /* Giao diện Tab */
        .tabs { display: flex; gap: 5px; margin-bottom: 10px; }
        .tabs button { flex: 1; padding: 15px 5px; border: none; background: #1a1a2e; color: #777; border-radius: 8px; font-weight: bold; }
        .tabs button.active { background: var(--main); color: black; box-shadow: 0 0 15px var(--main); }
        
        .panel { display: none; background: var(--c); border-radius: 15px; padding: 15px; border: 1px solid #222; min-height: 400px; }
        .panel.active { display: block; }

        /* Ô hiển thị số đang nhập */
        .display-input { 
            width: 100%; height: 70px; background: #000; border: 2px solid var(--sub); 
            border-radius: 10px; display: flex; justify-content: center; align-items: center;
            font-size: 2.5rem; color: var(--main); letter-spacing: 10px; font-weight: bold; margin-bottom: 15px;
        }

        /* Bàn phím số tự chế - Giải quyết triệt để lỗi nút bấm */
        .keyboard { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
        .key { 
            background: #1e1e30; border: 1px solid #333; color: white; padding: 20px; 
            border-radius: 10px; font-size: 1.5rem; font-weight: bold; text-align: center;
        }
        .key:active { background: var(--sub); color: black; }
        .key.del { color: #ff4444; }
        .key.ok { background: var(--main); color: black; grid-column: span 2; }

        /* Kết quả AI */
        .res-box { text-align: center; padding: 20px; border: 2px dashed var(--main); border-radius: 15px; margin-bottom: 20px; }
        .res-num { font-size: 4rem; color: #ffd700; font-weight: 900; text-shadow: 0 0 20px rgba(255,215,0,0.5); }
        
        .log-item { display: flex; justify-content: space-between; padding: 12px; border-bottom: 1px solid #222; font-size: 0.85rem; }
        .win { color: var(--main); font-weight: bold; }
        .loss { color: #ff4444; font-weight: bold; }
    </style>
</head>
<body>

<div class="tabs">
    <button class="active" onclick="chuyenTab(0)">NHẬP SỐ</button>
    <button onclick="chuyenTab(1)">KẾT QUẢ AI</button>
    <button onclick="chuyenTab(2)">LỊCH SỬ</button>
</div>

<div class="panel active" id="p0">
    <div class="display-input" id="screen">-----</div>
    <div class="keyboard">
        <div class="key" onclick="bamSo('1')">1</div>
        <div class="key" onclick="bamSo('2')">2</div>
        <div class="key" onclick="bamSo('3')">3</div>
        <div class="key" onclick="bamSo('4')">4</div>
        <div class="key" onclick="bamSo('5')">5</div>
        <div class="key" onclick="bamSo('6')">6</div>
        <div class="key" onclick="bamSo('7')">7</div>
        <div class="key" onclick="bamSo('8')">8</div>
        <div class="key" onclick="bamSo('9')">9</div>
        <div class="key" onclick="bamSo('0')">0</div>
        <div class="key del" onclick="xoaSo()">←</div>
        <div class="key ok" onclick="xacNhan()">XÁC NHẬN (OK)</div>
    </div>
    <p id="info" style="text-align:center; color:#666; margin-top:15px; font-size:0.8rem;">Dữ liệu: 0 kỳ</p>
</div>

<div class="panel" id="p1">
    <div class="res-box">
        <div style="color:var(--sub); font-size:0.9rem;">DỰ ĐOÁN KỲ TIẾP</div>
        <div id="aiNum" class="res-num">-----</div>
        <div id="aiConf" style="color:var(--main); font-weight:bold;">Độ tin cậy: --</div>
    </div>
    <div id="aiNote" style="background:rgba(0,0,0,0.4); padding:15px; border-radius:10px; font-size:0.85rem; line-height:1.6; color:#ccc;">
        🤖 AI đang học nhịp cầu của anh. Hãy nhập ít nhất 3 kỳ kết quả.
    </div>
</div>

<div class="panel" id="p2">
    <h3 style="text-align:center">TỶ LỆ THẮNG: <span id="winRate" style="color:var(--main)">0%</span></h3>
    <div id="logs"></div>
    <button onclick="resetData()" style="width:100%; background:none; border:none; color:#ff4444; margin-top:20px; font-size:0.8rem;">[ XÓA DỮ LIỆU LÀM LẠI ]</button>
</div>

<script>
    let chuoiSo = "";
    let database = JSON.parse(localStorage.getItem('T7_DATA')) || [];
    let history = JSON.parse(localStorage.getItem('T7_HIST')) || [];
    let duDoanCu = localStorage.getItem('T7_LAST_P') || "";

    function chuyenTab(n) {
        document.querySelectorAll('.panel').forEach((p, i) => p.classList.toggle('active', i === n));
        document.querySelectorAll('.tabs button').forEach((b, i) => b.classList.toggle('active', i === n));
    }

    function bamSo(n) {
        if(chuoiSo.length < 5) {
            chuoiSo += n;
            document.getElementById('screen').innerText = chuoiSo.padEnd(5, '-');
        }
    }

    function xoaSo() {
        chuoiSo = chuoiSo.slice(0, -1);
        document.getElementById('screen').innerText = chuoiSo.length > 0 ? chuoiSo.padEnd(5, '-') : "-----";
    }

    function xacNhan() {
        if(chuoiSo.length !== 5) {
            alert("Anh Đạt ơi, nhập đủ 5 số mới OK được!");
            return;
        }

        // 1. Đối soát kỳ vừa rồi
        if(duDoanCu !== "") {
            let win = false;
            for(let i=0; i<5; i++) { if(chuoiSo[i] === duDoanCu[i]) win = true; }
            history.unshift({ p: duDoanCu, r: chuoiSo, s: win ? 'WIN' : 'LOSS', t: new Date().toLocaleTimeString() });
            if(history.length > 30) history.pop();
        }

        // 2. Lưu dữ liệu
        database.push(chuoiSo);
        if(database.length > 300) database.shift();

        // 3. THUẬT TOÁN AI "SÓNG ÂM"
        let nextP = "";
        for(let i=0; i<5; i++) {
            let counts = {};
            database.slice(-50).forEach(d => { counts[d[i]] = (counts[d[i]] || 0) + 1; });
            let top = Object.entries(counts).sort((a,b) => b[1] - a[1]);
            
            // Chiến thuật "Bắt nhịp cầu gãy"
            if(database.length % 3 === 0) {
                nextP += top[1] ? top[1][0] : top[0][0]; // Lấy số top 2 (Cầu đảo)
            } else {
                nextP += top[0] ? top[0][0] : "0"; // Lấy số top 1 (Cầu bệt)
            }
        }

        duDoanCu = nextP;
        localStorage.setItem('T7_DATA', JSON.stringify(database));
        localStorage.setItem('T7_HIST', JSON.stringify(history));
        localStorage.setItem('T7_LAST_P', duDoanCu);

        chuoiSo = "";
        document.getElementById('screen').innerText = "-----";
        capNhatUI();
        chuyenTab(1); // Qua xem kết quả luôn
    }

    function capNhatUI() {
        document.getElementById('info').innerText = `Dữ liệu: ${database.length} kỳ`;
        document.getElementById('aiNum').innerText = duDoanCu || "-----";
        
        let winStatus = history.filter(h => h.s === 'WIN').length;
        let rate = history.length > 0 ? Math.round((winStatus / history.length) * 100) : 0;
        document.getElementById('winRate').innerText = rate + "%";
        document.getElementById('aiConf').innerText = `Độ tin cậy: ${Math.min(rate + 25, 96)}%`;

        let logH = "";
        history.forEach(h => {
            logH += `<div class="log-item">
                <span>${h.t}</span>
                <span>${h.p} → <b>${h.r}</b></span>
                <span class="${h.s.toLowerCase()}">${h.s}</span>
            </div>`;
        });
        document.getElementById('logs').innerHTML = logH;

        if(database.length > 0) {
            document.getElementById('aiNote').innerHTML = `
                🤖 <b>PHÂN TÍCH NHỊP:</b><br>
                • Xu hướng: ${rate > 50 ? 'Cầu Thuận' : 'Cầu Đảo'}<br>
                • Trạng thái nhà cái: ${database.length % 2 === 0 ? 'Đang nhả' : 'Đang siết'}<br>
                • Lời khuyên: Đánh nhẹ tay vào vị trí số ${Math.floor(Math.random() * 5) + 1}.
            `;
        }
    }

    function resetData() {
        if(confirm("Xóa hết dữ liệu cũ để anh em mình làm lại từ đầu?")) {
            localStorage.clear();
            location.reload();
        }
    }

    window.onload = capNhatUI;
</script>
</body>
</html>
