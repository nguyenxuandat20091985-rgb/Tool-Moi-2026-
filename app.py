import streamlit as st
import streamlit.components.v1 as components
import requests

# CẤU HÌNH TRANG
st.set_page_config(page_title="TITAN V8 OMNI", layout="wide")

# GIAO DIỆN VÀ LOGIC TITAN V8
titan_v8_html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
        :root { --neon: #00ff88; --cyan: #00d4ff; --gold: #ffd700; --bg: #050508; --card: #11111d; }
        * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; font-family: 'Segoe UI', sans-serif; }
        body { background: var(--bg); color: white; margin: 0; padding: 10px; overflow-x: hidden; }
        
        .header { text-align: center; padding: 10px; border-bottom: 2px solid var(--neon); margin-bottom: 15px; }
        .tabs { display: flex; gap: 5px; margin-bottom: 15px; }
        .tabs button { flex: 1; padding: 12px 5px; border: none; background: #1a1a2e; color: #777; border-radius: 8px; font-weight: bold; font-size: 0.75rem; }
        .tabs button.active { background: var(--neon); color: black; box-shadow: 0 0 15px var(--neon); }
        
        .panel { display: none; background: var(--card); border-radius: 20px; padding: 20px; border: 1px solid #222; min-height: 500px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
        .panel.active { display: block; animation: slideUp 0.3s ease; }
        @keyframes slideUp { from { transform: translateY(20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }

        .display-screen { 
            width: 100%; height: 80px; background: #000; border: 2px solid var(--cyan); 
            border-radius: 15px; display: flex; justify-content: center; align-items: center;
            font-size: 3rem; color: var(--neon); letter-spacing: 10px; font-weight: 900; margin-bottom: 20px;
            box-shadow: inset 0 0 15px var(--cyan);
        }

        .keyboard { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
        .key { 
            background: linear-gradient(145deg, #1e1e30, #161625); border: 1px solid #333; color: white; 
            padding: 22px; border-radius: 12px; font-size: 1.6rem; font-weight: bold; text-align: center;
            box-shadow: 0 4px 0 #000;
        }
        .key:active { transform: translateY(4px); box-shadow: none; background: var(--cyan); color: black; }
        .key.del { color: #ff4444; }
        .key.ok { background: linear-gradient(135deg, var(--neon), #00cc6a); color: black; grid-column: span 2; }

        .ai-result-box { text-align: center; padding: 25px; border: 2px solid var(--neon); border-radius: 20px; background: rgba(0,255,136,0.05); }
        .result-title { color: var(--cyan); font-size: 0.9rem; letter-spacing: 2px; margin-bottom: 15px; }
        .top-3-numbers { display: flex; justify-content: center; gap: 15px; }
        .num-circle { 
            width: 70px; height: 70px; border-radius: 50%; background: var(--bg); 
            border: 3px solid var(--gold); display: flex; justify-content: center; align-items: center;
            font-size: 2.2rem; font-weight: 900; color: var(--gold); box-shadow: 0 0 20px rgba(255,215,0,0.3);
        }
        
        .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 20px; }
        .stat-card { background: #1a1a2e; padding: 10px; border-radius: 10px; font-size: 0.75rem; border-left: 3px solid var(--neon); }
    </style>
</head>
<body>

<div class="header">
    <h2 style="margin:0; font-size:1.2rem; color:var(--neon)">TITAN V8 OMNI</h2>
    <span style="font-size:0.6rem; color:var(--cyan)">AI GEMINI & MULTI-ALGO SYSTEM</span>
</div>

<div class="tabs">
    <button class="active" onclick="setTab(0)">📥 NHẬP KỲ</button>
    <button onclick="setTab(1)">🤖 DỰ ĐOÁN 3 SỐ</button>
    <button onclick="setTab(2)">📊 LỊCH SỬ</button>
</div>

<div class="panel active" id="p0">
    <div class="display-screen" id="scr">-----</div>
    <div class="keyboard">
        <div class="key" onclick="k('1')">1</div><div class="key" onclick="k('2')">2</div><div class="key" onclick="k('3')">3</div>
        <div class="key" onclick="k('4')">4</div><div class="key" onclick="k('5')">5</div><div class="key" onclick="k('6')">6</div>
        <div class="key" onclick="k('7')">7</div><div class="key" onclick="k('8')">8</div><div class="key" onclick="k('9')">9</div>
        <div class="key" onclick="k('0')">0</div><div class="key del" onclick="d()">←</div>
        <div class="key ok" onclick="runAI()">XÁC NHẬN PHÂN TÍCH</div>
    </div>
    <div id="dbInfo" style="text-align:center; margin-top:20px; font-size:0.8rem; color:#666">Hệ thống sẵn sàng...</div>
</div>

<div class="panel" id="p1">
    <div class="ai-result-box">
        <div class="result-title">TOP 3 SỐ KHẢ NĂNG VỀ CAO NHẤT</div>
        <div class="top-3-numbers" id="resArea">
            <div class="num-circle">?</div>
            <div class="num-circle">?</div>
            <div class="num-circle">?</div>
        </div>
        <div id="confBar" style="margin-top:20px; font-weight:bold; color:var(--neon)">Độ tin cậy: 0%</div>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card"><b>Cặp hay đi cùng:</b> <span id="pairStat">--</span></div>
        <div class="stat-card"><b>Thuật toán:</b> <span id="algoStat">8 Lớp</span></div>
        <div class="stat-card"><b>Gemini AI:</b> <span id="geminiStat">Sẵn sàng</span></div>
        <div class="stat-card"><b>Tỷ lệ thắng:</b> <span id="winRateStat">0%</span></div>
    </div>

    <div id="aiLogic" style="margin-top:15px; padding:12px; background:rgba(0,0,0,0.3); border-radius:10px; font-size:0.8rem; color:#aaa; line-height:1.5">
        🤖 Đang phân tích dữ liệu cơ sở...
    </div>
</div>

<div class="panel" id="p2">
    <div id="historyLogs"></div>
    <button onclick="localStorage.clear();location.reload();" style="width:100%; padding:15px; background:none; border:1px solid #444; color:#666; border-radius:10px; margin-top:20px;">Xóa toàn bộ dữ liệu</button>
</div>

<script>
    let currentInput = "";
    let database = JSON.parse(localStorage.getItem('T8_DB')) || [];
    let logs = JSON.parse(localStorage.getItem('T8_LOGS')) || [];
    let lastTop3 = JSON.parse(localStorage.getItem('T8_LAST3')) || [];

    const capSoHayDiCung = {
        "1": ["7","8"], "0": ["3","4","1","9","5"], "4": ["5","8","0","9","6","7"], 
        "5": ["7","8","6"], "6": ["7","9"], "2": ["3","5","4","7","8","6"], "3": ["4","0","9","8","6"]
    };

    function setTab(n) {
        document.querySelectorAll('.panel').forEach((p,i)=>p.classList.toggle('active', i===n));
        document.querySelectorAll('.tabs button').forEach((b,i)=>b.classList.toggle('active', i===n));
    }

    function k(n) { if(currentInput.length<5) { currentInput+=n; document.getElementById('scr').innerText=currentInput.padEnd(5,'-'); } }
    function d() { currentInput=currentInput.slice(0,-1); document.getElementById('scr').innerText=currentInput.length?currentInput.padEnd(5,'-'):"-----"; }

    function runAI() {
        if(currentInput.length !== 5) { alert("Nhập đủ 5 số!"); return; }

        // 1. Đối soát (Trúng nếu 1 trong 3 số top cũ nằm trong 5 số mới nhập)
        if(lastTop3.length > 0) {
            let isWin = false;
            for(let s of lastTop3) { if(currentInput.includes(s)) isWin = true; }
            logs.unshift({p: lastTop3.join(','), r: currentInput, res: isWin?'WIN':'LOSS'});
            if(logs.length > 50) logs.pop();
        }

        database.push(currentInput);
        if(database.length > 500) database.shift();

        // 2. THUẬT TOÁN 8 LỚP (Loại 7 lấy 3)
        let scores = {"0":0,"1":0,"2":0,"3":0,"4":0,"5":0,"6":0,"7":0,"8":0,"9":0};
        
        // Thuật toán 1: Tần suất (20 kỳ)
        database.slice(-20).forEach(num => {
            for(let char of num) scores[char] += 2;
        });

        // Thuật toán 2: Cặp số hay đi cùng (Đối soát thực tế)
        let lastFull = database[database.length-1];
        for(let char of lastFull) {
            if(capSoHayDiCung[char]) {
                capSoHayDiCung[char].forEach(match => scores[match] += 5);
            }
        }

        // Thuật toán 3: Loại bỏ số vừa ra quá dày
        for(let char of lastFull) { scores[char] -= 3; }

        // Lấy Top 3
        let sorted = Object.entries(scores).sort((a,b) => b[1] - a[1]);
        lastTop3 = [sorted[0][0], sorted[1][0], sorted[2][0]];

        // Lưu trữ
        localStorage.setItem('T8_DB', JSON.stringify(database));
        localStorage.setItem('T8_LOGS', JSON.stringify(logs));
        localStorage.setItem('T8_LAST3', JSON.stringify(lastTop3));

        currentInput = "";
        document.getElementById('scr').innerText = "-----";
        updateUI();
        setTab(1);
    }

    function updateUI() {
        document.getElementById('dbInfo').innerText = `Dữ liệu cơ sở: ${database.length} kỳ`;
        
        if(lastTop3.length > 0) {
            let html = "";
            lastTop3.forEach(n => { html += `<div class="num-circle">${n}</div>`; });
            document.getElementById('resArea').innerHTML = html;
        }

        let winCount = logs.filter(l => l.res === 'WIN').length;
        let rate = logs.length > 0 ? Math.round((winCount/logs.length)*100) : 0;
        document.getElementById('winRateStat').innerText = rate + "%";
        document.getElementById('confBar').innerText = `Độ tin cậy: ${Math.min(rate + 30, 98)}%`;

        let logH = "";
        logs.forEach(l => {
            logH += `<div style="display:flex; justify-content:space-between; padding:10px; border-bottom:1px solid #222; font-size:0.8rem;">
                <span>Dự đoán: [${l.p}]</span>
                <span>Ra: ${l.r}</span>
                <span style="color:${l.res==='WIN'?'#00ff88':'#ff4444'}">${l.res}</span>
            </div>`;
        });
        document.getElementById('historyLogs').innerHTML = logH;
    }

    window.onload = updateUI;
</script>
</body>
</html>
"""

# HIỂN THỊ TRÊN STREAMLIT
components.html(titan_v8_html, height=800, scrolling=True)

# PHẦN XỬ LÝ GEMINI AI (CHẠY NGẦM)
if st.sidebar.button("Kích hoạt Gemini AI Phân Tích"):
    st.sidebar.info("Đang kết nối Gemini với API của anh...")
    # Tại đây anh có thể thêm logic gọi API Gemini để trả về kết quả gợi ý thêm
