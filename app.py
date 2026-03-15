import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="wide")

# Code TITAN V7 của anh em mình
titan_v7_code = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <style>
        :root { --main: #00ff88; --sub: #00d4ff; --bg: #050508; --c: #12121f; }
        * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; font-family: sans-serif; }
        body { background: var(--bg); color: white; margin: 0; padding: 5px; }
        .tabs { display: flex; gap: 5px; margin-bottom: 10px; }
        .tabs button { flex: 1; padding: 15px 5px; border: none; background: #1a1a2e; color: #777; border-radius: 8px; font-weight: bold; }
        .tabs button.active { background: var(--main); color: black; box-shadow: 0 0 15px var(--main); }
        .panel { display: none; background: var(--c); border-radius: 15px; padding: 15px; border: 1px solid #222; min-height: 450px; }
        .panel.active { display: block; }
        .display-input { width: 100%; height: 60px; background: #000; border: 2px solid var(--sub); border-radius: 10px; display: flex; justify-content: center; align-items: center; font-size: 2rem; color: var(--main); letter-spacing: 8px; margin-bottom: 10px; }
        .keyboard { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
        .key { background: #1e1e30; border: 1px solid #333; color: white; padding: 15px; border-radius: 10px; font-size: 1.2rem; font-weight: bold; text-align: center; }
        .key:active { background: var(--sub); color: black; }
        .key.ok { background: var(--main); color: black; grid-column: span 2; }
        .res-box { text-align: center; padding: 20px; border: 2px dashed var(--main); border-radius: 15px; }
        .res-num { font-size: 3.5rem; color: #ffd700; font-weight: 900; }
        .log-item { display: flex; justify-content: space-between; padding: 8px; border-bottom: 1px solid #222; font-size: 0.8rem; }
        .win { color: var(--main); } .loss { color: #ff4444; }
    </style>
</head>
<body>
    <div class="tabs">
        <button id="b0" class="active" onclick="tab(0)">NHẬP SỐ</button>
        <button id="b1" onclick="tab(1)">AI DỰ ĐOÁN</button>
        <button id="b2" onclick="tab(2)">LỊCH SỬ</button>
    </div>
    <div class="panel active" id="p0">
        <div class="display-input" id="screen">-----</div>
        <div class="keyboard">
            <div class="key" onclick="ins('1')">1</div><div class="key" onclick="ins('2')">2</div><div class="key" onclick="ins('3')">3</div>
            <div class="key" onclick="ins('4')">4</div><div class="key" onclick="ins('5')">5</div><div class="key" onclick="ins('6')">6</div>
            <div class="key" onclick="ins('7')">7</div><div class="key" onclick="ins('8')">8</div><div class="key" onclick="ins('9')">9</div>
            <div class="key" onclick="ins('0')">0</div><div class="key" style="color:red" onclick="del()">←</div>
            <div class="key ok" onclick="run()">XÁC NHẬN</div>
        </div>
    </div>
    <div class="panel" id="p1">
        <div class="res-box">
            <div id="aiNum" class="res-num">-----</div>
            <div id="aiNote" style="margin-top:10px; font-size:0.8rem; color:#aaa">Vui lòng nhập ít nhất 2 kỳ để AI tính sóng.</div>
        </div>
    </div>
    <div class="panel" id="p2"><div id="logs"></div></div>

    <script>
        let s = ""; let db = []; let hist = []; let last = "";
        function tab(n) {
            document.querySelectorAll('.panel').forEach((p,i)=>p.classList.toggle('active',i===n));
            document.querySelectorAll('.tabs button').forEach((b,i)=>b.classList.toggle('active',i===n));
        }
        function ins(n) { if(s.length<5){ s+=n; document.getElementById('screen').innerText=s.padEnd(5,'-'); } }
        function del() { s=s.slice(0,-1); document.getElementById('screen').innerText=s.length?s.padEnd(5,'-'):"-----"; }
        function run() {
            if(s.length!==5) return;
            if(last){
                let w=false; for(let i=0;i<5;i++) if(s[i]===last[i]) w=true;
                hist.unshift({p:last, r:s, st:w?'WIN':'LOSS'});
            }
            db.push(s); let nxt="";
            for(let i=0;i<5;i++){
                let c={}; db.slice(-20).forEach(d=>{c[d[i]]=(c[d[i]]||0)+1});
                let t=Object.entries(c).sort((a,b)=>b[1]-a[1]);
                nxt += (db.length%2==0) ? (t[1]?t[1][0]:t[0][0]) : t[0][0];
            }
            last=nxt; s=""; document.getElementById('screen').innerText="-----";
            document.getElementById('aiNum').innerText=last;
            let hHtml=""; hist.forEach(h=>{ hHtml+=`<div class="log-item"><span>${h.p}→${h.r}</span><span class="${h.st.toLowerCase()}">${h.st}</span></div>`});
            document.getElementById('logs').innerHTML=hHtml;
            tab(1);
        }
    </script>
</body>
</html>
"""

components.html(titan_v7_code, height=700, scrolling=True)
