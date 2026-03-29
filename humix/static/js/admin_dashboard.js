const startBtn    = document.getElementById('startBtn');
const stopBtn     = document.getElementById('stopBtn');
const camStatus   = document.getElementById('camStatus');
const srcTag      = document.getElementById('srcTag');
const camSrc      = document.getElementById('cameraSource');
const esp32Field  = document.getElementById('esp32Field');
const esp32Url    = document.getElementById('esp32Url');
const threshInput = document.getElementById('thresholdInput');
const confInput   = document.getElementById('confInput');
const confLabel   = document.getElementById('confLabel');
const saveBtn     = document.getElementById('saveBtn');
const feedImg     = document.getElementById('feedImg');
const feedPh      = document.getElementById('feedPh');
const alertBanner = document.getElementById('alertBanner');
const countVal    = document.getElementById('countVal');
const threshVal   = document.getElementById('threshVal');
const timeVal     = document.getElementById('timeVal');
const flowVal     = document.getElementById('flowVal');
const sessionCard = document.getElementById('sessionCard');
const graphPh     = document.getElementById('graphPh');
const ledG        = document.getElementById('ledG');
const ledY        = document.getElementById('ledY');
const ledR        = document.getElementById('ledR');

let statusInt=null, imageInt=null, graphInt=null;
let cameraActive=false, crowdChart=null;

// ── CHART ─────────────────────────────────────────────────
function initChart(){
  const ctx=document.getElementById('crowdChart').getContext('2d');
  crowdChart=new Chart(ctx,{
    type:'line',
    data:{labels:[],datasets:[
      {label:'People Count',data:[],borderColor:'#22c55e',
       backgroundColor:'rgba(34,197,94,.08)',borderWidth:2,
       pointRadius:3,pointBackgroundColor:'#22c55e',tension:.4,fill:true},
      {label:'Threshold',data:[],borderColor:'#ef4444',borderWidth:2,
       borderDash:[6,4],pointRadius:0,tension:0,fill:false}
    ]},
    options:{responsive:true,maintainAspectRatio:true,animation:{duration:400},
      plugins:{legend:{labels:{color:'#94a3b8',font:{size:12}}}},
      scales:{
        x:{ticks:{color:'#64748b',maxTicksLimit:8,maxRotation:0},
           grid:{color:'rgba(255,255,255,.04)'}},
        y:{beginAtZero:true,ticks:{color:'#64748b',stepSize:1},
           grid:{color:'rgba(255,255,255,.06)'}}
      }}
  });
}

async function updateChart(){
  if(!cameraActive||!crowdChart) return;
  try{
    const d=await fetch('/api/graph-data').then(r=>r.json());
    if(!d.labels.length) return;
    if(graphPh) graphPh.style.display='none';
    crowdChart.data.labels=d.labels;
    crowdChart.data.datasets[0].data=d.counts;
    crowdChart.data.datasets[1].data=d.labels.map(()=>d.threshold);
    crowdChart.data.datasets[0].pointBackgroundColor=d.alerts.map(a=>a?'#ef4444':'#22c55e');
    crowdChart.update();
  }catch{}
}

// ── LED DISPLAY ───────────────────────────────────────────
function updateLEDDisplay(count,threshold){
  ledG.classList.remove('active');
  ledY.classList.remove('active');
  ledR.classList.remove('active');
  if(count>=threshold)       ledR.classList.add('active');
  else if(count>=threshold*0.75) ledY.classList.add('active');
  else                       ledG.classList.add('active');
}

// ── SOURCE TOGGLE ─────────────────────────────────────────
camSrc.addEventListener('change',()=>{
  esp32Field.style.display=camSrc.value==='esp32'?'block':'none';
});
confInput.addEventListener('input',()=>{
  confLabel.textContent=(confInput.value/100).toFixed(2);
});

// ── START ─────────────────────────────────────────────────
startBtn.addEventListener('click',async()=>{
  startBtn.disabled=true; startBtn.textContent='Starting…';
  const src=camSrc.value, url=esp32Url?esp32Url.value.trim():'';
  if(src==='esp32'&&!url){
    showToast('Enter ESP32 URL','error');
    startBtn.disabled=false; startBtn.innerHTML='▶ Start'; return;
  }
  try{
    const d=await fetch('/api/camera/start',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({source:src,url})}).then(r=>r.json());
    if(d.success){
      cameraActive=true; startBtn.disabled=true; stopBtn.disabled=false;
      camStatus.textContent='Online'; camStatus.className='status-pill pill-online';
      srcTag.textContent=src==='esp32'?'📡 ESP32':'💻 Webcam';
      srcTag.style.display='inline';
      showFeed(); startPolling();
      showToast('Camera started','success');
    }else{ showToast(d.message||'Failed','error'); startBtn.disabled=false; startBtn.innerHTML='▶ Start'; }
  }catch{ showToast('Connection error','error'); startBtn.disabled=false; startBtn.innerHTML='▶ Start'; }
});

// ── STOP ──────────────────────────────────────────────────
stopBtn.addEventListener('click',async()=>{
  stopBtn.disabled=true; stopBtn.textContent='Stopping…';
  try{
    await fetch('/api/camera/stop',{method:'POST'});
    cameraActive=false; startBtn.disabled=false; startBtn.innerHTML='▶ Start';
    stopBtn.disabled=true; stopBtn.innerHTML='⏹ Stop';
    camStatus.textContent='Offline'; camStatus.className='status-pill pill-offline';
    srcTag.style.display='none';
    hideFeed(); stopPolling();
    countVal.textContent='0'; alertBanner.style.display='none';
    if(sessionCard) sessionCard.style.display='none';
    [ledG,ledY,ledR].forEach(l=>l.classList.remove('active'));
    showToast('Camera stopped','success');
  }catch{ showToast('Error','error'); stopBtn.disabled=false; stopBtn.innerHTML='⏹ Stop'; }
});

// ── SETTINGS ──────────────────────────────────────────────
saveBtn.addEventListener('click',async()=>{
  const t=parseInt(threshInput.value), c=parseInt(confInput.value)/100;
  if(t<1||t>5000){showToast('Threshold 1–5000','error');return;}
  try{
    await fetch('/api/settings/threshold',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({threshold:t})});
    await fetch('/api/settings/confidence',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({confidence:c})});
    showToast('Settings saved','success');
  }catch{showToast('Failed','error');}
});

// ── ARDUINO ───────────────────────────────────────────────
document.getElementById('arduinoBtn').addEventListener('click',async()=>{
  const port=document.getElementById('arduinoPort').value.trim();
  try{
    const d=await fetch('/api/arduino/connect',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({port})}).then(r=>r.json());
    showToast(d.message, d.success?'success':'error');
  }catch{showToast('Connection error','error');}
});

// ── POLLING ───────────────────────────────────────────────
function startPolling(){
  pollStatus(); statusInt=setInterval(pollStatus,2000);
  refreshImage(); imageInt=setInterval(refreshImage,2000);
  updateChart(); graphInt=setInterval(updateChart,3000);
}
function stopPolling(){
  clearInterval(statusInt); clearInterval(imageInt); clearInterval(graphInt);
}

async function pollStatus(){
  try{
    const d=await fetch('/api/status').then(r=>r.json());
    countVal.textContent=d.current_count; threshVal.textContent=d.threshold;
    timeVal.textContent=d.timestamp; flowVal.textContent=d.flow_direction||'—';
    alertBanner.style.display=d.alert_active?'block':'none';
    updateLEDDisplay(d.current_count,d.threshold);
    if(d.camera_active&&!cameraActive){cameraActive=true;showFeed();}
    else if(!d.camera_active&&cameraActive){cameraActive=false;hideFeed();}
    if(d.camera_active) pollStats();
  }catch{}
}

async function pollStats(){
  try{
    const d=await fetch('/api/session-stats').then(r=>r.json());
    if(d.session_start!=='N/A'&&sessionCard){
      sessionCard.style.display='block';
      document.getElementById('sesStart').textContent=d.session_start;
      document.getElementById('sesDur').textContent=d.session_duration+'s';
      document.getElementById('sesLogs').textContent=d.total_logs;
      document.getElementById('sesAlerts').textContent=d.alert_events;
      document.getElementById('sesPeak').textContent=d.max_crowd;
    }
  }catch{}
}

function refreshImage(){
  if(!cameraActive) return;
  feedImg.src='/latest_image?t='+Date.now();
  feedImg.onerror=()=>setTimeout(()=>{if(cameraActive)feedImg.src='/latest_image?t='+Date.now();},1000);
}
function showFeed(){
  feedPh.style.display='none'; feedImg.style.display='block';
  setTimeout(()=>{feedImg.src='/latest_image?t='+Date.now();},2200);
}
function hideFeed(){
  feedImg.style.display='none'; feedPh.style.display='flex'; feedImg.src='';
}

// ── THEME ─────────────────────────────────────────────────
const themeToggle=document.getElementById('themeToggle');
const savedTheme=localStorage.getItem('humix_theme')||'dark';
function applyTheme(t){
  if(t==='light'){document.body.classList.add('light');if(themeToggle)themeToggle.textContent='☀️';}
  else{document.body.classList.remove('light');if(themeToggle)themeToggle.textContent='🌙';}
}
applyTheme(savedTheme);
if(themeToggle) themeToggle.addEventListener('click',()=>{
  const nt=document.body.classList.contains('light')?'dark':'light';
  localStorage.setItem('humix_theme',nt); applyTheme(nt);
});

// ── TOAST ─────────────────────────────────────────────────
function showToast(msg,type){
  const t=document.createElement('div');
  t.textContent=msg;
  Object.assign(t.style,{position:'fixed',bottom:'28px',right:'28px',
    padding:'12px 22px',borderRadius:'10px',fontWeight:'700',fontSize:'.88rem',
    color:type==='success'?'#000':'#fff',
    background:type==='success'?'#22c55e':'#ef4444',
    boxShadow:'0 8px 32px rgba(0,0,0,.4)',zIndex:'9999',transition:'opacity .3s'});
  document.body.appendChild(t);
  setTimeout(()=>{t.style.opacity='0';setTimeout(()=>t.remove(),300);},3000);
}

// ── INIT ──────────────────────────────────────────────────
initChart(); pollStatus(); startPolling();
