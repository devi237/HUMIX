const liveStream  = document.getElementById('liveStream');
const feedPh      = document.getElementById('feedPh');
const alertBanner = document.getElementById('alertBanner');
const countVal    = document.getElementById('countVal');
const threshVal   = document.getElementById('threshVal');
const timeVal     = document.getElementById('timeVal');
const flowVal     = document.getElementById('flowVal');
const sessionCard = document.getElementById('sessionCard');
const graphPh     = document.getElementById('graphPh');
const alertSound  = document.getElementById('alertSound');
const liveTag     = document.getElementById('liveTag');

let statusInt=null, graphInt=null, cameraActive=false;
let crowdChart=null, isMuted=false;

// ── CHART ─────────────────────────────────────────────────
function initChart(){
  const ctx=document.getElementById('crowdChart').getContext('2d');
  crowdChart=new Chart(ctx,{
    type:'line',
    data:{labels:[],datasets:[
      {label:'People Count',data:[],borderColor:'#22c55e',
       backgroundColor:'rgba(34,197,94,.08)',borderWidth:2,
       pointRadius:3,tension:.4,fill:true},
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

// ── ALERT SOUND ───────────────────────────────────────────
let isSoundPlaying=false;
function handleAlertSound(alertActive){
  if(!alertSound) return;
  if(alertActive&&!isSoundPlaying){
    alertSound.play().then(()=>{isSoundPlaying=true;}).catch(()=>{
      document.addEventListener('click',()=>{
        if(alertActive){alertSound.play();isSoundPlaying=true;}
      },{once:true});
    });
  } else if(!alertActive&&isSoundPlaying){
    alertSound.pause(); alertSound.currentTime=0; isSoundPlaying=false;
  }
}

let isMuted2=false;
function toggleMute(){
  if(!alertSound) return;
  isMuted2=!isMuted2;
  alertSound.muted=isMuted2;
  document.getElementById('muteBtn').textContent=isMuted2?'🔇 Unmute':'🔊 Mute';
}

// ── POLLING ───────────────────────────────────────────────
function startPolling(){
  pollStatus(); statusInt=setInterval(pollStatus,2000);
  updateChart(); graphInt=setInterval(updateChart,3000);
}

async function pollStatus(){
  try{
    const d=await fetch('/api/status').then(r=>r.json());
    countVal.textContent=d.current_count; threshVal.textContent=d.threshold;
    timeVal.textContent=d.timestamp; flowVal.textContent=d.flow_direction||'—';
    alertBanner.style.display=d.alert_active?'flex':'none';
    handleAlertSound(d.alert_active);

    if(d.camera_active&&!cameraActive){
      cameraActive=true; showFeed(d.camera_source);
    } else if(!d.camera_active&&cameraActive){
      cameraActive=false; hideFeed();
    }
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

function showFeed(source){
  feedPh.style.display='none';
  liveStream.style.display='block';
  liveStream.src='/video_feed';
  if(liveTag) liveTag.textContent=source==='esp32'?'Updated every 2s':'Live stream';
}
function hideFeed(){
  liveStream.style.display='none'; liveStream.src='';
  feedPh.style.display='flex';
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

// ── INIT ──────────────────────────────────────────────────
initChart(); startPolling();
