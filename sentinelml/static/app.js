const $ = id => document.getElementById(id);
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = value => `${(value * 100).toFixed(1)}%`;
let snapshot, selected = 'EQ-001', timer = null, busy = false, historyRequest = 0;
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
  return data;
}
function toast(message) { $('toast').textContent = message; $('toast').hidden = false; setTimeout(() => $('toast').hidden = true, 5000); }
async function action(fn) {
  if (busy) return;
  busy = true;
  document.querySelectorAll('.controls button, #model-cards button, #scenario').forEach(el => el.disabled = true);
  try { $('error').hidden = true; await fn(); await refresh(); }
  catch (error) { $('error').textContent = error.message; $('error').hidden = false; pause(); }
  finally { busy = false; document.querySelectorAll('.controls button, #scenario').forEach(el => el.disabled = false); }
}
function pause() { clearInterval(timer); timer = null; $('play').textContent = '▶ Start replay'; $('stream-state').textContent = 'Replay paused'; }
async function advance() { const result = await api('/api/replay', {steps:1}); if (result.exhausted) { pause(); toast('Replay complete. The full evaluation period has been processed.'); } }
$('step').onclick = () => action(advance);
$('play').onclick = () => { if (timer) return pause(); $('play').textContent = 'Ⅱ Pause replay'; $('stream-state').textContent = 'Replay running · +1h / 3s'; action(advance); timer = setInterval(() => action(advance), 3000); };
$('scenario').onchange = () => action(async () => { await api('/api/scenario', {scenario:$('scenario').value}); toast('Scenario applies to the next replay step.'); });
$('rollback').onclick = () => action(async () => { const result = await api('/api/models/rollback', {}); toast(`Restored ${result.active}`); });
$('train').onclick = () => action(async () => { pause(); toast('Reproducing candidate on the original training partition…'); const result = await api('/api/training', {}); toast(`Candidate reproduced in ${result.elapsed_seconds}s. Deployment unchanged.`); });
$('sensor').onchange = renderHistory;
document.querySelectorAll('.nav').forEach(button => button.onclick = () => {
  document.querySelectorAll('.nav').forEach(el => el.classList.toggle('active', el===button));
  document.querySelectorAll('.tab').forEach(el => el.hidden = el.id !== button.dataset.tab);
  const titles = {overview:['Every signal tells a story.','Spot emerging failures. Keep your operations moving.'],models:['Better models. Controlled releases.','Compare performance before changing what runs in production.'],monitoring:['Trust starts with visibility.','Track data quality, changing conditions, and real outcomes.']};
  $('page-title').textContent=titles[button.dataset.tab][0]; $('page-subtitle').textContent=titles[button.dataset.tab][1]; $('breadcrumb').textContent=button.textContent.trim().toUpperCase();
});
async function refresh() {
  snapshot = await api('/api/summary');
  const s = snapshot;
  $('clock').textContent = new Date(s.simulated_time).toLocaleString('en-GB',{timeZone:'UTC',day:'2-digit',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit'})+' UTC';
  $('active-model').textContent=s.active_model;
  $('scenario').value=s.scenario;
  $('fleet-count').textContent=s.machines.length;
  $('attention-count').textContent=s.machines.filter(m=>m.alert&&!m.stale).length;
  $('prediction-count').textContent=s.total_predictions.toLocaleString();
  $('feature-health').textContent=s.monitoring.status==='warming_up'?'Warming up':s.monitoring.drift_detected?'Shift detected':'Stable';
  $('feature-health').style.fontSize='24px';
  $('feature-health').style.color=s.monitoring.drift_detected?'#c58324':'#278e7a';
  $('psi-caption').textContent=`Max PSI ${s.monitoring.max_psi.toFixed(2)} · ${s.monitoring.sample_count} windows`;
  $('fleet').innerHTML=s.machines.map((m,i)=>`<button class="machine ${m.alert?'warn':''} ${m.machine_id===selected?'selected':''}" data-machine="${esc(m.machine_id)}" aria-pressed="${m.machine_id===selected}"><div class="machine-top"><span class="machine-id">${esc(m.machine_id)}</span><span class="health ${m.stale?'stale':m.alert?'warn':''}">${m.stale?'STALE DATA':m.alert?'AT RISK':'HEALTHY'}</span></div><div class="machine-type">${['Hydraulic press','Rotary compressor','Cooling pump'][i%3]} · Line ${String(Math.floor(i/3)+1).padStart(2,'0')}</div><div class="risk-row"><strong>${pct(m.probability)}</strong><small>failure risk</small></div><div class="risk-bar"><i style="width:${Math.max(2,m.probability*100)}%"></i></div><div class="sensor-line"><span>TEMP <b>${m.sensors.temperature.toFixed(1)}°C</b></span><span>VIB <b>${m.sensors.vibration.toFixed(2)} mm/s</b></span></div></button>`).join('');
  document.querySelectorAll('[data-machine]').forEach(button=>button.onclick=()=>{selected=button.dataset.machine; document.querySelectorAll('[data-machine]').forEach(el=>{el.classList.toggle('selected',el===button);el.setAttribute('aria-pressed',el===button)});renderHistory();renderAttribution();});
  $('alerts').innerHTML=s.alerts.length?s.alerts.map(a=>`<div class="alert-row"><div class="alert-icon">!</div><div class="alert-body"><strong>${esc(a.machine_id)} · Elevated failure risk</strong><p>${esc(a.timestamp.slice(5,16).replace('T',' '))} UTC · ${esc(a.model_version)}</p><p>${a.outcome===null?'Outcome pending':a.outcome?'Failure observed within 6h':'No failure within 6h'}</p></div><span class="alert-prob">${pct(a.probability)}</span></div>`).join(''):'<div class="empty">No alerts in the replay yet.<br>Advance time to follow equipment health.</div>';
  renderModels(); renderMonitoring(); renderAttribution(); await renderHistory();
}
function renderAttribution() {
  const machine=snapshot.machines.find(m=>m.machine_id===selected); $('selected-title').textContent=selected;
  $('attributions').innerHTML=machine?machine.attribution.map(a=>`<div class="attr-row"><span>${esc(a.sensor)}</span><div class="attr-track"><i style="width:${Math.min(100,Math.abs(a.contribution)*100)}%;background:${a.contribution>0?'#d1a359':'#5ab09d'}"></i></div><b>${a.contribution>=0?'+':''}${(a.contribution*100).toFixed(1)} pp</b></div>`).join(''):'';
}
async function renderHistory() {
  const sequence=++historyRequest;
  try {
    const rows=await api(`/api/machines/${selected}/history`); if(sequence!==historyRequest)return;
    if(rows.length<2){$('chart').innerHTML='<div class="empty">Advance the replay to see sensor trends.</div>';return;}
    const key=$('sensor').value, values=rows.map(r=>r[key]);
    const lo=Math.min(...values),hi=Math.max(...values),pad=Math.max((hi-lo)*.2,.1),min=lo-pad,max=hi+pad;
    const points=values.map((v,i)=>[42+i/(values.length-1)*488,145-(v-min)/(max-min)*130]);
    const line=points.map(p=>p.join(',')).join(' ');
    $('chart').innerHTML=`<svg viewBox="0 0 545 165" preserveAspectRatio="none" role="img" aria-label="${esc(key)} sensor trend for ${esc(selected)}"><defs><linearGradient id="fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#188f7d" stop-opacity=".17"/><stop offset="100%" stop-color="#188f7d" stop-opacity=".01"/></linearGradient></defs>${[0,1,2,3].map(i=>{const y=15+i*43.33;return `<line x1="42" y1="${y}" x2="530" y2="${y}" stroke="#edf1f2" stroke-dasharray="3,4"/><text x="1" y="${y+3}" font-size="9" fill="#a1adb1">${(max-i*(max-min)/3).toFixed(1)}</text>`}).join('')}<polygon points="42,150 ${line} 530,150" fill="url(#fill)"/><polyline points="${line}" fill="none" stroke="#268f7d" stroke-width="2" vector-effect="non-scaling-stroke"/>${points.map((p,i)=>`<circle cx="${p[0]}" cy="${p[1]}" r="2.5" fill="#fff" stroke="#268f7d"><title>${esc(rows[i].timestamp)}: ${values[i].toFixed(3)}</title></circle>`).join('')}</svg>`;
    $('chart-range').textContent=`${rows[0].timestamp.slice(11,16)} — ${rows.at(-1).timestamp.slice(11,16)} · ${rows.length} observations`;
  } catch(error) { $('chart').textContent=error.message; }
}
function renderModels(){
 $('model-cards').innerHTML=snapshot.models.map(m=>`<article class="panel"><div class="model-top"><div><span class="eyebrow">${esc(m.algorithm.toUpperCase())}</span><h2 style="margin-top:8px">${esc(m.version)}</h2></div><span class="badge">${m.active?'● ACTIVE':'CANDIDATE'}</span></div><div class="metric-grid"><div><span>Test precision</span><strong>${pct(m.test.precision)}</strong></div><div><span>Test recall</span><strong>${pct(m.test.recall)}</strong></div><div><span>Mean lead time</span><strong>${m.test.mean_warning_lead_hours.toFixed(1)}h</strong></div></div><div class="model-meta">Test false alarm windows / machine: ${m.test.false_alarm_windows_per_machine.toFixed(2)}<br>Validation F1: ${m.validation.f1.toFixed(3)} · Alert threshold: ${pct(m.threshold)}<br>Training windows: ${m.training_windows.toLocaleString()} · Seed: ${m.seed}<br>Dataset SHA-256: ${esc(m.dataset_id.slice(0,24))}…<br>Features: ${esc(m.feature_version)}</div><p class="gate">${m.gate_passed?'✓ Validation gate passed':'✕ Validation gate failed'}</p><p class="model-meta">${esc(m.gate_rule)}</p>${m.active?'<button class="button secondary" disabled>Currently serving</button>':`<button class="button primary" data-approve="${esc(m.version)}" ${m.gate_passed?'':'disabled'}>Approve & deploy</button>`}</article>`).join('');
 document.querySelectorAll('[data-approve]').forEach(button=>button.onclick=()=>action(async()=>{const result=await api(`/api/models/${button.dataset.approve}/approve`,{});toast(`${result.active} approved and deployed.`);}));
 $('deployments').innerHTML=snapshot.deployments.map(d=>`<div class="audit-row"><strong>${esc(d.version)} · ${esc(d.action)}</strong><span>${esc(d.created_at.slice(0,19).replace('T',' '))} UTC</span></div>`).join('');
}
function renderMonitoring(){
 $('drift-features').innerHTML=snapshot.monitoring.feature_psi.slice(0,8).map(f=>`<div class="drift-row"><span>${esc(f.feature)}</span><strong style="color:${f.psi>.25?'#be8b42':'#268f7d'}">${f.psi.toFixed(3)}</strong></div>`).join('')||'<div class="empty">Collecting reference windows…</div>';
 $('checks').innerHTML=snapshot.checks.map(c=>`<div class="check-row"><b>${esc(c.severity.toUpperCase())} / ${esc(c.kind.toUpperCase())}</b>${esc(c.message)}</div>`).join('')||'<div class="empty">No data quality events recorded.</div>';
 $('label-status').textContent=`${snapshot.pending_labels} windows awaiting outcomes. Labels arrive after the 6-hour horizon, plus any scenario delay. Metrics include only observed labels.`;
 $('performance').innerHTML=snapshot.performance.length?`<table><thead><tr><th>Model</th><th>Windows</th><th>Precision</th><th>Recall</th><th>Event recall</th><th>Lead time</th></tr></thead><tbody>${snapshot.performance.map(p=>`<tr><td>${esc(p.version)}</td><td>${p.evaluated_windows}</td><td>${pct(p.precision)}</td><td>${pct(p.recall)}</td><td>${pct(p.event_recall)}</td><td>${p.mean_warning_lead_hours.toFixed(1)}h</td></tr>`).join('')}</tbody></table>`:'<div class="empty">Outcomes are not available yet. Advance at least six replay hours.</div>';
}
refresh().catch(error=>{$('error').hidden=false;$('error').textContent=error.message;});
