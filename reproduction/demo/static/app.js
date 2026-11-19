const state = {config:null,examples:[],filtered:[],currentId:null,current:null,source:"example",stage:"baseline",uploadFile:null};
const $ = id => document.getElementById(id);
const labels = {regular_no_merge:"普通",rowspan:"Rowspan",colspan:"Colspan",combined_merge:"组合合并",multiple_merges:"多合并",large_dense:"大型密集",formula:"公式",chinese:"中文",english:"英文",mixed_zh_en:"中英混合"};

async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `请求失败 (${response.status})`;
    try { message = (await response.json()).detail || message; } catch (_) {}
    throw new Error(message);
  }
  return response.json();
}
function showToast(message) {
  const element = $("toast");
  element.textContent = message;
  element.classList.remove("hidden");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => element.classList.add("hidden"), 4200);
}
const formatScore = value => typeof value === "number" ? value.toFixed(4) : "—";
const fact = (label, value) => `<div class="fact"><span>${label}</span><strong>${value ?? "—"}</strong></div>`;
function setStatus(id, text, kind) { const element=$(id); element.textContent=text; element.className=kind||""; }

function renderStages() {
  const rail = $("stageRail"); rail.innerHTML = "";
  state.config.stages.forEach(stage => {
    const button = document.createElement("button");
    button.className = `stage-button ${stage.available ? "" : "unavailable"} ${state.stage===stage.id ? "active" : ""}`;
    button.innerHTML = `<strong>${stage.label}</strong><span>${stage.badge}</span>`;
    button.onclick = () => { state.stage=stage.id; renderStages(); updateInferButton(); };
    rail.appendChild(button);
  });
}
function applyFilters() {
  const values=[$("structureFilter").value,$("contentFilter").value].filter(Boolean);
  state.filtered=state.examples.filter(example=>values.every(value=>example.tags.includes(value)));
  $("candidateCount").textContent=`当前条件下 ${state.filtered.length} / ${state.examples.length} 条样例`;
  if (!state.filtered.length) return showToast("当前筛选条件下没有样例");
  if (!state.filtered.some(example=>example.sample_id===state.currentId)) loadExample(state.filtered[0].sample_id);
}
function navigate(delta) {
  if (!state.filtered.length) return;
  let index=state.filtered.findIndex(example=>example.sample_id===state.currentId);
  index=(index+delta+state.filtered.length)%state.filtered.length;
  loadExample(state.filtered[index].sample_id);
}
function randomExample() {
  const candidates=state.filtered.filter(example=>example.sample_id!==state.currentId);
  if (!candidates.length) return showToast(state.filtered.length ? "当前条件下仅有一个样例" : "当前条件下没有样例");
  loadExample(candidates[Math.floor(Math.random()*candidates.length)].sample_id);
}

function renderSourceMetadata(detail) {
  $("sourceImage").src=detail.image_url; $("sourceImage").classList.remove("hidden"); $("imagePlaceholder").classList.add("hidden");
  $("sampleLabel").textContent=detail.sample_id||"自定义上传图片";
  $("tagList").innerHTML=(detail.tags||[]).filter(tag=>labels[tag]).slice(0,8).map(tag=>`<i>${labels[tag]}</i>`).join("");
}
function showAwaiting(detail=null) {
  if (detail) renderSourceMetadata(detail);
  $("resultContent").classList.add("hidden"); $("emptyState").classList.remove("hidden");
  $("originBadge").textContent="等待实时识别"; $("resultPanel").classList.toggle("upload-result",state.source==="upload"); updateInferButton();
}
async function loadExample(id) {
  state.currentId=id;
  try { const detail=await api(`/api/examples/${id}`); state.current=detail; showAwaiting(detail); }
  catch (error) { showToast(error.message); }
}
function setLoading(visible,title="正在运行 Baseline…") { $("loadingTitle").textContent=title; $("loadingOverlay").classList.toggle("hidden",!visible); }

function renderTable(id,html) {
  const host=$(id); host.innerHTML="";
  const canvas=document.createElement("div"); canvas.className="table-canvas";
  canvas.innerHTML=html||'<p class="note">未提取到完整表格。</p>'; host.appendChild(canvas);
}
function renderExtractedHtml(prediction) {
  const output=$("htmlOutput"); output.innerHTML="";
  const code=document.createElement("pre"); code.textContent=prediction.html||"未提取到完整 HTML 表格。"; output.appendChild(code);
}
function renderSequence(id,items) {
  const lane=$(id); lane.innerHTML="";
  (items||[]).forEach(item=>{
    const cell=document.createElement("span"); cell.className=`sequence-cell ${item.status}`;
    cell.title=`#${item.index} · ${item.signature}`;
    const index=document.createElement("small"),signature=document.createElement("b"); index.textContent=`#${item.index}`; signature.textContent=item.signature; cell.append(index,signature); lane.appendChild(cell);
  });
}
function matchBadge(label,matched,prediction,reference) {
  return `<div class="validity-check ${matched?"matched":"mismatch"}"><span>${label}</span><b>${matched?"匹配":"不匹配"}</b><small>PRED ${prediction}</small><small>REF ${reference}</small></div>`;
}
function renderReward(reward,hasReference) {
  const section=$("rewardSection");
  if (!hasReference) { section.classList.add("hidden"); return; }
  section.classList.remove("hidden");
  if (!reward?.available) {
    $("totalReward").textContent="不可用";
    ["validityValue","structureValue","contentValue"].forEach(id=>$(id).textContent="—");
    $("validityChecks").innerHTML=""; $("referenceSequence").innerHTML=""; $("predictionSequence").innerHTML=""; $("probeCalculation").textContent=""; $("rewardFormula").textContent="";
    $("rewardNote").textContent=`Reward 暂不可用：${reward?.reason||"未知原因"}`; return;
  }
  $("totalReward").textContent=formatScore(reward.reward);
  $("validityValue").textContent=formatScore(reward.validity); $("structureValue").textContent=formatScore(reward.structure); $("contentValue").textContent=formatScore(reward.content);
  const validity=reward.validity_detail;
  $("validityChecks").innerHTML=matchBadge("行列数",validity.row_col_match,`${validity.prediction.rows} × ${validity.prediction.cols}`,`${validity.reference.rows} × ${validity.reference.cols}`)+matchBadge("Merge 声明",validity.merge_match,validity.prediction.merge,validity.reference.merge);
  const probe=reward.one_d_probe_explanation; renderSequence("referenceSequence",probe.reference); renderSequence("predictionSequence",probe.prediction);
  $("probeCalculation").textContent=`正确结构前缀 / 参考 Cell 总数 = ${probe.calculation}`;
  $("probeBoundaryNote").textContent=probe.boundary_note||""; $("probeBoundaryNote").classList.toggle("hidden",!probe.boundary_note);
  const contributions=reward.aggregation.contributions.map(item=>`${item.weight.toFixed(1)} × ${item.score.toFixed(4)} = ${item.value.toFixed(4)}`).join("  +  ");
  $("rewardFormula").textContent=`${contributions}  →  ${formatScore(reward.reward)}`;
  $("rewardNote").textContent=state.config.reward_note;
}
function renderAnchor(anchor,evaluationMetrics) {
  $("evaluationTedsValue").textContent=formatScore(evaluationMetrics?.teds); $("evaluationTedsSValue").textContent=formatScore(evaluationMetrics?.teds_s);
  if(!anchor||anchor.status!=="ok") {
    $("anchorResult").classList.add("hidden");
    return;
  }
  $("anchorPredictionImage").src=anchor.prediction.input_render_url; $("anchorReferenceImage").src=anchor.reference.input_render_url;
  renderTable("anchorPredictionRender",anchor.prediction.safe_destylized_html); renderTable("anchorReferenceRender",anchor.reference.safe_destylized_html);
  $("anchorPredictionHtml").textContent=anchor.prediction.destylized_html; $("anchorReferenceHtml").textContent=anchor.reference.destylized_html;
  $("anchorContentScore").textContent=formatScore(anchor.content_reward); $("anchorResult").classList.remove("hidden");
}
function renderDetail(detail) {
  if (!detail.prediction) return showAwaiting(detail);
  const prediction=detail.prediction; renderSourceMetadata(detail);
  $("emptyState").classList.add("hidden"); $("resultContent").classList.remove("hidden");
  $("resultPanel").classList.toggle("upload-result",detail.source_type==="upload"); $("originBadge").textContent="实时 Baseline 结果";
  setStatus("completeStatus",prediction.has_complete_table?"通过":"未找到",prediction.has_complete_table?"ok":"bad");
  setStatus("structureStatus",prediction.structure.is_valid?"通过":"未通过",prediction.structure.is_valid?"ok":"bad");
  setStatus("truncateStatus",prediction.diagnostics?.suspected_truncation?"是":"否",prediction.diagnostics?.suspected_truncation?"warn":"ok");
  renderTable("predictionRender",prediction.safe_html);
  const hasReference=Boolean(detail.reference); $("referenceCard").classList.toggle("hidden",!hasReference); if(hasReference) renderTable("referenceRender",detail.reference.safe_html);
  renderReward(detail.reward,hasReference);
  if(hasReference) renderAnchor(prediction.anchor_destylization,prediction.evaluation_metrics);
  const structure=prediction.structure;
  $("structureFacts").innerHTML=fact("逻辑行数",structure.row_count)+fact("逻辑列数",structure.col_count)+fact("物理单元格",structure.physical_cell_count)+fact("合并区域",structure.merge_count);
  $("mergeList").textContent=structure.merge_count?structure.merged_cells.map(cell=>cell.protocol).join("\n"):"未发现合并单元格。";
  const runtime=prediction.runtime||{};
  $("runtimeFacts").innerHTML=fact("输入 Tokens",runtime.input_tokens)+fact("输出 Tokens",runtime.output_tokens)+fact("生成耗时",runtime.generation_seconds!=null?`${runtime.generation_seconds}s`:"—")+fact("峰值显存",runtime.peak_generation_memory_gib!=null?`${runtime.peak_generation_memory_gib} GiB`:"—");
  renderExtractedHtml(prediction); $("rawDetails").classList.toggle("hidden",detail.source_type==="upload"); $("rawOutput").textContent=prediction.raw_output||""; updateInferButton();
}

function switchSource(source) {
  state.source=source;
  document.querySelectorAll(".source-tab").forEach(tab=>tab.classList.toggle("active",tab.dataset.source===source));
  $("exampleControls").classList.toggle("hidden",source!=="example"); $("uploadControls").classList.toggle("hidden",source!=="upload");
  if (source==="example"&&state.currentId) loadExample(state.currentId);
  else if (source==="upload") {
    state.current=null; $("sourceImage").removeAttribute("src"); $("sourceImage").classList.add("hidden");
    $("imagePlaceholder").textContent="请选择一张表格图片"; $("imagePlaceholder").classList.remove("hidden");
    $("sampleLabel").textContent="自定义上传图片"; $("tagList").innerHTML=""; showAwaiting();
  }
  updateInferButton();
}
function updateInferButton() {
  const button=$("inferButton"),available=state.stage==="baseline";
  button.disabled=!available||(state.source==="upload"&&!state.uploadFile); button.textContent="识别";
  $("actionHint").textContent=!available?"当前阶段没有正式 checkpoint，推理已禁用。":state.source==="example"?"预测与 Reward 均在点击后现场生成，不读取 Phase 1 缓存结果。":"自定义图片没有 GT，仅展示识别与结构诊断，不计算 Reward。";
}
async function runInference() {
  if(state.stage!=="baseline") return;
  if(state.source==="example") {
    if(!state.currentId) return; setLoading(true,"正在识别并运行 Anchor 代理 Reward…");
    try { const detail=await api(`/api/examples/${state.currentId}/infer`,{method:"POST"}); state.current=detail; renderDetail(detail); showToast("Baseline、Anchor Destylization 与代理 Reward 已现场完成。"); }
    catch(error){showToast(error.message);} finally{setLoading(false);}
  } else {
    if(!state.uploadFile) return; setLoading(true,"正在识别上传图片…");
    try { const detail=await api("/api/infer-upload",{method:"POST",headers:{"Content-Type":state.uploadFile.type||"application/octet-stream"},body:state.uploadFile}); state.current=detail; renderDetail(detail); showToast("上传图片识别完成"); }
    catch(error){showToast(error.message);} finally{setLoading(false);}
  }
}
function chooseFile(file) {
  if(!file) return; state.uploadFile=file; $("sourceImage").src=URL.createObjectURL(file); $("sourceImage").classList.remove("hidden");
  $("imagePlaceholder").classList.add("hidden"); $("sampleLabel").textContent=file.name; showAwaiting(); updateInferButton();
}
function openImageModal() {
  const source=$("sourceImage"); if(!source.src||source.classList.contains("hidden")) return;
  $("modalImage").src=source.src; $("imageModal").classList.remove("hidden"); document.body.style.overflow="hidden";
}
function closeImageModal() { $("imageModal").classList.add("hidden"); document.body.style.overflow=""; }

async function init() {
  try {
    const [config,list]=await Promise.all([api("/api/config"),api("/api/examples")]); state.config=config; state.examples=list.examples; state.filtered=[...state.examples]; renderStages();
    ["structureFilter","contentFilter"].forEach(id=>$(id).onchange=applyFilters); $("previousButton").onclick=()=>navigate(-1); $("nextButton").onclick=()=>navigate(1); $("randomButton").onclick=randomExample;
    document.querySelectorAll(".source-tab").forEach(button=>button.onclick=()=>switchSource(button.dataset.source)); $("inferButton").onclick=runInference;
    const input=$("fileInput"),dropZone=$("dropZone"); input.onchange=()=>chooseFile(input.files[0]);
    ["dragenter","dragover"].forEach(event=>dropZone.addEventListener(event,interaction=>{interaction.preventDefault();dropZone.classList.add("drag");}));
    ["dragleave","drop"].forEach(event=>dropZone.addEventListener(event,interaction=>{interaction.preventDefault();dropZone.classList.remove("drag");}));
    dropZone.addEventListener("drop",event=>chooseFile(event.dataTransfer.files[0]));
    $("imageShell").onclick=openImageModal; $("modalClose").onclick=closeImageModal; $("imageModal").onclick=event=>{if(event.target===$("imageModal"))closeImageModal();};
    document.addEventListener("keydown",event=>{if(event.key==="Escape")closeImageModal();});
    $("candidateCount").textContent=`当前条件下 ${state.examples.length} / ${state.examples.length} 条样例`; await loadExample(state.examples[0].sample_id);
  } catch(error) { showToast(`Demo 初始化失败：${error.message}`); $("systemStatus").innerHTML="<span></span>服务异常"; }
}
document.addEventListener("DOMContentLoaded",init);
