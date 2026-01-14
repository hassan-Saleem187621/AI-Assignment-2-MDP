// Student-style JS implementation of GridWorld MDP (VI & PI)
// Keep it simple + readable for viva.

const NROWS = 5;
const NCOLS = 5;

// fixed walls and terminals (you can edit these to make it unique)
const WALLS = new Set(["1,1", "1,2", "2,2"]);
const TERMINALS = {
  "0,4": 10,
  "4,4": -10
};

const ACTIONS = ["U","D","L","R"];
const ARROW = {U:"↑", D:"↓", L:"←", R:"→"};
const MOVE = {U:[-1,0], D:[1,0], L:[0,-1], R:[0,1]};

let V = {};           // value table: key "r,c" -> number
let policy = {};      // policy table: key -> action
let iter = 0;
let lastDelta = 0;
let running = false;

// For Policy Iteration state
let piPolicy = {};    // current policy for PI
let piMode = "EVAL";  // just to keep it obvious for viva
let piEvalSweepsLeft = 0;

// ---------- DOM ----------
const gridEl = document.getElementById("grid");
const algoEl = document.getElementById("algo");
const gammaEl = document.getElementById("gamma");
const slipEl = document.getElementById("slip");
const stepRewardEl = document.getElementById("stepReward");
const maxItersEl = document.getElementById("maxIters");

const gammaValEl = document.getElementById("gammaVal");
const slipValEl = document.getElementById("slipVal");

const itOut = document.getElementById("itOut");
const deltaOut = document.getElementById("deltaOut");
const statusOut = document.getElementById("statusOut");

document.getElementById("btnReset").addEventListener("click", resetAll);
document.getElementById("btnStep").addEventListener("click", doOneStep);
document.getElementById("btnRun").addEventListener("click", runLoop);

gammaEl.addEventListener("input", ()=> gammaValEl.textContent = (+gammaEl.value).toFixed(2));
slipEl.addEventListener("input", ()=> slipValEl.textContent = (+slipEl.value).toFixed(2));

// ---------- Helpers ----------
function key(r,c){ return `${r},${c}`; }
function isWall(k){ return WALLS.has(k); }
function isTerminal(k){ return Object.prototype.hasOwnProperty.call(TERMINALS, k); }

function inBounds(r,c){ return r>=0 && r<NROWS && c>=0 && c<NCOLS; }

function nextStateDet(r,c,a){
  const k = key(r,c);
  if(isTerminal(k)) return k;

  const [dr,dc] = MOVE[a];
  const nr = r+dr, nc = c+dc;
  const nk = key(nr,nc);

  if(!inBounds(nr,nc)) return k;
  if(isWall(nk)) return k;
  return nk;
}

function reward(stateKey){
  if(isTerminal(stateKey)) return TERMINALS[stateKey];
  return parseFloat(stepRewardEl.value);
}

function transitions(stateKey, action){
  // 80% intended, 20% split among other 3 (or from slider)
  if(isTerminal(stateKey)) return [{p:1.0, ns:stateKey, r:0.0}];

  const slip = parseFloat(slipEl.value);
  const pInt = 1.0 - slip;
  const pOther = slip/3.0;

  const [r,c] = stateKey.split(",").map(Number);

  let out = [];
  const nsInt = nextStateDet(r,c,action);
  out.push({p:pInt, ns:nsInt, r:reward(nsInt)});

  for(const a2 of ACTIONS){
    if(a2 === action) continue;
    const ns2 = nextStateDet(r,c,a2);
    out.push({p:pOther, ns:ns2, r:reward(ns2)});
  }
  return out;
}

function allStates(){
  let s = [];
  for(let r=0;r<NROWS;r++){
    for(let c=0;c<NCOLS;c++){
      const k = key(r,c);
      if(!isWall(k)) s.push(k);
    }
  }
  return s;
}

function initTables(){
  V = {};
  policy = {};
  for(const s of allStates()){
    V[s] = 0.0;
    policy[s] = isTerminal(s) ? null : "R"; // default
  }
}

function greedyPolicyFromV(){
  const gamma = parseFloat(gammaEl.value);
  let pol = {};
  for(const s of allStates()){
    if(isTerminal(s)) { pol[s]=null; continue; }

    let bestA = "U";
    let bestQ = -1e9;
    for(const a of ACTIONS){
      let q = 0;
      for(const t of transitions(s,a)){
        q += t.p * (t.r + gamma * V[t.ns]);
      }
      if(q > bestQ){ bestQ=q; bestA=a; }
    }
    pol[s] = bestA;
  }
  return pol;
}

function clamp(x, lo, hi){ return Math.max(lo, Math.min(hi, x)); }

function colorForValue(v, vmin, vmax){
  // simple: map v to alpha, keep base blue-ish
  if(!isFinite(v)) return "rgba(148,163,184,0.08)";
  const t = (v - vmin) / (vmax - vmin + 1e-9);
  const a = clamp(0.12 + 0.55*t, 0.10, 0.70);
  return `rgba(96,165,250,${a})`;
}

// ---------- Rendering ----------
function renderGrid(){
  gridEl.innerHTML = "";

  // find min/max for color scale
  let vals = Object.values(V).filter(x => isFinite(x));
  let vmin = vals.length ? Math.min(...vals) : -1;
  let vmax = vals.length ? Math.max(...vals) :  1;

  for(let r=0;r<NROWS;r++){
    for(let c=0;c<NCOLS;c++){
      const k = key(r,c);

      const cell = document.createElement("div");
      cell.className = "cell";

      if(isWall(k)){
        cell.classList.add("wall");
        cell.innerHTML = `<div class="t">█</div>`;
        gridEl.appendChild(cell);
        continue;
      }

      if(isTerminal(k)){
        const tval = TERMINALS[k];
        cell.style.background = tval>0 ? "rgba(34,197,94,0.20)" : "rgba(251,113,133,0.16)";
        cell.innerHTML = `<div class="t">T(${tval})</div>`;
        gridEl.appendChild(cell);
        continue;
      }

      // normal state
      const v = V[k] ?? 0.0;
      cell.style.background = colorForValue(v, vmin, vmax);

      const a = policy[k] ?? null;

      cell.innerHTML = `
        <div class="v">${v.toFixed(2)}</div>
        <div class="a">${a ? ARROW[a] : ""}</div>
      `;
      gridEl.appendChild(cell);
    }
  }

  itOut.textContent = iter;
  deltaOut.textContent = lastDelta.toFixed(4);
}

// ---------- Algorithms (Step-by-step) ----------
function valueIterationSweep(){
  const gamma = parseFloat(gammaEl.value);
  let newV = {...V};
  let delta = 0;

  for(const s of allStates()){
    if(isTerminal(s)) { newV[s]=0.0; continue; }

    let best = -1e9;
    for(const a of ACTIONS){
      let q = 0;
      for(const t of transitions(s,a)){
        q += t.p * (t.r + gamma * V[t.ns]);
      }
      if(q > best) best = q;
    }
    newV[s] = best;
    delta = Math.max(delta, Math.abs(newV[s] - V[s]));
  }

  V = newV;
  lastDelta = delta;
  policy = greedyPolicyFromV();
}

function policyIterationStep(){
  // We'll do: a few evaluation sweeps, then one improvement.
  // This makes "Step" easy to explain.
  const gamma = parseFloat(gammaEl.value);

  if(piMode === "EVAL"){
    // do one evaluation sweep
    let newV = {...V};
    for(const s of allStates()){
      if(isTerminal(s)) { newV[s]=0.0; continue; }
      const a = piPolicy[s];
      let val = 0;
      for(const t of transitions(s,a)){
        val += t.p * (t.r + gamma * V[t.ns]);
      }
      newV[s] = val;
    }

    // delta just for display
    let d = 0;
    for(const s of allStates()){
      d = Math.max(d, Math.abs(newV[s]-V[s]));
    }
    V = newV;
    lastDelta = d;

    piEvalSweepsLeft -= 1;
    if(piEvalSweepsLeft <= 0){
      piMode = "IMPROVE";
    }
    // policy shown is current piPolicy while evaluating
    policy = {...piPolicy};
    return;
  }

  if(piMode === "IMPROVE"){
    let stable = true;

    for(const s of allStates()){
      if(isTerminal(s)) continue;

      const oldA = piPolicy[s];
      let bestA = oldA;
      let bestQ = -1e9;

      for(const a of ACTIONS){
        let q = 0;
        for(const t of transitions(s,a)){
          q += t.p * (t.r + gamma * V[t.ns]);
        }
        if(q > bestQ){ bestQ=q; bestA=a; }
      }

      piPolicy[s] = bestA;
      if(bestA !== oldA) stable = false;
    }

    policy = {...piPolicy};
    lastDelta = 0;

    // after improvement, switch back to evaluation
    piMode = "EVAL";
    piEvalSweepsLeft = 10; // student choice (simple fixed)
    if(stable){
      statusOut.textContent = "Converged (policy stable)";
      running = false;
    } else {
      statusOut.textContent = "PI: improved, evaluating…";
    }
  }
}

// ---------- Controls ----------
function resetAll(){
  initTables();
  iter = 0;
  lastDelta = 0;
  running = false;

  // PI init
  piPolicy = {};
  for(const s of allStates()){
    piPolicy[s] = isTerminal(s) ? null : "R";
  }
  piMode = "EVAL";
  piEvalSweepsLeft = 10;

  statusOut.textContent = "Ready";
  renderGrid();
}

function doOneStep(){
  const maxIters = parseInt(maxItersEl.value,10);
  if(iter >= maxIters){
    statusOut.textContent = "Reached max iterations";
    return;
  }

  const algo = algoEl.value;
  statusOut.textContent = algo === "VI" ? "Running VI step…" : `Running PI step (${piMode})…`;

  if(algo === "VI"){
    valueIterationSweep();
    if(lastDelta < 1e-4){
      statusOut.textContent = "Converged (delta small)";
      running = false;
    }
  } else {
    policyIterationStep();
  }

  iter += 1;
  renderGrid();
}

function sleep(ms){ return new Promise(res => setTimeout(res, ms)); }

async function runLoop(){
  if(running){
    running = false;
    statusOut.textContent = "Stopped";
    return;
  }
  running = true;
  statusOut.textContent = "Running… (click Run again to stop)";

  const maxIters = parseInt(maxItersEl.value,10);
  while(running && iter < maxIters){
    doOneStep();
    // small delay so you can "see" it
    await sleep(60);
    // stop if converged (VI)
    if(algoEl.value === "VI" && lastDelta < 1e-4){
      running = false;
    }
    // stop if PI says converged
    if(statusOut.textContent.includes("Converged")){
      running = false;
    }
  }
  if(iter >= maxIters){
    statusOut.textContent = "Reached max iterations";
  }
}

// start
gammaValEl.textContent = (+gammaEl.value).toFixed(2);
slipValEl.textContent = (+slipEl.value).toFixed(2);
resetAll();
