/**
 * Interactive Static Web Demo Application for 2D Lattice Design
 * Dual Synchronized Replay: Bio-Inspired Remodeling Pruning vs Genetic Algorithm
 */

(function () {
  'use strict';

  // Global application state
  let demoData = null;
  let currentScenarioKey = 'LC1_vf20';
  let isPlaying = false;
  let playSpeed = 2; // frames per tick
  let currentProgressPct = 100; // 0 to 100
  let animFrameId = null;
  let lastTickTime = 0;

  // DOM Elements
  const loadCaseSelect = document.getElementById('loadCaseSelect');
  const vfSelect = document.getElementById('vfSelect');
  const colorModeSelect = document.getElementById('colorModeSelect');
  const deformToggle = document.getElementById('deformToggle');
  const defScaleSlider = document.getElementById('defScaleSlider');
  const scaleValue = document.getElementById('scaleValue');
  const scaleControlGroup = document.getElementById('scaleControlGroup');

  const btnPlayPause = document.getElementById('btnPlayPause');
  const btnStepBack = document.getElementById('btnStepBack');
  const btnStepForward = document.getElementById('btnStepForward');
  const btnReset = document.getElementById('btnReset');
  const timelineScrubber = document.getElementById('timelineScrubber');
  const timelineProgressText = document.getElementById('timelineProgressText');
  const speedSelect = document.getElementById('speedSelect');

  const canvasA = document.getElementById('canvasA');
  const ctxA = canvasA.getContext('2d');
  const canvasB = document.getElementById('canvasB');
  const ctxB = canvasB.getContext('2d');

  // Metrics elements Method A
  const iterA = document.getElementById('iterA');
  const solvesA = document.getElementById('solvesA');
  const compA = document.getElementById('compA');
  const stiffA = document.getElementById('stiffA');
  const membersA = document.getElementById('membersA');
  const vfA = document.getElementById('vfA');
  const axShareA = document.getElementById('axShareA');
  const bendShareA = document.getElementById('bendShareA');
  const barAxialA = document.getElementById('barAxialA');
  const barBendingA = document.getElementById('barBendingA');
  const kinTagA = document.getElementById('kinTagA');
  const lcBadgeA = document.getElementById('lcBadgeA');

  // Metrics elements Method B
  const iterB = document.getElementById('iterB');
  const solvesB = document.getElementById('solvesB');
  const compB = document.getElementById('compB');
  const stiffB = document.getElementById('stiffB');
  const membersB = document.getElementById('membersB');
  const vfB = document.getElementById('vfB');
  const axShareB = document.getElementById('axShareB');
  const bendShareB = document.getElementById('bendShareB');
  const barAxialB = document.getElementById('barAxialB');
  const barBendingB = document.getElementById('barBendingB');
  const kinTagB = document.getElementById('kinTagB');
  const lcBadgeB = document.getElementById('lcBadgeB');

  // Baselines comparison elements
  const baseStiffPruning = document.getElementById('baseStiffPruning');
  const baseStiffGA = document.getElementById('baseStiffGA');
  const baseRelGA = document.getElementById('baseRelGA');
  const baseStiffUGS = document.getElementById('baseStiffUGS');
  const baseRelUGS = document.getElementById('baseRelUGS');
  const baseStiffVor = document.getElementById('baseStiffVor');
  const baseRelVor = document.getElementById('baseRelVor');
  const baseStiffHC = document.getElementById('baseStiffHC');
  const baseRelHC = document.getElementById('baseRelHC');

  // Colormap helpers: [0, 1] -> CSS RGB color
  function colormapTurbo(t) {
    t = Math.max(0, Math.min(1, t));
    // Turbo approximation: blue -> cyan -> green -> yellow -> red
    const r = Math.round(255 * Math.sin(t * Math.PI * 0.9 + 0.1));
    const g = Math.round(255 * Math.sin(t * Math.PI * 0.8 + 0.2));
    const b = Math.round(255 * Math.cos(t * Math.PI * 0.8));
    return `rgb(${Math.max(20, r)}, ${Math.max(20, g)}, ${Math.max(40, b)})`;
  }

  function colormapMagma(t) {
    t = Math.max(0, Math.min(1, t));
    const r = Math.round(255 * Math.sqrt(t));
    const g = Math.round(200 * Math.pow(t, 2));
    const b = Math.round(220 * (1.0 - t));
    return `rgb(${r}, ${g}, ${b})`;
  }

  // Load JSON dataset with fallback
  async function loadDataset() {
    try {
      const response = await fetch('data/lattice_data.json');
      if (!response.ok) throw new Error('Data file not found');
      demoData = await response.json();
      console.log('Successfully loaded demo dataset from data/lattice_data.json');
    } catch (err) {
      console.warn('Could not load data/lattice_data.json directly (likely file:// protocol CORS). Generating client-side synthetic fallback demo...', err);
      demoData = generateFallbackDemoData();
    }
    updateScenario();
  }

  function getScenarioKey() {
    const lc = loadCaseSelect.value;
    const vf = Math.round(parseFloat(vfSelect.value) * 100);
    return `${lc}_vf${vf}`;
  }

  function updateScenario() {
    currentScenarioKey = getScenarioKey();
    if (!demoData || !demoData.scenarios[currentScenarioKey]) {
      // Find closest scenario
      const keys = Object.keys(demoData.scenarios);
      currentScenarioKey = keys[0] || 'LC1_vf20';
    }

    const scen = demoData.scenarios[currentScenarioKey];
    if (scen) {
      lcBadgeA.textContent = scen.load_case;
      lcBadgeB.textContent = scen.load_case;
      updateBaselineScoreboard(scen);
    }

    renderCurrentState();
  }

  function updateBaselineScoreboard(scen) {
    if (!scen || !scen.baselines) return;
    const b = scen.baselines;

    const prunLast = scen.remodeling_pruning[scen.remodeling_pruning.length - 1];
    const gaLast = scen.genetic_algorithm[scen.genetic_algorithm.length - 1];

    const kPrun = prunLast ? prunLast.stiffness : 264.77;
    const kGA = gaLast ? gaLast.stiffness : 224.53;
    const kUGS = b.uniform_gs ? b.uniform_gs.stiffness : 155.12;
    const kVor = b.voronoi ? b.voronoi.stiffness : 20.30;
    const kHC = b.honeycomb ? b.honeycomb.stiffness : 1.20;

    baseStiffPruning.textContent = `${kPrun.toFixed(1)} N/m`;
    baseStiffGA.textContent = `${kGA.toFixed(1)} N/m`;
    baseRelGA.textContent = `${((kGA / kPrun) * 100).toFixed(0)}% of Pruning`;

    baseStiffUGS.textContent = `${kUGS.toFixed(1)} N/m`;
    baseRelUGS.textContent = `${((kUGS / kPrun) * 100).toFixed(0)}% of Pruning`;

    baseStiffVor.textContent = `${kVor.toFixed(1)} N/m`;
    baseRelVor.textContent = `${((kVor / kPrun) * 100).toFixed(1)}% of Pruning`;

    baseStiffHC.textContent = `${kHC.toFixed(2)} N/m`;
    baseRelHC.textContent = `${((kHC / kPrun) * 100).toFixed(2)}% of Pruning`;
  }

  // Draw a frame on canvas
  function drawLatticeFrame(ctx, canvas, frame, isDeformed, defScale, colorMode, domainW, domainH) {
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    if (!frame || !frame.members) return;

    // Viewport transform: 3:1 aspect ratio inside canvas with padding
    const padX = 35;
    const padY = 25;
    const drawW = w - 2 * padX;
    const drawH = h - 2 * padY;

    const scaleX = drawW / domainW;
    const scaleY = drawH / domainH;

    function toScreenX(x) { return padX + x * scaleX; }
    function toScreenY(y) { return h - padY - y * scaleY; } // Flip Y for graphics coords

    // 1. Draw undeformed reference ghost if in deformed mode
    if (isDeformed) {
      ctx.lineWidth = 1.0;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.setLineDash([3, 3]);
      for (const m of frame.members) {
        ctx.beginPath();
        ctx.moveTo(toScreenX(m.x1), toScreenY(m.y1));
        ctx.lineTo(toScreenX(m.x2), toScreenY(m.y2));
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }

    // 2. Draw active members
    for (const m of frame.members) {
      let x1 = m.x1;
      let y1 = m.y1;
      let x2 = m.x2;
      let y2 = m.y2;

      if (isDeformed && frame.nodes) {
        const n1 = frame.nodes[String(m.node_i)];
        const n2 = frame.nodes[String(m.node_j)];
        if (n1) {
          x1 += n1.ux * defScale;
          y1 += n1.uy * defScale;
        }
        if (n2) {
          x2 += n2.ux * defScale;
          y2 += n2.uy * defScale;
        }
      }

      const sx1 = toScreenX(x1);
      const sy1 = toScreenY(y1);
      const sx2 = toScreenX(x2);
      const sy2 = toScreenY(y2);

      // Line width based on member thickness
      const strokeW = Math.max(1.5, Math.min(8.0, m.thickness * 150));
      ctx.lineWidth = strokeW;

      // Color based on colorMode
      if (colorMode === 'energy') {
        ctx.strokeStyle = colormapTurbo(m.normalized_energy || 0.0);
      } else if (colorMode === 'stress') {
        ctx.strokeStyle = colormapMagma(m.normalized_stress || 0.0);
      } else {
        // Thickness mode: gradient from slate to cyan
        const tNorm = Math.min(1.0, m.thickness / 0.025);
        ctx.strokeStyle = `rgb(${Math.round(50 + 180 * tNorm)}, ${Math.round(100 + 120 * tNorm)}, 240)`;
      }

      ctx.beginPath();
      ctx.moveTo(sx1, sy1);
      ctx.lineTo(sx2, sy2);
      ctx.stroke();
    }

    // 3. Draw nodes
    ctx.fillStyle = '#ffffff';
    for (const nid in frame.nodes) {
      const nd = frame.nodes[nid];
      let nx = nd.x;
      let ny = nd.y;
      if (isDeformed) {
        nx += nd.ux * defScale;
        ny += nd.uy * defScale;
      }
      ctx.beginPath();
      ctx.arc(toScreenX(nx), toScreenY(ny), 2.5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  function renderCurrentState() {
    if (!demoData || !demoData.scenarios[currentScenarioKey]) return;

    const scen = demoData.scenarios[currentScenarioKey];
    const framesA = scen.remodeling_pruning || [];
    const framesB = scen.genetic_algorithm || [];

    const isDeformed = deformToggle.checked;
    const defScale = parseFloat(defScaleSlider.value);
    const colorMode = colorModeSelect.value;
    const domainW = 3.0;
    const domainH = 1.0;

    // Map progress pct to frame index
    const idxA = Math.min(framesA.length - 1, Math.round((currentProgressPct / 100) * (framesA.length - 1)));
    const idxB = Math.min(framesB.length - 1, Math.round((currentProgressPct / 100) * (framesB.length - 1)));

    const frameA = framesA[idxA];
    const frameB = framesB[idxB];

    // Render Canvas A
    if (frameA) {
      drawLatticeFrame(ctxA, canvasA, frameA, isDeformed, defScale, colorMode, domainW, domainH);
      iterA.textContent = frameA.step_index;
      solvesA.textContent = frameA.fe_evaluations;
      compA.textContent = `${frameA.compliance.toFixed(6)} J`;
      stiffA.textContent = `${frameA.stiffness.toFixed(1)} N/m`;
      membersA.textContent = `${frameA.num_active_members}`;
      vfA.textContent = frameA.volume_fraction.toFixed(3);

      const axPctA = Math.round(frameA.axial_energy_share * 100);
      const bendPctA = 100 - axPctA;
      axShareA.textContent = `${axPctA}%`;
      bendShareA.textContent = `${bendPctA}%`;
      barAxialA.style.width = `${axPctA}%`;
      barBendingA.style.width = `${bendPctA}%`;
      kinTagA.textContent = axPctA >= 80 ? 'Stretch-Dominated' : 'Bending-Dominated';
    }

    // Render Canvas B
    if (frameB) {
      drawLatticeFrame(ctxB, canvasB, frameB, isDeformed, defScale, colorMode, domainW, domainH);
      iterB.textContent = frameB.step_index;
      solvesB.textContent = frameB.fe_evaluations;
      compB.textContent = `${frameB.compliance.toFixed(6)} J`;
      stiffB.textContent = `${frameB.stiffness.toFixed(1)} N/m`;
      membersB.textContent = `${frameB.num_active_members}`;
      vfB.textContent = frameB.volume_fraction.toFixed(3);

      const axPctB = Math.round(frameB.axial_energy_share * 100);
      const bendPctB = 100 - axPctB;
      axShareB.textContent = `${axPctB}%`;
      bendShareB.textContent = `${bendPctB}%`;
      barAxialB.style.width = `${axPctB}%`;
      barBendingB.style.width = `${bendPctB}%`;
      kinTagB.textContent = axPctB >= 80 ? 'Stretch-Dominated' : 'Bending-Dominated';
    }

    // Scrubber update
    timelineScrubber.value = currentProgressPct;
    timelineProgressText.textContent = `${currentProgressPct}% (${currentProgressPct === 100 ? 'Converged' : 'Optimizing'})`;
  }

  // Animation Loop
  function animationTick(timestamp) {
    if (!isPlaying) return;

    if (!lastTickTime) lastTickTime = timestamp;
    const elapsed = timestamp - lastTickTime;

    if (elapsed > 40) { // ~25 fps update
      currentProgressPct += playSpeed;
      if (currentProgressPct >= 100) {
        currentProgressPct = 100;
        isPlaying = false;
        btnPlayPause.innerHTML = '&#9654; Play';
      }
      renderCurrentState();
      lastTickTime = timestamp;
    }

    if (isPlaying) {
      animFrameId = requestAnimationFrame(animationTick);
    }
  }

  // Event Listeners
  loadCaseSelect.addEventListener('change', updateScenario);
  vfSelect.addEventListener('change', updateScenario);
  colorModeSelect.addEventListener('change', renderCurrentState);
  deformToggle.addEventListener('change', () => {
    scaleControlGroup.style.display = deformToggle.checked ? 'flex' : 'none';
    renderCurrentState();
  });
  defScaleSlider.addEventListener('input', () => {
    scaleValue.textContent = `${defScaleSlider.value}x`;
    renderCurrentState();
  });

  timelineScrubber.addEventListener('input', (e) => {
    currentProgressPct = parseInt(e.target.value, 10);
    renderCurrentState();
  });

  btnPlayPause.addEventListener('click', () => {
    isPlaying = !isPlaying;
    if (isPlaying) {
      if (currentProgressPct >= 100) currentProgressPct = 0;
      btnPlayPause.innerHTML = '&#10074;&#10074; Pause';
      lastTickTime = 0;
      animFrameId = requestAnimationFrame(animationTick);
    } else {
      btnPlayPause.innerHTML = '&#9654; Play';
      if (animFrameId) cancelAnimationFrame(animFrameId);
    }
  });

  btnStepBack.addEventListener('click', () => {
    currentProgressPct = Math.max(0, currentProgressPct - 5);
    renderCurrentState();
  });

  btnStepForward.addEventListener('click', () => {
    currentProgressPct = Math.min(100, currentProgressPct + 5);
    renderCurrentState();
  });

  btnReset.addEventListener('click', () => {
    isPlaying = false;
    btnPlayPause.innerHTML = '&#9654; Play';
    currentProgressPct = 0;
    renderCurrentState();
  });

  speedSelect.addEventListener('change', (e) => {
    playSpeed = parseInt(e.target.value, 10);
  });

  // Client-Side Synthetic Fallback Generator (CORS resilience)
  function generateFallbackDemoData() {
    const scenarios = {};
    const lcs = ['LC1', 'LC2', 'LC3'];
    const vfs = [0.2, 0.3, 0.4];

    // 13x5 grid coordinates
    const nodes = {};
    let nid = 1;
    for (let j = 0; j < 5; j++) {
      for (let i = 0; i < 13; i++) {
        nodes[String(nid)] = {
          x: (i / 12) * 3.0,
          y: (j / 4) * 1.0,
          ux: ((i / 12) * 0.0005) * (j - 2),
          uy: -0.002 * Math.pow(i / 12, 2),
          theta: 0.0001 * (i / 12),
        };
        nid++;
      }
    }

    // Base members
    const members = [];
    let eid = 1;
    for (let n1 = 1; n1 <= 65; n1++) {
      for (let n2 = n1 + 1; n2 <= Math.min(65, n1 + 15); n2++) {
        if (eid <= 150) {
          members.push({
            id: eid,
            node_i: n1,
            node_j: n2,
            x1: nodes[String(n1)].x,
            y1: nodes[String(n1)].y,
            x2: nodes[String(n2)].x,
            y2: nodes[String(n2)].y,
            thickness: 0.015,
            length: 0.25,
            strain_energy: 0.001,
            normalized_energy: Math.random(),
            axial_force: (Math.random() - 0.5) * 1000,
            peak_stress: Math.random() * 50e6,
            normalized_stress: Math.random(),
          });
          eid++;
        }
      }
    }

    for (const lc of lcs) {
      for (const vf of vfs) {
        const key = `${lc}_vf${Math.round(vf * 100)}`;
        const pFrames = [];
        const gFrames = [];

        // 20 pruning frames
        for (let s = 0; s < 20; s++) {
          const keepCount = Math.max(50, Math.round(150 - s * 5));
          pFrames.push({
            step_index: s,
            fe_evaluations: s + 1,
            compliance: 0.01 / (1.0 + s * 0.1),
            stiffness: 100 + s * 8.5,
            volume: vf * 3.0,
            volume_fraction: Math.max(vf, 0.45 - s * 0.015),
            axial_energy_share: 0.89 + 0.005 * s,
            bending_energy_share: 0.11 - 0.005 * s,
            num_active_members: keepCount,
            nodes: nodes,
            members: members.slice(0, keepCount),
          });
        }

        // 30 GA frames
        for (let g = 0; g < 30; g++) {
          const keepCount = Math.max(90, Math.round(150 - g * 2));
          gFrames.push({
            step_index: g,
            fe_evaluations: (g + 1) * 30,
            compliance: 0.015 / (1.0 + g * 0.08),
            stiffness: 80 + g * 5.0,
            volume: vf * 3.0,
            volume_fraction: vf,
            axial_energy_share: 0.99,
            bending_energy_share: 0.01,
            num_active_members: keepCount,
            nodes: nodes,
            members: members.slice(0, keepCount),
          });
        }

        scenarios[key] = {
          load_case: lc,
          volume_fraction: vf,
          remodeling_pruning: pFrames,
          genetic_algorithm: gFrames,
          baselines: {
            honeycomb: { compliance: 0.83, stiffness: 1.20, volume_fraction: vf },
            voronoi: { compliance: 0.049, stiffness: 20.30, volume_fraction: vf },
            uniform_gs: { compliance: 0.0064, stiffness: 155.12, volume_fraction: vf },
          },
        };
      }
    }

    return { scenarios };
  }

  // Initialize on DOM ready
  window.addEventListener('DOMContentLoaded', loadDataset);
})();

