/* ═══════════════════════════════════════════════════════════════
   IceGuard AI — Demo 交互逻辑
   粒子动画 · 数据可视化 · 滚动动画 · 预警模拟
   ═══════════════════════════════════════════════════════════════ */

// ── Chart.js 全局配置 ──────────────────────────────────────
Chart.defaults.color = '#707eae';
Chart.defaults.borderColor = 'rgba(0,0,0,0.05)';
Chart.defaults.font.family = "'Inter', 'Noto Sans SC', sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.plugins.legend.labels.usePointStyle = true;
Chart.defaults.plugins.legend.labels.pointStyleWidth = 12;

// ══════════════════════════════════════════════════════════════
// 1. PARTICLE ANIMATION BACKGROUND
// ══════════════════════════════════════════════════════════════
class ParticleSystem {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.particles = [];
    this.connections = [];
    this.mouse = { x: null, y: null, radius: 150 };
    this.resize();
    this.init();
    this.bindEvents();
    this.animate();
  }

  resize() {
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
  }

  init() {
    const count = Math.min(80, Math.floor(window.innerWidth / 18));
    this.particles = [];
    for (let i = 0; i < count; i++) {
      this.particles.push({
        x: Math.random() * this.canvas.width,
        y: Math.random() * this.canvas.height,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        r: Math.random() * 2 + 0.5,
        alpha: Math.random() * 0.5 + 0.2,
      });
    }
  }

  bindEvents() {
    window.addEventListener('resize', () => { this.resize(); this.init(); });
    window.addEventListener('mousemove', (e) => {
      this.mouse.x = e.clientX;
      this.mouse.y = e.clientY;
    });
  }

  animate() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    const w = this.canvas.width, h = this.canvas.height;

    for (const p of this.particles) {
      // Mouse interaction
      if (this.mouse.x !== null) {
        const dx = p.x - this.mouse.x;
        const dy = p.y - this.mouse.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < this.mouse.radius) {
          const force = (this.mouse.radius - dist) / this.mouse.radius * 0.02;
          p.vx += dx * force;
          p.vy += dy * force;
        }
      }

      p.x += p.vx;
      p.y += p.vy;
      p.vx *= 0.99;
      p.vy *= 0.99;

      // Wrap around edges
      if (p.x < 0) p.x = w;
      if (p.x > w) p.x = 0;
      if (p.y < 0) p.y = h;
      if (p.y > h) p.y = 0;

      // Draw particle
      this.ctx.beginPath();
      this.ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      this.ctx.fillStyle = `rgba(59, 130, 246, ${p.alpha})`;
      this.ctx.fill();
    }

    // Draw connections
    for (let i = 0; i < this.particles.length; i++) {
      for (let j = i + 1; j < this.particles.length; j++) {
        const a = this.particles[i], b = this.particles[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 130) {
          this.ctx.beginPath();
          this.ctx.moveTo(a.x, a.y);
          this.ctx.lineTo(b.x, b.y);
          const alpha = (130 - dist) / 130 * 0.12;
          this.ctx.strokeStyle = `rgba(59, 130, 246, ${alpha})`;
          this.ctx.lineWidth = 0.5;
          this.ctx.stroke();
        }
      }
    }

    requestAnimationFrame(() => this.animate());
  }
}

// ══════════════════════════════════════════════════════════════
// 2. COUNTER ANIMATION
// ══════════════════════════════════════════════════════════════
function animateCounters() {
  const counters = document.querySelectorAll('.counter');
  counters.forEach(counter => {
    if (counter.dataset.animated) return;
    const target = parseFloat(counter.dataset.target);
    const decimals = parseInt(counter.dataset.decimal || '0');
    const duration = 2000;
    const start = performance.now();

    function update(now) {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 4); // ease-out quart
      const current = target * eased;
      
      if (decimals > 0) {
        counter.textContent = current.toFixed(decimals);
      } else {
        counter.textContent = Math.floor(current).toLocaleString();
      }

      if (progress < 1) {
        requestAnimationFrame(update);
      } else {
        if (decimals > 0) {
          counter.textContent = target.toFixed(decimals);
        } else {
          counter.textContent = target.toLocaleString();
        }
      }
    }
    counter.dataset.animated = '1';
    requestAnimationFrame(update);
  });
}

// ══════════════════════════════════════════════════════════════
// 3. SCROLL REVEAL ANIMATION
// ══════════════════════════════════════════════════════════════
function setupScrollReveal() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('active');
        // Trigger counter animation when stat cards become visible
        if (entry.target.querySelector('.counter')) {
          animateCounters();
        }
      }
    });
  }, { threshold: 0.15, rootMargin: '0px 0px -50px 0px' });

  document.querySelectorAll('.reveal').forEach(el => observer.observe(el));
}

// ══════════════════════════════════════════════════════════════
// 4. NAVBAR SCROLL EFFECT
// ══════════════════════════════════════════════════════════════
function setupNavbar() {
  const navbar = document.getElementById('navbar');
  let ticking = false;
  window.addEventListener('scroll', () => {
    if (!ticking) {
      requestAnimationFrame(() => {
        navbar.classList.toggle('scrolled', window.scrollY > 50);
        ticking = false;
      });
      ticking = true;
    }
  });
}

// ══════════════════════════════════════════════════════════════
// 5. GAUGE ANIMATION
// ══════════════════════════════════════════════════════════════
function animateGauges() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const circle = entry.target;
        const target = parseFloat(circle.dataset.target);
        setTimeout(() => {
          circle.style.strokeDashoffset = 364.4 - target;
        }, 300);
        observer.unobserve(circle);
      }
    });
  }, { threshold: 0.5 });

  document.querySelectorAll('.gauge-fill').forEach(el => observer.observe(el));
}

// ══════════════════════════════════════════════════════════════
// 6. CHARTS — Time Series
// ══════════════════════════════════════════════════════════════
function createTimeseriesChart() {
  const ctx = document.getElementById('timeseriesChart');
  if (!ctx) return;

  // Simulated ice thickness data (based on real patterns)
  const months = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月'];
  
  // Terminal 1 — severe winter icing
  const t1 = [18.5, 12.3, 2.1, 0, 0, 0, 0, 0, 0, 0, 1.2, 8.6];
  // Terminal 2 — moderate icing
  const t2 = [9.2, 6.8, 0.8, 0, 0, 0, 0, 0, 0, 0, 0.5, 4.3];
  // Terminal 3 — light icing
  const t3 = [4.5, 3.1, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 2.1];

  new Chart(ctx, {
    type: 'line',
    data: {
      labels: months,
      datasets: [
        {
          label: '终端 #03 (高海拔)',
          data: t1,
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          fill: true,
          tension: 0.4,
          borderWidth: 2.5,
          pointRadius: 4,
          pointHoverRadius: 7,
          pointBackgroundColor: '#3b82f6',
        },
        {
          label: '终端 #11 (中海拔)',
          data: t2,
          borderColor: '#8b5cf6',
          backgroundColor: 'rgba(139, 92, 246, 0.05)',
          fill: true,
          tension: 0.4,
          borderWidth: 2,
          pointRadius: 3,
          pointHoverRadius: 6,
          pointBackgroundColor: '#8b5cf6',
        },
        {
          label: '终端 #19 (低海拔)',
          data: t3,
          borderColor: '#06b6d4',
          backgroundColor: 'rgba(6, 182, 212, 0.05)',
          fill: true,
          tension: 0.4,
          borderWidth: 2,
          pointRadius: 3,
          pointHoverRadius: 6,
          pointBackgroundColor: '#06b6d4',
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        tooltip: {
          backgroundColor: 'rgba(255, 255, 255, 0.95)',
          titleColor: '#2b3674',
          bodyColor: '#475569',
          borderColor: 'rgba(59, 130, 246, 0.2)',
          borderWidth: 1,
          padding: 12,
          titleFont: { weight: '600' },
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)} mm`
          }
        }
      },
      scales: {
        y: {
          title: { display: true, text: '覆冰厚度 (mm)', color: '#64748b' },
          grid: { color: 'rgba(255,255,255,0.04)' },
          beginAtZero: true,
        },
        x: {
          grid: { display: false }
        }
      }
    }
  });
}

// ══════════════════════════════════════════════════════════════
// 7. CHARTS — Correlation Heatmap (as bar chart)
// ══════════════════════════════════════════════════════════════
function createCorrelationChart() {
  const ctx = document.getElementById('correlationChart');
  if (!ctx) return;

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['覆冰厚度\n↔ 覆冰比值', '覆冰厚度\n↔ 温度', '覆冰厚度\n↔ 湿度', '温度\n↔ 湿度'],
      datasets: [{
        label: 'Pearson 相关系数',
        data: [0.970, -0.205, 0.060, -0.128],
        backgroundColor: [
          'rgba(59, 130, 246, 0.7)',
          'rgba(239, 68, 68, 0.7)',
          'rgba(16, 185, 129, 0.5)',
          'rgba(245, 158, 11, 0.5)',
        ],
        borderColor: [
          '#3b82f6',
          '#ef4444',
          '#10b981',
          '#f59e0b',
        ],
        borderWidth: 1.5,
        borderRadius: 8,
        barPercentage: 0.6,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(255, 255, 255, 0.95)',
          titleColor: '#2b3674',
          bodyColor: '#475569',
          callbacks: {
            label: (ctx) => `r = ${ctx.parsed.x.toFixed(3)}`
          }
        }
      },
      scales: {
        x: {
          min: -0.5,
          max: 1.0,
          grid: { color: 'rgba(255,255,255,0.04)' },
          title: { display: true, text: '相关系数 r', color: '#64748b' }
        },
        y: {
          grid: { display: false },
          ticks: { font: { size: 11 } }
        }
      }
    }
  });
}

// ══════════════════════════════════════════════════════════════
// 8. CHARTS — Distribution
// ══════════════════════════════════════════════════════════════
function createDistributionChart() {
  const ctx = document.getElementById('distributionChart');
  if (!ctx) return;

  new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['无覆冰样本 (76.9%)', '有覆冰样本 (23.1%)'],
      datasets: [{
        data: [76.9, 23.1],
        backgroundColor: [
          'rgba(100, 116, 139, 0.4)',
          'rgba(59, 130, 246, 0.7)',
        ],
        borderColor: [
          'rgba(100, 116, 139, 0.6)',
          '#3b82f6',
        ],
        borderWidth: 2,
        hoverOffset: 8,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '65%',
      plugins: {
        legend: {
          position: 'bottom',
          labels: { padding: 20 }
        },
        tooltip: {
          backgroundColor: 'rgba(255, 255, 255, 0.95)',
          titleColor: '#2b3674',
          bodyColor: '#475569',
        }
      }
    }
  });
}

// ══════════════════════════════════════════════════════════════
// 9. CHARTS — Window Analysis
// ══════════════════════════════════════════════════════════════
function createWindowChart() {
  const ctx = document.getElementById('windowChart');
  if (!ctx) return;

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['0-6h', '6-12h', '12-24h', '24-48h'],
      datasets: [
        {
          label: 'Acc@0.2 (%)',
          data: [60.2, 69.9, 77.0, 78.6],
          backgroundColor: 'rgba(59, 130, 246, 0.6)',
          borderColor: '#3b82f6',
          borderWidth: 1.5,
          borderRadius: 8,
          yAxisID: 'y',
        },
        {
          label: 'MAE (mm)',
          data: [5.92, 3.97, 3.11, 3.37],
          type: 'line',
          borderColor: '#f59e0b',
          backgroundColor: 'rgba(245, 158, 11, 0.1)',
          borderWidth: 2.5,
          pointRadius: 5,
          pointBackgroundColor: '#f59e0b',
          tension: 0.3,
          fill: true,
          yAxisID: 'y1',
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        tooltip: {
          backgroundColor: 'rgba(255, 255, 255, 0.95)',
          titleColor: '#2b3674',
          bodyColor: '#475569',
          borderColor: 'rgba(59, 130, 246, 0.2)',
          borderWidth: 1,
        }
      },
      scales: {
        y: {
          position: 'left',
          title: { display: true, text: 'Acc@0.2 (%)', color: '#3b82f6' },
          grid: { color: 'rgba(255,255,255,0.04)' },
          min: 40,
          max: 90,
        },
        y1: {
          position: 'right',
          title: { display: true, text: 'MAE (mm)', color: '#f59e0b' },
          grid: { drawOnChartArea: false },
          min: 0,
          max: 8,
        },
        x: {
          grid: { display: false }
        }
      }
    }
  });
}

// ══════════════════════════════════════════════════════════════
// 10. CHARTS — Error Distribution
// ══════════════════════════════════════════════════════════════
function createErrorChart() {
  const ctx = document.getElementById('errorChart');
  if (!ctx) return;

  // Simulated error distribution (bell-shaped, peaked near 0)
  const bins = ['0-0.05', '0.05-0.1', '0.1-0.15', '0.15-0.2', '0.2-0.5', '0.5-1.0', '1.0-2.0', '2.0-5.0', '>5.0'];
  const counts = [35.2, 18.1, 10.5, 7.6, 3.8, 2.1, 4.2, 8.5, 10.0];

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: bins,
      datasets: [{
        label: '样本占比 (%)',
        data: counts,
        backgroundColor: counts.map((v, i) =>
          i < 4 ? 'rgba(16, 185, 129, 0.6)' :
          i < 6 ? 'rgba(245, 158, 11, 0.6)' :
          'rgba(239, 68, 68, 0.5)'
        ),
        borderColor: counts.map((v, i) =>
          i < 4 ? '#10b981' :
          i < 6 ? '#f59e0b' :
          '#ef4444'
        ),
        borderWidth: 1.5,
        borderRadius: 6,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(255, 255, 255, 0.95)',
          titleColor: '#2b3674',
          bodyColor: '#475569',
          callbacks: {
            title: (items) => `误差区间: ${items[0].label} mm`,
            label: (ctx) => `占比: ${ctx.parsed.y}%`
          }
        }
      },
      scales: {
        y: {
          title: { display: true, text: '样本占比 (%)', color: '#64748b' },
          grid: { color: 'rgba(255,255,255,0.04)' },
          beginAtZero: true,
        },
        x: {
          title: { display: true, text: '预测误差 (mm)', color: '#64748b' },
          grid: { display: false },
          ticks: { font: { size: 10 } }
        }
      }
    }
  });
}

// ══════════════════════════════════════════════════════════════
// 11. ALERT SIMULATION
// ══════════════════════════════════════════════════════════════
function runAlertSimulation() {
  const output = document.getElementById('simOutput');
  const btn = document.getElementById('runSimBtn');
  btn.disabled = true;
  btn.textContent = '⏳ 模拟中...';
  output.innerHTML = '';

  const scenarios = [
    { thickness: 0.12, ratio: 0.004, vlm: 'no', delay: 0 },
    { thickness: 1.35, ratio: 0.042, vlm: 'unknow', delay: 2000 },
    { thickness: 3.82, ratio: 0.120, vlm: 'yes', delay: 4000 },
    { thickness: 8.56, ratio: 0.268, vlm: 'yes', delay: 6000 },
  ];

  const levelConfig = [
    { name: '正常', color: '#10b981', emoji: '✅' },
    { name: '注意', color: '#f59e0b', emoji: '⚡' },
    { name: '预警', color: '#f97316', emoji: '⚠️' },
    { name: '紧急', color: '#ef4444', emoji: '🚨' },
  ];

  scenarios.forEach((s, idx) => {
    setTimeout(() => {
      // Determine level
      let level, reason;
      if (s.vlm === 'yes' && s.thickness > 5) {
        level = 3;
        reason = `时序预测=${s.thickness}mm(>5mm) 且 VLM双重确认覆冰`;
      } else if (s.vlm === 'yes' && s.thickness > 2) {
        level = 3;
        reason = `时序预测=${s.thickness}mm(>2mm) 且 VLM确认覆冰`;
      } else if (s.vlm === 'yes' || s.thickness > 2) {
        level = 2;
        reason = s.vlm === 'yes' ? 'VLM图像识别发现覆冰' : `预测厚度=${s.thickness}mm(>2mm)`;
      } else if (s.thickness > 0.5 || s.vlm === 'unknow') {
        level = 1;
        reason = s.vlm === 'unknow' ? 'VLM识别不确定，建议人工复核' : `预测${s.thickness}mm(0.5~2mm)`;
      } else {
        level = 0;
        reason = `预测${s.thickness}mm 正常，VLM未发现覆冰`;
      }

      const cfg = levelConfig[level];
      const line = document.createElement('span');
      line.className = 'sim-line';
      line.style.animationDelay = `${idx * 0.1}s`;
      line.innerHTML = `<span style="color:${cfg.color}">${cfg.emoji} [Level ${level} · ${cfg.name}]</span> ` +
        `覆冰厚度=<span style="color:#3b82f6">${s.thickness}mm</span> | ` +
        `比值=<span style="color:#8b5cf6">${s.ratio}</span> | ` +
        `VLM=<span style="color:${s.vlm==='yes'?'#ef4444':s.vlm==='no'?'#10b981':'#f59e0b'}">${s.vlm}</span> | ` +
        `<span style="color:var(--text-muted)">${reason}</span>\n`;
      output.appendChild(line);

      // Highlight corresponding card
      document.querySelectorAll('.alert-level-card').forEach(c => c.classList.remove('active'));
      const card = document.querySelector(`[data-level="${level}"]`);
      if (card) card.classList.add('active');

      if (idx === scenarios.length - 1) {
        setTimeout(() => {
          btn.disabled = false;
          btn.textContent = '▶ 运行模拟';
          const summary = document.createElement('span');
          summary.className = 'sim-line';
          summary.style.animationDelay = '0.2s';
          summary.innerHTML = `\n<span style="color:var(--accent-cyan)">━━━ 模拟完成 ━━━ 4个场景已全部演示，覆盖从正常到紧急全部预警等级</span>`;
          output.appendChild(summary);
        }, 800);
      }
    }, s.delay);
  });
}

// ══════════════════════════════════════════════════════════════
// 12. ALERT CARD HOVER INTERACTIONS
// ══════════════════════════════════════════════════════════════
function setupAlertCards() {
  document.querySelectorAll('.alert-level-card').forEach(card => {
    card.addEventListener('click', () => {
      document.querySelectorAll('.alert-level-card').forEach(c => c.classList.remove('active'));
      card.classList.add('active');
    });
  });
}

// ══════════════════════════════════════════════════════════════
// 13. LIVE DATA SIMULATION (subtle dashboard updates)
// ══════════════════════════════════════════════════════════════
function startLiveSimulation() {
  // Periodically update small elements to give a "live" feel
  setInterval(() => {
    const dot = document.querySelector('.hero-badge .dot');
    if (dot) {
      dot.style.background = Math.random() > 0.1 ? '#10b981' : '#f59e0b';
    }
  }, 3000);
}

// ══════════════════════════════════════════════════════════════
// 15. DYNAMIC PREDICTION & IMAGE MONITOR
// ══════════════════════════════════════════════════════════════
function createDynamicPredictionChart(predictions) {
  const ctx = document.getElementById('predictionChart');
  if (!ctx || !predictions) return;

  const labels = predictions.map(p => p.hour);
  const data = predictions.map(p => p.thickness);

  new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: '预测覆冰厚度 (mm)',
        data: data,
        borderColor: '#e31a1a',
        backgroundColor: 'rgba(227, 26, 26, 0.1)',
        fill: true,
        tension: 0.4,
        borderWidth: 2.5,
        pointRadius: 0,
        pointHoverRadius: 6,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        tooltip: {
          backgroundColor: 'rgba(255, 255, 255, 0.95)',
          titleColor: '#2b3674',
          bodyColor: '#475569',
          borderColor: 'rgba(0,0,0,0.08)',
          borderWidth: 1,
          padding: 12,
        }
      },
      scales: {
        y: {
          title: { display: true, text: '预测覆冰厚度 (mm)' },
          beginAtZero: true,
        },
        x: {
          grid: { display: false },
          ticks: { maxTicksLimit: 12 }
        }
      }
    }
  });
}

function startDynamicImageMonitor(images) {
  if (!images || images.length === 0) return;

  const imgEl = document.getElementById('live-monitor-img');
  const badgeEl = document.getElementById('live-monitor-badge');
  const alertBox = document.getElementById('monitor-alert-box');
  const alertIcon = document.getElementById('alert-icon');
  const alertText = document.getElementById('alert-text');
  const timeEl = document.getElementById('monitor-time');
  const resultEl = document.getElementById('monitor-result');

  let currentIndex = 0;

  for (let i = images.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [images[i], images[j]] = [images[j], images[i]];
  }

  function updateMonitor() {
    const imgData = images[currentIndex];
    imgEl.style.opacity = '0.5';
    
    setTimeout(() => {
      imgEl.src = imgData.path;
      imgEl.onload = () => {
        imgEl.style.opacity = '1';
        const now = new Date();
        timeEl.textContent = now.toLocaleTimeString('zh-CN', { hour12: false });
        
        if (imgData.isIce) {
          badgeEl.className = 'img-badge yes';
          badgeEl.innerHTML = '⚠ 覆冰检出 · 异常';
          badgeEl.style.background = 'rgba(227, 26, 26, 0.9)';
          badgeEl.style.color = '#fff';
          
          alertBox.style.background = 'rgba(227, 26, 26, 0.1)';
          alertBox.style.borderColor = 'rgba(227, 26, 26, 0.3)';
          alertIcon.innerHTML = '🚨';
          alertText.innerHTML = '发现覆冰！立即报警';
          alertText.style.color = '#e31a1a';
          
          resultEl.innerHTML = '<span style="color:#e31a1a">存在积雪或结冰</span>';
        } else {
          badgeEl.className = 'img-badge no';
          badgeEl.innerHTML = '✓ 状态正常 · 安全';
          badgeEl.style.background = 'rgba(1, 181, 116, 0.9)';
          badgeEl.style.color = '#fff';
          
          alertBox.style.background = 'rgba(1, 181, 116, 0.1)';
          alertBox.style.borderColor = 'rgba(1, 181, 116, 0.3)';
          alertIcon.innerHTML = '✅';
          alertText.innerHTML = '线路状态正常';
          alertText.style.color = '#01b574';
          
          resultEl.innerHTML = '<span style="color:#01b574">无覆冰现象</span>';
        }
      };
      
      imgEl.onerror = () => {
        console.error('Failed to load image:', imgData.path);
        badgeEl.innerHTML = '⚠ 图像加载失败';
        badgeEl.className = 'img-badge yes';
        badgeEl.style.background = 'rgba(100, 100, 100, 0.9)';
        currentIndex = (currentIndex + 1) % images.length;
      };

      currentIndex = (currentIndex + 1) % images.length;
    }, 300);
  }

  updateMonitor();
  setInterval(updateMonitor, 4000);
}

function loadDynamicData() {
  fetch('data.json?t=' + new Date().getTime())
    .then(res => res.json())
    .then(data => {
      createDynamicPredictionChart(data.predictions);
      startDynamicImageMonitor(data.images);
    })
    .catch(err => console.error('Failed to load data:', err));
}

// ══════════════════════════════════════════════════════════════
// 14. SMOOTH SCROLL FOR NAV LINKS
// ══════════════════════════════════════════════════════════════
function setupSmoothScroll() {
  document.querySelectorAll('.nav-links a').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const target = document.querySelector(link.getAttribute('href'));
      if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
}

// ══════════════════════════════════════════════════════════════
// INITIALIZATION
// ══════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  // Particle background
  new ParticleSystem('particles-canvas');

  // UI interactions
  setupNavbar();
  setupScrollReveal();
  setupSmoothScroll();
  setupAlertCards();
  animateGauges();
  startLiveSimulation();
  loadDynamicData();

  // Charts (delayed slightly for smooth page load)
  setTimeout(() => {
    createTimeseriesChart();
    createCorrelationChart();
    createDistributionChart();
    createWindowChart();
    createErrorChart();
  }, 500);

  // Trigger counter animation for hero section
  setTimeout(animateCounters, 800);
});
