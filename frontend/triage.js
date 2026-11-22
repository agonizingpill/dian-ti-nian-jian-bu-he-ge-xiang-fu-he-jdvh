/* Floating supervisor panel for triage-tree version switching. */
const TriageUI = (() => {
  const getToken = () => localStorage.getItem('lift_token');
  const getUser = () => JSON.parse(localStorage.getItem('lift_user') || 'null');

  function el(tag, text) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    return node;
  }

  async function api(path, options = {}) {
    const res = await fetch(path, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(getToken() ? { Authorization: `Bearer ${getToken()}` } : {}),
        ...(options.headers || {}),
      },
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = data.detail;
      throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail || data));
    }
    return data;
  }

  const launcher = el('button', '分流树版本');
  launcher.type = 'button';
  launcher.id = 'triage-open';
  launcher.hidden = true;
  const dialog = el('dialog');
  dialog.id = 'triage-panel';
  document.body.append(launcher, dialog);

  function statusBadge(row) {
    if (row.load_status === 'ready') return '<span class="triage-pill">可加载</span>';
    if (row.load_status === 'error') return '<span class="triage-pill" style="background:#fbe4e1;color:#9b2c22;border-color:#eab8b1">加载失败</span>';
    return '<span class="triage-pill">待验证</span>';
  }

  function activePill(version) {
    const cls = version && version.includes('v2') ? 'triage-pill v2' : 'triage-pill';
    return `<span class="${cls}">${version || '—'}</span>`;
  }

  async function render() {
    const user = getUser();
    const isSupervisor = user?.role === 'supervisor';
    const [{ versions, active }, ] = await Promise.all([api('/api/triage/versions')]);
    const activeVersion = active?.active_version;
    dialog.replaceChildren();

    const header = el('header');
    header.append(el('h2', '分流树版本管理'));
    const closeBtn = el('button', '关闭');
    closeBtn.addEventListener('click', () => dialog.close());
    header.append(closeBtn);
    dialog.append(header);

    const activeLine = el('div', '');
    activeLine.className = 'active-line';
    activeLine.innerHTML =
      `<span>当前生效版本：${activePill(activeVersion)}</span>` +
      `<span class="muted">切换人：${active?.switched_by_name || '系统'} · ${active?.switched_at || ''}</span>` +
      `<span class="muted">原因：${active?.reason || ''}</span>`;
    dialog.append(activeLine);

    const feedback = el('p');
    feedback.setAttribute('role', 'status');
    dialog.append(feedback);

    for (const row of versions) {
      const card = el('div');
      card.className = 'version-card' + (row.version === activeVersion ? ' is-active' : '');
      const head = el('div');
      head.className = 'version-head';
      head.innerHTML =
        `<div><code>${row.version}</code> ${row.version === activeVersion ? '<strong>（生效中）</strong>' : ''}<br>` +
        `<small class="muted">${row.title} · ${row.built_in ? '内置版本' : '登记版本'}</small></div>` +
        statusBadge(row);
      card.append(head);
      card.append(el('p', row.description));
      if (row.load_error) {
        const err = el('p', `加载错误：${row.load_error}`);
        err.className = 'load-error';
        card.append(err);
      }

      const actions = el('div');
      actions.className = 'row-actions';
      const diffBtn = el('button', '检验项差异预览');
      diffBtn.addEventListener('click', () => showDiff(row.version, feedback));
      actions.append(diffBtn);

      if (isSupervisor && row.version !== activeVersion) {
        const switchBtn = el('button', '切换到此版本');
        switchBtn.className = 'primary';
        if (row.load_status === 'error') switchBtn.disabled = true;
        switchBtn.addEventListener('click', () => showSwitchForm(row, feedback));
        actions.append(switchBtn);
      }
      card.append(actions);
      dialog.append(card);
    }
    if (!isSupervisor) {
      const note = el('p', '仅主管可切换版本；当前账号可查看生效版本与各单实际版本。');
      note.className = 'muted';
      dialog.append(note);
    }
    await renderJobList(dialog);
  }

  async function renderJobList(dialogNode) {
    const wrap = el('div');
    wrap.className = 'job-list-wrap';
    wrap.append(el('h3', '各复核单实际使用的版本'));
    try {
      const data = await api('/api/triage/analysis-jobs?limit=100');
      if (!data.jobs.length) {
        wrap.append(el('p', '暂无复核单。'));
      } else {
        const table = el('table');
        table.className = 'data-table';
        table.innerHTML = '<thead><tr><th>复核单</th><th>设备</th><th>状态</th><th>建议</th><th>实际版本</th></tr></thead>';
        const tbody = el('tbody');
        for (const job of data.jobs) {
          const tr = el('tr');
          tr.innerHTML =
            `<td>#${job.id}</td><td>${job.asset_code}（#${job.lift_id}）</td>` +
            `<td>${job.status}</td><td>${job.prediction || '—'}</td>` +
            `<td><code>${job.model_version}</code></td>`;
          tbody.append(tr);
        }
        table.append(tbody);
        wrap.append(table);
      }
    } catch (error) {
      const p = el('p', `复核单版本读取失败：${error.message}`);
      p.className = 'load-error';
      wrap.append(p);
    }
    dialogNode.append(wrap);
  }

  function showSwitchForm(row, feedback) {
    const existing = dialog.querySelector('.switch-form');
    if (existing) existing.remove();
    const form = el('form');
    form.className = 'switch-form';
    const label = el('label', '切换原因（不少于 4 个字）');
    const input = el('input');
    input.required = true;
    input.minLength = 4;
    input.maxLength = 500;
    input.placeholder = row.version.includes('v2') ? '主管试用新版分流树' : '新版建议不合适，切回上一版';
    label.append(input);
    const submit = el('button', `确认切换到 ${row.version}`);
    submit.type = 'submit';
    submit.className = 'primary';
    form.append(label, submit);
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      submit.disabled = true;
      feedback.className = '';
      feedback.textContent = '正在校验并切换…';
      try {
        await api(`/api/triage/versions/${encodeURIComponent(row.version)}/activate`, {
          method: 'POST',
          body: JSON.stringify({ version: row.version, reason: input.value }),
        });
        feedback.className = 'ok';
        feedback.textContent = `已切换到 ${row.version}，新接复核单按此版本计算。`;
        await render();
      } catch (error) {
        feedback.className = 'danger';
        feedback.textContent = error.message;
        submit.disabled = false;
      }
    });
    const card = [...dialog.querySelectorAll('.version-card')]
      .find((node) => node.querySelector('code')?.textContent === row.version);
    card.append(form);
    input.focus();
  }

  async function showDiff(version, feedback) {
    feedback.className = '';
    feedback.textContent = '正在计算两版建议差异…';
    try {
      const data = await api(`/api/triage/versions/${encodeURIComponent(version)}/diff`);
      const old = dialog.querySelector('.diff-wrap');
      if (old) old.remove();
      const wrap = el('div');
      wrap.className = 'diff-wrap';
      wrap.append(el('h3', `差异预览（${data.baseline_version} → ${data.target_version}）`));
      const summary = el('p', `共 ${data.rows.length} 组近期检验项，其中 ${data.divergent_count} 组建议不同。`);
      wrap.append(summary);
      if (!data.rows.length) {
        wrap.append(el('p', '暂无复核任务可用于对比，可先在检验复核页提交一单。'));
      } else {
        const table = el('table');
        table.className = 'diff-table data-table';
        table.innerHTML = '<thead><tr><th>复核单</th><th>特征</th><th>当前版本建议</th><th>目标版本建议</th><th>是否不同</th></tr></thead>';
        const tbody = el('tbody');
        for (const r of data.rows) {
          const tr = el('tr');
          const f = r.features;
          tr.innerHTML =
            `<td>#${r.analysis_job_id}（电梯 ${r.lift_id}）</td>` +
            `<td class="muted">关键失败 ${f.critical_fail_count} · 均分 ${f.average_score}<br>失败项 ${f.failed_item_count} · 资料缺口 ${f.documentation_gap}</td>` +
            `<td>${r.baseline_prediction}</td><td>${r.target_prediction}</td>` +
            `<td class="${r.diverges ? 'diverges' : 'same'}">${r.diverges ? '不同' : '一致'}</td>`;
          tbody.append(tr);
        }
        table.append(tbody);
        wrap.append(table);
      }
      dialog.append(wrap);
      feedback.textContent = '';
    } catch (error) {
      feedback.className = 'danger';
      feedback.textContent = error.message;
    }
  }

  launcher.addEventListener('click', async () => {
    dialog.showModal();
    try {
      await render();
    } catch (error) {
      dialog.replaceChildren();
      const p = el('p', error.message);
      p.style.color = '#9b2c22';
      dialog.append(p);
    }
  });

  function syncVisibility() {
    let signedIn = false;
    try { signedIn = Boolean(getToken()); } catch { /* ignore */ }
    launcher.hidden = !signedIn;
    if (!signedIn && dialog.open) dialog.close();
  }
  syncVisibility();
  window.addEventListener('storage', syncVisibility);
  setInterval(syncVisibility, 600);

  return { render, activePill };
})();
