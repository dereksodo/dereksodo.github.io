// 简单的暗色/亮色主题切换，记录到 localStorage
(function () {
  const key = "theme";
  const saved = localStorage.getItem(key);
  const prefersLight = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
  if (saved) {
    document.documentElement.setAttribute('data-theme', saved);
  } else {
    document.documentElement.setAttribute('data-theme', prefersLight ? 'light' : 'dark');
  }

  const btn = document.getElementById('themeToggle');
  if (btn) {
    btn.addEventListener('click', () => {
      const curr = document.documentElement.getAttribute('data-theme');
      const next = curr === 'light' ? 'dark' : 'light';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem(key, next);
    });
  }

  // 平滑滚动到文章列表
  const ghost = document.querySelector('.btn.ghost');
  if (ghost) {
    ghost.addEventListener('click', (e) => {
      const id = ghost.getAttribute('href');
      if (id && id.startsWith('#')) {
        e.preventDefault();
        const el = document.querySelector(id);
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  }
})();