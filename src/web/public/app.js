const socket = io();

const taskList = document.getElementById('taskList');
const consoleOutput = document.getElementById('consoleOutput');
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const outputPathInput = document.getElementById('outputPath');
const selectOutputPathBtn = document.getElementById('selectOutputPath');
const clearCompletedBtn = document.getElementById('clearCompleted');
const clearConsoleBtn = document.getElementById('clearConsole');

let currentTaskId = null;

async function loadSettings() {
  try {
    const response = await fetch('/api/settings');
    if (response.ok) {
      const settings = await response.json();
      if (settings.outputPath) {
        outputPathInput.value = settings.outputPath;
      }
    }
  } catch (error) {
    console.error('加载设置失败:', error);
  }
}

loadSettings();

socket.on('queue-update', (queue) => {
  renderTasks(queue);
});

socket.on('task-progress', (task) => {
  updateTaskProgress(task);
});

socket.on('log', (log) => {
  addLog(log);
});

socket.on('task-start', (task) => {
  addLog({
    message: `[任务开始] ${task.originalName}`,
    type: 'info',
    timestamp: new Date().toLocaleTimeString('en-US', { hour12: false })
  });
});

socket.on('task-complete', (task) => {
  addLog({
    message: `[任务完成] ${task.originalName}`,
    type: 'success',
    timestamp: new Date().toLocaleTimeString('en-US', { hour12: false })
  });
});

socket.on('task-error', (task) => {
  addLog({
    message: `[任务失败] ${task.originalName}: ${task.error}`,
    type: 'error',
    timestamp: new Date().toLocaleTimeString('en-US', { hour12: false })
  });
});

function renderTasks(queue) {
  if (queue.length === 0) {
    taskList.innerHTML = '<div class="empty-state">暂无任务</div>';
    return;
  }

  taskList.innerHTML = queue.map(task => {
    const statusIcon = getStatusIcon(task.status);
    const statusClass = task.status;
    const progressBar = `
      <div class="task-progress">
        <div class="progress-bar">
          <div class="progress-fill" style="width: ${task.progress}%">
            ${Math.round(task.progress)}%
          </div>
        </div>
        ${task.speed > 0 ? `<div class="task-speed"><span>速度: ${task.speed} 段/秒</span></div>` : ''}
      </div>
    `;

    const controlButtons = `
      <button onclick="removeTask('${task.id}')" style="padding: 2px 8px; font-size: 10px; border-color: #f00; color: #f00;">删除</button>
    `;

    return `
      <div class="task-item ${statusClass}" data-id="${task.id}">
        <div class="task-header">
          <span class="task-name">${statusIcon} ${task.originalName}</span>
          <span class="task-status ${statusClass}">${getStatusText(task.status)}</span>
        </div>
        ${progressBar}
        <div class="task-speed">
          <span>${task.inputPath}</span>
          ${controlButtons}
        </div>
      </div>
    `;
  }).join('');
}

function updateTaskProgress(task) {
  const taskElement = document.querySelector(`[data-id="${task.id}"]`);
  if (!taskElement) return;

  const progressFill = taskElement.querySelector('.progress-fill');
  const speedSpan = taskElement.querySelector('.task-speed span');
  
  if (progressFill) {
    progressFill.style.width = `${task.progress}%`;
    progressFill.textContent = `${Math.round(task.progress)}%`;
  }
  
  if (speedSpan && task.speed > 0) {
    speedSpan.textContent = `速度: ${task.speed} 段/秒`;
  }
}

function getStatusIcon(status) {
  const icons = {
    pending: '⏳',
    processing: '🔄',
    completed: '✅',
    error: '❌'
  };
  return icons[status] || '📄';
}

function getStatusText(status) {
  const texts = {
    pending: '等待中',
    processing: '翻译中',
    completed: '已完成',
    error: '失败'
  };
  return texts[status] || status;
}

function addLog(log) {
  const entry = document.createElement('div');
  entry.className = `log-entry ${log.type}`;
  entry.innerHTML = `<span class="timestamp">[${log.timestamp}]</span>${log.message}`;
  consoleOutput.appendChild(entry);
  consoleOutput.scrollTop = consoleOutput.scrollHeight;
  
  if (log.type === 'error') {
    console.error(log.message);
  } else {
    console.log(log.message);
  }
}

uploadZone.addEventListener('click', () => {
  fileInput.click();
});

uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', async (e) => {
  e.preventDefault();
  uploadZone.classList.remove('dragover');
  
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    await uploadFiles(Array.from(files));
  }
});

fileInput.addEventListener('change', async (e) => {
  const files = Array.from(e.target.files);
  if (files.length > 0) {
    await uploadFiles(files);
  }
  fileInput.value = '';
});

async function uploadFiles(files) {
  for (const file of files) {
    if (!file.name.match(/\.(md|markdown)$/i)) {
      addLog({
        message: `[错误] 不支持的文件类型: ${file.name}`,
        type: 'error',
        timestamp: new Date().toLocaleTimeString('en-US', { hour12: false })
      });
      continue;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      addLog({
        message: `[上传] ${file.name}`,
        type: 'info',
        timestamp: new Date().toLocaleTimeString('en-US', { hour12: false })
      });

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      
      if (response.ok) {
        await createTask(data);
      } else {
        throw new Error(data.error || '上传失败');
      }
    } catch (error) {
      addLog({
        message: `[错误] ${file.name}: ${error.message}`,
        type: 'error',
        timestamp: new Date().toLocaleTimeString('en-US', { hour12: false })
      });
    }
  }
}

async function createTask(fileData) {
  try {
    const response = await fetch('/api/task', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        filename: fileData.filename,
        originalName: fileData.originalName
      })
    });

    const task = await response.json();
    
    if (!response.ok) {
      throw new Error(task.error || '创建任务失败');
    }
    
    addLog({
      message: `[任务已添加] ${task.originalName}`,
      type: 'success',
      timestamp: new Date().toLocaleTimeString('en-US', { hour12: false })
    });
  } catch (error) {
    addLog({
      message: `[错误] 创建任务失败: ${error.message}`,
      type: 'error',
      timestamp: new Date().toLocaleTimeString('en-US', { hour12: false })
    });
  }
}

async function removeTask(taskId) {
  if (!confirm('确定要删除这个任务吗？')) return;
  
  try {
    const response = await fetch(`/api/queue/${taskId}`, {
      method: 'DELETE'
    });
    const data = await response.json();
    
    if (data.success) {
      const statusText = data.status ? `(${getStatusText(data.status)})` : '';
      addLog({
        message: `[删除] 任务已删除 ${statusText}`,
        type: 'warning',
        timestamp: new Date().toLocaleTimeString('en-US', { hour12: false })
      });
    }
  } catch (error) {
    console.error('删除任务失败:', error);
  }
}

selectOutputPathBtn.addEventListener('click', async () => {
  const path = outputPathInput.value.trim();
  if (!path) {
    alert('请输入输出路径');
    return;
  }

  try {
    const response = await fetch('/api/output-path', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ path })
    });

    const data = await response.json();
    
    if (response.ok) {
      addLog({
        message: `[设置] 输出路径已更新: ${data.path}`,
        type: 'success',
        timestamp: new Date().toLocaleTimeString('en-US', { hour12: false })
      });
    } else {
      throw new Error(data.error || '设置失败');
    }
  } catch (error) {
    alert(`设置输出路径失败: ${error.message}`);
  }
});

clearCompletedBtn.addEventListener('click', async () => {
  try {
    const response = await fetch('/api/queue', {
      method: 'DELETE'
    });
    
    if (response.ok) {
      addLog({
        message: `[清理] 已清除所有已完成任务`,
        type: 'info',
        timestamp: new Date().toLocaleTimeString('en-US', { hour12: false })
      });
    }
  } catch (error) {
    console.error('清除任务失败:', error);
  }
});

clearConsoleBtn.addEventListener('click', () => {
  consoleOutput.innerHTML = '';
});

addLog({
  message: '[系统] mtrans Web 界面已启动',
  type: 'info',
  timestamp: new Date().toLocaleTimeString('en-US', { hour12: false })
});
