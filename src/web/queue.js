import { EventEmitter } from 'events';
import fs from 'fs';

export class TaskQueue extends EventEmitter {
  constructor() {
    super();
    this.queue = [];
    this.isProcessing = false;
    this.currentTask = null;
    this.outputPath = 'output';
  }

  addTask(task) {
    const taskWithId = {
      id: Date.now() + Math.random().toString(36).substr(2, 9),
      status: 'pending',
      progress: 0,
      speed: 0,
      createdAt: new Date(),
      shouldStop: false,
      abortController: new AbortController(),
      ...task
    };
    
    this.queue.push(taskWithId);
    this.emit('task-added', taskWithId);
    
    if (!this.isProcessing) {
      this.processNext();
    }
    
    return taskWithId;
  }

  setOutputPath(path) {
    this.outputPath = path;
  }

  deleteInputFile(task) {
    if (task && task.inputPath) {
      try {
        if (fs.existsSync(task.inputPath)) {
          fs.unlinkSync(task.inputPath);
        }
      } catch (error) {
        console.warn(`无法删除输入文件 ${task.inputPath}:`, error.message);
      }
    }
  }

  updateTaskStatus(taskId, status, data = {}) {
    const task = this.queue.find(t => t.id === taskId);
    if (task) {
      task.status = status;
      Object.assign(task, data);
      this.emit('task-updated', task);
    }
  }

  updateTaskProgress(taskId, progress, speed = 0) {
    const task = this.queue.find(t => t.id === taskId);
    if (task) {
      task.progress = progress;
      task.speed = speed;
      this.emit('task-progress', task);
    }
  }

  async processNext() {
    if (this.isProcessing) return;
    
    const pendingTask = this.queue.find(t => t.status === 'pending');
    if (!pendingTask) {
      this.isProcessing = false;
      this.currentTask = null;
      return;
    }

    this.isProcessing = true;
    this.currentTask = pendingTask;
    
    try {
      this.updateTaskStatus(pendingTask.id, 'processing');
      this.emit('task-start', pendingTask);
      
      await pendingTask.execute(pendingTask);
      
      this.updateTaskStatus(pendingTask.id, 'completed', {
        completedAt: new Date()
      });
      this.deleteInputFile(pendingTask);
      this.emit('task-complete', pendingTask);
      
    } catch (error) {
      this.updateTaskStatus(pendingTask.id, 'error', {
        error: error.message,
        completedAt: new Date()
      });
      this.deleteInputFile(pendingTask);
      this.emit('task-error', pendingTask);
    }
    
    this.isProcessing = false;
    this.currentTask = null;
    this.processNext();
  }

  getQueue() {
    return this.queue;
  }

  getTask(taskId) {
    return this.queue.find(t => t.id === taskId);
  }

  removeTask(taskId) {
    const index = this.queue.findIndex(t => t.id === taskId);
    if (index !== -1) {
      const task = this.queue[index];
      
      if (task.status === 'processing') {
        console.log(`[停止任务] ${task.originalName} (${task.id}) - 任务状态: ${task.status}`);
        console.log(`[停止任务] ${task.originalName} - 设置 shouldStop = true`);
        console.log(`[停止任务] ${task.originalName} - 调用 abortController.abort()`);
        
        task.shouldStop = true;
        task.abortController.abort();
        
        console.log(`[停止任务] ${task.originalName} - 中断信号已发送`);
        console.log(`[停止任务] ${task.originalName} - abortSignal.aborted: ${task.abortController.signal.aborted}`);
      }
      
      console.log(`[删除任务] ${task.originalName} (${task.id}) - 删除输入文件: ${task.inputPath}`);
      this.deleteInputFile(task);
      
      console.log(`[删除任务] ${task.originalName} (${task.id}) - 从队列中移除`);
      this.queue.splice(index, 1);
      this.emit('task-removed', task);
      
      if (!this.isProcessing) {
        console.log(`[队列] 开始处理下一个任务`);
        this.processNext();
      }
      
      return true;
    }
    
    console.log(`[删除任务] 任务 ID ${taskId} 不存在于队列中`);
    return false;
  }

  clearCompleted() {
    const completed = this.queue.filter(t => t.status === 'completed' || t.status === 'error');
    completed.forEach(task => {
      this.removeTask(task.id);
    });
    this.emit('queue-cleared');
  }

  clearPaused() {
    const pausedTasks = this.queue.filter(t => t.status === 'paused');
    if (pausedTasks.length > 0) {
      console.log(`清理 ${pausedTasks.length} 个暂停的任务...`);
      pausedTasks.forEach(task => {
        console.log(`  - ${task.originalName}`);
        this.removeTask(task.id);
      });
      this.emit('queue-cleared');
    }
  }
}

export const taskQueue = new TaskQueue();
