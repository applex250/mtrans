import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import routes from './routes.js';
import { taskQueue } from './queue.js';
import { loadSettings } from './settings.js';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const settings = loadSettings();
const PORT = process.env.WEB_PORT || settings.webPort || 3000;

const app = express();
const server = createServer(app);
const io = new Server(server);

taskQueue.setOutputPath(settings.outputPath);

app.set('io', io);

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/api', routes);

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

io.on('connection', (socket) => {
  console.log('客户端已连接');
  
  socket.emit('queue-update', taskQueue.getQueue());
  
  socket.on('disconnect', () => {
    console.log('客户端已断开');
  });
});

taskQueue.on('task-added', (task) => {
  io.emit('queue-update', taskQueue.getQueue());
});

taskQueue.on('task-updated', (task) => {
  io.emit('queue-update', taskQueue.getQueue());
});

taskQueue.on('task-progress', (task) => {
  io.emit('task-progress', task);
});

taskQueue.on('task-start', (task) => {
  io.emit('task-start', task);
});

taskQueue.on('task-complete', (task) => {
  io.emit('task-complete', task);
  io.emit('queue-update', taskQueue.getQueue());
});

taskQueue.on('task-error', (task) => {
  io.emit('task-error', task);
  io.emit('queue-update', taskQueue.getQueue());
});

taskQueue.on('task-removed', (task) => {
  io.emit('queue-update', taskQueue.getQueue());
});

taskQueue.on('queue-cleared', () => {
  io.emit('queue-update', taskQueue.getQueue());
});

taskQueue.clearPaused();

server.listen(PORT, () => {
  const MAX_CONCURRENCY = parseInt(process.env.MAX_CONCURRENCY || '5', 10);
  console.log('='.repeat(60));
  console.log('mtrans Web 服务器');
  console.log('='.repeat(60));
  console.log(`服务器运行在 http://localhost:${PORT}`);
  console.log(`并发翻译数: ${MAX_CONCURRENCY}`);
  console.log(`输出目录: ${settings.outputPath}`);
  console.log('='.repeat(60));
  console.log('按 Ctrl+C 停止服务器');
});
