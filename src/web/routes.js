import express from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { taskQueue } from './queue.js';
import { loadSettings, saveSettings, updateSetting } from './settings.js';
import { parseMarkdown, parseFilteredMarkdown } from '../parsers/markdownParser.js';
import { extractDirection, designSegments } from '../translator/segmentDesigner.js';
import { Translator } from '../translator/translator.js';
import { filterSpecialElements, saveFilteredContent } from '../filters/elementFilter.js';

const router = express.Router();

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = 'input';
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueName = Date.now() + '-' + path.basename(file.originalname);
    cb(null, uniqueName);
  }
});

const upload = multer({ storage });

router.post('/upload', upload.single('file'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }
  res.json({ 
    filename: req.file.filename,
    originalName: req.file.originalname,
    path: req.file.path 
  });
});

router.post('/output-path', (req, res) => {
  const { path: outputPath } = req.body;
  if (!outputPath) {
    return res.status(400).json({ error: 'Output path is required' });
  }
  
  if (!fs.existsSync(outputPath)) {
    fs.mkdirSync(outputPath, { recursive: true });
  }
  
  taskQueue.setOutputPath(outputPath);
  updateSetting('outputPath', outputPath);
  res.json({ success: true, path: outputPath });
});

router.get('/settings', (req, res) => {
  const settings = loadSettings();
  res.json(settings);
});

router.get('/queue', (req, res) => {
  const queue = taskQueue.getQueue();
  res.json(queue);
});

router.post('/task', async (req, res) => {
  const { filename, originalName } = req.body;
  const io = req.app.get('io');
  
  if (!filename) {
    return res.status(400).json({ error: 'Filename is required' });
  }
  
  const inputPath = path.join('input', filename);
  const outputPath = path.join(taskQueue.outputPath, `${path.parse(filename).name}_translated.md`);
  
  const task = taskQueue.addTask({
    filename,
    originalName: originalName || filename,
    inputPath,
    outputPath,
    execute: async (task) => {
      const logger = (message, type = 'info') => {
        const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
        io.emit('log', { taskId: task.id, message, type, timestamp });
      };
      
      logger(`开始翻译: ${task.originalName}`);
      logger(`输入文件: ${task.inputPath}`);
      logger(`输出文件: ${task.outputPath}`);
      
      try {
        logger('步骤 1/6: 解析原始 Markdown...');
        const originalStructure = parseMarkdown(task.inputPath);
        logger(`  - 原始总行数: ${originalStructure.totalLines}`);
        logger(`  - 标题数: ${originalStructure.headings.length}`);

        logger('步骤 2/6: 过滤特殊元素并保存中间文件...');
        const content = fs.readFileSync(task.inputPath, 'utf-8');
        const lines = content.split('\n');
        const { filteredLines, elementMap, lineMapping, stats } = filterSpecialElements(lines, originalStructure);
        logger(`  - 过滤前: ${lines.length} 行`);
        logger(`  - 过滤后: ${filteredLines.length} 行`);
        logger(`  - 表格: ${stats.tablesFiltered} 个`);
        logger(`  - 图片: ${stats.imagesFiltered} 个`);
        logger(`  - 代码块: ${stats.codeBlocksFiltered} 个`);
        
        const { filteredFile, elementMapFile, lineMappingFile, statsFile } = saveFilteredContent(
          filteredLines, 
          elementMap, 
          lineMapping, 
          stats, 
          task.inputPath
        );
        logger(`  - 过滤文件: ${filteredFile}`);
        logger(`  - 元素映射: ${elementMapFile}`);

        logger('步骤 3/6: 解析过滤后的 Markdown...');
        const filteredStructure = parseFilteredMarkdown(filteredLines);
        logger(`  - 过滤后总行数: ${filteredStructure.totalLines}`);
        logger(`  - 标题数: ${filteredStructure.headings.length}`);
        logger(`  - 段落数: ${filteredStructure.paragraphs.length}`);
        logger(`  - Abstract长度: ${filteredStructure.abstract.length} 字符`);

        logger('步骤 4/6: 提取专业方向...');
        const direction = await extractDirection(filteredStructure);
        logger(`  专业方向: ${direction}`);

        logger('步骤 5/6: 设计分段方案...');
        const { segments, reason } = await designSegments(filteredStructure, direction);
        logger(`  分段数: ${segments.length}`);
        logger(`  分段理由: ${reason}`);
        logger('  分段详情:');
        segments.forEach((seg, index) => {
          const [start, end] = seg;
          const segmentLines = end - start + 1;
          logger(`    分段 ${index + 1}: 行 ${start} - ${end} (共 ${segmentLines} 行)`);
        });

        logger('步骤 6/6: 翻译并还原...');
        const MAX_CONCURRENCY = parseInt(process.env.MAX_CONCURRENCY || '2', 10);
        const translator = new Translator(MAX_CONCURRENCY, task.outputPath);

        translator.onProgress = (current, total, speed) => {
          const progress = (current / total) * 100;
          taskQueue.updateTaskProgress(task.id, progress, speed);
          logger(`翻译进度 |${'█'.repeat(Math.floor(progress / 10))}${'░'.repeat(10 - Math.floor(progress / 10))}| ${Math.floor(progress)}% | ${current}/${total} 段 | 速度: ${speed} 段/秒`);
        };

        translator.onLog = (message) => {
          logger(message);
        };

        await translator.translateAll(segments, filteredLines, direction, elementMap);

        if (originalStructure.referencesSection.excluded && originalStructure.referencesSection.content) {
          logger(`  - 追加参考文献...`);
          translator.appendReferences(originalStructure.referencesSection.content);
        }

        logger(`翻译完成，已保存到: ${task.outputPath}`);

        logger('清理中间文件...');
        const tempFiles = [filteredFile, elementMapFile, lineMappingFile, statsFile];
        let deletedCount = 0;
        for (const tempFile of tempFiles) {
          try {
            if (fs.existsSync(tempFile)) {
              fs.unlinkSync(tempFile);
              deletedCount++;
            }
          } catch (err) {
            logger(`  警告: 无法删除文件 ${tempFile}`, 'warn');
          }
        }
        logger(`  已删除 ${deletedCount} 个中间文件`);
        
      } catch (error) {
        logger(`错误: ${error.message}`, 'error');
        throw error;
      }
    }
  });
  
  res.json(task);
});

router.post('/queue/pause/:id', (req, res) => {
  const { id } = req.params;
  const success = taskQueue.pauseTask(id);
  res.json({ success });
});

router.post('/queue/resume/:id', (req, res) => {
  const { id } = req.params;
  const success = taskQueue.resumeTask(id);
  res.json({ success });
});

router.delete('/queue/:id', (req, res) => {
  const { id } = req.params;
  const success = taskQueue.removeTask(id);
  res.json({ success });
});

router.delete('/queue', (req, res) => {
  taskQueue.clearCompleted();
  res.json({ success: true });
});

export default router;
