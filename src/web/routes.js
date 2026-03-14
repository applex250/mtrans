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
import {
  isSupportedFileType,
  isMarkdownFile,
  needsMinerUConversion,
  getFileType,
  convertDocument
} from '../services/minerUService.js';

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

  const originalName = req.file.originalname;

  // Check if file type is supported
  if (!isSupportedFileType(originalName)) {
    // Delete uploaded file if not supported
    fs.unlinkSync(req.file.path);
    return res.status(400).json({
      error: 'Unsupported file type',
      supportedTypes: '.md, .markdown, .pdf, .doc, .docx, .ppt, .pptx, .png, .jpg, .jpeg, .gif, .bmp'
    });
  }

  const fileType = getFileType(originalName);
  const needsConversion = needsMinerUConversion(originalName);
  const isMarkdown = isMarkdownFile(originalName);

  res.json({
    filename: req.file.filename,
    originalName: req.file.originalname,
    path: req.file.path,
    fileType,
    needsConversion,
    isMarkdown
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
  const { filename, originalName, needsConversion, fileType } = req.body;
  const io = req.app.get('io');

  if (!filename) {
    return res.status(400).json({ error: 'Filename is required' });
  }

  const inputPath = path.join('input', filename);
  // For files needing conversion, output will be inside converted folder
  // For direct markdown files, use original behavior
  let outputPath;
  if (needsConversion) {
    const convertedDir = path.join(taskQueue.outputPath, `${path.parse(filename).name}_converted`);
    // Ensure the output converted directory exists before translation starts
    if (!fs.existsSync(convertedDir)) {
      fs.mkdirSync(convertedDir, { recursive: true });
    }
    outputPath = path.join(convertedDir, `${path.parse(filename).name}_translated.md`);
  } else {
    outputPath = path.join(taskQueue.outputPath, `${path.parse(filename).name}_translated.md`);
  }

  const task = taskQueue.addTask({
    filename,
    originalName: originalName || filename,
    inputPath,
    outputPath,
    fileType: fileType || 'markdown',
    needsConversion: needsConversion || false,
    execute: async (task) => {
      const isAbortError = (error) => {
        return error.name === 'AbortError' ||
               (error.name === 'DOMException' && error.message.includes('Aborted')) ||
               error.message.includes('Aborted') ||
               error.message.includes('The user aborted a request') ||
               error.message.includes('Task was cancelled') ||
               error.message.includes('user aborted a request');
      };

      const checkTaskStatus = () => {
        if (task.shouldStop || (task.abortController && task.abortController.signal.aborted)) {
          console.log(`[中断检查] ${task.originalName} - 检测到停止信号`);
          const error = new Error('Task stopped');
          error.isTaskStopped = true;
          throw error;
        }
      };

      const logger = (message, type = 'info') => {
        const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
        io.emit('log', { taskId: task.id, message, type, timestamp });
      };

      let tempFiles = [];
      let currentInputPath = task.inputPath;

      const cleanupTempFiles = (message = '') => {
        if (tempFiles.length > 0) {
          if (message) logger(message, 'warning');
          let deletedCount = 0;
          for (const tempFile of tempFiles) {
            try {
              if (fs.existsSync(tempFile)) {
                const stats = fs.statSync(tempFile);
                if (stats.isDirectory()) {
                  // Recursively delete directory
                  fs.rmSync(tempFile, { recursive: true, force: true });
                  console.log(`[清理] 已删除目录: ${tempFile}`);
                  logger(`  - 已删除目录: ${tempFile}`, 'info');
                } else {
                  fs.unlinkSync(tempFile);
                  console.log(`[清理] 已删除: ${tempFile}`);
                  logger(`  - 已删除: ${tempFile}`, 'info');
                }
                deletedCount++;
              }
            } catch (err) {
              const errorMsg = `无法删除 ${tempFile}`;
              console.warn(`[警告] ${errorMsg}:`, err.message);
              logger(`  - ${errorMsg}`, 'warn');
            }
          }
          if (deletedCount > 0) {
            console.log(`[清理] 总计已删除 ${deletedCount} 个临时项目`);
            logger(`  已删除 ${deletedCount} 个临时项目`, 'info');
          }
          tempFiles = [];
        }
      };

      // Calculate total steps based on conversion needs
      const totalSteps = task.needsConversion ? 7 : 6;
      const stepOffset = task.needsConversion ? 1 : 0;

      logger(`开始翻译: ${task.originalName}`);
      logger(`文件类型: ${task.fileType}`);
      logger(`输入文件: ${task.inputPath}`);
      logger(`输出文件: ${task.outputPath}`);
      console.log(`[开始] ${task.originalName} - 开始翻译任务`);

      try {
        checkTaskStatus();

        // Step 0: MinerU conversion if needed
        // Track input converted dir for later cleanup (after copying to output)
        let inputConvertedDir = null;
        if (task.needsConversion) {
          logger(`步骤 1/${totalSteps}: MinerU 文档转换...`);
          console.log(`[步骤 1] ${task.originalName} - MinerU 文档转换`);

          inputConvertedDir = path.join('input', `${path.parse(task.filename).name}_converted`);
          if (!fs.existsSync(inputConvertedDir)) {
            fs.mkdirSync(inputConvertedDir, { recursive: true });
          }

          try {
            const convertedMdPath = await convertDocument(
              task.inputPath,
              inputConvertedDir,
              logger,
              task.abortController.signal
            );

            currentInputPath = convertedMdPath;
            // Don't add to tempFiles yet - we'll copy to output first, then cleanup
            logger(`  转换后的 Markdown: ${convertedMdPath}`);
          } catch (conversionError) {
            logger(`MinerU 转换失败: ${conversionError.message}`, 'error');
            throw new Error(`MinerU conversion failed: ${conversionError.message}`);
          }

          checkTaskStatus();
        }

        logger(`步骤 ${1 + stepOffset}/${totalSteps}: 解析原始 Markdown...`);
        console.log(`[步骤 ${1 + stepOffset}] ${task.originalName} - 解析原始 Markdown`);
        const originalStructure = parseMarkdown(currentInputPath);
        logger(`  - 原始总行数: ${originalStructure.totalLines}`);
        logger(`  - 标题数: ${originalStructure.headings.length}`);

        checkTaskStatus();

        checkTaskStatus();

        logger(`步骤 ${2 + stepOffset}/${totalSteps}: 过滤特殊元素并保存中间文件...`);
        console.log(`[步骤 ${2 + stepOffset}] ${task.originalName} - 过滤特殊元素并保存中间文件`);
        const content = fs.readFileSync(currentInputPath, 'utf-8');
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
          currentInputPath
        );

        tempFiles.push(filteredFile, elementMapFile, lineMappingFile, statsFile);
        console.log(`[临时文件] ${task.originalName} - 已创建临时文件:`);
        tempFiles.forEach(f => console.log(`  - ${f}`));

        logger(`  - 过滤文件: ${filteredFile}`);
        logger(`  - 元素映射: ${elementMapFile}`);

        checkTaskStatus();

        checkTaskStatus();

        logger(`步骤 ${3 + stepOffset}/${totalSteps}: 解析过滤后的 Markdown...`);
        console.log(`[步骤 ${3 + stepOffset}] ${task.originalName} - 解析过滤后的 Markdown`);
        const filteredStructure = parseFilteredMarkdown(filteredLines);
        logger(`  - 过滤后总行数: ${filteredStructure.totalLines}`);
        logger(`  - 标题数: ${filteredStructure.headings.length}`);
        logger(`  - 段落数: ${filteredStructure.paragraphs.length}`);
        logger(`  - Abstract长度: ${filteredStructure.abstract.length} 字符`);

        checkTaskStatus();

        checkTaskStatus();

        logger(`步骤 ${4 + stepOffset}/${totalSteps}: 提取专业方向...`);
        console.log(`[步骤 ${4 + stepOffset}] ${task.originalName} - 提取专业方向`);
        const direction = await extractDirection(filteredStructure);
        logger(`  专业方向: ${direction}`);

        checkTaskStatus();

        checkTaskStatus();

        logger(`步骤 ${5 + stepOffset}/${totalSteps}: 设计分段方案...`);
        console.log(`[步骤 ${5 + stepOffset}] ${task.originalName} - 设计分段方案`);
        const { segments, reason } = await designSegments(filteredStructure, direction);
        logger(`  分段数: ${segments.length}`);
        logger(`  分段理由: ${reason}`);
        logger('  分段详情:');
        segments.forEach((seg, index) => {
          const [start, end] = seg;
          const segmentLines = end - start + 1;
          logger(`    分段 ${index + 1}: 行 ${start} - ${end} (共 ${segmentLines} 行)`);
        });

        checkTaskStatus();

        checkTaskStatus();

        logger(`步骤 ${6 + stepOffset}/${totalSteps}: 翻译并还原...`);
        console.log(`[步骤 ${6 + stepOffset}] ${task.originalName} - 翻译并还原`);
        const MAX_CONCURRENCY = parseInt(process.env.MAX_CONCURRENCY || '2', 10);
        const translator = new Translator(MAX_CONCURRENCY, task.outputPath);

        tempFiles.push(translator.tempFile);
        console.log(`[临时文件] ${task.originalName} - 已添加临时文件: ${translator.tempFile}`);

        translator.onProgress = (current, total, speed) => {
          const progress = (current / total) * 100;
          taskQueue.updateTaskProgress(task.id, progress, speed);
          logger(`翻译进度 |${'█'.repeat(Math.floor(progress / 10))}${'░'.repeat(10 - Math.floor(progress / 10))}| ${Math.floor(progress)}% | ${current}/${total} 段 | 速度: ${speed} 段/秒`);

          checkTaskStatus();
        };

        translator.onLog = (message) => {
          logger(message);
        };

        taskQueue.updateTaskProgress(task.id, 0, 0);

        await translator.translateAll(segments, filteredLines, direction, elementMap, task.abortController.signal);

        checkTaskStatus();

        if (originalStructure.referencesSection.excluded && originalStructure.referencesSection.content) {
          logger(`  - 追加参考文献...`);
          translator.appendReferences(originalStructure.referencesSection.content);
        }

        logger(`翻译完成，已保存到: ${task.outputPath}`);
        console.log(`[完成] ${task.originalName} - 翻译完成，已保存到: ${task.outputPath}`);

        // If MinerU was used, copy the converted folder to output
        if (inputConvertedDir) {
          const outputConvertedDir = path.dirname(task.outputPath);
          logger(`复制 converted 文件夹到输出目录...`);

          const copyDir = (src, dest, skipFile = null) => {
            fs.mkdirSync(dest, { recursive: true });
            const entries = fs.readdirSync(src, { withFileTypes: true });
            for (const entry of entries) {
              const srcPath = path.join(src, entry.name);
              const destPath = path.join(dest, entry.name);
              if (entry.isDirectory()) {
                copyDir(srcPath, destPath);
              } else {
                // Skip the translated output file if it already exists
                if (skipFile && destPath === skipFile) {
                  continue;
                }
                fs.copyFileSync(srcPath, destPath);
              }
            }
          };

          copyDir(inputConvertedDir, outputConvertedDir, task.outputPath);
          logger(`  已复制到: ${outputConvertedDir}`);
          console.log(`[复制] ${task.originalName} - 已复制 converted 文件夹到输出目录`);

          // Now mark input converted dir for cleanup
          tempFiles.push(inputConvertedDir);
        }

        logger('清理中间文件...');
        console.log(`[清理] ${task.originalName} - 清理中间文件`);
        cleanupTempFiles('');

      } catch (error) {
        if (error.isTaskStopped || isAbortError(error)) {
          console.log(`[中断] ${task.originalName} - 任务已停止`);
          logger('[中断] 任务已停止', 'warning');

          console.log(`[清理] ${task.originalName} - 中断时清理临时文件`);
          logger('清理临时文件...', 'warning');
          cleanupTempFiles('');

          throw error;
        }

        console.error(`[错误] ${task.originalName} - ${error.message}`);
        logger(`错误: ${error.message}`, 'error');

        console.log(`[清理] ${task.originalName} - 错误时清理临时文件`);
        logger('清理临时文件...', 'warning');
        cleanupTempFiles('');

        throw error;
      }
    }
  });

  res.json(task);
});

router.delete('/queue/:id', (req, res) => {
  const { id } = req.params;
  const task = taskQueue.getTask(id);
  const status = task ? task.status : null;
  const success = taskQueue.removeTask(id);
  res.json({ success, status });
});

router.delete('/queue', (req, res) => {
  taskQueue.clearCompleted();
  res.json({ success: true });
});

export default router;
