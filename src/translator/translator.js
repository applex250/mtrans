import pLimit from 'p-limit';
import cliProgress from 'cli-progress';
import fs from 'fs';
import path from 'path';
import { chatWithRetry } from '../llm/zhipuClient.js';
import { restoreSpecialElements } from '../filters/elementFilter.js';

const TRANSLATE_PROMPT = (direction, content, startLine, endLine) => {
  return `专业方向：${direction}

翻译规则：
0. 翻译为简体中文
1. 保持学术风格，缩写/术语不翻译
2. 表格/图片/公式/代码块保持原样
3. 保持markdown结构
4. 直接输出翻译，无额外文字
5. 段落间用空行分隔

翻译片段（行${startLine}-${endLine}）：
${content}`;
};

export class Translator {
  constructor(maxConcurrency = 5, outputFile) {
    this.limit = pLimit(maxConcurrency);
    this.progressBar = null;
    this.translations = new Map();
    this.outputFile = outputFile;
    this.tempFile = `${outputFile}.temp`;
    this.completedSegments = 0;
    this.onProgress = null;
    this.onLog = null;
  }

  createProgressBar(totalSegments) {
    this.progressBar = new cliProgress.SingleBar({
      format: '翻译进度 |{bar}| {percentage}% | {value}/{total} 段 | 速度: {speed} 段/秒',
      barCompleteChar: '█',
      barIncompleteChar: '░',
      hideCursor: true
    }, cliProgress.Presets.shades_classic);

    this.startTime = Date.now();
    this.progressBar.start(totalSegments, 0);
    
    this.completedSegments = 0;
    
    fs.writeFileSync(this.tempFile, '', 'utf-8');
  }

  updateProgressBar() {
    if (this.progressBar) {
      const elapsed = (Date.now() - this.startTime) / 1000;
      const speed = this.progressBar.value / elapsed;
      this.progressBar.increment(undefined, { speed: speed.toFixed(2) });
      
      if (this.onProgress) {
        const total = this.progressBar.getTotal();
        const current = this.progressBar.value;
        this.onProgress(current, total, speed.toFixed(2));
      }
    }
  }

  stopProgressBar() {
    if (this.progressBar) {
      this.progressBar.stop();
    }
  }

  async translateSegment(segmentIndex, segmentLines, direction, abortSignal = null) {
    const [startLine, endLine] = segmentLines;
    const content = segmentLines.join('\n');
    
    try {
      if (abortSignal && abortSignal.aborted) {
        throw new Error('Translation aborted');
      }
      
      const prompt = TRANSLATE_PROMPT(direction, content, startLine, endLine);
      const translated = await chatWithRetry(prompt, 3, `translate_${segmentIndex}`);
      
      if (abortSignal && abortSignal.aborted) {
        throw new Error('Translation aborted');
      }
      
      this.translations.set(segmentIndex, {
        startLine,
        endLine,
        translated
      });
      
      const segmentData = `<!-- SEGMENT:${segmentIndex} -->\n${translated}\n\n`;
      fs.appendFileSync(this.tempFile, segmentData, 'utf-8');
      
      this.updateProgressBar();
      return translated;
    } catch (error) {
      if (abortSignal && abortSignal.aborted) {
        console.log(`\n分段翻译已取消 (行${startLine}-${endLine})`);
        return null;
      }
      
      console.error(`\n分段翻译失败 (行${startLine}-${endLine}): ${error.message}`);
      this.translations.set(segmentIndex, {
        startLine,
        endLine,
        translated: content,
        error: error.message
      });
      
      const segmentData = `<!-- SEGMENT:${segmentIndex} -->\n${content}\n\n`;
      fs.appendFileSync(this.tempFile, segmentData, 'utf-8');
      
      this.updateProgressBar();
      return content;
    }
  }

  async translateAll(segments, lines, direction, elementMap = null, abortSignal = null) {
    if (segments.length === 0) {
      console.log('  无需要翻译的内容');
      return '';
    }
    
    this.createProgressBar(segments.length);
    
    const tasks = segments.map((segment, index) => {
      return this.limit(() => {
        const startIdx = segment[0] - 1;
        const endIdx = segment[1] - 1;
        const segmentLines = lines.slice(startIdx, endIdx + 1);
        return this.translateSegment(index, segmentLines, direction, abortSignal);
      });
    });

    await Promise.all(tasks);
    
    if (abortSignal && abortSignal.aborted) {
      console.log('\n翻译已取消，停止组装最终文件');
      this.stopProgressBar();
      try {
        if (fs.existsSync(this.tempFile)) {
          fs.unlinkSync(this.tempFile);
        }
      } catch (err) {
        console.warn('警告: 无法删除临时文件', this.tempFile);
      }
      throw new Error('Translation aborted');
    }
    
    this.stopProgressBar();
    
    let finalContent = this.assembleAndWriteFinal(segments);
    
    if (elementMap && elementMap.size > 0) {
      finalContent = this.restoreElements(finalContent, elementMap);
      fs.writeFileSync(this.outputFile, finalContent.trim(), 'utf-8');
    }
    
    return finalContent;
  }

  assembleAndWriteFinal(segments) {
    const tempContent = fs.readFileSync(this.tempFile, 'utf-8');
    const segmentRegex = /<!-- SEGMENT:(\d+) -->\n([\s\S]*?)(?=\n<!-- SEGMENT:|$)/g;
    const segmentMap = new Map();

    let match;
    while ((match = segmentRegex.exec(tempContent)) !== null) {
      const index = parseInt(match[1]);
      const content = match[2].trim();
      segmentMap.set(index, content);
    }

    let finalContent = '';
    for (let i = 0; i < segments.length; i++) {
      if (segmentMap.has(i)) {
        finalContent += segmentMap.get(i) + '\n\n';
      }
    }

    fs.writeFileSync(this.outputFile, finalContent.trim(), 'utf-8');

    try {
      fs.unlinkSync(this.tempFile);
    } catch (err) {
      console.warn('警告: 无法删除临时文件', this.tempFile);
    }

    return finalContent;
  }

  restoreElements(content, elementMap) {
    return restoreSpecialElements(content, elementMap);
  }

  appendReferences(referencesContent) {
    if (!referencesContent || referencesContent.trim().length === 0) {
      return;
    }
    
    const content = fs.readFileSync(this.outputFile, 'utf-8');
    const finalContent = content.trim() + '\n\n' + referencesContent.trim();
    fs.writeFileSync(this.outputFile, finalContent, 'utf-8');
  }
}
