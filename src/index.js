import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import { parseMarkdown } from './parsers/markdownParser.js';
import { extractDirection, designSegments } from './translator/segmentDesigner.js';
import { Translator } from './translator/translator.js';

dotenv.config();

const MAX_CONCURRENCY = parseInt(process.env.MAX_CONCURRENCY || '5', 10);

async function translatePaper(inputFile, outputFile) {
  console.log('='.repeat(60));
  console.log('Markdown 学术文献翻译工具');
  console.log('='.repeat(60));
  console.log();

  console.log(`输入文件: ${inputFile}`);
  console.log(`输出文件: ${outputFile}`);
  console.log(`并发数: ${MAX_CONCURRENCY}`);
  console.log();

  try {
    console.log('步骤 1/5: 解析 Markdown...');
    const structure = parseMarkdown(inputFile);
    console.log(`  - 总行数: ${structure.totalLines}`);
    console.log(`  - 标题数: ${structure.headings.length}`);
    console.log(`  - 段落数: ${structure.paragraphs.length}`);
    console.log(`  - Abstract长度: ${structure.abstract.length} 字符`);
    console.log();

    console.log('步骤 2/5: 提取专业方向...');
    const direction = await extractDirection(structure);
    console.log(`  专业方向: ${direction}`);
    console.log();

    console.log('步骤 3/5: 设计分段方案...');
    const { segments, reason } = await designSegments(structure, direction);
    console.log(`  分段数: ${segments.length}`);
    console.log(`  分段理由: ${reason}`);
    console.log('  分段详情:');
    segments.forEach((seg, index) => {
      const [start, end] = seg;
      const lines = end - start + 1;
      console.log(`    分段 ${index + 1}: 行 ${start} - ${end} (共 ${lines} 行)`);
    });
    console.log();

    console.log('步骤 4/5: 开始翻译...');
    const content = fs.readFileSync(inputFile, 'utf-8');
    const lines = content.split('\n');

    const translator = new Translator(MAX_CONCURRENCY, outputFile);
    await translator.translateAll(segments, lines, direction);
    console.log();

    console.log('步骤 5/5: 处理参考文献...');
    if (structure.referencesSection.excluded && structure.referencesSection.content) {
      console.log(`  参考文献位置: 行${structure.referencesSection.startLine}-${structure.referencesSection.endLine}`);
      translator.appendReferences(structure.referencesSection.content);
      console.log('  参考文献已追加到文件末尾');
    } else {
      console.log('  未检测到参考文献章节');
    }
    console.log(`  翻译完成，已保存到: ${outputFile}`);
    console.log();

    console.log('='.repeat(60));
    console.log('翻译完成！');
    console.log('='.repeat(60));

    return outputFile;
  } catch (error) {
    console.error();
    console.error('错误:', error.message);
    console.error();
    process.exit(1);
  }
}

function main() {
  const args = process.argv.slice(2);
  
  if (args.length === 0) {
    console.log('用法: node src/index.js <input.md> [output.md]');
    console.log('示例: node src/index.js input/paper.md');
    console.log('      node src/index.js input/paper.md output/translated.md');
    process.exit(1);
  }

  const inputFile = args[0];
  
  if (!fs.existsSync(inputFile)) {
    console.error(`错误: 输入文件不存在: ${inputFile}`);
    process.exit(1);
  }

  const inputBasename = path.basename(inputFile, path.extname(inputFile));
  const outputFile = args[1] || path.join('output', `${inputBasename}_translated.md`);

  const outputDir = path.dirname(outputFile);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  translatePaper(inputFile, outputFile);
}

main();
