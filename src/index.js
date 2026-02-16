import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import { parseMarkdown, parseFilteredMarkdown } from './parsers/markdownParser.js';
import { extractDirection, designSegments } from './translator/segmentDesigner.js';
import { Translator } from './translator/translator.js';
import { filterSpecialElements, saveFilteredContent } from './filters/elementFilter.js';

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
    console.log('步骤 1/6: 解析原始 Markdown...');
    const originalStructure = parseMarkdown(inputFile);
    console.log(`  - 原始总行数: ${originalStructure.totalLines}`);
    console.log(`  - 标题数: ${originalStructure.headings.length}`);
    console.log();

    console.log('步骤 2/6: 过滤特殊元素并保存中间文件...');
    const content = fs.readFileSync(inputFile, 'utf-8');
    const lines = content.split('\n');
    const { filteredLines, elementMap, lineMapping, stats } = filterSpecialElements(lines, originalStructure);
    console.log(`  - 过滤前: ${lines.length} 行`);
    console.log(`  - 过滤后: ${filteredLines.length} 行`);
    console.log(`  - 表格: ${stats.tablesFiltered} 个`);
    console.log(`  - 图片: ${stats.imagesFiltered} 个`);
    console.log(`  - 代码块: ${stats.codeBlocksFiltered} 个`);
    
    const { filteredFile, elementMapFile, lineMappingFile, statsFile } = saveFilteredContent(
      filteredLines, 
      elementMap, 
      lineMapping, 
      stats, 
      inputFile
    );
    console.log(`  - 过滤文件: ${filteredFile}`);
    console.log(`  - 元素映射: ${elementMapFile}`);
    console.log();

    console.log('步骤 3/6: 解析过滤后的 Markdown...');
    const filteredStructure = parseFilteredMarkdown(filteredLines);
    console.log(`  - 过滤后总行数: ${filteredStructure.totalLines}`);
    console.log(`  - 标题数: ${filteredStructure.headings.length}`);
    console.log(`  - 段落数: ${filteredStructure.paragraphs.length}`);
    console.log(`  - Abstract长度: ${filteredStructure.abstract.length} 字符`);
    console.log();

    console.log('步骤 4/6: 提取专业方向...');
    const direction = await extractDirection(filteredStructure);
    console.log(`  专业方向: ${direction}`);
    console.log();

    console.log('步骤 5/6: 设计分段方案...');
    const { segments, reason } = await designSegments(filteredStructure, direction);
    console.log(`  分段数: ${segments.length}`);
    console.log(`  分段理由: ${reason}`);
    console.log('  分段详情:');
    segments.forEach((seg, index) => {
      const [start, end] = seg;
      const segmentLines = end - start + 1;
      console.log(`    分段 ${index + 1}: 行 ${start} - ${end} (共 ${segmentLines} 行)`);
    });
    console.log();

    console.log('步骤 6/6: 翻译并还原...');
    const translator = new Translator(MAX_CONCURRENCY, outputFile);
    await translator.translateAll(segments, filteredLines, direction, elementMap);
    
    if (originalStructure.referencesSection.excluded && originalStructure.referencesSection.content) {
      console.log(`  - 追加参考文献...`);
      translator.appendReferences(originalStructure.referencesSection.content);
    }
    console.log(`  翻译完成，已保存到: ${outputFile}`);
    console.log();

    console.log('清理中间文件...');
    const tempDir = path.join(process.cwd(), 'temp');
    const tempFiles = [
      filteredFile,
      elementMapFile,
      lineMappingFile,
      statsFile
    ];
    
    let deletedCount = 0;
    for (const tempFile of tempFiles) {
      try {
        if (fs.existsSync(tempFile)) {
          fs.unlinkSync(tempFile);
          deletedCount++;
        }
      } catch (err) {
        console.warn(`  警告: 无法删除文件 ${tempFile}`);
      }
    }
    console.log(`  已删除 ${deletedCount} 个中间文件`);
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
