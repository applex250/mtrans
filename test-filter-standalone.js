import { parseMarkdown } from './src/parsers/markdownParser.js';
import { filterSpecialElements } from './src/filters/elementFilter.js';
import { restoreSpecialElements } from './src/filters/elementFilter.js';
import fs from 'fs';

function testFilter(inputFile) {
  console.log('=== 过滤程序独立测试 ===\n');
  
  console.log('步骤 1: 读取测试文件...');
  const lines = fs.readFileSync(inputFile, 'utf-8').split('\n');
  console.log(`  - 文件: ${inputFile}`);
  console.log(`  - 总行数: ${lines.length}\n`);
  
  console.log('步骤 2: 解析 Markdown 结构...');
  const structure = parseMarkdown(inputFile);
  console.log(`  - 标题数: ${structure.headings.length}`);
  console.log(`  - 表格: ${structure.specialElements.tables.length}`);
  console.log(`  - 代码块: ${structure.specialElements.codeBlocks.length}`);
  console.log(`  - 图片: ${structure.specialElements.images.length}\n`);
  
  console.log('步骤 3: 执行过滤...');
  const filterResult = filterSpecialElements(lines, structure);
  console.log(`  - filteredLines 长度: ${filterResult.filteredLines.length}`);
  console.log(`  - lineMapping 大小: ${filterResult.lineMapping.size}`);
  console.log(`  - elementMap 大小: ${filterResult.elementMap.size}`);
  console.log(`  - 表格过滤: ${filterResult.stats.tablesFiltered} 个`);
  console.log(`  - 代码块过滤: ${filterResult.stats.codeBlocksFiltered} 个`);
  console.log(`  - 图片过滤: ${filterResult.stats.imagesFiltered} 个`);
  console.log(`  - 节省行数: ${filterResult.stats.totalLinesSaved} 行\n`);
  
  console.log('步骤 4: 输出详细信息...');
  
  console.log('\n--- elementMap 详细信息 ---');
  for (const [placeholder, content] of filterResult.elementMap) {
    const linesCount = content.split('\n').length;
    console.log(`\n${placeholder}`);
    console.log(`  原始内容行数: ${linesCount}`);
    console.log(`  内容预览: ${content.substring(0, 50)}...`);
  }
  
  console.log('\n--- lineMapping 详细信息（前30个） ---');
  let count = 0;
  for (const [filteredLine, originalLine] of filterResult.lineMapping) {
    console.log(`  过滤后行 ${filteredLine} -> 原始行 ${originalLine}`);
    count++;
    if (count >= 30) {
      console.log(`  ...（共 ${filterResult.lineMapping.size} 个映射）`);
      break;
    }
  }
  
  console.log('\n--- filteredLines 预览（前50行） ---');
  for (let i = 0; i < Math.min(50, filterResult.filteredLines.length); i++) {
    const line = filterResult.filteredLines[i];
    let prefix = '';
    if (line.startsWith('<!-- TABLE:')) prefix = '[TABLE] ';
    else if (line.startsWith('<!-- IMAGE:')) prefix = '[IMAGE] ';
    else if (line.startsWith('<!-- CODE:')) prefix = '[CODE] ';
    console.log(`${(i + 1).toString().padStart(3, '0')}: ${prefix}${line.substring(0, 70)}`);
  }
  
  console.log('\n\n步骤 5: 测试还原...');
  const mockTranslatedContent = filterResult.filteredLines.slice(0, 20).join('\n');
  console.log('  - 模拟翻译内容（前20行）...\n');
  
  const restoredContent = restoreSpecialElements(mockTranslatedContent, filterResult.elementMap);
  console.log('  - 还原内容（前20行）:\n');
  const restoredLines = restoredContent.split('\n').slice(0, 20);
  restoredLines.forEach((line, idx) => {
    console.log(`${(idx + 1).toString().padStart(3, '0')}: ${line.substring(0, 70)}`);
  });
  
  console.log('\n\n步骤 6: 保存测试结果...');
  
  const filteredOutputFile = 'test-filtered.md';
  fs.writeFileSync(filteredOutputFile, filterResult.filteredLines.join('\n'), 'utf-8');
  console.log(`  - 过滤后内容已保存到: ${filteredOutputFile}`);
  
  const filteredOutputFileTxt = 'test-filtered.txt';
  fs.writeFileSync(filteredOutputFileTxt, filterResult.filteredLines.join('\n'), 'utf-8');
  console.log(`  - 过滤后内容(TXT)已保存到: ${filteredOutputFileTxt}`);
  
  const elementMapOutputFile = 'test-element-map.json';
  const elementMapObj = {};
  for (const [key, value] of filterResult.elementMap) {
    elementMapObj[key] = value;
  }
  fs.writeFileSync(elementMapOutputFile, JSON.stringify(elementMapObj, null, 2), 'utf-8');
  console.log(`  - 元素映射已保存到: ${elementMapOutputFile}`);
  
  const lineMappingOutputFile = 'test-line-mapping.json';
  const lineMappingObj = {};
  for (const [key, value] of filterResult.lineMapping) {
    lineMappingObj[key] = value;
  }
  fs.writeFileSync(lineMappingOutputFile, JSON.stringify(lineMappingObj, null, 2), 'utf-8');
  console.log(`  - 行号映射已保存到: ${lineMappingOutputFile}`);
  
  const reportOutputFile = 'test-report.txt';
  const reportContent = `
=== 过滤程序测试报告 ===

测试文件: ${inputFile}

=== 解析结果 ===
- 标题数: ${structure.headings.length}
- 表格: ${structure.specialElements.tables.length}
- 代码块: ${structure.specialElements.codeBlocks.length}
- 图片: ${structure.specialElements.images.length}

=== 过滤结果 ===
- filteredLines 长度: ${filterResult.filteredLines.length}
- lineMapping 大小: ${filterResult.lineMapping.size}
- elementMap 大小: ${filterResult.elementMap.size}
- 表格过滤: ${filterResult.stats.tablesFiltered} 个
- 代码块过滤: ${filterResult.stats.codeBlocksFiltered} 个
- 图片过滤: ${filterResult.stats.imagesFiltered} 个
- 节省行数: ${filterResult.stats.totalLinesSaved} 行

=== 问题诊断 ===
${diagnoseProblems(filterResult, structure)}
  `;
  fs.writeFileSync(reportOutputFile, reportContent.trim(), 'utf-8');
  console.log(`  - 完整报告已保存到: ${reportOutputFile}\n`);
  
  console.log('=== 测试完成 ===\n');
  return filterResult;
}

function diagnoseProblems(filterResult, structure) {
  const problems = [];
  
  if (filterResult.lineMapping.size !== filterResult.filteredLines.length) {
    problems.push(`❌ lineMapping 不完整: ${filterResult.lineMapping.size} < ${filterResult.filteredLines.length}`);
  } else {
    problems.push(`✅ lineMapping 完整: ${filterResult.lineMapping.size} = ${filterResult.filteredLines.length}`);
  }
  
  const totalFiltered = filterResult.stats.tablesFiltered + 
                    filterResult.stats.imagesFiltered + 
                    filterResult.stats.codeBlocksFiltered;
  if (filterResult.elementMap.size !== totalFiltered) {
    problems.push(`❌ elementMap 大小不一致: ${filterResult.elementMap.size} ≠ ${totalFiltered}`);
  } else {
    problems.push(`✅ elementMap 大小一致: ${filterResult.elementMap.size} = ${totalFiltered}`);
  }
  
  const tableRowsSaved = structure.specialElements.tables.reduce((sum, t) => {
    return sum + (t.endLine - t.startLine);
  }, 0);
  const codeRowsSaved = structure.specialElements.codeBlocks.reduce((sum, c) => {
    return sum + (c.endLine - c.startLine);
  }, 0);
  const expectedSaved = tableRowsSaved + codeRowsSaved;
  const actualSaved = filterResult.stats.totalLinesSaved;
  const diff = Math.abs(expectedSaved - actualSaved);
  
  if (diff > 10) {
    problems.push(`⚠️  节省行数差异较大: 预期 ${expectedSaved}，实际 ${actualSaved}（差异 ${diff}）`);
  } else {
    problems.push(`✅ 节省行数合理: ${actualSaved}`);
  }
  
  return problems.join('\n');
}

const inputFile = 'input/sample.md';
testFilter(inputFile);
