import { chatWithRetry } from '../llm/zhipuClient.js';

const DIRECTION_PROMPT = (abstract) => {
  return `分析以下abstract，用一句话（中文）概括专业方向，50字以内。
Abstract: ${abstract}`;
};

function calculateSectionWordCount(structure, currentHeading, nextHeading) {
  const startLine = currentHeading.line;
  const endLine = nextHeading ? nextHeading.line - 1 : structure.totalLines;
  
  if (structure.referencesSection.excluded) {
    if (startLine >= structure.referencesSection.startLine) {
      return 0;
    }
    if (endLine > structure.referencesSection.startLine) {
      return calculateSectionWordCount(structure, currentHeading, { line: structure.referencesSection.startLine });
    }
  }
  
  let wordCount = 0;
  for (const para of structure.paragraphs) {
    if (para.endLine < startLine) continue;
    if (para.startLine > endLine) break;
    wordCount += para.wordCount;
  }
  
  return wordCount;
}

const SEGMENT_PROMPT = (structure, direction) => {
  const headings = structure.headings
    .filter(h => {
      const lowerText = h.text.toLowerCase();
      return h.level <= 2 && 
             lowerText !== 'references' && 
             lowerText !== 'bibliography' && 
             lowerText !== '参考文献';
    })
    .map((h, idx, arr) => {
      const nextHeading = arr[idx + 1];
      const wordCount = calculateSectionWordCount(structure, h, nextHeading);
      return `${'#'.repeat(h.level)} ${h.text} (行${h.line}, ${wordCount}字)`;
    })
    .join('\n');
  
  let extraInfo = '';
  if (structure.referencesSection.excluded) {
    extraInfo = `\n注意：参考文献部分（行${structure.referencesSection.startLine}-${structure.referencesSection.endLine}）已排除，不参与分段。`;
  }
  
  return `专业方向：${direction}

文档结构：
${headings}${extraInfo}

总行数：${structure.totalLines}

请按照二级标题进行分段（速度优先）：
- 每个二级标题（##）作为一段的开始
- 一段的结束是下一个二级标题之前的一行
- Abstract单独作为一段
- 分段范围不超过总行数（${structure.totalLines}）

请返回JSON格式的分段结果，格式为：
{
  "segments": [[startLine, endLine], ...],
  "reason": "分段理由"
}

示例：
{
  "segments": [[1, 28], [29, 64], [65, 100]],
  "reason": "按照二级标题分段：Abstract、1. Introduction、2. Related Work等"
}`;
};

export async function extractDirection(structure) {
  if (!structure.abstract || structure.abstract.trim().length === 0) {
    return '学术文献';
  }
  const direction = await chatWithRetry(DIRECTION_PROMPT(structure.abstract), 3, 'direction');
  return direction.trim();
}

export async function designSegments(structure, direction) {
  const response = await chatWithRetry(SEGMENT_PROMPT(structure, direction), 3, 'segment');
  
  try {
    const jsonMatch = response.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('No JSON found in response');
    }
    const result = JSON.parse(jsonMatch[0]);
    
    if (!result.segments || !Array.isArray(result.segments)) {
      throw new Error('Invalid segments format');
    }
    
    return {
      segments: result.segments,
      reason: result.reason || 'LLM自动分段'
    };
  } catch (error) {
    console.error(`LLM分段失败，使用备用方案: ${error.message}`);
    return fallbackSegment(structure);
  }
}

function fallbackSegment(structure) {
  let totalLines = structure.totalLines;
  const segmentSize = 100;
  const segments = [];
  
  if (totalLines === 0) {
    return {
      segments: [],
      reason: '文档无内容（可能只有参考文献）'
    };
  }
  
  for (let start = 1; start <= totalLines; start += segmentSize) {
    const end = Math.min(start + segmentSize - 1, totalLines);
    segments.push([start, end]);
  }
  
  return {
    segments,
    reason: '备用分段方案（每100行一段）'
  };
}
