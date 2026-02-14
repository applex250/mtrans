import { chatWithRetry } from '../llm/zhipuClient.js';

const DIRECTION_PROMPT = (abstract) => {
  return `分析以下abstract，用一句话（中文）概括专业方向，50字以内。
Abstract: ${abstract}`;
};

function filterValidHeadings(headings, referencesSection) {
  const excludedKeywords = ['references', 'bibliography', '参考文献'];
  
  return headings.filter(h => {
    const lowerText = h.text.toLowerCase();
    return !excludedKeywords.includes(lowerText) &&
           h.line < referencesSection.startLine;
  });
}

function calculateHeadingSegments(structure, validHeadings) {
  const segments = [];
  
  validHeadings.forEach((heading, index) => {
    const nextHeading = validHeadings[index + 1];
    const endLine = nextHeading ? nextHeading.line - 1 : structure.totalLines;
    
    let charCount = 0;
    for (const para of structure.paragraphs) {
      if (para.endLine < heading.line) continue;
      if (para.startLine > endLine) break;
      charCount += para.wordCount;
    }
    
    segments.push({
      heading,
      startLine: heading.line,
      endLine,
      charCount,
      level: heading.level,
      text: heading.text
    });
  });
  
  return segments;
}

function findBestSplitPoint(segments, targetChars, currentChars) {
  let bestIndex = segments.length - 1;
  let bestDiff = Infinity;
  let runningTotal = 0;
  
  for (let i = segments.length - 1; i >= 0; i--) {
    runningTotal += segments[i].charCount;
    const diff = Math.abs(runningTotal - targetChars);
    
    if (diff < bestDiff) {
      bestDiff = diff;
      bestIndex = i;
    } else {
      break;
    }
  }
  
  return bestIndex;
}

function createSegmentsByCharCount(headingSegments, targetChars) {
  const segments = [];
  let currentSegmentStart = headingSegments[0]?.startLine || 1;
  let currentChars = 0;
  
  for (let i = 0; i < headingSegments.length; i++) {
    const seg = headingSegments[i];
    
    if (seg.charCount > targetChars) {
      if (currentChars > 0) {
        const segmentEnd = headingSegments[i - 1].endLine;
        segments.push([currentSegmentStart, segmentEnd]);
        currentSegmentStart = seg.startLine;
        currentChars = 0;
      }
      segments.push([seg.startLine, seg.endLine]);
      currentSegmentStart = headingSegments[i + 1]?.startLine || seg.endLine + 1;
      continue;
    }
    
    currentChars += seg.charCount;
    
    if (currentChars >= targetChars) {
      const bestSplitIndex = findBestSplitPoint(
        headingSegments.slice(0, i + 1),
        targetChars,
        currentChars
      );
      
      const splitSeg = headingSegments[bestSplitIndex];
      segments.push([currentSegmentStart, splitSeg.endLine]);
      
      currentSegmentStart = headingSegments[bestSplitIndex + 1]?.startLine || splitSeg.endLine + 1;
      currentChars = headingSegments.slice(bestSplitIndex + 1, i + 1)
        .reduce((sum, s) => sum + s.charCount, 0);
    }
  }
  
  if (currentChars > 0 || segments.length === 0) {
    const lastSegment = headingSegments[headingSegments.length - 1];
    const lastEndLine = lastSegment ? lastSegment.endLine : 0;
    if (lastEndLine >= currentSegmentStart) {
      segments.push([currentSegmentStart, lastEndLine]);
    }
  }
  
  return segments;
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
    reason: '备用分段方案（无标题文档，每100行一段）'
  };
}

export async function extractDirection(structure) {
  if (!structure.abstract || structure.abstract.trim().length === 0) {
    return '学术文献';
  }
  const direction = await chatWithRetry(DIRECTION_PROMPT(structure.abstract), 3, 'direction');
  return direction.trim();
}

export async function designSegments(structure, direction) {
  const validHeadings = filterValidHeadings(structure.headings, structure.referencesSection);
  
  if (validHeadings.length === 0) {
    return fallbackSegment(structure);
  }
  
  const headingSegments = calculateHeadingSegments(structure, validHeadings);
  
  const segments = createSegmentsByCharCount(headingSegments, 10000);
  
  return {
    segments,
    reason: `程序化分段（每段约10000字符，共${segments.length}段）`
  };
}
