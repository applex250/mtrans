import fs from 'fs';
import path from 'path';

export function parseMarkdown(filePath) {
  const content = fs.readFileSync(filePath, 'utf-8');
  const lines = content.split('\n');

  const structure = {
    abstract: '',
    headings: [],
    paragraphs: [],
    referencesSection: {
      startLine: 0,
      endLine: 0,
      content: '',
      excluded: false
    },
    specialElements: {
      tables: [],
      codeBlocks: [],
      mathBlocks: [],
      images: []
    }
  };

  let inAbstract = false;
  let inReferences = false;
  let currentParagraph = [];
  let paraStartLine = 0;
  let foundAbstractByHeading = false;
  let first10000Chars = '';

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const lineNum = i + 1;

    if (first10000Chars.length < 10000) {
      first10000Chars += line + '\n';
    }

    const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      const text = headingMatch[2].trim();
      structure.headings.push({ level, text, line: lineNum });

      const lowerText = text.toLowerCase();
      if (lowerText === 'abstract' || lowerText === '摘要') {
        inAbstract = true;
        foundAbstractByHeading = true;
      } else if (inAbstract) {
        inAbstract = false;
      }

      if (lowerText === 'references' || lowerText === 'bibliography' || lowerText === '参考文献') {
        inReferences = true;
        structure.referencesSection.startLine = lineNum;
      }
      continue;
    }

    if (inAbstract) {
      structure.abstract += line + '\n';
      continue;
    }

    const htmlTableMatch = line.match(/<table\b/i);
    if (htmlTableMatch) {
      const tableEnd = findHtmlTableEnd(lines, i);
      structure.specialElements.tables.push({
        startLine: lineNum,
        endLine: tableEnd
      });
      i = tableEnd - 1;
      continue;
    }

    const tableMatch = line.match(/^\|.*\|$/);
    if (tableMatch) {
      const tableEnd = findTableEnd(lines, i);
      structure.specialElements.tables.push({
        startLine: lineNum,
        endLine: tableEnd
      });
      i = tableEnd - 1;
      continue;
    }

    const codeBlockStart = line.match(/^```(\w*)$/);
    if (codeBlockStart) {
      const codeBlockEnd = findCodeBlockEnd(lines, i);
      structure.specialElements.codeBlocks.push({
        startLine: lineNum,
        endLine: codeBlockEnd
      });
      i = codeBlockEnd - 1;
      continue;
    }

    const mathBlockStart = line.match(/^\$\$/);
    if (mathBlockStart) {
      const mathBlockEnd = findMathBlockEnd(lines, i);
      structure.specialElements.mathBlocks.push({
        startLine: lineNum,
        endLine: mathBlockEnd
      });
      i = mathBlockEnd - 1;
      continue;
    }

    const imageMatch = line.match(/!\[.*?\]\(.*?\)/);
    if (imageMatch) {
      structure.specialElements.images.push({ line: lineNum });
    }

    if (line.trim()) {
      if (currentParagraph.length === 0) {
        paraStartLine = lineNum;
      }
      currentParagraph.push(line);
    } else {
      if (currentParagraph.length > 0 && !inReferences) {
        const paraText = currentParagraph.join('\n');
        structure.paragraphs.push({
          startLine: paraStartLine,
          endLine: lineNum - 1,
          text: paraText,
          wordCount: paraText.replace(/\s/g, '').length
        });
        currentParagraph = [];
      }
    }

    if (inReferences) {
      structure.referencesSection.content += line + '\n';
    }
  }

  if (!foundAbstractByHeading && first10000Chars.length > 0) {
    const abstractMatch = first10000Chars.match(/\babstract\b/i);

    if (abstractMatch) {
      const abstractStartIndex = abstractMatch.index;
      const abstractContent = first10000Chars.substring(abstractStartIndex);

      const nextHeadingMatch = abstractContent.match(/^#{1,6}\s+/m);
      if (nextHeadingMatch) {
        structure.abstract = abstractContent.substring(0, nextHeadingMatch.index).trim();
      } else {
        structure.abstract = abstractContent.trim();
      }
    }
  }

  if (currentParagraph.length > 0 && !inReferences) {
    const paraText = currentParagraph.join('\n');
    structure.paragraphs.push({
      startLine: paraStartLine,
      endLine: lines.length,
      text: paraText,
      wordCount: paraText.replace(/\s/g, '').length
    });
  }

  if (structure.referencesSection.startLine > 0) {
    structure.referencesSection.endLine = lines.length;
    structure.referencesSection.excluded = true;
    structure.totalLines = structure.referencesSection.startLine - 1;
  } else {
    structure.totalLines = lines.length;
  }

  return structure;
}

function findTableEnd(lines, startLine) {
  for (let i = startLine + 1; i < lines.length; i++) {
    if (!lines[i].match(/^\|.*\|$/)) {
      return i + 1;
    }
  }
  return lines.length;
}

function findHtmlTableEnd(lines, startLine) {
  for (let i = startLine; i < lines.length; i++) {
    if (lines[i].match(/<\/table>/i)) {
      return i + 1;
    }
  }
  return lines.length;
}

function findCodeBlockEnd(lines, startLine) {
  for (let i = startLine + 1; i < lines.length; i++) {
    if (lines[i].match(/^```$/)) {
      return i + 1;
    }
  }
  return lines.length;
}

function findMathBlockEnd(lines, startLine) {
  for (let i = startLine + 1; i < lines.length; i++) {
    if (lines[i].match(/^\$\$$/)) {
      return i + 1;
    }
  }
  return lines.length;
}

export function parseFilteredMarkdown(filteredLines) {
  const structure = {
    abstract: '',
    headings: [],
    paragraphs: [],
    totalLines: filteredLines.length,
    referencesSection: {
      startLine: filteredLines.length + 1,
      endLine: filteredLines.length,
      content: '',
      excluded: false
    }
  };

  let inAbstract = false;
  let inReferences = false;
  let currentParagraph = [];
  let paraStartLine = 0;
  let foundAbstractByHeading = false;
  let first10000Chars = '';

  for (let i = 0; i < filteredLines.length; i++) {
    const line = filteredLines[i];
    const lineNum = i + 1;

    if (first10000Chars.length < 10000) {
      first10000Chars += line + '\n';
    }

    const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      const text = headingMatch[2].trim();
      structure.headings.push({ level, text, line: lineNum });

      const lowerText = text.toLowerCase();
      if (lowerText === 'abstract' || lowerText === '摘要') {
        inAbstract = true;
        foundAbstractByHeading = true;
      } else if (inAbstract) {
        inAbstract = false;
      }

      if (lowerText === 'references' || lowerText === 'bibliography' || lowerText === '参考文献') {
        inReferences = true;
        structure.referencesSection.startLine = lineNum;
      }
      continue;
    }

    if (inAbstract) {
      structure.abstract += line + '\n';
      continue;
    }

    if (line.trim()) {
      if (currentParagraph.length === 0) {
        paraStartLine = lineNum;
      }
      currentParagraph.push(line);
    } else {
      if (currentParagraph.length > 0 && !inReferences) {
        const paraText = currentParagraph.join('\n');
        structure.paragraphs.push({
          startLine: paraStartLine,
          endLine: lineNum - 1,
          text: paraText,
          wordCount: paraText.replace(/\s/g, '').length
        });
        currentParagraph = [];
      }
    }
  }

  if (!foundAbstractByHeading && first10000Chars.length > 0) {
    const abstractMatch = first10000Chars.match(/\babstract\b/i);

    if (abstractMatch) {
      const abstractStartIndex = abstractMatch.index;
      const abstractContent = first10000Chars.substring(abstractStartIndex);

      const nextHeadingMatch = abstractContent.match(/^#{1,6}\s+/m);
      if (nextHeadingMatch) {
        structure.abstract = abstractContent.substring(0, nextHeadingMatch.index).trim();
      } else {
        structure.abstract = abstractContent.trim();
      }
    }
  }

  if (currentParagraph.length > 0 && !inReferences) {
    const paraText = currentParagraph.join('\n');
    structure.paragraphs.push({
      startLine: paraStartLine,
      endLine: filteredLines.length,
      text: paraText,
      wordCount: paraText.replace(/\s/g, '').length
    });
  }

  if (structure.referencesSection.startLine <= filteredLines.length) {
    structure.referencesSection.endLine = filteredLines.length;
    structure.referencesSection.excluded = true;
    structure.totalLines = structure.referencesSection.startLine - 1;
  }

  return structure;
}

export function extractSegment(lines, segments) {
  const segmentTexts = segments.map(seg => {
    const startIdx = seg[0] - 1;
    const endIdx = seg[1] - 1;
    return lines.slice(startIdx, endIdx).join('\n');
  });
  return segmentTexts;
}
