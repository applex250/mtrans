export function filterSpecialElements(lines, structure) {
  const elementMap = new Map();
  const filteredLines = [];
  const lineMapping = new Map();
  
  let tableCounter = 0;
  let imageCounter = 0;
  let codeCounter = 0;
  let filteredLineCount = 0;
  
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const originalLineNum = i + 1;
    
    const tableMatch = line.match(/^\|.*\|$/);
    if (tableMatch) {
      const tableEnd = findTableEnd(lines, i);
      const tableContent = lines.slice(i, tableEnd + 1).join('\n');
      const placeholder = `<!-- TABLE:${++tableCounter} -->`;
 
      elementMap.set(placeholder, tableContent);
      filteredLines.push(placeholder);
      lineMapping.set(++filteredLineCount, originalLineNum);
 
      i = tableEnd + 1;
      continue;
    }
    
    const codeBlockStart = line.match(/^```(\w*)$/);
    if (codeBlockStart) {
      const codeEnd = findCodeBlockEnd(lines, i);
      const codeContent = lines.slice(i, codeEnd + 1).join('\n');
      const placeholder = `<!-- CODE:${++codeCounter} -->`;
 
      elementMap.set(placeholder, codeContent);
      filteredLines.push(placeholder);
      lineMapping.set(++filteredLineCount, originalLineNum);
 
      i = codeEnd + 1;
      continue;
    }
    
    const imageMatch = line.match(/!\[.*?\]\(.*?\)/);
    if (imageMatch) {
      const placeholder = `<!-- IMAGE:${++imageCounter} -->`;
      
      elementMap.set(placeholder, line);
      filteredLines.push(placeholder);
      lineMapping.set(++filteredLineCount, originalLineNum);
      
      i++;
      continue;
    }
    
    filteredLines.push(line);
    lineMapping.set(++filteredLineCount, originalLineNum);
    i++;
  }
  
  return {
    filteredLines,
    elementMap,
    lineMapping,
    stats: {
      tablesFiltered: tableCounter,
      imagesFiltered: imageCounter,
      codeBlocksFiltered: codeCounter,
      totalLinesSaved: lines.length - filteredLines.length
    }
  };
}

export function restoreSpecialElements(translatedContent, elementMap) {
  let restoredContent = translatedContent;
  
  const placeholders = Array.from(elementMap.keys());
  placeholders.sort((a, b) => {
    const typeA = a.match(/<!-- (TABLE|IMAGE|CODE):/)[1];
    const typeB = b.match(/<!-- (TABLE|IMAGE|CODE):/)[1];
    const typeOrder = { 'TABLE': 1, 'IMAGE': 2, 'CODE': 3 };
    return typeOrder[typeA] - typeOrder[typeB];
  });
  
  for (const placeholder of placeholders) {
    if (restoredContent.includes(placeholder)) {
      restoredContent = restoredContent.replace(placeholder, elementMap.get(placeholder));
    }
  }
  
  return restoredContent;
}

function findTableEnd(lines, startLine) {
  for (let i = startLine + 1; i < lines.length; i++) {
    if (!lines[i].match(/^\|.*\|$/)) {
      return i - 1;
    }
  }
  return lines.length - 1;
}

function findCodeBlockEnd(lines, startLine) {
  for (let i = startLine + 1; i < lines.length; i++) {
    if (lines[i].match(/^```$/)) {
      return i;
    }
  }
  return lines.length - 1;
}
