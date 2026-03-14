/**
 * MinerU API Service
 * Handles document conversion using MinerU API
 * API Documentation: https://mineru.net/apiManage/docs
 */

import fs from 'fs';
import path from 'path';
import https from 'https';
import http from 'http';
import { pipeline } from 'stream/promises';
import { createWriteStream } from 'fs';
import os from 'os';
import AdmZip from 'adm-zip';

const MINERU_API_BASE = 'https://mineru.net';
const MINERU_API_TOKEN = process.env.MINERU_API_TOKEN || '';

// Supported file extensions
const MARKDOWN_EXTENSIONS = ['.md', '.markdown'];
const MINERU_EXTENSIONS = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.png', '.jpg', '.jpeg', '.gif', '.bmp'];
const ALL_SUPPORTED_EXTENSIONS = [...MARKDOWN_EXTENSIONS, ...MINERU_EXTENSIONS];

/**
 * Check if a file is a markdown file
 * @param {string} fileName - File name to check
 * @returns {boolean} True if markdown file
 */
export function isMarkdownFile(fileName) {
  const ext = path.extname(fileName).toLowerCase();
  return MARKDOWN_EXTENSIONS.includes(ext);
}

/**
 * Check if a file needs MinerU conversion
 * @param {string} fileName - File name to check
 * @returns {boolean} True if needs conversion
 */
export function needsMinerUConversion(fileName) {
  const ext = path.extname(fileName).toLowerCase();
  return MINERU_EXTENSIONS.includes(ext);
}

/**
 * Check if a file type is supported
 * @param {string} fileName - File name to check
 * @returns {boolean} True if supported
 */
export function isSupportedFileType(fileName) {
  const ext = path.extname(fileName).toLowerCase();
  return ALL_SUPPORTED_EXTENSIONS.includes(ext);
}

/**
 * Get file type category
 * @param {string} fileName - File name
 * @returns {string} File type category
 */
export function getFileType(fileName) {
  const ext = path.extname(fileName).toLowerCase();
  if (MARKDOWN_EXTENSIONS.includes(ext)) return 'markdown';
  if (ext === '.pdf') return 'pdf';
  if (['.doc', '.docx'].includes(ext)) return 'word';
  if (['.ppt', '.pptx'].includes(ext)) return 'powerpoint';
  if (['.png', '.jpg', '.jpeg', '.gif', '.bmp'].includes(ext)) return 'image';
  return 'unknown';
}

/**
 * Make HTTP request using fetch API
 * @param {string} url - Request URL
 * @param {Object} options - Request options
 * @returns {Promise<Object>} Response data
 */
async function makeRequest(url, options = {}) {
  const fetchOptions = {
    method: options.method || 'GET',
    headers: {
      'Content-Type': 'application/json',
      'Accept': '*/*',
      ...options.headers
    }
  };

  if (options.body) {
    fetchOptions.body = typeof options.body === 'string' ? options.body : JSON.stringify(options.body);
  }

  console.log('[MinerU] Fetch:', url);
  console.log('[MinerU] Options:', JSON.stringify(fetchOptions, null, 2));

  const response = await fetch(url, fetchOptions);
  const data = await response.json();

  console.log('[MinerU] Response status:', response.status);
  console.log('[MinerU] Response data:', JSON.stringify(data, null, 2));

  return { status: response.status, data };
}

/**
 * Apply for upload URLs from MinerU API
 * @param {Array} files - Array of file objects with fileName property
 * @returns {Promise<Object>} Response with batch_id and upload URLs
 */
export async function applyUploadUrls(files) {
  if (!MINERU_API_TOKEN) {
    throw new Error('MINERU_API_TOKEN not configured. Please set the MINERU_API_TOKEN environment variable.');
  }

  // Format: { files: [{ name: "filename.pdf" }], model_version: "vlm" }
  const filesArray = files.map((f, index) => ({
    name: f.fileName || f.name,
    data_id: `mtrans_${Date.now()}_${index}`
  }));

  const requestBody = { files: filesArray, model_version: 'vlm' };

  const response = await makeRequest(`${MINERU_API_BASE}/api/v4/file-urls/batch`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${MINERU_API_TOKEN}`
    },
    body: requestBody
  });

  return response.data;
}

/**
 * Upload file to presigned URL
 * @param {string} uploadUrl - Presigned upload URL
 * @param {string} localPath - Local file path
 * @param {Function} logger - Logger function
 * @returns {Promise<void>}
 */
export async function uploadFile(uploadUrl, localPath, logger = console.log) {
  return new Promise((resolve, reject) => {
    const fileStream = fs.createReadStream(localPath);
    const fileSize = fs.statSync(localPath).size;
    const urlObj = new URL(uploadUrl);
    const client = urlObj.protocol === 'https:' ? https : http;

    const options = {
      hostname: urlObj.hostname,
      port: urlObj.port || (urlObj.protocol === 'https:' ? 443 : 80),
      path: urlObj.pathname + urlObj.search,
      method: 'PUT',
      headers: {
        'Content-Length': fileSize
      }
    };

    logger(`  上传文件: ${localPath} (${(fileSize / 1024).toFixed(2)} KB)`);

    const req = client.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        if (res.statusCode >= 200 && res.statusCode < 300) {
          logger(`  上传完成: HTTP ${res.statusCode}`);
          resolve();
        } else {
          reject(new Error(`Upload failed: HTTP ${res.statusCode} - ${data}`));
        }
      });
    });

    req.on('error', (err) => {
      logger(`  上传错误: ${err.message}`, 'error');
      reject(err);
    });

    fileStream.pipe(req);
  });
}

/**
 * Poll batch result until completion
 * @param {string} batchId - Batch ID to poll
 * @param {Function} logger - Logger function
 * @param {AbortSignal} signal - Abort signal for cancellation
 * @returns {Promise<Object>} Batch result with full_zip_url
 */
export async function pollBatchResult(batchId, logger = console.log, signal = null) {
  const maxPolls = 120; // 10 minutes max (5 second intervals)
  let pollCount = 0;

  while (pollCount < maxPolls) {
    if (signal && signal.aborted) {
      throw new Error('Polling aborted');
    }

    try {
      const response = await makeRequest(`${MINERU_API_BASE}/api/v4/extract-results/batch/${batchId}`, {
        headers: {
          'Authorization': `Bearer ${MINERU_API_TOKEN}`
        }
      });

      const result = response.data;

      // Check API response format: { code: 0, data: { batch_id, extract_result: [...] } }
      if (result.code !== 0) {
        throw new Error(`API error: ${result.msg || 'Unknown error'}`);
      }

      const extractResults = result.data?.extract_result || [];
      if (extractResults.length === 0) {
        logger(`  等待任务开始...`);
      } else {
        const firstResult = extractResults[0];
        const status = firstResult.state || 'unknown';
        const progress = firstResult.extract_progress;

        if (progress) {
          logger(`  转换状态: ${status} (${progress.extracted_pages}/${progress.total_pages} 页)`);
        } else {
          logger(`  转换状态: ${status}`);
        }

        if (status === 'done') {
          const fullZipUrl = firstResult.full_zip_url;
          if (fullZipUrl) {
            logger(`  转换完成`);
            return { fullZipUrl, result };
          }
        }

        if (status === 'failed') {
          const errMsg = firstResult.err_msg || 'MinerU conversion failed';
          throw new Error(errMsg);
        }
      }

      // Wait before next poll
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(resolve, 5000);
        if (signal) {
          signal.addEventListener('abort', () => {
            clearTimeout(timeout);
            reject(new Error('Polling aborted'));
          });
        }
      });

      pollCount++;
    } catch (error) {
      if (error.message.includes('aborted')) {
        throw error;
      }
      logger(`  轮询错误: ${error.message}`, 'warning');
      await new Promise(resolve => setTimeout(resolve, 5000));
      pollCount++;
    }
  }

  throw new Error('MinerU conversion timeout');
}

/**
 * Download file from URL
 * @param {string} url - Download URL
 * @param {string} outputPath - Output file path
 * @param {Function} logger - Logger function
 * @returns {Promise<void>}
 */
async function downloadFile(url, outputPath, logger = console.log) {
  return new Promise((resolve, reject) => {
    const urlObj = new URL(url);
    const client = urlObj.protocol === 'https:' ? https : http;

    logger(`  下载文件: ${url}`);

    const file = createWriteStream(outputPath);

    client.get(url, (response) => {
      if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
        // Handle redirect
        file.close();
        fs.unlinkSync(outputPath);
        downloadFile(response.headers.location, outputPath, logger).then(resolve).catch(reject);
        return;
      }

      if (response.statusCode !== 200) {
        file.close();
        fs.unlinkSync(outputPath);
        reject(new Error(`Download failed: HTTP ${response.statusCode}`));
        return;
      }

      response.pipe(file);

      file.on('finish', () => {
        file.close();
        logger(`  下载完成: ${outputPath}`);
        resolve();
      });
    }).on('error', (err) => {
      file.close();
      fs.unlinkSync(outputPath);
      reject(err);
    });
  });
}

/**
 * Download and extract markdown from zip
 * @param {string} zipUrl - URL of the zip file
 * @param {string} outputPath - Output directory path
 * @param {Function} logger - Logger function
 * @returns {Promise<string>} Path to extracted markdown file
 */
export async function downloadAndExtractMarkdown(zipUrl, outputPath, logger = console.log) {
  // Create temp directory
  const tempDir = path.join(os.tmpdir(), `mineru_${Date.now()}`);
  fs.mkdirSync(tempDir, { recursive: true });

  const zipPath = path.join(tempDir, 'result.zip');

  // Helper function to recursively copy directory
  const copyDir = (src, dest) => {
    fs.mkdirSync(dest, { recursive: true });
    const entries = fs.readdirSync(src, { withFileTypes: true });
    for (const entry of entries) {
      const srcPath = path.join(src, entry.name);
      const destPath = path.join(dest, entry.name);
      if (entry.isDirectory()) {
        copyDir(srcPath, destPath);
      } else {
        fs.copyFileSync(srcPath, destPath);
      }
    }
  };

  try {
    // Download zip
    await downloadFile(zipUrl, zipPath, logger);

    // Extract zip
    logger(`  解压文件...`);
    const zip = new AdmZip(zipPath);
    zip.extractAllTo(tempDir, true);

    // Find the extracted content directory (usually contains full.md and images/)
    const findContentDir = (dir) => {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          // Check if this directory contains full.md
          const subEntries = fs.readdirSync(fullPath, { withFileTypes: true });
          if (subEntries.some(e => e.name === 'full.md')) {
            return fullPath;
          }
          // Recursively search
          const result = findContentDir(fullPath);
          if (result) return result;
        }
      }
      return null;
    };

    let contentDir = findContentDir(tempDir);
    if (!contentDir) {
      // Check if full.md is directly in tempDir
      const directMdPath = path.join(tempDir, 'full.md');
      if (fs.existsSync(directMdPath)) {
        contentDir = tempDir;
      } else {
        throw new Error('full.md not found in extracted zip');
      }
    }

    // Copy all files (including images folder) to output path
    logger(`  复制所有文件（包括图片）...`);
    copyDir(contentDir, outputPath);

    const outputMdPath = path.join(outputPath, 'full.md');
    logger(`  已提取所有文件到: ${outputPath}`);
    logger(`  Markdown 文件: ${outputMdPath}`);

    // Check if images folder exists
    const imagesDir = path.join(outputPath, 'images');
    if (fs.existsSync(imagesDir)) {
      const imageCount = fs.readdirSync(imagesDir).length;
      logger(`  图片文件夹: ${imagesDir} (${imageCount} 个文件)`);
    }

    // Cleanup temp directory
    fs.rmSync(tempDir, { recursive: true, force: true });

    return outputMdPath;
  } catch (error) {
    // Cleanup on error
    try {
      fs.rmSync(tempDir, { recursive: true, force: true });
    } catch (e) {
      // Ignore cleanup errors
    }
    throw error;
  }
}

/**
 * Convert document using MinerU API
 * Full workflow: apply -> upload -> poll -> download -> extract
 * @param {string} inputPath - Input file path
 * @param {string} outputPath - Output directory path
 * @param {Function} logger - Logger function
 * @param {AbortSignal} signal - Abort signal for cancellation
 * @returns {Promise<string>} Path to converted markdown file
 */
export async function convertDocument(inputPath, outputPath, logger = console.log, signal = null) {
  const fileName = path.basename(inputPath);

  logger(`MinerU 转换开始: ${fileName}`);

  // Step 1: Apply for upload URL
  logger('  步骤 1/4: 申请上传链接...');
  const uploadResponse = await applyUploadUrls([{ fileName }]);

  // Response format: { code: 0, data: { batch_id, file_urls: [...] } }
  if (uploadResponse.code !== 0) {
    throw new Error(`Failed to get upload URL: ${uploadResponse.msg || 'Unknown error'}`);
  }

  const batchId = uploadResponse.data?.batch_id;
  const fileUrls = uploadResponse.data?.file_urls || [];

  if (!batchId || fileUrls.length === 0) {
    throw new Error('Failed to get upload URL from MinerU');
  }

  const uploadUrl = fileUrls[0];
  logger(`  Batch ID: ${batchId}`);

  if (signal && signal.aborted) {
    throw new Error('Conversion aborted');
  }

  // Step 2: Upload file
  logger('  步骤 2/4: 上传文件...');
  await uploadFile(uploadUrl, inputPath, logger);

  if (signal && signal.aborted) {
    throw new Error('Conversion aborted');
  }

  // Step 3: Poll for result
  logger('  步骤 3/4: 等待转换完成...');
  const result = await pollBatchResult(batchId, logger, signal);

  if (signal && signal.aborted) {
    throw new Error('Conversion aborted');
  }

  // Step 4: Download and extract
  logger('  步骤 4/4: 下载并提取 Markdown...');
  const mdPath = await downloadAndExtractMarkdown(result.fullZipUrl, outputPath, logger);

  logger(`MinerU 转换完成: ${mdPath}`);

  return mdPath;
}

export default {
  isMarkdownFile,
  needsMinerUConversion,
  isSupportedFileType,
  getFileType,
  applyUploadUrls,
  uploadFile,
  pollBatchResult,
  downloadAndExtractMarkdown,
  convertDocument
};
