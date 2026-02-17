import dotenv from 'dotenv';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';

dotenv.config();

const apiKey = process.env.ZHIPU_API_KEY;
const model = process.env.ZHIPU_MODEL || 'GLM-4.7-FlashX';
const API_URL = 'https://open.bigmodel.cn/api/coding/paas/v4/chat/completions';

if (!apiKey) {
  throw new Error('ZHIPU_API_KEY not found in environment variables');
}

function generateToken(apiKey, expSeconds = 3600) {
  const [id, secret] = apiKey.split('.');
  
  const now = Date.now();
  const exp = now + expSeconds * 1000;
  
  const header = { alg: 'HS256', sign_type: 'SIGN' };
  const payload = { api_key: id, exp: exp, timestamp: now };
  
  const encodedHeader = base64urlEncode(JSON.stringify(header));
  const encodedPayload = base64urlEncode(JSON.stringify(payload));
  
  const signature = crypto
    .createHmac('sha256', secret)
    .update(`${encodedHeader}.${encodedPayload}`)
    .digest('base64url');
  
  return `${encodedHeader}.${encodedPayload}.${signature}`;
}

function base64urlEncode(str) {
  return Buffer.from(str)
    .toString('base64')
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '');
}

async function callChatAPI(messages, options = {}, signal = null) {
  const token = generateToken(apiKey);
  
  const response = await fetch(API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify({
      model,
      messages,
      temperature: 0.7,
      ...options
    }),
    signal: signal
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API error ${response.status}: ${errorText}`);
  }

  const data = await response.json();
  return data.choices[0].message.content;
}

export async function chat(prompt, options = {}) {
  try {
    return await callChatAPI([{ role: 'user', content: prompt }], options);
  } catch (error) {
    throw new Error(`Zhipu API error: ${error.message}`);
  }
}

export async function chatWithRetry(prompt, maxRetries = 3, label = 'prompt', abortSignal = null) {
  const isAbortError = (error) => {
    return error.name === 'AbortError' || 
           (error.name === 'DOMException' && error.message.includes('Aborted')) ||
           error.message.includes('Aborted') ||
           error.message.includes('The user aborted a request') ||
           error.message.includes('Task was cancelled') ||
           error.message.includes('user aborted a request');
  };
  
  const cleanupDebugFile = () => {
    try {
      if (fs.existsSync(debugFile)) {
        fs.unlinkSync(debugFile);
        console.log(`[清理] 已删除 debug 文件: ${debugFile}`);
      }
    } catch (err) {
      console.warn(`[警告] 无法删除 debug 文件 ${debugFile}:`, err.message);
    }
  };
  
  if (abortSignal && abortSignal.aborted) {
    const error = new Error('Aborted: Task was cancelled before starting');
    error.name = 'AbortError';
    throw error;
  }
  
  const timestamp = Date.now();
  const random = Math.random().toString(36).substring(7);
  const debugFile = path.join('debug', `${label}_${timestamp}_${random}.txt`);
  
  try {
    fs.writeFileSync(debugFile, prompt, 'utf-8');
  } catch (err) {
    console.warn(`[警告] 无法写入 debug 文件 ${debugFile}:`, err.message);
  }

  for (let i = 0; i < maxRetries; i++) {
    if (abortSignal && abortSignal.aborted) {
      console.log(`[中断] ${label} - 检测到中断信号，取消重试`);
      cleanupDebugFile();
      const error = new Error('Aborted: Task was cancelled');
      error.name = 'AbortError';
      throw error;
    }
    
    try {
      const result = await chat(prompt, { signal: abortSignal });
      
      console.log(`[成功] ${label} - 翻译完成`);
      cleanupDebugFile();
      return result;
      
    } catch (error) {
      if (isAbortError(error) || (abortSignal && abortSignal.aborted)) {
        console.log(`[中断] ${label} - API 请求已取消`);
        cleanupDebugFile();
        const abortError = new Error('Aborted: Task was cancelled');
        abortError.name = 'AbortError';
        throw abortError;
      }
      
      if (i === maxRetries - 1) {
        console.log(`[失败] ${label} - 达到最大重试次数`);
        cleanupDebugFile();
        throw error;
      }
      
      console.error(`[重试] ${label} - 第 ${i + 1}/${maxRetries} 次尝试失败: ${error.message}`);
      
      if (abortSignal && abortSignal.aborted) {
        console.log(`[中断] ${label} - 等待重试时检测到中断信号`);
        cleanupDebugFile();
        const abortError = new Error('Aborted: Task was cancelled');
        abortError.name = 'AbortError';
        throw abortError;
      }
      
      await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
    }
  }
}

export { model };
