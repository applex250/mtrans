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

async function callChatAPI(messages, options = {}) {
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
    })
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

export async function chatWithRetry(prompt, maxRetries = 3, label = 'prompt') {
  const timestamp = Date.now();
  const random = Math.random().toString(36).substring(7);
  const debugFile = path.join('debug', `${label}_${timestamp}_${random}.txt`);
  
  try {
    fs.writeFileSync(debugFile, prompt, 'utf-8');
  } catch (err) {
    console.warn('警告: 无法写入debug文件', err);
  }

  for (let i = 0; i < maxRetries; i++) {
    try {
      const result = await chat(prompt);
      
      try {
        if (fs.existsSync(debugFile)) {
          fs.unlinkSync(debugFile);
        }
      } catch (err) {
        console.warn('警告: 无法删除debug文件', err);
      }
      
      return result;
    } catch (error) {
      if (i === maxRetries - 1) {
        try {
          if (fs.existsSync(debugFile)) {
            fs.unlinkSync(debugFile);
          }
        } catch (err) {
          console.warn('警告: 无法删除debug文件', err);
        }
        throw error;
      }
      console.error(`Retry ${i + 1}/${maxRetries}: ${error.message}`);
      await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
    }
  }
}

export { model };
