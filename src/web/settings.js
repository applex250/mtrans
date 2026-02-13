import fs from 'fs';
import path from 'path';

const SETTINGS_FILE = 'settings.json';

const DEFAULT_SETTINGS = {
  outputPath: 'output',
  webPort: 3000
};

function loadSettings() {
  try {
    if (fs.existsSync(SETTINGS_FILE)) {
      const content = fs.readFileSync(SETTINGS_FILE, 'utf-8');
      return { ...DEFAULT_SETTINGS, ...JSON.parse(content) };
    }
  } catch (error) {
    console.warn('无法加载设置文件，使用默认设置:', error.message);
  }
  return { ...DEFAULT_SETTINGS };
}

function saveSettings(settings) {
  try {
    fs.writeFileSync(SETTINGS_FILE, JSON.stringify(settings, null, 2), 'utf-8');
    return true;
  } catch (error) {
    console.error('保存设置失败:', error.message);
    return false;
  }
}

function updateSetting(key, value) {
  const settings = loadSettings();
  settings[key] = value;
  return saveSettings(settings);
}

function getSetting(key) {
  const settings = loadSettings();
  return settings[key];
}

export {
  loadSettings,
  saveSettings,
  updateSetting,
  getSetting,
  DEFAULT_SETTINGS
};
