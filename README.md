# mtrans - Markdown 学术文献翻译工具

基于智谱 GLM-4.7 的英文学术文献 Markdown 翻译工具，支持智能分段、并发翻译和实时进度显示。
# 推荐使用方法
https://mineru.net/client 进行pdf转markdown，

然后用该项目进行翻译。
## 特性

- ✅ 智能分段：LLM 自动分析文档结构，设计最优分段方案
- ✅ 并发翻译：支持多段并行翻译，速度优先
- ✅ 实时进度：可视化进度条，显示翻译速度
- ✅ 术语一致性：提取专业方向，保持术语翻译一致性
- ✅ 特殊元素保留：表格、公式、代码块、图片保持原样
- 🌐 Web 界面：黑白极客风格的可视化界面，支持拖拽上传和任务队列管理

## 安装

### 1. 克隆项目

```bash
git clone <repository-url>
cd mtrans
```

### 2. 安装依赖

```bash
npm install
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填写你的智谱 API Key：

```env
ZHIPU_API_KEY=your_api_key_here
ZHIPU_MODEL=glm-4
MAX_CONCURRENCY=5
```

## 使用

### 基本用法

翻译输入文件，自动保存到 `output/` 目录：

```bash
npm test
```

或直接运行：

```bash
node src/index.js input/sample.md
```

### 指定输出文件

```bash
node src/index.js input/paper.md output/translated.md
```

### Web 界面使用

启动 Web 服务器：

```bash
npm run web
```

浏览器访问：`http://localhost:3000`

#### Web 界面功能

- **拖拽上传**：直接拖拽 Markdown 文件到上传区域
- **任务队列**：左侧面板显示所有翻译任务及进度
- **实时日志**：右侧控制台显示详细的翻译步骤和日志
- **自定义输出路径**：可设置统一的输出目录（自动保存到 settings.json）
- **任务管理**：支持暂停、继续、删除翻译任务
- **批量翻译**：支持同时上传多个文档，自动依次执行
- **自动清理**：翻译完成后自动删除 `input/` 目录中的临时文件
- **配置持久化**：设置自动保存到本地，重启后保持配置

#### Web 界面快捷操作

- **暂停任务**：点击任务旁的"暂停"按钮
- **继续任务**：点击任务旁的"继续"按钮
- **删除任务**：点击任务旁的"删除"按钮（仅非执行中任务）
- **清除已完成**：点击底部"清除已完成"按钮清理已完成任务

## 工作流程

1. **解析 Markdown**：提取 abstract、标题、段落和特殊元素
2. **提取专业方向**：分析 abstract，生成专业方向描述
3. **设计分段方案**：LLM 根据文档结构智能分段
4. **并发翻译**：多段并行翻译，实时显示进度
5. **保存结果**：拼接翻译内容，输出到目标文件

## 项目结构

```
mtrans/
├── src/
│   ├── parsers/
│   │   └── markdownParser.js    # Markdown 解析器
│   ├── llm/
│   │   └── zhipuClient.js       # 智谱 API 客户端
│   ├── translator/
│   │   ├── segmentDesigner.js   # 分段设计器
│   │   └── translator.js        # 翻译器
│   ├── web/
│   │   ├── server.js            # Web 服务器
│   │   ├── routes.js            # API 路由
│   │   ├── queue.js             # 任务队列管理
│   │   ├── settings.js          # 配置文件管理
│   │   └── public/
│   │       ├── index.html       # 主页面
│   │       ├── style.css        # 极客风格样式
│   │       └── app.js           # 前端逻辑
│   └── index.js                 # 命令行主入口
├── input/                       # 上传文件临时目录（自动清理）
├── output/                      # 翻译结果输出目录
├── debug/                       # 调试文件目录
├── .env.example                 # 环境变量模板
├── settings.json                # 用户配置文件（该文件会在用户修改设置时自动生成）
└── package.json
```

## 配置说明

### 环境变量

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `ZHIPU_API_KEY` | 智谱 API Key | 必填 |
| `ZHIPU_MODEL` | 使用的模型 | glm-4 |
| `MAX_CONCURRENCY` | 并发翻译段数 | 5 |
| `WEB_PORT` | Web 服务器端口 | 3000 |

### 配置文件

`settings.json` 该文件会在用户修改设置时,自动生成，包含以下配置：

```json
{
  "outputPath": "output",
  "webPort": 3000
}
```

- `outputPath`: 翻译结果输出路径
- `webPort`: Web 服务器端口

该文件会在 Web 界面修改设置时自动更新。

## 翻译规则

1. 保持学术风格，缩写/术语不翻译
2. 表格/图片/公式/代码块保持原样
3. 保持 markdown 结构
4. 直接输出翻译，无额外文字

## 示例

输入 `input/sample.md` 中的英文文献，输出翻译后的中文版本：

```bash
npm test
```

## 注意事项

- 确保 `.env` 文件中的 API Key 有效
- 翻译过程需要网络连接
- 长文档翻译可能需要较长时间
- 失败的分段会记录到日志并保留原文
- Web 界面上传的文件会自动保存到 `input/` 目录，翻译完成后自动删除
- `settings.json` 配置文件会在首次运行时自动生成，请勿将其提交到版本控制
- 建议将 `.env` 和 `settings.json` 添加到 `.gitignore` 以保护敏感信息

## 依赖

- `dotenv`: 环境变量管理
- `cli-progress`: 进度条显示
- `p-limit`: 并发控制
- `express`: Web 服务器框架
- `socket.io`: WebSocket 实时通信
- `multer`: 文件上传中间件

## 许可证

MIT
