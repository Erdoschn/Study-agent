# AI 出题按钮：行为与文件存储

统一 Web UI 的 **AI 出题**操作使用独立 API 端点，不会把出题指令注入普通聊天提示词。

## 工作流程

1. 只有至少一轮完整的用户消息和助手回答时才启用按钮。
2. 点击后立即在对话状态区显示“正在出题，请稍等”，同时锁住发送框和出题按钮。
3. 后端使用 config/prompts.json 中的 study_agent.question_generation 生成首题；同一对话完成新一轮交流后，再点击“重新出题”会使用 study_agent.question_regeneration，并要求产生实质不同的新题。
4. 题目与参考答案写入独立 Markdown 文件。普通聊天返回题目预览和可点击的文件链接，不返回参考答案正文。
5. 服务端会再次核验本对话完成轮数，避免绕过 UI 后在同一轮重复出题。成功后只有观察到新一轮已完成交流，才能继续出题。

## 文件路径与隐私边界

生成文件保存到仓库根目录 出题/ 下，并按规范化用户名的 SHA-256 标识分目录，例如：

    出题/<profile-sha256>/question-YYYYMMDD_HHMMSS-<id>.md
    出题/<profile-sha256>/answer-YYYYMMDD_HHMMSS-<id>.md

目录权限尽量设为私有；该目录、生成文件与状态清单都加入 Git 忽略规则，不会作为源码自动提交。题目文件不包含参考答案；答案 Markdown 文件链接回对应题目。Web UI 的文件链接通过本地 API 读取文件，点击时会通过同一 UI 的 API 授权头完成读取，并在新标签页中以纯文本安全显示 Markdown。

用户名是本地 Profile 分区键，不是网络身份认证。若 API 绑定非 loopback 地址，必须继续使用既有的 API 密钥或可信认证层；不能把 SHA-256 目录名视作访问控制。

## 提示词配置

- study_agent.question_generation：首题提示词。
- study_agent.question_regeneration：重新出题提示词。
- study_agent.assessment：保留给现有显式 Assessment API/Tool 的兼容提示词，不再被 TaskAnalyzer/Reasoner 当作普通聊天内自动出题流程。

## 当前限制

- 生成模型返回的参考答案被单独存放，但当前流程并没有生产级的独立数学/事实核验器；文件分离不代表答案已被独立证明正确。
- 路径链接受同一 API 的来源检查和授权策略保护；本地文件本身不会上传到远程仓库。
