# ML Engineering：Training 与 Insights 简体中文版

[Stas Bekman《Machine Learning Engineering Open Book》](https://github.com/stas00/ml-engineering) 的非官方中文译本，覆盖 `training/` 与 `insights/` 下全部 16 篇 Markdown 文档（含子目录），对应原文约 5.46 万英文词，含代码。原文固定于 `fb75494c0d9e536f048e27be2869a77fe65ad9f4`，获取日期 2026-10-10；逐文件 SHA-256 见 `source-manifest.json`。

按相邻 `ultrascale-playbook-zh-tw` 项目的组织和阅读样式制作：奶油色底、紫色点缀、章节卡片、侧栏目录、上一章／下一章、全文阅读、系统深色模式以及本地公式渲染。虽然参考目录名包含 zh-tw，其正文采用简体中文，本译本也使用简体。

## 阅读

```bash
cd /Users/lezewei/project/study/blog/ml-engineering-zh-cn
bash run.sh
```

打开 <http://127.0.0.1:8645>。如端口被占用，可用 `PORT=8646 bash run.sh`。构建只需要 Python 3 标准库，不需要 npm 或在线 CDN。也可直接打开 `webapp/index.html`；页面内嵌正文，脚本和样式均在本地。

## 目录

- `content/training/`：训练导读、模型并行、性能优化、容错、可复现性、训练不稳定性与损失曲线、检查点、超参数、数据类型、模拟多节点、重训 Hub 模型、数据集。
- `content/insights/`：AI 工程经验、云服务商选择、GPU 升级决策。
- `content/` 同时保留这两个目录的原始图片、辅助脚本和基准测试文件；代码与图片内的英文标签保持原样。
- `src/ml-engineering/`：完整原文仓库快照，便于中英对照。
- `webapp/`：生成后的阅读网站；章节间链接指向中文版，站内锚点保留英文原文 ID。
- `webapp/source-assets/`：阅读页引用的本地图片与代码资源，包括来自翻译范围外的辅助文件。
- `build.py`：从 Markdown 重建网站。修改 `content/` 后运行 `python3 build.py`。
- `scripts/check_translation.py`：检查文档覆盖、标题层级、代码块、原文链接保留和网站本地跳转。

Markdown 保留原始相对链接与英文锚点，完整跳转体验以生成后的网页为准。翻译范围之外的文档链接在网页中指向固定版本的 GitHub 原文。

## 翻译与验证说明

这是正文中文翻译，保留原文论述顺序、技术限定、作者第一人称、数据、公式和参考资料；第一人称“我／我们”指原作者或其团队。代码块、代码注释、命令输出及配套脚本保留原文。图像标题／替代文字翻译为中文，图中英文未重绘。原文时点的版本、硬件结果、价格和建议不改写为当前事实；疑似原文问题也没有擅自改动代码。

结构校验通过：16 篇、262 个标题、129 个围栏代码块；代码块与原文逐字一致，原 Markdown 链接目标均保留。浏览器检查了首页、性能章节和全文页；全文页无失效的页内锚点、未加载图片或 KaTeX 错误，并显示两条脚注。结果见 `translation-check.json`。这些检查验证的是翻译结构与阅读功能，不代表独立复现原文的训练实验或技术结论。

## 署名与许可

原文作者为 Stas Bekman 及项目贡献者。原文来源：<https://github.com/stas00/ml-engineering>。本项目将其指定章节译为中文，并调整了本地阅读导航与呈现方式；不是作者认可或发布的官方版本。

原文与本译文按 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 提供，许可副本见 `LICENSE-CC-BY-SA`。引用和再分发时请保留署名、原作链接、许可及修改说明。代码文件沿用上游文件自身的许可声明。

阅读模板的样式与本地渲染资源来自用户指定的 `../ultrascale-playbook-zh-tw`；其 Apache 2.0 许可副本保存在 `LICENSE-template`。第三方库 Marked、KaTeX 保留原文件及其许可信息。
