## 项目开发

安装依赖：

```bash
npm install
```

启动：

```bash
npm run dev
```

访问：

```bash
页面：http://localhost:3000/rag
接口：http://localhost:3000/rag/01_simple_rag
```

## chroma 版本要求

由于服务端 chromadb >= 1.0.0 存在兼容性问题，请使用以下版本：

```bash
pip install chromadb==0.6.3
```

然后客户端 chromadb 限制版本为 2.0.1：

```bash
npm install --save chromadb@2.0.1
```

## 项目说明

本项目是 AI 应用开发实践项目，将会涵盖 AI 应用开发中涉及到各种技术，包括但不限于提示工程、RAG、模型微调、Agent 等等。

### 入门基础系列

- [面对汹涌的 AI 浪潮，前端开发者该如何破局？](https://mp.weixin.qq.com/s/7OZAuw9QMQWViXi-59k8gA)
- [前端 x AI：从了解提示工程、RAG 和微调开始](https://mp.weixin.qq.com/s/b9ROm1cU41BCFYRlq2Nfbw)
- [前端 x AI：基于 Ollama、DeepSeek、Chroma、LangChain 搭建 AI 应用开发框架](https://mp.weixin.qq.com/s/qtlq-Iu7chqpMEbahnR64w)

### 提示工程系列

待更新，敬请期待...

### RAG 系列

- [RAG 系列（一）：一文让你由浅到深搞懂 RAG 实现](https://mp.weixin.qq.com/s/WbDPo0JM40qasuZzXSdi_Q)
- [RAG 系列（二）：基于 DeepSeek + Chroma + LangChain 开发一个简单 RAG 系统
  ](https://mp.weixin.qq.com/s/yZLqv_YJFbK0jYMA_6DMVQ)
- [RAG 系列（三）：系统评估 - 构造 QA 测试数据集](https://mp.weixin.qq.com/s?__biz=MzAwMjgzNTAxMA==&mid=2650407042&idx=1&sn=5bf699ac9963143732314a54e24ff6c8&chksm=82caf7afb5bd7eb92bd7eda5831354e822b6214b43e3e5917898cedc3b80f475b989c077e50f&cur_album_id=3879147818837032971&scene=189#wechat_redirect)
- [RAG 系列（四）：系统评估 - 五个主流评估指标详解](https://mp.weixin.qq.com/s?__biz=MzAwMjgzNTAxMA==&mid=2650407054&idx=1&sn=55aacdc9a453a4d8352f62e2c182c68e&chksm=82caf7a3b5bd7eb5e633084712c436976d809357ef2b99f3ed1cfca06c33b73a5078a2a9ba85&cur_album_id=3879147818837032971&scene=189#wechat_redirect)
- [RAG 系列（五）：系统评估 - 基于 LLM-as-judge 实现评估系统](https://mp.weixin.qq.com/s?__biz=MzAwMjgzNTAxMA==&mid=2650407071&idx=1&sn=7964554f13eadda3c3ce592ba8a54faf&chksm=82caf7b2b5bd7ea48f46540f7f674ea5accb3680acb208504a2f6ace8bd589a826302aa4d960&cur_album_id=3879147818837032971&scene=189#wechat_redirect)
- [RAG 系列（六）：问题优化 - 意图识别&同义改写&多视角分解&补充上下文](https://mp.weixin.qq.com/s?__biz=MzAwMjgzNTAxMA==&mid=2650407082&idx=1&sn=8c4e5aff9ef7e31c8f5f750b4efe403f&chksm=82caf787b5bd7e919654692c9ea4d1c5204388388a9bea8bf2af60f6e3f41ca03ec751cd14fe&scene=178&cur_album_id=3920944561060528133&search_click_id=#rd)

### 模型微调系列

待更新，敬请期待...

### Agent 系列

待更新，敬请期待...

## 微信公众号

欢迎大家关注我的微信公众号【燃哥讲 AI】，分享前端 x AI 的前沿技术，注重理论结合实践，助你成为 AI 浪潮中的超级个体！

![微信公众号：燃哥讲AI](https://github.com/user-attachments/assets/465ecea4-52e6-4917-baf1-c6debdd19db1)
