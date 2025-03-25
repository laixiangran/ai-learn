import { initChroma, initOllamaLLM } from '@/app/rag/utils';
import { PromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import {
  RunnablePassthrough,
  RunnableSequence,
} from '@langchain/core/runnables';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  // 处理问题
  const question = searchParams.get('question');
  if (!question) {
    return Response.json({ error: 'Missing input question' });
  }

  // 初始化向量数据库
  const chromadb = initChroma();

  // 创建检索器
  const retriever = chromadb.asRetriever();

  // 创建生成器（初始化大模型）
  const ollamaLLM = initOllamaLLM('qwen2.5:14b');

  // 设置提示模版
  const prompt = PromptTemplate.fromTemplate(
    '你是负责回答问题的助手。使用以下检索到的上下文片段来回答问题。如果你不知道答案，就说你不知道。\n\n上下文：{context}\n\n问题：{question}\n\n回答：'
  );

  // 使用 LCEL 构建 RAG 链
  const ragChain = RunnableSequence.from([
    {
      context: retriever.pipe((docs) => {
        // 文档列表使用 \n\n 拼接为字符串
        return docs.map((doc) => doc.pageContent).join('\n\n');
      }),
      question: new RunnablePassthrough(),
    },
    prompt,
    ollamaLLM,
    new StringOutputParser(),
  ]);

  // 使用 RAG 链生成答案
  const answer = await ragChain.invoke(question);
  return Response.json({ answer });
}
