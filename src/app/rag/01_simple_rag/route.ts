import { initChroma, initOllamaLLM } from '@/app/utils';
import { PromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import {
  RunnablePassthrough,
  RunnableSequence,
} from '@langchain/core/runnables';

/**
 * 使用 langchain 实现的简单 RAG 系统
 * @param request
 * @returns
 */
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const question = searchParams.get('question');
  if (!question) {
    return Response.json({ error: 'Missing input question' });
  }

  // 文件解析
  const loader = new PDFLoader(
    'src/app/data/2024少儿编程教育行业发展趋势报告.pdf',
    {
      splitPages: false,
    }
  );
  const docs = await loader.load();

  // 文件切分
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  });
  const texts = await textSplitter.splitDocuments(docs);

  // 初始化向量数据库
  const chromadb = initChroma('collection_simple_rag');

  // 文档向量化
  await chromadb.addDocuments(texts);

  // 创建检索器
  const retriever = chromadb.asRetriever();

  // 创建生成器（初始化大模型）
  const ollamaLLM = initOllamaLLM();

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
