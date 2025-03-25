import {
  Chroma,
  ChromaLibArgs,
} from '@langchain/community/vectorstores/chroma';
import { ChatOllama, OllamaEmbeddings } from '@langchain/ollama';
import { EmbeddingsInterface } from '@langchain/core/embeddings';
import { PromptTemplate } from '@langchain/core/prompts';

export async function initRagPrompt(context: string, question: string) {
  const promptTemplate = PromptTemplate.fromTemplate(
    '你是负责回答问题的助手。使用以下检索到的上下文片段来回答问题。如果你不知道答案，就说你不知道。\n\n上下文：{context}\n\n问题：{question}\n\n回答：'
  );
  return promptTemplate.format({ context, question });
}

export function initOllamaLLM(model = 'deepseek-r1:14b') {
  return new ChatOllama({ model });
}

export function initOllamaEmbeddings(model = 'nomic-embed-text') {
  return new OllamaEmbeddings({ model });
}

export function initChroma(
  embeddings: EmbeddingsInterface = initOllamaEmbeddings(),
  args: ChromaLibArgs = {
    collectionName: 'rag_collection',
    url: 'http://localhost:8000',
  }
) {
  return new Chroma(embeddings, args);
}
