import {
  Chroma,
  ChromaLibArgs,
} from '@langchain/community/vectorstores/chroma';
import { ChatOllama, OllamaEmbeddings } from '@langchain/ollama';
import { EmbeddingsInterface } from '@langchain/core/embeddings';
import { PromptTemplate } from '@langchain/core/prompts';
import { Workbook } from 'exceljs';

export async function initRagPrompt(context: string, question: string) {
  const promptTemplate = PromptTemplate.fromTemplate(
    '你是负责回答问题的助手。使用以下检索到的上下文片段来回答问题。如果你不知道答案，就说你不知道。\n\n上下文：{context}\n\n问题：{question}\n\n回答：'
  );
  return promptTemplate.format({ context, question });
}

export function initOllamaLLM(model = 'qwen2.5:14b') {
  return new ChatOllama({ model });
}

export function initOllamaEmbeddings(model = 'nomic-embed-text') {
  return new OllamaEmbeddings({ model });
}

export function initChroma(
  collectionName: string = `collection_${Date.now()}`,
  embeddings: EmbeddingsInterface = initOllamaEmbeddings()
) {
  const args: ChromaLibArgs = {
    collectionName,
    url: 'http://localhost:8000',
  };
  return new Chroma(embeddings, args);
}

export async function saveToExcel(data: any[], filePath: string) {
  const workbook = new Workbook();
  const worksheet = workbook.addWorksheet('Sheet1');
  const titles = Object.keys(data[0]);
  worksheet.addRow(titles);
  const values = data.map((item) => Object.values(item));
  values.forEach((value) => {
    worksheet.addRow(value);
  });
  await workbook.xlsx.writeFile(filePath);
}

export async function readFromExcel(filePath: string, sheetIndex: number = 1) {
  const workbook = new Workbook();
  await workbook.xlsx.readFile(filePath);
  const worksheet = workbook.getWorksheet(sheetIndex);
  const texts = [];
  worksheet?.eachRow((row) => {
    texts.push(row.values);
  });
  const keys = texts.shift();
  const values = texts.map((row) => {
    const obj = {};
    row.forEach((value, index) => {
      obj[keys[index]] = value;
    });
    return obj;
  });
  return values;
}
