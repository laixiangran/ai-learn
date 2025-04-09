import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { initOllamaLLM, saveToExcel, readFromExcel } from '@/app/utils';
import { v4 as uuidv4 } from 'uuid';

/**
 * 数据加载
 * @param request
 * @returns
 */
export async function GET(request: Request) {
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
  texts.forEach((text) => {
    // 给每个文档生成uuid
    text.metadata.uuid = uuidv4();
  });

  // 文档持久化
  const baseTextPath = 'src/app/data/doc_base.xlsx';
  await saveToExcel(
    texts.map((text) => {
      return {
        doc: JSON.stringify(text),
      };
    }),
    baseTextPath
  );

  // 读取切分后的文档
  const baseTextData = await readFromExcel(baseTextPath);
  baseTextData.forEach((text) => {
    text.doc = JSON.parse(text.doc);
  });

  // QA 数据集抽取
  const allData = [];
  const errorData = [];
  const baseQaPath = `src/app/data/qa_base.xlsx`;
  const errorTextPath = `src/app/data/doc_error.xlsx`;
  // 本地资源有限，因此单个处理完再处理下一个
  while (baseTextData.length > 0) {
    const baseText = baseTextData[0];
    try {
      const { pageContent: document, metadata } = baseText.doc;
      const prompt = `
      我会给你一段文本，你需要阅读这段文本，分别针对这段文本生成8个问题、用户回答这个问题的上下文，和基于上下文对问题的回答。

      说明：
      1. 问题要与这段文本相关，不要询问类似“这个问题的答案在哪一章”这样的问题；
      2. 上下文必须与原始文本的内容保持一致，不要进行缩写、扩写、改写、摘要、替换词语等；
      3. 答案请保持完整且简洁，无须重复问题。答案要能够独立回答问题，而不是引用现有的章节、页码等；
      4. 返回结果以JSON形式组织，格式为[{"question": "...", "context": ..., "answer": "..."}, ...]；
      5. 如果当前文本主要是目录，或者是一些人名、地址、电子邮箱等没有办法生成有意义的问题时，可以返回[]。

      文本：
      ${document}

      回答：
      `;
      const ollamaLLM = initOllamaLLM();
      const res = await ollamaLLM.invoke(prompt);
      const regex = new RegExp('\\[(.*?)\\]', 's');
      const match = res.content.match(regex);
      if (match) {
        const data = JSON.parse(`[${match[1]}]`);
        data.forEach((item) => {
          item.doc = document;
          item.uuid = metadata.uuid;
        });
        allData.push(data);

        // 保存QA数据集
        await saveToExcel(allData.flat(), baseQaPath);
        baseTextData.shift();
      }
    } catch (error) {
      console.log('error', error);
      errorData.push({ doc: JSON.stringify(baseText.doc) });
      await saveToExcel(errorData, errorTextPath);
      baseTextData.shift();
    }
  }

  // QA 数据集质量检查
  const checkQaPath = 'src/app/data/qa_grade.xlsx';
  const baseQaData = await readFromExcel(baseQaPath);
  const allQas = [];
  while (baseQaData.length > 0) {
    const baseQa = baseQaData[0];
    const prompt = `
      你是一个优秀的教师，你的任务是根据问题和参考答案来进行打分。

      说明：
      1. 你需要根据所出的问题以及参考答案进行打分，并给出打分理由，分值是一个int类型的值，取值范围为1-5；
      2. 好的问题，应该是询问事实、观点等，而不是类似于“这一段描述了什么”；
      3. 好的答案，应该能够直接回答问题，而不是给出在原文中的引用，例如“在第3页中”等；
      4. 结果请以JSON形式组织，格式为如下：{"score": ..., "reason": ...}。

      问题：
      ${baseQa.question}

      参考答案：
      ${baseQa.answer}

      请打分：
      `;
    const ollamaLLM = initOllamaLLM();
    const res = await ollamaLLM.invoke(prompt);
    const regex = new RegExp('\\{(.*?)\\}', 's');
    const match = res.content.match(regex);
    if (match) {
      const data = JSON.parse(`{${match[1]}}`);
      // 保存QA数据集
      allQas.push({ ...baseQa, ...data });
      await saveToExcel(allQas, checkQaPath);
    }
    baseQaData.shift();
  }

  // 筛选出分数大于3的记录并分为测试集和训练集
  const data = await readFromExcel(checkQaPath);
  const testData = [];
  const trainData = [];
  data.forEach((item, i: number) => {
    if (item.score > 3) {
      if (i % 2 === 0 && testData.length < 100) {
        testData.push(item);
      } else {
        trainData.push(item);
      }
    }
  });
  saveToExcel(testData, 'src/app/data/qa_test.xlsx');
  saveToExcel(trainData, 'src/app/data/qa_train.xlsx');

  return Response.json({
    success: true,
    data: 'qa数据集处理中...',
  });
}
