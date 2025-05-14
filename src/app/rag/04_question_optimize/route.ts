import {
  initOllamaLLM,
  initChroma,
  initOllamaEmbeddings,
  readJsonFile,
  saveJsonFile,
} from '@/app/utils';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';

// 基础配置，根据实际情况进行修改
const generateModel = initOllamaLLM('qwen2.5:14b'); // 生成 LLM 模型
const evaluateModel = initOllamaLLM('qwen2.5:14b'); // 评估 LLM 模型
const embeddingModel = initOllamaEmbeddings('nomic-embed-text'); // 向量模型
const collectionName = 'collection_rag_evaluator_04'; // 向量数据集合名称
const chromadb = initChroma(collectionName, embeddingModel); // 向量数据库
const qaPath = 'src/app/data/qa_test_20_evaluate_v1.1.json'; // 评估数据
const textSplitterParams = {
  chunkSize: 500, // 文本切分大小
  chunkOverlap: 50, // 文本切分重叠大小
};
const topK = 3; // 检索的上下文数量
const vectorFilter = undefined; // 向量查询过滤条件，默认不使用
// const vectorFilter = {
//   category: '少儿编程',
// };

function formatToJson(res) {
  let result = null;
  try {
    result = JSON.parse(res.content);
  } catch (error) {
    console.error('error: ', error);
  }
  return result;
}

/**
 * 将文本拆分为多个句子（关键信息）
 * @param text 待拆分的文本
 * @returns
 */
async function statementSplit(text) {
  const prompt = `
  你是一个语言专家，你的任务是将以下文本拆分为多个独立的句子，每个句子独立表达一个完整含义，同时保留原意的逻辑连贯性。
  
  说明：
  1. 严格按以下JSON格式返回：["句子1", "句子2", ...]，不能输出其他无关内容。
  
  文本：
  ${text}
  
  回答：
  
  `;
  const res = await generateModel.invoke(prompt);
  const data = formatToJson(res) || [];
  console.log('statementSplit: ', data);
  return data;
}

/**
 * 根据答案推导出多个模拟问题
 * @param text
 * @returns
 */
async function simulationQuestion(text) {
  const prompt = `
  你是一个语言专家，你的任务是根据以下答案的核心内容来生成3个用户可能问的问题。
  
  说明：
  1. 严格按以下JSON格式返回：["问题1", "问题2", ...]，不能输出其他无关内容。
  
  答案：
  ${text}
  
  回答：
  
  `;
  const res = await generateModel.invoke(prompt);
  const data = formatToJson(res) || [];
  console.log('simulationQuestions: ', data);
  return data;
}

/**
 * pdf文件解析
 * @param filePath
 * @returns
 */
async function loadPdf(filePath: string) {
  const loader = new PDFLoader(filePath, {
    splitPages: false,
  });
  const docs = await loader.load();
  return docs;
}

/**
 * 文件切分
 * @param docs
 * @returns
 */
async function splitDocuments(docs) {
  const textSplitter = new RecursiveCharacterTextSplitter(textSplitterParams);
  return await textSplitter.splitDocuments(docs);
}

/**
 * 初始化向量模型和向量数据库，并将文档存储到向量数据库
 * @param texts
 * @returns
 */
async function addDocuments(texts) {
  const res = await chromadb.addDocuments(texts);
  return {
    chromadb,
    documents: res,
  };
}

async function llmAnswerByQaData(question: string) {
  // 检索上下文
  const docs = await chromadb.similaritySearchWithScore(
    question,
    topK,
    vectorFilter
  );
  const retrievedContext = docs.map((doc) => doc[0].pageContent);

  // 构造提示词
  const answerPrompt = `
你是负责回答问题的专家，请严格按照以下检索到的上下文片段来回答问题。如果你不知道答案，就说你不知道。

上下文：
${retrievedContext.join('\n')}

问题：
${question}

回答：

`;

  // LLM 回答
  const answerRes = await generateModel.invoke(answerPrompt);
  return { answer: answerRes.content, retrievedContext };
}

/**
 * 上下文召回率评估器。
 * 实现步骤：
 * 1. 将参考答案拆分成多个句子（关键信息）；
 * 2. 逐个分析每个句子（关键信息）是否可归因于给定的上下文；
 * 3. 根据每个句子（关键信息）的得分，计算上下文召回率。
 * @param evaluateData 评估数据
 * @param evaluateLLM 评估 LLM
 * @returns
 */
async function contextRecallEvaluator(evaluateData, evaluateLLM) {
  // retrievedContext 检索到的上下文
  // referenceAnswer 参考答案
  // referenceAnswerStatements 参考答案拆分出的多个句子（关键信息）
  if (!evaluateData.answer) {
    const answerObj = await llmAnswerByQaData(evaluateData.question);
    evaluateData = {
      ...evaluateData,
      ...answerObj,
    };
  }
  if (!evaluateData.referenceAnswerStatements) {
    evaluateData.referenceAnswerStatements = await statementSplit(
      evaluateData.referenceAnswer
    );
  }
  const allRes = [];
  // 逐个分析每个句子（关键信息）是否可归因于给定的上下文
  const newReferenceAnswerStatements = [
    ...evaluateData.referenceAnswerStatements,
  ];
  while (newReferenceAnswerStatements.length > 0) {
    const statement = newReferenceAnswerStatements.shift();
    const prompt = `
你是一个语言专家，你的任务是分析句子是否可归因于给定的上下文。

说明：
1. 如果句子不能归因于上下文，则得分为0；
2. 如果句子能够归因于上下文，则得分为1；
3. 严格按以下JSON格式返回：{"score": "得分"}，不能输出其他无关内容。

句子：
${statement}

上下文：
${evaluateData.retrievedContext.join('\n')}

回答：

`;
    const llmRes = await evaluateModel.invoke(prompt);
    const data = formatToJson(llmRes);
    allRes.push({
      score: data?.score ? +data.score : 0,
      statement,
    });
    console.log('contextRecallEvaluator: ', allRes);
  }

  // 根据每个句子（关键信息）的得分，计算上下文召回率
  const score =
    allRes.reduce((score, cur) => {
      score += +cur.score;
      return score;
    }, 0) / allRes.length;

  return {
    evaluateData,
    evaluateResult: {
      score,
      data: allRes,
    },
  };
}

/**
 * 上下文相关性评估器
 * 实现步骤：
 * 1. 逐个分析每个上下文片段是否与问题相关；
 * 2. 根据每个上下文片段的得分，计算上下文相关性。
 * @param evaluateData 评估数据
 * @returns
 */
async function contextRelevanceEvaluator(evaluateData) {
  // question 问题
  // retrievedContext 检索到的上下文
  if (!evaluateData.answer) {
    const answerObj = await llmAnswerByQaData(evaluateData.question);
    evaluateData = {
      ...evaluateData,
      ...answerObj,
    };
  }
  const allRes = [];
  // 逐个分析每个上下文片段是否与问题相关
  const newRetrievedContext = [...evaluateData.retrievedContext];
  while (newRetrievedContext.length > 0) {
    const context = newRetrievedContext.shift();
    const prompt = `
你是一个语言专家，你的任务是确定上下文是否与问题有关。

说明：
1. 如果上下文与问题无关，则得分为0；
2. 如果上下文与问题有关，则得分为1；
3. 严格按以下JSON格式返回：{"score": "得分"}，不能输出其他无关内容。

问题：
${evaluateData.question}

上下文：
${context}

回答：

`;
    const llmRes = await evaluateModel.invoke(prompt);
    const data = formatToJson(llmRes);
    allRes.push({
      score: data?.score ? +data.score : 0,
      context,
    });
    console.log('contextRelevanceEvaluator: ', allRes);
  }
  // 根据每个上下文片段的得分，计算上下文相关性
  const score =
    allRes.reduce((score, cur) => {
      score += +cur.score;
      return score;
    }, 0) / allRes.length;
  return {
    evaluateData,
    evaluateResult: {
      score,
      data: allRes,
    },
  };
}

/**
 * 答案忠实度评估器
 * 实现步骤：
 * 1. 将实际答案拆分成多个句子（事实）；
 * 2. 逐个分析每个句子（事实）是否可归因于给定的上下文；
 * 3. 根据每个句子（事实）的得分，计算答案忠实度。
 * @param evaluateData 评估数据
 * @returns
 */
async function faithfulnessEvaluator(evaluateData) {
  // retrievedContext 检索到的上下文
  // answer 实际答案
  // answerStatements 实际答案拆分出的多个句子（事实）
  if (!evaluateData.answer) {
    const answerObj = await llmAnswerByQaData(evaluateData.question);
    evaluateData = {
      ...evaluateData,
      ...answerObj,
    };
  }
  if (!evaluateData.answerStatements) {
    evaluateData.answerStatements = await statementSplit(evaluateData.answer);
  }
  const allRes = [];
  // 逐个分析每个句子（事实）是否可归因于给定的上下文
  const newAnswerStatements = [...evaluateData.answerStatements];
  while (newAnswerStatements.length > 0) {
    const statement = newAnswerStatements.shift();
    const prompt = `
你是一个语言专家，你的任务是分析句子是否可归因于给定的上下文。

说明：
1. 如果句子不能归因于上下文，则得分为0；
2. 如果句子能够归因于上下文，则得分为1；
3. 严格按以下JSON格式返回：{"score": "得分"}，不能输出其他无关内容。

句子：
${statement}

上下文：
${evaluateData.retrievedContext.join('\n')}

回答：

`;
    const llmRes = await evaluateModel.invoke(prompt);
    const data = formatToJson(llmRes);
    allRes.push({
      score: data?.score ? +data.score : 0,
      statement,
    });
    console.log('faithfulnessEvaluator: ', allRes);
  }
  // 根据每个句子（事实）的得分，计算答案忠实度
  const score =
    allRes.reduce((score, cur) => {
      score += +cur.score;
      return score;
    }, 0) / allRes.length;
  return {
    evaluateData,
    evaluateResult: {
      score,
      data: allRes,
    },
  };
}

/**
 * 答案相关性评估器
 * 实现步骤：
 * 1. 根据实际答案推导出多个模拟问题；
 * 2. 逐个分析每个模拟问题是否与原问题相似；
 * 3. 根据每个模拟问题的得分，计算答案相关性。
 * @param evaluateData 评估数据
 * @returns
 */
async function answerRelevanceEvaluator(evaluateData) {
  // question 问题
  // answer 实际答案
  // simulationQuestions 根据实际答案推导出的多个模拟问题
  if (!evaluateData.answer) {
    const answerObj = await llmAnswerByQaData(evaluateData.question);
    evaluateData = {
      ...evaluateData,
      ...answerObj,
    };
  }
  if (!evaluateData.simulationQuestions) {
    evaluateData.simulationQuestions = await simulationQuestion(
      evaluateData.answer
    );
  }
  const allRes = [];
  // 逐个分析每个模拟问题是否与原问题相似
  const newSimulationQuestions = [...evaluateData.simulationQuestions];
  while (newSimulationQuestions.length > 0) {
    const simulationQuestion = newSimulationQuestions.shift();
    const prompt = `
你是一个语言专家，你的任务是分析模拟问题和实际问题是否相似。

说明：
1. 如果模拟问题与实际问题不相似，则得分为0；
2. 如果模拟问题与实际问题相似，则得分为1；
3. 严格按以下JSON格式返回：{"score": "相似度"}，不能输出其他无关内容。

模拟问题：
${simulationQuestion}

实际问题：
${evaluateData.question}

回答：

`;
    const llmRes = await evaluateModel.invoke(prompt);
    const data = formatToJson(llmRes);
    allRes.push({
      score: data?.score ? +data.score : 0,
      simulationQuestion,
    });
    console.log('answerRelevanceEvaluator: ', allRes);
  }
  // 根据每个模拟问题的得分，计算答案相关性
  const score =
    allRes.reduce((score, cur) => {
      score += +cur.score;
      return score;
    }, 0) / allRes.length;
  return {
    evaluateData,
    evaluateResult: {
      score,
      data: allRes,
    },
  };
}

/**
 * 答案正确性评估器
 * 实现步骤：
 * 1. 将参考答案拆分成多个句子（关键信息）；
 * 2. 逐个分析每个句子（关键信息）是否可归因于给定的实际答案；
 * 3. 根据每个句子（关键信息）的得分，计算答案正确性。
 * @param evaluateData 评估数据
 * @returns
 */
async function answerCorrectnessEvaluator(evaluateData) {
  // answer 实际答案
  // referenceAnswer 参考答案
  // referenceAnswerStatements 参考答案拆分出的多个句子（关键信息）
  if (!evaluateData.answer) {
    const answerObj = await llmAnswerByQaData(evaluateData.question);
    evaluateData = {
      ...evaluateData,
      ...answerObj,
    };
  }
  if (!evaluateData.referenceAnswerStatements) {
    evaluateData.referenceAnswerStatements = await statementSplit(
      evaluateData.referenceAnswer
    );
  }
  const allRes = [];
  // 逐个分析每个句子（关键信息）是否可归因于给定的实际答案
  const newReferenceAnswerStatements = [
    ...evaluateData.referenceAnswerStatements,
  ];
  while (newReferenceAnswerStatements.length > 0) {
    const statement = newReferenceAnswerStatements.shift();
    const prompt = `
你是一个语言专家，你的任务是分析句子是否可归因于给定的实际答案。

说明：
1. 如果句子不能归因于实际答案，则得分为0；
2. 如果句子能够归因于实际答案，则得分为1；
3. 严格按以下JSON格式返回：{"score": "得分"}，不能输出其他无关内容。

句子：
${statement}

实际答案：
${evaluateData.answer}

回答：

`;
    const llmRes = await evaluateModel.invoke(prompt);
    const data = formatToJson(llmRes);
    allRes.push({
      score: data?.score ? +data.score : 0,
      statement,
    });
    console.log('answerCorrectnessEvaluator: ', allRes);
  }
  // 根据每个句子（关键信息）的得分，计算答案正确性
  const score =
    allRes.reduce((score, cur) => {
      score += +cur.score;
      return score;
    }, 0) / allRes.length;
  return {
    evaluateData,
    evaluateResult: {
      score,
      data: allRes,
    },
  };
}

async function bathEvaluator(indexName: string, qaDatas: any) {
  const evaluatorMap = {
    contextRecall: contextRecallEvaluator,
    contextRelevance: contextRelevanceEvaluator,
    faithfulness: faithfulnessEvaluator,
    answerRelevance: answerRelevanceEvaluator,
    answerCorrectness: answerCorrectnessEvaluator,
  };
  const evaluator = evaluatorMap[indexName];

  const res = [];
  while (qaDatas.length > 0) {
    const data = qaDatas.shift();
    if (!data[indexName]) {
      const { evaluateData, evaluateResult } = await evaluator(data);
      res.push({
        ...data,
        ...evaluateData,
        [indexName]: evaluateResult,
      });
    } else {
      res.push(data);
    }
  }

  // 将 LLM 回答结果保存到文件中
  await saveJsonFile(JSON.stringify(res), qaPath);

  // 计算最终指标数据
  const score =
    res.reduce((score, cur) => {
      score += cur[indexName].score || 0;
      return score;
    }, 0) / res.length;

  return { [indexName]: +score.toFixed(2) };
}

/**
 * 知识库构建
 * @returns
 */
async function knowledgeConstruction() {
  const pdfs = [
    {
      path: 'src/app/data/2024少儿编程教育行业发展趋势报告.pdf',
      category: '少儿编程',
    },
    {
      path: 'src/app/data/2021年低代码行业研究报告.pdf',
      category: '低代码',
    },
  ];
  for (const pdf of pdfs) {
    const { path, category } = pdf;
    const pdfContent = await loadPdf(path);
    const documents = await splitDocuments(pdfContent);
    for (const document of documents) {
      document.metadata.category = category;
    }
    await addDocuments(documents);
  }
}

/**
 * RAG 评估系统实现
 * @param request
 * @returns
 */
export async function GET(request: Request) {
  // 1. 知识库构建
  // await knowledgeConstruction();

  // 2. 指标评估
  const data: any = {};
  const indexs = [
    'contextRecall',
    'contextRelevance',
    'faithfulness',
    'answerRelevance',
    'answerCorrectness',
  ];
  while (indexs.length > 0) {
    const indexName = indexs.shift() || '';
    const qaDatas = await readJsonFile(qaPath);
    const indexRes = await bathEvaluator(indexName, qaDatas);
    data[indexName] = indexRes[indexName];
  }
  console.log('评估得分：', data);
  return Response.json({
    success: true,
    data,
  });
}
