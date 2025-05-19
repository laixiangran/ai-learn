import {
  initOllamaLLM,
  initChroma,
  initOllamaEmbeddings,
  readJsonFile,
  saveJsonFile,
  formatToJson,
} from '@/app/utils';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';

// 评估版本
const evaluateVersion: string = 'v1.0';

// 基础配置，根据实际情况进行修改
const generateModel = initOllamaLLM('qwen2.5:14b'); // 生成 LLM 模型
const evaluateModel = initOllamaLLM('qwen2.5:14b'); // 评估 LLM 模型
const embeddingModel = initOllamaEmbeddings('nomic-embed-text'); // 向量模型
const collectionName = 'collection_rag_evaluator_04'; // 向量数据集合名称
const chromadb = initChroma(collectionName, embeddingModel); // 向量数据库
let qaTestPath = 'src/app/data/qa_test_10.json'; // 测试数据
const qaEvalPath = `src/app/data/qa_test_10_evaluate_${evaluateVersion}.json`; // 评估数据
const textSplitterParams = {
  chunkSize: 500, // 文本切分大小
  chunkOverlap: 50, // 文本切分重叠大小
};
const commonVectorFilter = {
  category: '少儿编程',
};
// 不同版本对应的配置
const configs: Record<string, any> = {
  'v1.0': {
    topK: 3, // 检索的上下文数量
    vectorFilter: commonVectorFilter, // 检索条件
  },
  'v2.0': {
    topK: 3,
    vectorFilter: undefined,
  },
  'v2.1': {
    topK: 6,
    vectorFilter: undefined,
  },
  'v3.0': {
    topK: 3,
    vectorFilter: commonVectorFilter,
  },
  'v3.1': {
    topK: 6,
    vectorFilter: commonVectorFilter,
  },
  'v4.0': {
    topK: 3,
    vectorFilter: commonVectorFilter,
  },
  'v4.1': {
    topK: 6,
    vectorFilter: commonVectorFilter,
  },
  'v5.0': {
    topK: 3,
    vectorFilter: commonVectorFilter,
  },
  'v5.1': {
    topK: 6,
    vectorFilter: commonVectorFilter,
  },
  'v6.0': {
    topK: 3,
    vectorFilter: commonVectorFilter,
  },
  'v6.1': {
    topK: 6,
    vectorFilter: commonVectorFilter,
  },
};

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
 * 提取出答案所有的关键信息
 * @param text 答案文本
 * @returns
 */
async function statementSplit(text) {
  const prompt = `
你是一个语言专家，你的任务提取出以下答案所有的关键信息。

说明：
1. 严格按以下JSON格式返回：["关键信息1", "关键信息2", ...]，不能输出其他无关内容。

答案：
${text}

回答：

  `;
  const res = await generateModel.invoke(prompt);
  const data = formatToJson(res.content) || [];
  console.log('statementSplit: ', data);
  return data;
}

/**
 * 根据答案推导出多个模拟问题
 * @param text 答案文本
 * @param num 模拟问题数量
 * @returns
 */
async function simulationQuestion(text, num) {
  const prompt = `
你是一个语言专家，你的任务是根据以下答案的核心内容来生成${num}个用户可能问的问题。

说明：
1. 严格按以下JSON格式返回：["问题1", "问题2", ...]，不能输出其他无关内容。

答案：
${text}

回答：

  `;
  const res = await generateModel.invoke(prompt);
  const data = formatToJson(res.content) || [];
  console.log('simulationQuestions: ', data);
  return data;
}

/**
 * 处理问题及回答问题
 * @param evaluateData 评估数据
 * @returns
 */
async function llmAnswerByQaData(evaluateData) {
  const allQuestions = [evaluateData.question];

  // 同义改写
  if (evaluateVersion.includes('v4')) {
    allQuestions.push(...evaluateData.synonymyQuestions);
  }

  // 多视角分解
  if (evaluateVersion.includes('v5')) {
    allQuestions.push(...evaluateData.subQuestions);
  }

  // 补充上下文
  if (evaluateVersion.includes('v6')) {
    allQuestions.push(evaluateData.supplementaryContext);
  }

  const allDocs = [];
  const topK = configs[evaluateVersion]?.topK;
  const vectorFilter = configs[evaluateVersion]?.vectorFilter;
  while (allQuestions.length > 0) {
    const question = allQuestions.shift();
    const docs = await chromadb.similaritySearchWithScore(
      question,
      topK,
      vectorFilter
    );
    allDocs.push(...docs);
  }

  // 根据文档 id 去重并按文档相似度升序排列，最终取 topK 个文档作为上下文
  const uniqueDocs = Array.from(
    new Map(allDocs.map((doc) => [doc[0].id, doc])).values()
  );
  uniqueDocs.sort((a, b) => a[1] - b[1]);
  const retrievedContext = uniqueDocs
    .slice(0, topK)
    .map((doc) => doc[0].pageContent);

  // 构造提示词
  const answerPrompt = `
你是负责回答问题的专家，请严格按照以下检索到的上下文片段来回答问题。如果你不知道答案，就说你不知道。

上下文：
${retrievedContext.join('\n')}

问题：
${evaluateData.question}

回答：

`;

  // LLM 回答
  const answerRes = await generateModel.invoke(answerPrompt);
  const data = { answer: answerRes.content, retrievedContext };
  console.log('llmAnswer: ', data);
  return data;
}

/**
 * 上下文召回率评估器。
 * 实现步骤：
 * 1. 将参考答案拆分成多个关键信息；
 * 2. 逐个分析每个关键信息是否可归因于给定的上下文；
 * 3. 根据每个关键信息的得分，计算上下文召回率。
 * @param evaluateData 评估数据
 * @returns
 */
async function contextRecallEvaluator(evaluateData) {
  // retrievedContext 检索到的上下文
  // referenceAnswer 参考答案
  // referenceAnswerStatements 参考答案拆分出的多个关键信息
  if (!evaluateData.answer) {
    const answerObj = await llmAnswerByQaData(evaluateData);
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
  // 逐个分析每个关键信息是否可归因于给定的上下文
  const newReferenceAnswerStatements = [
    ...evaluateData.referenceAnswerStatements,
  ];
  while (newReferenceAnswerStatements.length > 0) {
    const statement = newReferenceAnswerStatements.shift();
    const prompt = `
你是一个语言专家，你的任务是分析关键信息是否可归因于给定的上下文。

说明：
1. 如果关键信息不能归因于上下文，则得分为0；
2. 如果关键信息能够归因于上下文，则得分为1；
3. 严格按以下JSON格式返回：{"score": "得分"}，不能输出其他无关内容。

关键信息：
${statement}

上下文：
${evaluateData.retrievedContext.join('\n')}

回答：

`;
    const llmRes = await evaluateModel.invoke(prompt);
    const data = formatToJson(llmRes.content);
    allRes.push({
      score: data?.score ? +data.score : 0,
      statement,
    });
    console.log('contextRecallEvaluator: ', allRes);
  }

  // 根据每个关键信息的得分，计算上下文召回率
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
    const data = formatToJson(llmRes.content);
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
 * 1. 将实际答案拆分成多个关键信息；
 * 2. 逐个分析每个关键信息是否可归因于给定的上下文；
 * 3. 根据每个关键信息的得分，计算答案忠实度。
 * @param evaluateData 评估数据
 * @returns
 */
async function faithfulnessEvaluator(evaluateData) {
  // retrievedContext 检索到的上下文
  // answer 实际答案
  // answerStatements 实际答案拆分出的多个关键信息
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
  // 逐个分析每个关键信息是否可归因于给定的上下文
  const newAnswerStatements = [...evaluateData.answerStatements];
  while (newAnswerStatements.length > 0) {
    const statement = newAnswerStatements.shift();
    const prompt = `
你是一个语言专家，你的任务是分析关键信息是否可归因于给定的上下文。

说明：
1. 如果关键信息不能归因于上下文，则得分为0；
2. 如果关键信息能够归因于上下文，则得分为1；
3. 严格按以下JSON格式返回：{"score": "得分"}，不能输出其他无关内容。

关键信息：
${statement}

上下文：
${evaluateData.retrievedContext.join('\n')}

回答：

`;
    const llmRes = await evaluateModel.invoke(prompt);
    const data = formatToJson(llmRes.content);
    allRes.push({
      score: data?.score ? +data.score : 0,
      statement,
    });
    console.log('faithfulnessEvaluator: ', allRes);
  }
  // 根据每个关键信息的得分，计算答案忠实度
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
    const data = formatToJson(llmRes.content);
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
 * 1. 将参考答案拆分成多个关键信息；
 * 2. 逐个分析每个关键信息是否可归因于给定的实际答案；
 * 3. 根据每个关键信息的得分，计算答案正确性。
 * @param evaluateData 评估数据
 * @returns
 */
async function answerCorrectnessEvaluator(evaluateData) {
  // answer 实际答案
  // referenceAnswer 参考答案
  // referenceAnswerStatements 参考答案拆分出的多个关键信息
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
  // 逐个分析每个关键信息是否可归因于给定的实际答案
  const newReferenceAnswerStatements = [
    ...evaluateData.referenceAnswerStatements,
  ];
  while (newReferenceAnswerStatements.length > 0) {
    const statement = newReferenceAnswerStatements.shift();
    const prompt = `
你是一个语言专家，你的任务是分析关键信息是否可归因于给定的实际答案。

说明：
1. 如果关键信息不能归因于实际答案，则得分为0；
2. 如果关键信息能够归因于实际答案，则得分为1；
3. 严格按以下JSON格式返回：{"score": "得分"}，不能输出其他无关内容。

关键信息：
${statement}

实际答案：
${evaluateData.answer}

回答：

`;
    const llmRes = await evaluateModel.invoke(prompt);
    const data = formatToJson(llmRes.content);
    allRes.push({
      score: data?.score ? +data.score : 0,
      statement,
    });
    console.log('answerCorrectnessEvaluator: ', allRes);
  }
  // 根据每个关键信息的得分，计算答案正确性
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
  await saveJsonFile(JSON.stringify(res), qaEvalPath);

  // 计算最终指标数据
  const score =
    res.reduce((score, cur) => {
      score += cur[indexName].score || 0;
      return score;
    }, 0) / res.length;

  return { [indexName]: +score.toFixed(2) };
}

/**
 * 问题优化 - 意图识别
 * @param question 原始问题
 * @returns
 */
async function intentRecognition(question) {
  const prompt = `
你是一个语言专家，你的任务是分析下面的问题是属于哪个领域。

说明：
1. 无法判断时，默认为“少儿编程”；
2. 只需要回答领域名称，不要输出其他内容。

领域列表：
["少儿编程", "低代码"]

问题：
${question}

回答：

  `;
  const res = await generateModel.invoke(prompt);
  const data = res.content;
  console.log('intentRecognition: ', data);
  return data;
}

/**
 * 问题优化 - 同义改写
 * @param question 原始问题
 * @param num 同义改写后的同义问题数量
 * @returns
 */
async function synonymyRewritten(question, num = 3) {
  const prompt = `
你是一个语言专家，你的任务是将给定的原始问题改写成${num}个语义相同但表达方式不同的问题。

说明：
1. 严格按以下JSON格式返回：["问题1", "问题2", ...]，不能输出其他无关内容。

原始问题：
${question}

回答：

  `;
  const res = await generateModel.invoke(prompt);
  const data = formatToJson(res.content) || [];
  console.log('synonymyRewritten: ', data);
  return data;
}

/**
 * 问题优化 - 多视角分解
 * @param question 原始问题
 * @param num 多视角分解后的子问题数量
 * @returns
 */
async function subRewritten(question, num = 3) {
  const prompt = `
你是一个语言专家，你的任务是将给定的原始问题分解成${num}个不同视角的子问题。

说明：
1. 严格按以下JSON格式返回：["问题1", "问题2", ...]，不能输出其他无关内容。

原始问题：
${question}

回答：

  `;
  const res = await generateModel.invoke(prompt);
  const data = formatToJson(res.content) || [];
  console.log('subRewritten: ', data);
  return data;
}

/**
 * 问题优化 - 补充上下文
 * @param question 原始问题
 * @param maxLen 补充上下文的最大字符长度
 * @returns
 */
async function contextSupplement(question, maxLen = 200) {
  const prompt = `
你是一个语言专家，你的任务是根据给定的原始问题，生成一段与原始问题相关的背景信息。

说明：
1. 背景信息最大不超过${maxLen}个字符；
2. 只要输出背景信息，不能输出其他无关内容。

原始问题：
${question}

回答：

  `;
  const res = await generateModel.invoke(prompt);
  const data = res.content;
  console.log('supplementContext: ', data);
  return data;
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
  const defaultIndexs = [
    'contextRecall',
    'contextRelevance',
    'faithfulness',
    'answerRelevance',
    'answerCorrectness',
  ];
  const indexs = [...defaultIndexs];
  while (indexs.length > 0) {
    const indexName = indexs.shift() || '';
    const qaDatas = await readJsonFile(qaTestPath);
    const indexRes = await bathEvaluator(indexName, qaDatas);
    qaTestPath = qaEvalPath;
    console.log(indexName, indexRes);
    data[indexName] = indexRes[indexName];
  }
  console.log(`版本${evaluateVersion}`, '评估得分：', data);
  return Response.json({
    success: true,
    data,
  });
}
