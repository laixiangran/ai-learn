import {
  initOllamaLLM,
  initChroma,
  initOllamaEmbeddings,
  readJsonFile,
  saveJsonFile,
} from '@/app/utils';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';

function formatToJson(res) {
  let result = null;
  try {
    result = JSON.parse(res.content);
  } catch (error) {
    console.error('error: ', error);
  }
  return result;
}

async function statementSplit(text, evaluateLLM) {
  const prompt = `
  你是一个语言专家，你的任务是将以下文本拆分为多个独立的句子，每个句子独立表达一个完整含义，同时保留原意的逻辑连贯性。
  
  说明：
  1. 严格按以下JSON格式返回：["句子1", "句子2", ...]，不能输出其他无关内容。
  
  文本：
  ${text}
  
  回答：
  
  `;
  const res = await evaluateLLM.invoke(prompt);
  const data = formatToJson(res) || [];
  console.log('statements: ', data);
  return data;
}

async function batchStatementSplit(dim: string) {
  const llm = initOllamaLLM('qwen2.5:14b');
  const path = 'src/app/data/qa_test_20_base_evaluate.json';
  const datas = await readJsonFile(path);
  const res = [];
  while (datas.length > 0) {
    const data = datas.shift();
    const statements = await statementSplit(data[dim], llm);
    res.push({
      ...data,
      [`${dim}Statements`]: statements,
    });
  }
  // 将 LLM 回答结果保存到文件中
  await saveJsonFile(JSON.stringify(res), path);
  return res;
}

async function simulationQuestion(text, evaluateLLM) {
  const prompt = `
  你是一个语言专家，你的任务是根据以下答案的核心内容来生成3个用户可能问的问题。
  
  说明：
  1. 严格按以下JSON格式返回：["问题1", "问题2", ...]，不能输出其他无关内容。
  
  答案：
  ${text}
  
  回答：
  
  `;
  const res = await evaluateLLM.invoke(prompt);
  const data = formatToJson(res) || [];
  console.log('questions: ', data);
  return data;
}

async function batchSimulationQuestion() {
  const llm = initOllamaLLM('qwen2.5:14b');
  const path = 'src/app/data/qa_test_20_base_evaluate.json';
  const datas = await readJsonFile(path);
  const res = [];
  while (datas.length > 0) {
    const data = datas.shift();
    const questions = await simulationQuestion(data.answer, llm);
    res.push({
      ...data,
      simulationQuestions: questions,
    });
  }
  // 将 LLM 回答结果保存到文件中
  await saveJsonFile(JSON.stringify(res), path);
  return res;
}

async function loadPdf(filePath) {
  const loader = new PDFLoader(filePath, {
    splitPages: false,
  });
  const docs = await loader.load();
  return docs;
}

async function splitDocuments(docs) {
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  });
  return await textSplitter.splitDocuments(docs);
}

async function addDocuments(texts) {
  const collectionName = 'collection_rag_evaluator';
  const embeddings = initOllamaEmbeddings('nomic-embed-text');
  const chromadb = initChroma(collectionName, embeddings);
  const res = await chromadb.addDocuments(texts);
  return {
    chromadb,
    documents: res,
  };
}

async function llmAnswerByQaData() {
  const chromadb = initChroma(
    'collection_rag_evaluator',
    initOllamaEmbeddings('nomic-embed-text')
  );
  const llm = initOllamaLLM('qwen2.5:14b');
  const testQaData = await readJsonFile('src/app/data/qa_test_20.json');
  const res = [];
  while (testQaData.length > 0) {
    const testQa: any = testQaData[0];
    const { question, referenceAnswer } = testQa;

    // 检索上下文
    const docs = await chromadb.similaritySearchWithScore(question, 3);
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
    const answerRes = await llm.invoke(answerPrompt);
    const result = {
      question,
      retrievedContext,
      answer: answerRes.content,
      referenceAnswer,
    };
    res.push(result);
    testQaData.shift();
  }
  // 将 LLM 回答结果保存到文件中
  await saveJsonFile(
    JSON.stringify(res),
    'src/app/data/qa_test_20_base_evaluate.json'
  );
  return res;
}

/**
 * 上下文召回率评估器
 * 实现步骤：
 * 1. 将参考答案拆分成多个句子；
 * 2. 逐个分析每个句子是否可归因于给定的上下文；
 * 3. 计算上下文召回率。上下文召回率 = 上下文可归因的参考答案句子数量 / 参考答案总句子数量；
 * 4. 上下文召回率取值在 0 到 1 之间，数值越高表示检索到的上下文覆盖越全面。
 * @param evaluateData 评估数据
 * @param evaluateLLM 评估 LLM
 * @returns
 */
async function contextRecallEvaluator(evaluateData, evaluateLLM) {
  const { retrievedContext, referenceAnswer, referenceAnswerStatements } =
    evaluateData;
  let newStatements = [];
  if (!referenceAnswerStatements) {
    newStatements = await statementSplit(referenceAnswer, evaluateLLM);
  } else {
    newStatements = [...referenceAnswerStatements];
  }
  const allRes = [];
  while (newStatements.length > 0) {
    const statement = newStatements.shift();
    const prompt = `
你是一个语言专家，你的任务是分析下面的句子是否可归因于提供的上下文并输出得分。

说明：
1. 如果句子完全不能归因于上下文，则得分为0；
2. 如果句子部分归因于上下文，则得分为0.5；
3. 如果句子完全能归因于上下文，则得分为1；
4. 严格按以下JSON格式返回：{"score": "得分"}，不能输出其他无关内容。

句子：
${statement}

上下文：
${retrievedContext.join('\n')}

回答：

`;
    const llmRes = await evaluateLLM.invoke(prompt);
    const data = formatToJson(llmRes);
    allRes.push({
      score: data?.score ? +data.score : 0,
      statement,
    });
    console.log('allRes: ', allRes);
  }
  const score =
    allRes.reduce((score, cur) => {
      score += +cur.score;
      return score;
    }, 0) / allRes.length;
  return {
    score,
    data: allRes,
  };
}

/**
 * 上下文相关性评估器
 * 实现步骤：
 * 1. 分析并获取上下文中与问题相关的片段数量；
 * 2. 计算上下文相关性。上下文相关性 = 上下文中与问题相关的片段数量 / 上下文中片段总数量；
 * 3. 上下文相关性：取值在 0 到 1 之间，数值越高表示检索到的上下文相关性越高。
 * @param evaluateData 评估数据
 * @param evaluateLLM 评估 LLM
 * @returns
 */
async function contextRelevanceEvaluator(evaluateData, evaluateLLM) {
  const { question, retrievedContext } = evaluateData;
  const newRetrievedContext = [...retrievedContext];
  const allRes = [];
  while (newRetrievedContext.length > 0) {
    const context = newRetrievedContext.shift();
    const prompt = `
你是一个语言专家，你的任务是确定上下文是否包含问题的相关信息并输出得分，不要依赖你以前对这个问题的了解，严格按照问题和提供的上下文来回答。

说明：
1. 如果上下文不包含任何与问题相关的信息，则得分为0；
2. 如果上下文部分包含与问题相关的信息，则得分为0.5；
3. 如果上下文完全包含与问题相关的信息，则得分为1；
4. 严格按以下JSON格式返回：{"score": "得分"}，不能输出其他无关内容。

问题：
${question}

上下文：
${context}

回答：

`;
    const llmRes = await evaluateLLM.invoke(prompt);
    const data = formatToJson(llmRes);
    allRes.push({
      score: data?.score ? +data.score : 0,
      context,
    });
    console.log('allRes: ', allRes);
  }
  const score =
    allRes.reduce((score, cur) => {
      score += +cur.score;
      return score;
    }, 0) / allRes.length;
  return {
    score,
    data: allRes,
  };
}

/**
 * 答案忠实度评估器
 * 实现步骤：
 * 1. 将实际答案拆分成多个句子；
 * 2. 逐个分析每个句子是否可归因于给定的上下文；
 * 3. 计算答案忠实度：答案忠实度 = 上下文可归因的答案句子数量 / 答案总句子数量；
 * 4. 答案忠实度取值在 0 到 1 之间，数值越高表示实际答案越严格基于检索到的上下文。
 * @param evaluateData 评估数据
 * @param evaluateLLM 评估 LLM
 * @returns
 */
async function faithfulnessEvaluator(evaluateData, evaluateLLM) {
  const { retrievedContext, answer, answerStatements } = evaluateData;
  let newStatements = [];
  if (!answerStatements) {
    newStatements = await statementSplit(answer, evaluateLLM);
  } else {
    newStatements = [...answerStatements];
  }
  const allRes = [];
  while (newStatements.length > 0) {
    const statement = newStatements.shift();
    const prompt = `
你是一个语言专家，你的任务是分析下面的句子是否可归因于提供的上下文并输出得分。

说明：
1. 如果句子完全不能归因于上下文，则得分为0；
2. 如果句子部分归因于上下文，则得分为0.5；
3. 如果句子完全能归因于上下文，则得分为1；
4. 严格按以下JSON格式返回：{"score": "得分"}，不能输出其他无关内容。

句子：
${statement}

上下文：
${retrievedContext.join('\n')}

回答：

`;
    const llmRes = await evaluateLLM.invoke(prompt);
    const data = formatToJson(llmRes);
    allRes.push({
      score: data?.score ? +data.score : 0,
      statement,
    });
    console.log('allRes: ', allRes);
  }
  const score =
    allRes.reduce((score, cur) => {
      score += +cur.score;
      return score;
    }, 0) / allRes.length;
  return {
    score,
    data: allRes,
  };
}

/**
 * 答案相关性评估器
 * 实现步骤：
 * 1. 根据实际答案推导出多个模拟问题；
 * 2. 逐个分析每个模拟问题是否与原问题相关；
 * 3. 计算答案相关性：答案相关性 = 与实际问题相关的模拟问题数量 / 实际答案推导出的模拟问题总数量；
 * 4. 答案相关性取值在 0 到 1 之间，数值越高表示实际答案更直接完整回答用户问题。
 * @param evaluateData 评估数据
 * @param evaluateLLM 评估 LLM
 * @returns
 */
async function answerRelevanceEvaluator(evaluateData, evaluateLLM) {
  const { question, answer, simulationQuestions } = evaluateData;
  let newSimulationQuestions = [];
  if (!simulationQuestions) {
    newSimulationQuestions = await simulationQuestion(answer, evaluateLLM);
  } else {
    newSimulationQuestions = [...simulationQuestions];
  }
  const allRes = [];
  while (newSimulationQuestions.length > 0) {
    const simulationQuestion = newSimulationQuestions.shift();
    const prompt = `
你是一个语言专家，你的任务是分析模拟问题和实际问题表达的意思是否相似。

说明：
1. 如果模拟问题与实际问题不相似，则得分为0；
2. 如果模拟问题与实际问题部分相似，则得分为0.5；
3. 如果模拟问题与实际问题相似，则得分为1；
4. 严格按以下JSON格式返回：{"score": "相似度"}，不能输出其他无关内容。

模拟问题：
${simulationQuestion}

实际问题：
${question}

回答：

`;
    const llmRes = await evaluateLLM.invoke(prompt);
    const data = formatToJson(llmRes);
    allRes.push({
      score: data?.score ? +data.score : 0,
      simulationQuestion,
    });
    console.log('allRes: ', allRes);
  }
  const score =
    allRes.reduce((score, cur) => {
      score += +cur.score;
      return score;
    }, 0) / allRes.length;
  return {
    score,
    data: allRes,
  };
}

/**
 * 答案正确率评估器
 * 实现步骤：
 * 1. 将参考答案拆分成多个句子；
 * 2. 逐个分析每个句子是否可归因于给定的答案；
 * 3. 计算答案准确性。答案准确性 = 实际答案覆盖的关键信息数量 / 参考答案中关键信息总数量；
 * 4. 答案准确性取值在 0 到 1 之间，数值越高表示实际答案与参考答案匹配度越高，准确性也就越高。
 * @param evaluateData 评估数据
 * @param evaluateLLM 评估 LLM
 * @returns
 */
async function answerCorrectnessEvaluator(evaluateData, evaluateLLM) {
  const { answer, referenceAnswer, referenceAnswerStatements } = evaluateData;
  let newStatements = [];
  if (!referenceAnswerStatements) {
    newStatements = await statementSplit(referenceAnswer, evaluateLLM);
  } else {
    newStatements = [...referenceAnswerStatements];
  }
  const allRes = [];
  while (newStatements.length > 0) {
    const statement = newStatements.shift();
    const prompt = `
你是一个语言专家，你的任务是分析下面的句子是否可归因于提供的上下文并输出得分。

说明：
1. 如果句子完全不能归因于上下文，则得分为0；
2. 如果句子部分归因于上下文，则得分为0.5；
3. 如果句子完全能归因于上下文，则得分为1；
4. 严格按以下JSON格式返回：{"score": "得分"}，不能输出其他无关内容。

句子：
${statement}

上下文：
${answer}

回答：

`;
    const llmRes = await evaluateLLM.invoke(prompt);
    const data = formatToJson(llmRes);
    allRes.push({
      score: data?.score ? +data.score : 0,
      statement,
    });
    console.log('allRes: ', allRes);
  }
  const score =
    allRes.reduce((score, cur) => {
      score += +cur.score;
      return score;
    }, 0) / allRes.length;
  return {
    score,
    data: allRes,
  };
}

async function bathEvaluator(indexName: string) {
  const evaluatorMap = {
    contextRecall: contextRecallEvaluator,
    contextRelevance: contextRelevanceEvaluator,
    faithfulness: faithfulnessEvaluator,
    answerRelevance: answerRelevanceEvaluator,
    answerCorrectness: answerCorrectnessEvaluator,
  };
  const evaluator = evaluatorMap[indexName];
  if (!evaluator) {
    return { [indexName]: 0 };
  }
  const path = 'src/app/data/qa_test_20_base_evaluate.json';
  const datas = await readJsonFile(path);
  let res = [];
  // 删除数据中已有的指标评估数据并重新计算
  // datas.forEach((data) => {
  //   delete data[indexName];
  // });
  if (datas[0]?.[indexName]) {
    res = datas;
  } else {
    const llm = initOllamaLLM('qwen2.5:14b');
    while (datas.length > 0) {
      const data = datas.shift();
      const evaluateRes = await evaluator(data, llm);
      res.push({
        ...data,
        [indexName]: evaluateRes,
      });
    }

    // 将 LLM 回答结果保存到文件中
    await saveJsonFile(JSON.stringify(res), path);
  }

  // 计算最终上下文召回率
  const score =
    res.reduce((score, cur) => {
      score += cur[indexName].score;
      return score;
    }, 0) / res.length;

  return { [indexName]: +score.toFixed(1) };
}

/**
 * RAG 评估系统实现
 * @param request
 * @returns
 */
export async function GET(request: Request) {
  // 1. 文件解析
  // const docs = await loadPdf(
  //   'src/app/data/2024少儿编程教育行业发展趋势报告.pdf'
  // );

  // 2. 文件切分
  // const texts = await splitDocuments(docs);

  // 3. 初始化向量模型和向量数据库，并将文档存储到向量数据库
  // const { chromadb, documents } = await addDocuments(texts);

  // 4. LLM 基于 QA 测试数据集回答
  // const llmAnswerRes = await llmAnswerByQaData();

  // 5.1 将参考答案批量拆分成多个句子
  // await batchStatementSplit('referenceAnswer');

  // 5.2 将实际答案批量拆分成多个句子
  // await batchStatementSplit('answer');

  // 5.3 将实际答案批量推导出多个模拟问题
  // await batchSimulationQuestion();

  // 6. 指标评估
  const indexs = [
    'contextRecall',
    'contextRelevance',
    'faithfulness',
    'answerRelevance',
    'answerCorrectness',
  ];
  const data: any = {};
  while (indexs.length > 0) {
    const indexName = indexs.shift() || '';
    const indexRes = await bathEvaluator(indexName);
    data[indexName] = indexRes[indexName];
  }

  return Response.json({
    success: true,
    data,
  });
}
