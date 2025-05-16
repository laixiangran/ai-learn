import { formatToJson } from '@/app/utils';

/**
 * 提取出答案所有的关键信息
 * @param text 答案文本
 * @param generateModel 生成模型
 * @returns
 */
export async function statementSplit(text, generateModel) {
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
  return data;
}

/**
 * 根据答案推导出多个模拟问题
 * @param text 答案文本
 * @param num 模拟问题数量
 * @param generateModel 生成模型
 * @returns
 */
export async function simulationQuestion(text, num, generateModel) {
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
  return data;
}

/**
 * 上下文召回率评估器。
 * 实现步骤：
 * 1. 将参考答案拆分成多个关键信息；
 * 2. 逐个分析每个关键信息是否可归因于给定的上下文；
 * 3. 根据每个关键信息的得分，计算上下文召回率。
 * @param evaluateData 评估数据
 * @param evaluateModel 评估模型
 * @returns
 */
export async function contextRecallEvaluator(evaluateData, evaluateModel) {
  // retrievedContext：检索到的上下文
  // referenceAnswerStatements：参考答案拆分出的多个关键信息
  const { retrievedContext, referenceAnswerStatements } = evaluateData;

  // 逐个分析每个关键信息是否可归因于给定的上下文
  const allRes = [];
  const newReferenceAnswerStatements = [...referenceAnswerStatements];
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
${retrievedContext.join('\n')}

回答：

`;
    const llmRes = await evaluateModel.invoke(prompt);
    const data = formatToJson(llmRes.content);
    allRes.push({
      score: data?.score ? +data.score : 0,
      statement,
    });
  }

  // 根据每个关键信息的得分，计算上下文召回率
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
 * 1. 逐个分析每个上下文片段是否与问题相关；
 * 2. 根据每个上下文片段的得分，计算上下文相关性。
 * @param evaluateData 评估数据
 * @param evaluateModel 评估模型
 * @returns
 */
export async function contextRelevanceEvaluator(evaluateData, evaluateModel) {
  // question：问题
  // retrievedContext：检索到的上下文
  const { question, retrievedContext } = evaluateData;

  // 逐个分析每个上下文片段是否与问题相关
  const allRes = [];
  const newRetrievedContext = [...retrievedContext];
  while (newRetrievedContext.length > 0) {
    const context = newRetrievedContext.shift();
    const prompt = `
你是一个语言专家，你的任务是确定上下文是否与问题有关。

说明：
1. 如果上下文与问题无关，则得分为0；
2. 如果上下文与问题有关，则得分为1；
3. 严格按以下JSON格式返回：{"score": "得分"}，不能输出其他无关内容。

问题：
${question}

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
  }

  // 根据每个上下文片段的得分，计算上下文相关性
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
 * 1. 将实际答案拆分成多个关键信息；
 * 2. 逐个分析每个关键信息是否可归因于给定的上下文；
 * 3. 根据每个关键信息的得分，计算答案忠实度。
 * @param evaluateData 评估数据
 * @param evaluateModel 评估模型
 * @returns
 */
export async function faithfulnessEvaluator(evaluateData, evaluateModel) {
  // retrievedContext：检索到的上下文
  // answerStatements：实际答案拆分出的多个关键信息
  const { retrievedContext, answerStatements } = evaluateData;

  // 逐个分析每个关键信息是否可归因于给定的上下文
  const allRes = [];
  const newAnswerStatements = [...answerStatements];
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
${retrievedContext.join('\n')}

回答：

`;
    const llmRes = await evaluateModel.invoke(prompt);
    const data = formatToJson(llmRes.content);
    allRes.push({
      score: data?.score ? +data.score : 0,
      statement,
    });
  }

  // 根据每个关键信息的得分，计算答案忠实度
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
 * 2. 逐个分析每个模拟问题是否与原问题相似；
 * 3. 根据每个模拟问题的得分，计算答案相关性。
 * @param evaluateData 评估数据
 * @param evaluateModel 评估模型
 * @returns
 */
export async function answerRelevanceEvaluator(evaluateData, evaluateModel) {
  // question：问题
  // simulationQuestions：根据实际答案推导出的多个模拟问题
  const { question, simulationQuestions } = evaluateData;

  // 逐个分析每个模拟问题是否与原问题相似
  const allRes = [];
  const newSimulationQuestions = [...simulationQuestions];
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
${question}

回答：

`;
    const llmRes = await evaluateModel.invoke(prompt);
    const data = formatToJson(llmRes.content);
    allRes.push({
      score: data?.score ? +data.score : 0,
      simulationQuestion,
    });
  }

  // 根据每个模拟问题的得分，计算答案相关性
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
 * 答案正确性评估器
 * 实现步骤：
 * 1. 将参考答案拆分成多个关键信息；
 * 2. 逐个分析每个关键信息是否可归因于给定的实际答案；
 * 3. 根据每个关键信息的得分，计算答案正确性。
 * @param evaluateData 评估数据
 * @param evaluateModel 评估模型
 * @returns
 */
export async function answerCorrectnessEvaluator(evaluateData, evaluateModel) {
  // answer:实际答案
  // referenceAnswerStatements：参考答案拆分出的多个关键信息
  const { answer, referenceAnswerStatements } = evaluateData;

  // 逐个分析每个关键信息是否可归因于给定的实际答案
  const allRes = [];
  const newReferenceAnswerStatements = [...referenceAnswerStatements];
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
${answer}

回答：

`;
    const llmRes = await evaluateModel.invoke(prompt);
    const data = formatToJson(llmRes.content);
    allRes.push({
      score: data?.score ? +data.score : 0,
      statement,
    });
  }

  // 根据每个关键信息的得分，计算答案正确性
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
