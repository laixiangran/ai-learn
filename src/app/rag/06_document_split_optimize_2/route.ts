import {
  initOllamaLLM,
  initChroma,
  initOllamaEmbeddings,
  readJsonFile,
  saveJsonFile,
  formatToJson,
  readFile,
} from '@/app/utils';
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { DocxLoader } from '@langchain/community/document_loaders/fs/docx';
import {
  RecursiveCharacterTextSplitter,
  MarkdownTextSplitter,
} from 'langchain/text_splitter';

/**
 * doc/docx 文件解析
 * @param filePath
 * @returns
 */
async function loadDoc(filePath: string, fileExt: any) {
  const loader = new DocxLoader(filePath, {
    type: fileExt,
  });
  const docs = await loader.load();
  return docs;
}

/**
 * pdf文件解析
 * @param filePath
 * @returns
 */
async function loadPdf(filePath: string, splitPages: boolean = true) {
  const loader = new PDFLoader(filePath, {
    splitPages,
  });
  const docs = await loader.load();
  return docs;
}

/**
 * 文件切分：RecursiveCharacterTextSplitter
 * @param docs
 * @returns
 */
async function splitDocuments_v50(docs, config) {
  const textSplitter = new RecursiveCharacterTextSplitter(textSplitterParams);
  console.log('textSplitter: ', textSplitter);
  const documents = await textSplitter.splitDocuments(docs);
  documents.forEach((doc) => {
    doc.metadata.pageContent = doc.pageContent;
  });
  await saveJsonFile(
    JSON.stringify(documents),
    `src/app/data/split_data/rag_split_${config.category}.json`
  );
  return documents;
}

/**
 * 文件切分：MarkdownTextSplitter
 * @param docs
 * @returns
 */
async function splitDocuments_v51(docs, config) {
  const textSplitter = new MarkdownTextSplitter(textSplitterParams);
  console.log('textSplitter: ', textSplitter);
  const documents = await textSplitter.splitDocuments(docs);
  documents.forEach((doc) => {
    doc.metadata.pageContent = doc.pageContent;
  });
  await saveJsonFile(
    JSON.stringify(documents),
    `src/app/data/split_data/rag_split_${config.category}.json`
  );
  return documents;
}

/**
 * 文件切分：在 splitDocuments_v51 的基础上合并 chunkSize 小于 100 的文档块
 * @param docs
 * @returns
 */
async function splitDocuments_v52(docs, config) {
  const documents = await splitDocuments_v51(docs, config);
  for (let i = 0; i < documents.length; i++) {
    const doc = documents[i];
    // 长度小于100的文档
    if (doc.pageContent.length < 100) {
      // 如果不是最后一个元素，则合并到下一个元素
      if (i < documents.length - 1) {
        console.log('合并内容:', doc.pageContent, '到下一个');
        documents[i + 1].pageContent =
          doc.pageContent + '\n' + documents[i + 1].pageContent;
      }

      // 删除当前元素
      documents.splice(i, 1);
      i--; // 回退索引以适应数组缩短的情况
    }
  }
  await saveJsonFile(
    JSON.stringify(documents),
    `src/app/data/split_data/rag_split_${config.category}.json`
  );
  return documents;
}

/**
 * 文件切分：在 splitDocuments_v52 的基础上给每个文档块添加标题
 * @param docs
 * @returns
 */
async function splitDocuments_v53(docs, config) {
  const documents = await splitDocuments_v52(docs, config);
  // 获取每个文档块的标题并添加到元数据中
  for (let i = 0; i < documents.length; i++) {
    const doc = documents[i];
    const lines = doc.pageContent.split('\n');
    const headers = [
      {
        key: 'header5',
        value: '##### ',
      },
      {
        key: 'header4',
        value: '#### ',
      },
      {
        key: 'header3',
        value: '### ',
      },
      {
        key: 'header2',
        value: '## ',
      },
      {
        key: 'header1',
        value: '# ',
      },
    ];
    headers.forEach((header) => {
      const { key, value } = header;
      doc.metadata[key] = [];

      // 提取每一行的标题
      for (const line of lines) {
        if (line.startsWith(value)) {
          doc.metadata[key].push(
            line.replace(new RegExp(`/^${value}/`), '').trim()
          );
        }
      }
      // 如果当前文档没有对应标题，则取前一个文档的对应标题的第一个，并加入到当前文档中
      if (i > 0) {
        if (doc.metadata[key].length === 0) {
          const preHeader = documents[i - 1].metadata[key][0];
          if (preHeader) {
            doc.pageContent = preHeader + '\n\n' + doc.pageContent;
            doc.metadata[key] = [preHeader];
          }
        }
      }
    });
  }
  await saveJsonFile(
    JSON.stringify(documents),
    `src/app/data/split_data/rag_split_${config.category}.json`
  );
  return documents;
}

/**
 * 文件切分：在 splitDocuments_v53 的基础上给每个文档块添加模拟问题
 * @param docs
 * @param config
 * @returns
 */
async function splitDocuments_v54(docs, config) {
  const documents = await splitDocuments_v53(docs, config);
  let i = 0;
  while (i < documents.length) {
    const doc = documents[i];
    const simulationQuestions = await simulationQuestion(
      doc?.metadata.pageContent
    );
    doc.metadata.simulationQuestions = simulationQuestions;
    doc.pageContent = `${doc.metadata.simulationQuestions.join('\n')}\n\n${
      doc.pageContent
    }`;
    console.log(doc.pageContent);
    i++;
  }
  await saveJsonFile(
    JSON.stringify(documents),
    `src/app/data/split_data/rag_split_${config.category}.json`
  );
  return documents;
}

/**
 * 文件切分：在 splitDocuments_v53 的基础上将每个文档块的模拟问题单独向量化
 * @param docs
 * @param config
 * @returns
 */
async function splitDocuments_v55(docs, config) {
  const documents = await readJsonFile(
    'src/app/data/split_data/rag_split_v5.4.json'
  );
  const newDocuments = [];
  documents.forEach((doc) => {
    doc.metadata.category = config.category;
    newDocuments.push({
      ...doc,
      pageContent: doc.metadata.simulationQuestions.join('\n'),
    });
  });
  return newDocuments;
}

/**
 * 文件切分：在 splitDocuments_v53 的基础上将每个文档块的模拟问题单独向量化
 * @param docs
 * @param config
 * @returns
 */
async function splitDocuments_v56(docs, config) {
  const documents = await readJsonFile(
    'src/app/data/split_data/rag_split_v5.4.json'
  );
  const newDocuments = [];
  documents.forEach((doc, i) => {
    doc.pageContent = doc.metadata.pageContent;
    doc.metadata.category = `${config.category}-document`;
    doc.metadata.id = `doc-${i}`;
    const newDoc = {
      ...doc,
      metadata: {
        ...doc.metadata,
        category: `${config.category}-question`,
      },
      pageContent: doc.metadata.simulationQuestions.join('\n'),
    };
    newDocuments.push(newDoc);
  });
  return [...documents, ...newDocuments];
}

/**
 * @param texts
 * 初始化向量模型和向量数据库，并将文档存储到向量数据库
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
async function knowledgeConstruction(isAddDocuments = false) {
  const files = [
    {
      path: 'src/app/data/2024少儿编程教育行业发展趋势报告.md',
      category: 'v5.0',
      splitFunc: splitDocuments_v50,
    },
    {
      path: 'src/app/data/2024少儿编程教育行业发展趋势报告.md',
      category: 'v5.1',
      splitFunc: splitDocuments_v51,
    },
    {
      path: 'src/app/data/2024少儿编程教育行业发展趋势报告.md',
      category: 'v5.2',
      splitFunc: splitDocuments_v52,
    },
    {
      path: 'src/app/data/2024少儿编程教育行业发展趋势报告.md',
      category: 'v5.3',
      splitFunc: splitDocuments_v53,
    },
    // {
    //   path: 'src/app/data/2024少儿编程教育行业发展趋势报告.md',
    //   category: 'v5.4',
    //   splitFunc: splitDocuments_v54,
    // },
    // {
    //   path: 'src/app/data/2024少儿编程教育行业发展趋势报告.md',
    //   category: 'v5.5',
    //   splitFunc: splitDocuments_v55,
    // },
    // {
    //   path: 'src/app/data/2024少儿编程教育行业发展趋势报告.md',
    //   category: 'v5.6',
    //   splitFunc: splitDocuments_v56,
    // },
  ];
  const allDocuments = [];
  for (const file of files) {
    const { path, category, splitFunc } = file;
    let fileContent;
    if (path.includes('.pdf')) {
      fileContent = await loadPdf(path);
      fileContent.forEach((item) => {
        item.metadata.category = category;
      });
    } else {
      fileContent = await readFile(path);
      fileContent = [
        {
          pageContent: fileContent,
          metadata: {
            category: category,
          },
        },
      ];
    }
    const documents = await splitFunc(fileContent, file);
    console.log(
      JSON.stringify(
        documents.map((doc) => {
          return doc.pageContent.length;
        })
      )
    );
    allDocuments.push(documents);
    if (isAddDocuments) {
      await addDocuments(documents);
    }
  }
  return allDocuments;
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
 * 根据上下文推导出多个模拟问题
 * @param text 上下文
 * @param num 模拟问题数量
 * @returns
 */
async function simulationQuestion(text, num = 3) {
  const prompt = `
你是一个语言专家，你的任务是根据提供的上下文来生成${num}个用户可能问的问题。

说明：
1. 严格按以下JSON格式返回：["问题1", "问题2", ...]，不能输出其他无关内容。

上下文：
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
  const allDocs = [];
  const vectorFilter = {
    category: evaluateVersion,
  };
  while (allQuestions.length > 0) {
    const question = allQuestions.shift();
    if (evaluateVersion === 'v5.6') {
      const docs1 = await chromadb.similaritySearchWithScore(question, topK, {
        ...vectorFilter,
        category: `${evaluateVersion}-document`,
      });
      const docs2 = await chromadb.similaritySearchWithScore(question, topK, {
        ...vectorFilter,
        category: `${evaluateVersion}-question`,
      });
      docs2.forEach((doc2) => {
        const isExist = docs1.some((doc1) => {
          return doc1[0].metadata.id === doc2[0].metadata.id;
        });
        if (!isExist) {
          docs1.push(doc2);
        }
      });
      allDocs.push(...docs1);
    } else {
      const docs = await chromadb.similaritySearchWithScore(
        question,
        topK,
        vectorFilter
      );
      allDocs.push(...docs);
    }
  }

  console.log(allDocs.length, allDocs);

  const retrievedContext = allDocs.map((doc) => doc[0].metadata.pageContent);

  // 构造提示词
  const answerPrompt = `
你是负责回答问题的专家，请严格根据提供的上下文片段来回答问题。
如果你无法从提供的上下文中直接得出答案，你只需要回复 “我无法根据现有信息回答这个问题”，不用回复其他解释信息。

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
async function contextRecallEvaluator(evaluateData, isPrintReason = false) {
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
3. 严格按以下JSON格式返回：${
      isPrintReason ? '{"score":"得分","reason":"依据"}' : '{"score":"得分"}'
    }，不能输出其他无关内容。

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
      reason: data?.reason || '',
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
async function contextRelevanceEvaluator(evaluateData, isPrintReason = false) {
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
3. 严格按以下JSON格式返回：${
      isPrintReason ? '{"score":"得分","reason":"依据"}' : '{"score":"得分"}'
    }，不能输出其他无关内容。

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
      reason: data?.reason || '',
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
async function faithfulnessEvaluator(evaluateData, isPrintReason = false) {
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
  const allRes = [];
  const notAnswerStr = '我无法根据现有信息回答这个问题';
  if (!evaluateData.answer.includes(notAnswerStr)) {
    if (!evaluateData.answerStatements) {
      evaluateData.answerStatements = await statementSplit(evaluateData.answer);
    }
    // 逐个分析每个关键信息是否可归因于给定的上下文
    const newAnswerStatements = [...evaluateData.answerStatements];
    while (newAnswerStatements.length > 0) {
      const statement = newAnswerStatements.shift();
      const prompt = `
  你是一个语言专家，你的任务是分析关键信息是否可归因于给定的上下文。
  
  说明：
  1. 如果关键信息不能归因于上下文，则得分为0；
  2. 如果关键信息能够归因于上下文，则得分为1；
  3. 严格按以下JSON格式返回：${
    isPrintReason ? '{"score":"得分","reason":"依据"}' : '{"score":"得分"}'
  }，不能输出其他无关内容。
  
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
        reason: data?.reason || '',
        statement,
      });
      console.log('faithfulnessEvaluator: ', allRes);
    }
  } else {
    allRes.push({
      score: 0.8,
      reason: isPrintReason ? notAnswerStr : '',
      statement: notAnswerStr,
    });
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
async function answerRelevanceEvaluator(evaluateData, isPrintReason = false) {
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
  const allRes = [];
  const notAnswerStr = '我无法根据现有信息回答这个问题';
  if (!evaluateData.answer.includes(notAnswerStr)) {
    if (!evaluateData.simulationQuestions) {
      evaluateData.simulationQuestions = await simulationQuestion(
        evaluateData.answer
      );
    }
    // 逐个分析每个模拟问题是否与原问题相似
    const newSimulationQuestions = [...evaluateData.simulationQuestions];
    while (newSimulationQuestions.length > 0) {
      const simulationQuestion = newSimulationQuestions.shift();
      const prompt = `
你是一个语言专家，你的任务是分析模拟问题和实际问题是否相似。

说明：
1. 如果模拟问题与实际问题不相似，则得分为0；
2. 如果模拟问题与实际问题相似，则得分为1；
3. 严格按以下JSON格式返回：${
        isPrintReason ? '{"score":"得分","reason":"依据"}' : '{"score":"得分"}'
      }，不能输出其他无关内容。

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
        reason: data?.reason || '',
        simulationQuestion,
      });
      console.log('answerRelevanceEvaluator: ', allRes);
    }
  } else {
    allRes.push({
      score: 0.8,
      reason: isPrintReason ? notAnswerStr : '',
      simulationQuestion: [],
    });
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
async function answerCorrectnessEvaluator(evaluateData, isPrintReason = false) {
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
3. 严格按以下JSON格式返回：${
      isPrintReason ? '{"score":"得分","reason":"依据"}' : '{"score":"得分"}'
    }，不能输出其他无关内容。

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
      reason: data?.reason || '',
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
      const isPrintReason = false;
      const { evaluateData, evaluateResult } = await evaluator(
        data,
        isPrintReason
      );
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

async function startBathEvaluator() {
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
  return data;
}

// 基础配置，根据实际情况进行修改
const evaluateVersion: string = 'v5.1'; // 评估版本
const topK = 3;
const textSplitterParams = {
  chunkSize: 500, // 文本切分大小
  chunkOverlap: 50, // 文本切分重叠大小
};
const generateModel = initOllamaLLM('qwen2.5:14b'); // 生成 LLM 模型
const evaluateModel = initOllamaLLM('qwen2.5:14b'); // 评估 LLM 模型
const embeddingModel = initOllamaEmbeddings('nomic-embed-text'); // 向量模型
const collectionName = 'collection_rag_evaluator_0502'; // 向量数据集合名称
const chromadb = initChroma(collectionName, embeddingModel); // 向量数据库
let qaTestPath = `src/app/data/qa_test_10.json`; // 测试数据
const qaEvalPath = `src/app/data/rag_evaluate_${evaluateVersion}.json`; // 评估数据
// qaTestPath = qaEvalPath; // TODO 临时

/**
 * RAG 评估系统实现
 * @param request
 * @returns
 */
export async function GET(request: Request) {
  let data = {};

  // 1. 知识库构建
  data = await knowledgeConstruction(false);

  // 2. 指标评估
  // data = await startBathEvaluator();

  return Response.json({
    success: true,
    data,
  });
}
