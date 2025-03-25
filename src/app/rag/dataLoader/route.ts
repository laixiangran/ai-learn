import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { initChroma } from '@/app/rag/utils';

export async function GET(request: Request) {
  console.log('request: ', request);
  const loader = new PDFLoader('public/example.pdf', { splitPages: false });
  const docs = await loader.load();
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const texts = await textSplitter.splitDocuments(docs);
  const chromadb = initChroma();
  const documents = await chromadb.addDocuments(texts);
  return Response.json(documents);
}
