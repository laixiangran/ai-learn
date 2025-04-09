'use client';

import { useState } from 'react';
import styles from './page.module.css';

export default function Home() {
  const [question, setQuestion] = useState('');
  const [message, setMessage] = useState<string>();
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setLoading(true);
    if (question.trim() === '') return;
    try {
      const res = await fetch(
        `http://localhost:3000/rag/01_simple_rag?question=${question}`
      );
      res.json().then((data) => {
        setMessage(data?.answer);
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.ragPage}>
      <main className={styles.ragMain}>
        <h1>RAG 系统开发实践</h1>
        <form onSubmit={handleSubmit}>
          <input
            type='text'
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder='请输入你的问题'
          />
          &nbsp;&nbsp;
          <button type='submit' disabled={loading}>
            {loading ? '正在加载...' : '发送'}
          </button>
        </form>
        <div>
          <strong>AI 回答：</strong>
          <p>{message}</p>
        </div>
      </main>
    </div>
  );
}
