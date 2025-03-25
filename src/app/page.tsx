import styles from './page.module.css';

export default function Home() {
  return (
    <div className={styles.homePage}>
      <main className={styles.homeMain}>
        <h1>燃哥说前端</h1>
        <h2>跟着燃哥学AI，助你成为AI浪潮中的超级个体！</h2>
        <div>
          <a href='/rag'>RAG 系统开发实践</a>
        </div>
      </main>
    </div>
  );
}
