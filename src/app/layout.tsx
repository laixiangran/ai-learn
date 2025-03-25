import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'AI 开发实践',
  description: 'AI 开发实践',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang='zh-cn'>
      <body>{children}</body>
    </html>
  );
}
