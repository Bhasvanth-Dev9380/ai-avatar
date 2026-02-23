import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Avatar — Voice Call",
  description: "Talk to your AI avatar in real-time",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
