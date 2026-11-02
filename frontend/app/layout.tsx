import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PaperTrail AI — Research Intelligence Workspace",
  description:
    "Multi-paper scientific research workspace with verifiable citations, cross-paper comparative reasoning, and fail-closed abstention.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full flex flex-col font-sans antialiased text-slate-900 bg-slate-50">
        {children}
      </body>
    </html>
  );
}
