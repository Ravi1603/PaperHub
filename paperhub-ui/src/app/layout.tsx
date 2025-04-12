"use client";
import React from "react";
import { Roboto } from "next/font/google";
import type { Metadata } from "next";
import Header from "@/components/header";
import ChatWidget from "@/components/Chatbox/ChatBox";
import  '@/components/Chatbox/ChatBox.css';
import "./globals.css";

const roboto = Roboto({
  subsets: ["latin"],
  weight: ["300", "400", "500", "700", "900"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "PaperHub",
  description: "AI-powered research paper search engine",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="shortcut icon" href="/favicon.png" type="image/png" />
      </head>
      <body className="flex flex-col min-h-screen">
        <Header />
        <div className="flex-grow overflow-y-auto">
          {children}
        </div>
        {/* ChatWidget is inserted globally so it will appear on every page */}
        <ChatWidget />
        <footer className="w-full text-center text-sm text-black py-4 bg-transparent">
  <div className="max-w-7xl mx-auto px-4">
    © {new Date().getFullYear()} PaperHub. Big Data Science Spring MS Capstone Project. Kennesaw State University
  </div>
</footer>




      </body>
    </html>
  );
}
