"use client";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { MagnifyingGlassIcon, BookOpenIcon } from "@heroicons/react/24/outline";
import ChatWidget from "@/components/Chatbox/ChatBox"; // Optional

export default function HomePage() {
  const router = useRouter();
  const [search, setSearch] = useState("");

  const handleSearch = () => {
    if (search.trim()) {
      router.push(`/search?q=${encodeURIComponent(search.trim())}`);
    }
  };

  return (
    <div className="flex flex-col min-h-screen">
      <main className="flex-grow flex flex-col items-center justify-center px-4 text-center bg-gradient-to-br from-[#bbd0ff] via-[#b8c0ff] to-[#ffd6ff] text-black">
        {/* Logo + Title */}
        <div className="flex flex-col items-center gap-2 mb-10">
          <h1 className="text-5xl md:text-6xl font-bold text-[#2b4f7e] flex items-center gap-3">
            <BookOpenIcon className="w-10 h-10 text-[#2b4f7e]" />
            PaperHub
          </h1>
          <p className="text-lg md:text-xl text-[#375e78] font-medium tracking-wide">
            Discover research papers smarter & faster
          </p>
        </div>

        {/* Search Bar */}
        <div className="w-full px-6 md:px-10 max-w-3xl">
          <div className="flex items-center gap-3 bg-white shadow-lg border border-gray-200 rounded-full px-6 py-3 transition-all focus-within:ring-2 ring-[#3a7ca5]">
            <BookOpenIcon className="w-6 h-6 text-[#3a7ca5]" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder="Search papers, authors, or keywords..."
              className="flex-grow text-base md:text-lg text-gray-800 outline-none border-none bg-transparent placeholder-gray-400"
            />
            <button
              onClick={handleSearch}
              className="bg-[#3a7ca5] hover:bg-[#2f6b94] text-white rounded-full w-10 h-10 flex items-center justify-center transition-colors"
            >
              <MagnifyingGlassIcon className="w-5 h-5" />
            </button>
          </div>
        </div>
      </main>

      {/* Optional Chat Widget */}
      <ChatWidget />
    </div>
  );
}
