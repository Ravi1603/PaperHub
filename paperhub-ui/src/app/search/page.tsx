"use client";

import { useSearchParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { MagnifyingGlassIcon } from "@heroicons/react/24/outline";
import PaperCard from "@/components/PaperCard";
import RelatedPapersSidebar from "../../components/RelatedPapersSidebar";
import ChatWidget from "@/components/Chatbox/ChatBox";
import "@/components/RelatedBox.css";

type Paper = {
  title: string;
  abstract: string;
  id: string;
  source: string;
  citations: number;
};

export default function SearchPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const query = searchParams.get("q") || "";
  const [searchInput, setSearchInput] = useState(query);
  const [results, setResults] = useState<Paper[]>([]);
  const [loading, setLoading] = useState(false);
  const [related, setRelated] = useState<Paper[]>([]);
  const [relatedLoading, setRelatedLoading] = useState(false);

  useEffect(() => {
    const fetchResults = async () => {
      if (query.trim()) {
        setLoading(true);
        try {
          const response = await fetch("http://localhost:5000/recommend", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query }),
          });

          const rawData = await response.text();
          const data = JSON.parse(rawData);
          if (data.recommendations) {
            setResults(
              data.recommendations.map((rec: any) => ({
                title: rec.title,
                abstract: rec.abstract,
                id: rec.id,
                source: rec.source,
                citations: rec.citations,
              }))
            );
          } else {
            setResults([]);
          }
        } catch (error) {
          console.error("Error fetching recommendations:", error);
          setResults([]);
        } finally {
          setLoading(false);
        }
      } else {
        setResults([]);
      }
    };

    fetchResults();
  }, [query]);

  const handleSearch = () => {
    if (searchInput.trim()) {
      router.push(`/search?q=${encodeURIComponent(searchInput.trim())}`);
    }
  };

  const handleTitleClick = async (title: string, abstract: string) => {
    setRelatedLoading(true);
    const newQuery = `${title} ${abstract}`;
    try {
      const response = await fetch("http://localhost:5000/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: newQuery }),
      });

      const data = await response.json();
      const limited = (data.recommendations || []).slice(0, 4).map((rec: any) => ({
        title: rec.title,
        abstract: rec.abstract,
        id: rec.id,
        source: rec.source,
        citations: rec.citations,
      }));

      setRelated(limited);
    } catch (error) {
      console.error("Error fetching related papers:", error);
      setRelated([]);
    } finally {
      setRelatedLoading(false);
    }
  };

  const showRelated = related.length > 0;

  return (
    <main
      className={`grid min-h-screen bg-gradient-to-br from-[#bbd0ff] via-[#b8c0ff] to-[#ffd6ff] p-6 gap-6 ${
        showRelated ? "grid-cols-[360px_1fr_360px]" : "grid-cols-[1fr_360px]"
      }`}
    >
      {/* Related Papers (left column) */}
      {showRelated && (
        <RelatedPapersSidebar papers={related} loading={relatedLoading} />
      )}

      {/* Main Results (center column) */}
      <div className="space-y-6">
        <div className="flex items-center bg-white border border-gray-300 rounded-full px-6 py-2 shadow">
          <input
            type="text"
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
            placeholder="Search papers..."
            className="flex-grow bg-transparent outline-none text-gray-800"
          />
          <button
            onClick={handleSearch}
            className="ml-3 bg-[#3a7ca5] hover:bg-[#2f6b94] rounded-full w-10 h-10 flex items-center justify-center"
          >
            <MagnifyingGlassIcon className="w-5 h-5 text-white" />
          </button>
        </div>

        {loading ? (
          <p className="text-gray-600 text-center">Generating...</p>
        ) : results.length > 0 ? (
          <>
            <h1 className="text-2xl font-semibold text-[#2b4f7e]">
              Results for: <span className="text-gray-800">{query}</span>
            </h1>
            <div className="space-y-4">
              {results.map((paper, i) => (
                <div key={i} onClick={() => handleTitleClick(paper.title, paper.abstract)}>
                  <PaperCard paper={paper} />
                </div>
              ))}
            </div>
          </>
        ) : (
          <p className="text-gray-500 text-center">No results found for "{query}"</p>
        )}
      </div>

      {/* Chatbot (right column) */}
      <div className="sticky top-8 h-fit">
        <ChatWidget />
      </div>
    </main>
  );
}
