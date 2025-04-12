"use client"

import { useSearchParams, useRouter } from "next/navigation"
import { useEffect, useState } from "react"
import { MagnifyingGlassIcon } from "@heroicons/react/24/outline"
import PaperCard from "@/components/PaperCard"

type Paper = {
  title: string
  abstract: string
  id: string         // <- changed from `link` to `id`
  source: string
  citations: number
}

export default function SearchPage() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const query = searchParams.get("q") || ""
  const [searchInput, setSearchInput] = useState(query)
  const [results, setResults] = useState<Paper[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const fetchResults = async () => {
      if (query.trim()) {
        setLoading(true)
        try {
          const response = await fetch("http://localhost:5000/recommend", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ query }),
          })

          const rawData = await response.text()
          console.log("Raw response:", rawData)
          const data = JSON.parse(rawData)

          if (data.recommendations) {
            setResults(data.recommendations.map((rec: any) => ({
              title: rec.title,
              abstract: rec.abstract,
              id: rec.id,                   // <- changed from `link`
              source: rec.source,
              citations: rec.citations,
            })))
          } else {
            setResults([])
          }
        } catch (error) {
          console.error("Error fetching recommendations:", error)
          setResults([])
        } finally {
          setLoading(false)
        }
      } else {
        setResults([])
      }
    }

    fetchResults()
  }, [query])

  const handleSearch = () => {
    if (searchInput.trim()) {
      router.push(`/search?q=${encodeURIComponent(searchInput.trim())}`)
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-[#bbd0ff] via-[#b8c0ff] to-[#ffd6ff] p-4">
      {/* Search bar */}
      <div className="w-full mb-6">
        <div className="flex items-center max-w-3xl mx-auto bg-white border border-gray-300 rounded-full px-6 py-2 shadow">
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
      </div>

      {/* Results */}
      <div className="max-w-3xl mx-auto">
        {loading ? (
          <p className="text-gray-600 text-center">Generating...</p>
        ) : results.length > 0 ? (
          <>
            <h1 className="text-2xl font-semibold text-[#2b4f7e] mb-4">
              Results for: <span className="text-gray-800">{query}</span>
            </h1>
            {results.map((paper, i) => (
              <PaperCard key={i} paper={paper} />
            ))}
          </>
        ) : (
          <p className="text-gray-500 text-center">No results found for "{query}"</p>
        )}
      </div>
    </main>
  )
}
