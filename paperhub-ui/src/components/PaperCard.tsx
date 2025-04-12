import { BookmarkIcon } from "@heroicons/react/24/solid"
import { useState, useEffect } from "react"

type Paper = {
  title: string
  abstract: string
  link: string
  source: string
  citations: number
}

export default function PaperCard({ paper }: { paper: Paper }) {
  const [favorited, setFavorited] = useState(false)

  useEffect(() => {
    const saved = JSON.parse(localStorage.getItem("favorites") || "[]")
    setFavorited(saved.some((p: Paper) => p.title === paper.title))
  }, [paper.title])

  const toggleFavorite = () => {
    const saved = JSON.parse(localStorage.getItem("favorites") || "[]")
    const updated = favorited
      ? saved.filter((p: Paper) => p.title !== paper.title)
      : [...saved, paper]

    localStorage.setItem("favorites", JSON.stringify(updated))
    setFavorited(!favorited)
  }

  // Generate PDF link by replacing 'abs' with 'pdf'
  const pdfLink = paper.link ? paper.link.replace('/abs/', '/pdf/') : ''

  console.log("Paper link:", paper.link) // Log the original link
  console.log("PDF link:", pdfLink) // Log the generated PDF link

  return (
    <div className="bg-white rounded-lg shadow-md p-6 mb-6 space-y-2">
      <h2 className="text-lg font-semibold text-[#3a7ca5]">
        <a href={paper.link} target="_blank" rel="noopener noreferrer" className="hover:underline">
          {paper.title}
        </a>
      </h2>
      <p className="text-sm text-gray-800 line-clamp-3">
  <span className="font-semibold">Abstract:</span> {paper.abstract}
</p>



      <div className="flex justify-between items-center text-sm mt-3">
        <p className="text-gray-500">{paper.citations} Citations</p>
        <div className="flex items-center gap-4">
          {pdfLink && (
            <a href={pdfLink} target="_blank" rel="noopener noreferrer" className="text-blue-600 underline">
              [PDF] {paper.source}
            </a>
          )}
          <button onClick={toggleFavorite}>
            <BookmarkIcon className={`w-5 h-5 ${favorited ? "text-[#3a7ca5]" : "text-gray-400"}`} />
          </button>
        </div>
      </div>
    </div>
  )
}