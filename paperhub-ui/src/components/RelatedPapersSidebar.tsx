"use client";

import PaperCard from "./PaperCard";

type Paper = {
  title: string;
  abstract: string;
  id: string;
  source: string;
  citations: number;
};

type Props = {
  papers: Paper[];
  loading: boolean;
};

export default function RelatedPapersSidebar({ papers, loading }: Props) {
  return (
    <aside className="w-[360px] bg-transparent p-4">
      <h2 className="text-lg font-semibold text-[#3a7ca5] mb-4 flex items-center gap-2">
        <span className="text-2xl">📘</span> Related Papers
      </h2>

      {loading ? (
        <p className="text-sm text-gray-600">Generating related papers...</p>
      ) : (
        <div className="space-y-4">
          {papers.map((paper, i) => (
            <a
              key={i}
              href={
                paper.id.includes("/abs/")
                  ? paper.id.replace("/abs/", "/pdf/") + ".pdf"
                  : paper.id
              }
              target="_blank"
              rel="noopener noreferrer"
              className="block"
            >
              <PaperCard paper={paper} />
            </a>
          ))}
        </div>
      )}
    </aside>
  );
}