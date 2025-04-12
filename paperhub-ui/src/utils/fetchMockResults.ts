export async function fetchMockResults(query: string) {
    return [
      {
        title: "The Impact of Quantum Computing on Present Cryptography",
        authors: ["Vasileios Mavroeidis", "Kamer Vishi", "Mateusz Zych"],
        abstract: "This paper elucidates the implications of quantum computing...",
        link: "https://arxiv.org/abs/1234.5678",
        source: "arxiv.org",
      },
      {
        title: "Quantum-Resistant Blockchain Protocols",
        authors: ["C. Easttom"],
        abstract: "This review explores the use of post-quantum cryptographic techniques...",
        link: "https://ieeexplore.ieee.org/document/5678",
        source: "ieee.org",
      },
    ]
  }
  