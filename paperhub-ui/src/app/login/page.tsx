// app/login/page.tsx
"use client"
import { useState, useEffect } from "react"
import { useRouter} from "next/navigation"

export default function LoginPage() {
  const router = useRouter()

  useEffect(() => {
    const user = localStorage.getItem("user")
    if (user) {
      router.replace("/") // Skip login if already logged in
    }
  }, [])
  

  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState("")

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault()

    // TEMP: Fake user login
    if (email === "user@example.com" && password === "password") {
      localStorage.setItem("user", JSON.stringify({ email }))
      router.push("/")
    } else {
      setError("Invalid credentials.")
    }
  }

  return (
    <main className="min-h-screen flex items-center justify-center bg-gray-50 px-4">
      <form
        onSubmit={handleLogin}
        className="w-full max-w-sm bg-white p-8 rounded-xl shadow-lg space-y-6"
      >
        <h2 className="text-2xl font-bold text-center text-[#3a7ca5]">Log In to PaperHub</h2>

        {error && <p className="text-sm text-red-500 text-center">{error}</p>}

        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Email"
          required
          className="w-full p-3 border border-gray-300 rounded-md outline-none focus:ring-2 focus:ring-[#3a7ca5]"
        />

        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Password"
          required
          className="w-full p-3 border border-gray-300 rounded-md outline-none focus:ring-2 focus:ring-[#3a7ca5]"
        />

        <button
          type="submit"
          className="w-full bg-[#81c3d7] text-white font-semibold py-2 rounded-md hover:bg-[#6bb4cd] transition"
        >
          Log In
        </button>
      </form>
    </main>
  )
}
