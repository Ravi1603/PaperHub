// components/Header.tsx
"use client"
import { useEffect, useState } from "react"
import Link from "next/link"
import { BookOpenIcon } from "@heroicons/react/24/outline"

export default function Header() {
  const [user, setUser] = useState<{ email: string } | null>(null)
  const [isMounted, setIsMounted] = useState(false)

  useEffect(() => {
    setIsMounted(true)
    const storedUser = localStorage.getItem("user")
    if (storedUser) {
      try {
        setUser(JSON.parse(storedUser))
      } catch (error) {
        console.error("Failed to parse user data:", error)
      }
    }
  }, [])

  const handleLogout = () => {
    localStorage.removeItem("user")
    setUser(null)
    window.location.href = "/" // Forces reload + logout
  }

  return (
    <header className="w-full px-8 py-4 flex justify-between items-center border-b bg-white">
      {/* 👈 Left side: Logo link */}
      <Link href="/" className="flex items-center gap-2 hover:opacity-80">
        <BookOpenIcon className="w-6 h-6 text-[#3a7ca5]" />
        <h1 className="text-xl font-bold text-[#3a7ca5]">PaperHub</h1>
      </Link>

      {/* 👉 Right side: Login or Sign Out */}
      {isMounted && (
        <>
          {user ? (
            <div className="flex items-center gap-4">
              <p className="text-sm text-gray-700">Welcome, {user.email}</p>
              <button
                onClick={handleLogout}
                className="text-blue-600 hover:underline text-sm font-medium"
              >
                Sign Out
              </button>
            </div>
          ) : (
            <Link
              href="/login"
              className="text-blue-600 hover:underline text-sm font-medium"
            >
              Log In
            </Link>
          )}
        </>
      )}
    </header>
  )
}