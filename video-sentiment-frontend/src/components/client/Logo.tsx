"use client";

import Link from "next/link";

export default function Logo() {
  return (
    <Link href="/" className="flex items-center gap-2 hover:opacity-90">
      {/* Logo combining video and sentiment analysis */}
      <div className="flex h-8 w-8 items-center justify-center rounded-md bg-gradient-to-br from-blue-600 to-purple-600 text-white">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="currentColor"
          className="h-5 w-5"
        >
          {/* Play button + waveform */}
          <path d="M5.507 4.048A3 3 0 017.785 3h8.43a3 3 0 012.278 1.048l1.722 2.008A4.533 4.533 0 0019.5 6h-15c-.243 0-.482.02-.715.056l1.722-2.008z" />
          <path
            fillRule="evenodd"
            d="M1.5 10.5a3 3 0 013-3h15a3 3 0 110 6h-15a3 3 0 01-3-3zm15 0a.75.75 0 11-1.5 0 .75.75 0 011.5 0zm2.25.75a.75.75 0 100-1.5.75.75 0 000 1.5zM4.5 15a3 3 0 100 6h15a3 3 0 100-6h-15zm11.25 3.75a.75.75 0 100-1.5.75.75 0 000 1.5zM19.5 18a.75.75 0 11-1.5 0 .75.75 0 011.5 0z"
            clipRule="evenodd"
          />
        </svg>
      </div>
      <span className="text-lg font-semibold text-gray-900">VibeScan</span>
    </Link>
  );
}
