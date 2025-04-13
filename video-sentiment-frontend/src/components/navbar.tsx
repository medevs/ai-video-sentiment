"use client";

import { useSession } from "next-auth/react";
import { useState, useEffect } from "react";
import Logo from "./client/Logo";
import NavLinks from "./client/NavLinks";

export default function Navbar() {
  const { data: session, status } = useSession();
  const [mounted, setMounted] = useState(false);
  
  // Only show auth-dependent UI after component has mounted on the client
  useEffect(() => {
    setMounted(true);
  }, []);

  // This helps prevent hydration errors by ensuring consistent rendering
  if (!mounted) {
    return (
      <nav className="flex h-16 items-center justify-between border-b border-gray-200 px-4 sm:px-10">
        <div className="flex items-center gap-3">
          <Logo />
        </div>
        <div className="flex items-center gap-4">
          <div className="h-8 w-20 animate-pulse rounded-md bg-gray-200"></div>
        </div>
      </nav>
    );
  }

  const isAuthenticated = status === "authenticated";

  return (
    <nav className="flex h-16 items-center justify-between border-b border-gray-200 px-4 sm:px-10">
      <div className="flex items-center gap-3">
        <Logo />
      </div>

      <div className="flex items-center gap-4">
        {status === "loading" ? (
          <div className="h-8 w-20 animate-pulse rounded-md bg-gray-200"></div>
        ) : isAuthenticated ? (
          <>
            <div className="hidden sm:block">
              <span className="mr-2 text-sm text-gray-600">
                Hello, {session?.user?.name ?? "User"}
              </span>
            </div>
            <NavLinks isAuthenticated={true} />
          </>
        ) : (
          <NavLinks isAuthenticated={false} />
        )}
      </div>
    </nav>
  );
}