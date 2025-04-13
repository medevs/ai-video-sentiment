"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import LogoutButton from "./LogoutButton";

type NavLinksProps = {
  isAuthenticated?: boolean;
};

export default function NavLinks({ isAuthenticated = false }: NavLinksProps) {
  const pathname = usePathname();

  return (
    <div className="flex items-center gap-4">
      {isAuthenticated ? (
        <>
          <Link
            href="/dashboard"
            className={`rounded-md px-3 py-1.5 text-sm font-medium transition ${
              pathname.startsWith("/dashboard")
                ? "bg-blue-50 text-blue-600"
                : "text-gray-700 hover:bg-gray-100 hover:text-blue-600"
            }`}
          >
            Dashboard
          </Link>
          <LogoutButton />
        </>
      ) : (
        <>
          {pathname !== "/signup" && (
            <Link
              href="/signup"
              className="rounded-md px-3 py-1.5 text-sm font-medium text-gray-700 transition hover:bg-gray-100 hover:text-blue-600"
            >
              Sign up
            </Link>
          )}
          {pathname !== "/login" && (
            <Link
              href="/login"
              className="rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white transition hover:bg-blue-700"
            >
              Log in
            </Link>
          )}
        </>
      )}
    </div>
  );
}
