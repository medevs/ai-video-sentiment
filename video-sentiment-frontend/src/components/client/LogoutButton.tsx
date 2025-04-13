"use client";

import { signOut } from "next-auth/react";

type LogoutButtonProps = {
  className?: string;
};

export default function LogoutButton({ className = "" }: LogoutButtonProps) {
  const handleLogout = async () => {
    await signOut({ callbackUrl: "/" });
  };

  return (
    <button
      onClick={handleLogout}
      className={`rounded-md bg-gray-100 px-3 py-1.5 text-sm font-medium text-gray-700 transition hover:bg-gray-200 ${className}`}
    >
      Log out
    </button>
  );
}
