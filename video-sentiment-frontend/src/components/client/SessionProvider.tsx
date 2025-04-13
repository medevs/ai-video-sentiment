"use client";

import { SessionProvider as NextAuthSessionProvider } from "next-auth/react";

type SessionProviderProps = {
  children: React.ReactNode;
};

export default function SessionProvider({ children }: SessionProviderProps) {
  return <NextAuthSessionProvider>{children}</NextAuthSessionProvider>;
}
