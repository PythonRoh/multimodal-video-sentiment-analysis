import NextAuth from "next-auth";
import { authConfig } from "./config.edge";

export const { auth, handlers, signIn, signOut } = NextAuth(authConfig);
