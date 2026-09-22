import { ApiError } from "./api";

/** Do not expose transport jargon (or arbitrary backend error bodies) as UI copy. */
export function requestErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return error.status >= 500
      ? "The library couldn’t complete this request. Try again in a moment."
      : "The library couldn’t load these results. Check your search and filters, then retry.";
  }
  return "Couldn’t connect to the library. Check your connection, then retry.";
}
