import { useCallback, useRef, useState } from "react";
import type { ChatEventUnion } from "../api";

/**
 * Measures generation speed for the current response.
 *
 * The app's backend does not forward Ollama's eval_count/eval_duration to the
 * UI, so this times the stream instead: the clock starts at the first content
 * chunk and each subsequent chunk counts as a token. Ollama emits one token
 * per chunk, so the count is accurate in practice, though it is an estimate
 * rather than the server's own figure.
 *
 * Time-to-first-token is deliberately excluded. That interval covers prompt
 * processing, which on this hardware can be slower than generation, and
 * folding it in would make short replies look far worse than they are.
 */
export interface GenerationSpeed {
  /** Rate for the most recently completed response, tokens/second. */
  completed: number | null;
  /** Rate so far for a response still streaming, tokens/second. */
  live: number | null;
  /** Feed every chat event here. */
  observe: (event: ChatEventUnion) => void;
  /** Call when starting a new response so the previous rate is cleared. */
  reset: () => void;
}

export function useGenerationSpeed(): GenerationSpeed {
  const [completed, setCompleted] = useState<number | null>(null);
  const [live, setLive] = useState<number | null>(null);

  const startedAt = useRef<number | null>(null);
  const tokens = useRef(0);

  const finish = useCallback(() => {
    if (startedAt.current !== null) {
      const elapsed = (performance.now() - startedAt.current) / 1000;
      // One chunk gives no interval to measure across.
      if (elapsed > 0 && tokens.current > 1) {
        setCompleted(tokens.current / elapsed);
      }
    }
    startedAt.current = null;
    tokens.current = 0;
    setLive(null);
  }, []);

  const observe = useCallback(
    (event: ChatEventUnion) => {
      if (event.eventName === "done") {
        finish();
        return;
      }

      // Only content counts. Thinking, tool calls and download progress are
      // not generated tokens in the sense a user cares about here.
      if (event.eventName !== "chat") return;
      const content = (event as { content?: string }).content;
      if (!content) return;

      if (startedAt.current === null) {
        startedAt.current = performance.now();
        tokens.current = 0;
        return; // first chunk marks t=0, it is not yet an interval
      }

      tokens.current += 1;

      const elapsed = (performance.now() - startedAt.current) / 1000;
      // Settle briefly before showing a live figure, otherwise the first
      // reading swings wildly.
      if (elapsed > 0.4) {
        setLive(tokens.current / elapsed);
      }
    },
    [finish],
  );

  const reset = useCallback(() => {
    startedAt.current = null;
    tokens.current = 0;
    setLive(null);
    setCompleted(null);
  }, []);

  return { completed, live, observe, reset };
}
