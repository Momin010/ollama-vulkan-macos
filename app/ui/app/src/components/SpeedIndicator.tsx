interface SpeedIndicatorProps {
  /** Rate of the last completed response, tokens/second. */
  completed: number | null;
  /** Rate of a response currently streaming, tokens/second. */
  live: number | null;
}

/**
 * Shows generation speed just above the message input.
 *
 * Worth having on this hardware specifically: throughput falls by roughly 40%
 * within a minute of sustained load as the GPU throttles, so the rate you get
 * depends on how long you have been going. Without a readout that is invisible.
 */
export default function SpeedIndicator({
  completed,
  live,
}: SpeedIndicatorProps) {
  const streaming = live !== null;
  const value = streaming ? live : completed;

  if (value === null) return null;

  return (
    <div className="mx-auto w-full max-w-[768px] flex justify-end px-4 pb-1 select-none">
      <span
        className="text-xs text-neutral-400 dark:text-neutral-500 tabular-nums"
        title={
          streaming
            ? "Generation speed so far for this response"
            : "Generation speed of the last response"
        }
      >
        {streaming && (
          <span className="inline-block mr-1.5 h-1.5 w-1.5 rounded-full bg-green-500 align-middle animate-pulse" />
        )}
        {value.toFixed(1)} tok/s
      </span>
    </div>
  );
}
