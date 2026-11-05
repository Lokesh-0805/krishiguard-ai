export function Logo({ className = "size-8" }: { className?: string }) {
  return (
    <svg viewBox="0 0 64 64" className={className} aria-hidden>
      <rect width="64" height="64" rx="16" className="fill-primary" />
      <path d="M18 44c0-15 11-26 30-28-1 19-12 30-27 30" className="fill-primary-foreground/85" />
      <path d="M18 46c6-8 13-15 22-21" strokeWidth="3" fill="none" strokeLinecap="round" className="stroke-primary" />
    </svg>
  );
}
