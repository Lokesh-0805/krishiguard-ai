// Decorative hero illustration: a stylised leaf being scanned, with attention "hot spots" over lesions.
// Purely illustrative — it contains no data or results.
export function LeafScan() {
  return (
    <div className="relative mx-auto aspect-square w-full max-w-md" aria-hidden>
      <div className="absolute inset-6 rounded-[2.5rem] bg-gradient-to-br from-secondary via-card to-accent/60 shadow-xl ring-1 ring-border" />
      <svg viewBox="0 0 400 400" className="absolute inset-0 size-full">
        <defs>
          <linearGradient id="leaf" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="oklch(0.72 0.15 140)" />
            <stop offset="1" stopColor="oklch(0.48 0.12 152)" />
          </linearGradient>
          <radialGradient id="hot" cx="0.5" cy="0.5" r="0.5">
            <stop offset="0" stopColor="oklch(0.75 0.19 45)" stopOpacity="0.85" />
            <stop offset="0.55" stopColor="oklch(0.85 0.17 90)" stopOpacity="0.35" />
            <stop offset="1" stopColor="oklch(0.85 0.17 90)" stopOpacity="0" />
          </radialGradient>
          <linearGradient id="beam" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="white" stopOpacity="0" />
            <stop offset="0.5" stopColor="white" stopOpacity="0.55" />
            <stop offset="1" stopColor="white" stopOpacity="0" />
          </linearGradient>
          <clipPath id="leafClip">
            <path d="M200 60 C 300 90, 340 190, 300 290 C 270 350, 200 360, 200 360 C 200 360, 130 350, 100 290 C 60 190, 100 90, 200 60 Z" />
          </clipPath>
        </defs>
        <g transform="rotate(-18 200 210)">
          <path d="M200 60 C 300 90, 340 190, 300 290 C 270 350, 200 360, 200 360 C 200 360, 130 350, 100 290 C 60 190, 100 90, 200 60 Z" fill="url(#leaf)" />
          <g stroke="oklch(0.86 0.08 140)" strokeOpacity="0.7" strokeWidth="3" fill="none" strokeLinecap="round">
            <path d="M200 70 L200 365" />
            {[110, 150, 190, 230, 270, 310].map((y, i) => (
              <g key={y}>
                <path d={`M200 ${y} Q ${240 + i * 4} ${y - 18}, ${270 - Math.abs(i - 2.5) * 10} ${y - 40}`} />
                <path d={`M200 ${y} Q ${160 - i * 4} ${y - 18}, ${130 + Math.abs(i - 2.5) * 10} ${y - 40}`} />
              </g>
            ))}
          </g>
          <g clipPath="url(#leafClip)">
            {[[250, 170, 18], [160, 230, 14], [235, 265, 22], [175, 150, 10]].map(([x, y, r]) => (
              <g key={`${x}-${y}`}>
                <circle cx={x} cy={y} r={r} fill="oklch(0.42 0.06 60)" />
                <circle cx={x} cy={y} r={r * 0.55} fill="oklch(0.3 0.04 50)" />
                <circle cx={x} cy={y} r={r + 7} fill="none" stroke="oklch(0.78 0.13 95)" strokeWidth="3" strokeOpacity="0.8" />
              </g>
            ))}
            {[[250, 170, 70], [235, 265, 80], [160, 230, 55]].map(([x, y, r]) => (
              <circle key={`h${x}`} cx={x} cy={y} r={r} fill="url(#hot)" className="animate-pulse" />
            ))}
            <rect x="40" y="-60" width="320" height="60" fill="url(#beam)">
              <animate attributeName="y" values="40;360;40" dur="4.5s" repeatCount="indefinite" />
            </rect>
          </g>
        </g>
        {[[52, 52], [348, 52], [52, 348], [348, 348]].map(([x, y], i) => (
          <path
            key={i}
            d={`M${x} ${y + (y < 200 ? 26 : -26)} L${x} ${y} L${x + (x < 200 ? 26 : -26)} ${y}`}
            stroke="oklch(0.47 0.115 152)"
            strokeWidth="4"
            fill="none"
            strokeLinecap="round"
          />
        ))}
      </svg>
    </div>
  );
}
