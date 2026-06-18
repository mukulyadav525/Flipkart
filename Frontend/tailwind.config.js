/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        display: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
      colors: {
        // Civic palette — restrained, official, high-contrast
        surface: {
          DEFAULT: '#0f172a',   // slate-900 — primary background
          raised:  '#1e293b',   // slate-800 — card surface
          elevated:'#334155',   // slate-700 — hover, active, elevated
        },
        border: {
          DEFAULT: '#1e293b',   // slate-800
          muted:   '#334155',   // slate-700
        },
        // Violation severity
        violation: {
          critical: '#f87171',  // red-400
          high:     '#fb923c',  // orange-400
          medium:   '#fbbf24',  // amber-400
          low:      '#94a3b8',  // slate-400
        },
      },
      animation: {
        'fade-in': 'fadeIn 0.15s ease-out',
        'slide-up': 'slideUp 0.2s ease-out',
      },
      keyframes: {
        fadeIn:  { from: { opacity: '0' },               to: { opacity: '1' } },
        slideUp: { from: { transform: 'translateY(8px)', opacity: '0' }, to: { transform: 'translateY(0)', opacity: '1' } },
      },
    },
  },
  plugins: [],
}
