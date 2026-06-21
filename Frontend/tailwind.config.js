/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans:    ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        // Stencilled / condensed — control-desk labels & gauges
        stencil: ['Oswald', 'Inter', 'sans-serif'],
        // Terminal / CRT readouts & licence plates
        mono:    ['"Share Tech Mono"', 'ui-monospace', 'monospace'],
        // Typewriter — paper challans & rubber stamps
        type:    ['"Special Elite"', 'Courier', 'monospace'],
        // Engraved brass — command-center headings
        serif:   ['"Playfair Display"', 'Georgia', 'serif'],
      },
      colors: {
        brass:    { DEFAULT: '#c9a227', light: '#f4d97b', dark: '#8a6d1b' },
        steel:    { DEFAULT: '#6b7280', light: '#9ca3af', dark: '#2b2f36' },
        leather:  { DEFAULT: '#5a3a22', light: '#7a4f2e', dark: '#2e1d11' },
        mahogany: { DEFAULT: '#3b1f17', light: '#5c3324', dark: '#1c0d08' },
        crt:      { green: '#39ff14', amber: '#ffb000', dim: '#0a1a0a' },
        ink:      { red: '#9b1c1c', blue: '#1e3a5f' },
        plate:    { yellow: '#f4c20d', white: '#e8e8e0' },
      },
      animation: {
        'fade-in':   'fadeIn 0.2s ease-out',
        'slide-up':  'slideUp 0.25s ease-out',
        'flicker':   'flicker 4s infinite',
        'led-pulse': 'ledPulse 1.6s ease-in-out infinite',
        'scan':      'scan 6s linear infinite',
        'stamp':     'stampIn 0.35s cubic-bezier(.2,1.4,.4,1) both',
        'needle':    'needle 0.6s cubic-bezier(.2,1.2,.3,1) both',
      },
      keyframes: {
        fadeIn:  { from: { opacity: '0' }, to: { opacity: '1' } },
        slideUp: { from: { transform: 'translateY(10px)', opacity: '0' }, to: { transform: 'translateY(0)', opacity: '1' } },
        flicker: {
          '0%,100%':   { opacity: '1' },
          '92%':       { opacity: '1' },
          '93%':       { opacity: '0.7' },
          '94%':       { opacity: '1' },
          '96%':       { opacity: '0.85' },
          '97%':       { opacity: '1' },
        },
        ledPulse: { '0%,100%': { opacity: '1' }, '50%': { opacity: '0.45' } },
        scan:     { from: { backgroundPosition: '0 0' }, to: { backgroundPosition: '0 100%' } },
        stampIn:  {
          '0%':   { transform: 'scale(2.4) rotate(-14deg)', opacity: '0' },
          '60%':  { opacity: '0.95' },
          '100%': { transform: 'scale(1) rotate(-9deg)', opacity: '0.9' },
        },
        needle:   { from: { transform: 'rotate(-90deg)' } },
      },
    },
  },
  plugins: [],
}
