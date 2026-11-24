/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        industrial: {
          50: '#F8FAFC',
          100: '#F1F5F9',
          200: '#E2E8F0',
          300: '#CBD5E1',
          400: '#94A3B8',
          500: '#64748B',
          600: '#475569',
          700: '#334155',
          800: '#1E293B',
          900: '#0F172A',
        },
        safety: {
          green: '#16A34A',
          greenLight: '#DCFCE7',
          greenBorder: '#86EFAC',
          amber: '#D97706',
          amberLight: '#FEF3C7',
          amberBorder: '#FCD34D',
          red: '#DC2626',
          redLight: '#FEE2E2',
          redBorder: '#FCA5A5',
          blue: '#2563EB',
          blueLight: '#EFF6FF',
          blueBorder: '#93C5FD',
        }
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'Roboto Mono', 'monospace'],
      }
    },
  },
  plugins: [],
}
