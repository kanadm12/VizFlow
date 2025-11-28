/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        gray: {
          850: '#1a1a1a',
          900: '#0f0f0f',
          950: '#0a0a0a',
        }
      }
    },
  },
  plugins: [],
  darkMode: 'class',
}
