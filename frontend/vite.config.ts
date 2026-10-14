import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/ask': 'http://127.0.0.1:8000',
      '/schema': 'http://127.0.0.1:8000',
      '/explain': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
      '/.well-known': 'http://127.0.0.1:8000',
    },
  },
})
