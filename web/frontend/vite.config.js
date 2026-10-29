import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import path from 'path';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { proxy: { '/api': 'https://ectrc.rafaew.tech/' } },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
