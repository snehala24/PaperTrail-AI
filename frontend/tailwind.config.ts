import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#eef3ff",
          100: "#dfe8ff",
          200: "#c5d6fd",
          300: "#9ebaf8",
          400: "#7196ef",
          500: "#4b74e2",
          600: "#3457d5",
          700: "#2b45b8",
          800: "#283c95",
          900: "#263675",
        },
        ink: {
          900: "#101828",
          800: "#1d2939",
          700: "#344054",
          600: "#475467",
          500: "#667085",
          400: "#98a2b3",
          300: "#d0d5dd",
          200: "#e4e7ec",
          100: "#f2f4f7",
          50: "#f8fafc",
        },
        canvas: "#f6f7f9",
      },
      fontFamily: {
        sans: [
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "SF Mono",
          "Cascadia Mono",
          "Consolas",
          "Liberation Mono",
          "monospace",
        ],
      },
      boxShadow: {
        "2xs": "0 1px 1px rgba(16, 24, 40, 0.04)",
        xs: "0 1px 2px rgba(16, 24, 40, 0.06)",
        sm: "0 1px 2px rgba(16, 24, 40, 0.05), 0 1px 3px rgba(16, 24, 40, 0.04)",
        md: "0 2px 6px -1px rgba(16, 24, 40, 0.07), 0 1px 3px rgba(16, 24, 40, 0.05)",
        lg: "0 8px 24px -6px rgba(16, 24, 40, 0.12), 0 2px 6px rgba(16, 24, 40, 0.05)",
        card: "0 1px 2px rgba(16, 24, 40, 0.04), 0 1px 3px rgba(16, 24, 40, 0.03)",
        "card-hover":
          "0 6px 20px -4px rgba(16, 24, 40, 0.10), 0 2px 6px rgba(16, 24, 40, 0.04)",
      },
      keyframes: {
        "fade-in": {
          from: { opacity: "0" },
          to: { opacity: "1" },
        },
        "rise-in": {
          from: { opacity: "0", transform: "translateY(6px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "fade-in": "fade-in 180ms ease-out both",
        "rise-in": "rise-in 220ms cubic-bezier(0.16, 1, 0.3, 1) both",
      },
    },
  },
  plugins: [],
};

export default config;
