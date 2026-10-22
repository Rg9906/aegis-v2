import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource-variable/geist";
import "@fontsource-variable/geist-mono";
import "@fontsource/instrument-serif/400.css";
import "@fontsource/instrument-serif/400-italic.css";
import "./styles/tokens.css";
import "./styles/app.css";
import "./styles/sections.css";
import App from "./App";

// Apply the saved theme before first paint (falls back to dark)
try {
  const saved = localStorage.getItem("aegis-theme");
  if (saved === "light" || saved === "dark") document.documentElement.dataset.theme = saved;
} catch {
  /* storage blocked: default theme */
}
if (!document.documentElement.dataset.theme) document.documentElement.dataset.theme = "dark";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
