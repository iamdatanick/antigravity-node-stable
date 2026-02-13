import { Routes, Route, Navigate } from "react-router";
import AppShell from "./components/layout/AppShell";
import ErrorBoundary from "./components/ErrorBoundary";
import Dashboard from "./pages/Dashboard";
import Chat from "./pages/Chat";
import HybridChat from "./pages/HybridChat";
import Logs from "./pages/Logs";
import Memory from "./pages/Memory";
import Query from "./pages/Query";
import Workflows from "./pages/Workflows";
import Budget from "./pages/Budget";
import Services from "./pages/Services";
import Settings from "./pages/Settings";

export default function App() {
  return (
    <ErrorBoundary fallbackTitle="Antigravity Node encountered an error">
      <Routes>
        <Route path="/hybrid" element={<HybridChat />} />
        <Route
          path="/*"
          element={
            <AppShell>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/chat" element={<Chat />} />
                <Route path="/logs" element={<Logs />} />
                <Route path="/memory" element={<Memory />} />
                <Route path="/query" element={<Query />} />
                <Route path="/workflows" element={<Workflows />} />
                <Route path="/budget" element={<Budget />} />
                <Route path="/services" element={<Services />} />
                <Route path="/settings" element={<Settings />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </AppShell>
          }
        />
      </Routes>
    </ErrorBoundary>
  );
}